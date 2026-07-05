#!/usr/bin/env python3
"""Overconfidence Validation — BFN Self-Confidence vs AF2 Ground Truth.

Compares BFN's self-assessed confidence (pLDDT, ipTM) against independent
AF2 multimer predictions to measure overconfidence calibration.

Workflow:
  Phase 1: BFN CDR design with each checkpoint → intermediate JSON
  Phase 2: AF2 multimer validation on each design → enriched JSON
  Phase 3: Overconfidence ratio analysis → report

Usage:
  # Full pipeline (Phase 1 + 2 + 3)
  python validate_overconfidence.py --all

  # Phase 1 only (BFN design, GPU required)
  python validate_overconfidence.py --phase1

  # Phase 2 + 3 only (AF2 validation + analysis, needs Phase 1 JSON)
  python validate_overconfidence.py --phase2

  # Phase 3 only (analysis, runs on CPU)
  python validate_overconfidence.py --phase3

  # Custom comparison
  python validate_overconfidence.py --all --samples 20 --models phase3,phase4,idp_combined

Checkpoints compared (3-way):
  phase3       : no negative samples, disorder head baseline
  phase4       : 40% negative samples (expected overcorrection)
  idp_combined : fully frozen backbone + encoder dropout, best val metrics
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.resolve()

CHECKPOINTS = {
    'phase3': {
        'path': 'logs/bfn_phase3_disorder_2026_05_31__14_52_47/checkpoints/best.pt',
        'label': 'Phase 3 (baseline, no neg)',
        'desc': 'Disorder head baseline, freeze backbone + 2 encoder layers',
    },
    'phase4': {
        'path': 'logs/bfn_phase4_negative_xpu_2026_05_31__18_27_56/checkpoints/best.pt',
        'label': 'Phase 4 (40% negative)',
        'desc': '40% scrambled negative samples, forced low confidence',
    },
    'phase5': {
        'path': 'logs/bfn_phase5_calibrate_xpu_2026_05_31__21_40_45/checkpoints/best.pt',
        'label': 'Phase 5 (30% neg, calibrated)',
        'desc': '30% negative samples with calibration tuning',
    },
    'idp_combined': {
        'path': 'logs/bfn_idp_combined_cuda_2026_06_02__13_48_09/checkpoints/best.pt',
        'label': 'IDP Combined (best val)',
        'desc': 'Frozen backbone, encoder dropout 0.05, quality filtered',
    },
    'v11_seqconf': {
        'path': 'logs/bfn_v11_seqconf_xpu_2026_06_19__23_01_18/checkpoints/best.pt',
        'label': 'V11 SeqConf (sequence-aware bypass)',
        'desc': 'Entropy+max_prob bypass heads, trained 2000 iters',
    },
}

DESIGN_CONFIG = {
    'scaffold_pdb': 'data/misfolding_targets/5IMK.pdb',
    'scaffold_chain': 'B',
    'target_pdb': 'data/misfolding_targets/2NAO_model1_A_1-42.pdb',
    'target_chain': 'A',
    'cdr_spec': 'B:26-33,51-58,97-113',
    'context_chains': None,  # None = Complex mode (epitope visible)
    'n_samples': 10,
    'design_top_n': 5,  # top-N designs to send to AF2
}

AF2_CONFIG = {
    'num_recycle': 3,
    'use_jax': True,  # JAX-native runner; False = colabfold CLI
    'colabfold_exe': 'colabfold_batch',
    'timeout_per_design': 1800,
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def update_app_config(ckpt_path):
    """Temporarily update app_config.yaml to point to a specific checkpoint."""
    import yaml
    cfg_path = PROJECT_ROOT / 'app_config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return cfg, cfg['models']['bfn']['checkpoint']


def set_app_checkpoint(ckpt_path):
    """Update the checkpoint path in app_config.yaml."""
    import yaml
    cfg_path = PROJECT_ROOT / 'app_config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg['models']['bfn']['checkpoint'] = str(ckpt_path)
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)


def verify_checkpoints(models):
    """Verify that all specified checkpoint files exist."""
    missing = []
    for name in models:
        ckpt = PROJECT_ROOT / CHECKPOINTS[name]['path']
        if not ckpt.exists():
            missing.append((name, CHECKPOINTS[name]['path']))
    return missing


# ── Phase 1: BFN Design ──────────────────────────────────────────────────────

def run_phase1(models, output_path, device='cuda'):
    """Run BFN CDR design with each checkpoint. Saves intermediate JSON."""
    import numpy as np
    import yaml

    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / 'modules'))

    from bfn_loader import run_bfn_design, load_bfn, has_disorder_head
    from idp_antibody_design import (
        _extract_sequence_from_pdb,
        _parse_cdr_ranges,
        graft_cdrs,
    )

    scaffold_pdb = str(PROJECT_ROOT / DESIGN_CONFIG['scaffold_pdb'])
    target_pdb = str(PROJECT_ROOT / DESIGN_CONFIG['target_pdb'])
    n_samples = DESIGN_CONFIG['n_samples']
    cdr_spec = DESIGN_CONFIG['cdr_spec']

    # Extract sequences for later AF2 validation
    epi_seq = _extract_sequence_from_pdb(target_pdb, DESIGN_CONFIG['target_chain'])
    scaffold_seq = _extract_sequence_from_pdb(
        scaffold_pdb, DESIGN_CONFIG['scaffold_chain']
    )
    cdr_ranges = _parse_cdr_ranges(cdr_spec)

    # Save original checkpoint
    _, orig_ckpt = update_app_config(None)

    results = {
        'config': DESIGN_CONFIG,
        'af2_config': AF2_CONFIG,
        'epi_seq': epi_seq,
        'scaffold_seq': scaffold_seq,
        'cdr_ranges': [[s, e, l] for s, e, l in cdr_ranges],
        'models': {},
    }

    for name in models:
        ckpt_info = CHECKPOINTS[name]
        ckpt_path = PROJECT_ROOT / ckpt_info['path']

        print(f"\n{'=' * 70}")
        print(f"  Phase 1: {ckpt_info['label']}")
        print(f"  Checkpoint: {ckpt_info['path']}")
        print(f"{'=' * 70}")

        set_app_checkpoint(ckpt_path)

        try:
            # Force reload model with new checkpoint
            import bfn_loader
            bfn_loader._bfn_model = None
            bfn_loader._bfn_config = None
            model, config = load_bfn(device)

            t0 = time.time()
            designs = run_bfn_design(
                scaffold_pdb, cdr_spec,
                num_samples=n_samples,
                stochastic=True,
                context_chains=DESIGN_CONFIG['context_chains'],
                device=device,
            )
            elapsed = time.time() - t0

            bfn_plddts = [d['plddt'] for d in designs]
            bfn_iptms = [d['iptm'] for d in designs]

            print(f"  Generated {len(designs)} designs in {elapsed:.0f}s")
            print(f"  BFN pLDDT: {np.mean(bfn_plddts):.4f} "
                  f"+/- {np.std(bfn_plddts):.4f}")
            print(f"  BFN ipTM:  {np.mean(bfn_iptms):.4f} "
                  f"+/- {np.std(bfn_iptms):.4f}")

            # Graft CDRs for AF2 input
            af2_input = []
            for i, d in enumerate(designs[:DESIGN_CONFIG['design_top_n']]):
                full_ab, mutations = graft_cdrs(
                    scaffold_seq, d['sequence'], cdr_ranges
                )
                af2_input.append({
                    'rank': i + 1,
                    'sequence': d['sequence'],
                    'full_ab': full_ab,
                    'bfn_plddt': d['plddt'],
                    'bfn_iptm': d['iptm'],
                    'bfn_ppl': d.get('ppl', 0),
                    'bfn_pae': d.get('pae', 0),
                    'bfn_entropy': d.get('entropy', 0),
                    'disorder_score': d.get('disorder_score'),
                    'mutations': len(mutations),
                })
                print(f"  [{i+1}] BFN pLDDT={d['plddt']:.4f} "
                      f"ipTM={d['iptm']:.4f} PPL={d.get('ppl', 0):.1f} "
                      f"| {len(mutations)} muts | {d['sequence'][:30]}...")

            results['models'][name] = {
                'checkpoint': ckpt_info['path'],
                'label': ckpt_info['label'],
                'desc': ckpt_info['desc'],
                'n_designs': len(designs),
                'bfn_plddt_mean': float(np.mean(bfn_plddts)),
                'bfn_plddt_std': float(np.std(bfn_plddts)),
                'bfn_iptm_mean': float(np.mean(bfn_iptms)),
                'bfn_iptm_std': float(np.std(bfn_iptms)),
                'af2_input': af2_input,
            }

        finally:
            # Restore original checkpoint
            set_app_checkpoint(orig_ckpt)

    # Save intermediate results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 1 results saved to: {output_path}")

    return results


# ── Phase 2: AF2 Validation ──────────────────────────────────────────────────

def run_phase2(input_path, output_path):
    """Run AF2 multimer on all designs from Phase 1. Enriches JSON."""
    with open(input_path) as f:
        data = json.load(f)

    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / 'modules'))

    from af2_jax_runner import run_multimer_prediction

    epi_seq = data['epi_seq']
    models = data['models']
    af2_config = data.get('af2_config', AF2_CONFIG)

    print(f"\n{'=' * 70}")
    print(f"  Phase 2: AF2 Multimer Validation")
    print(f"  Epitope: {len(epi_seq)} AA")
    print(f"  Recycle: {af2_config.get('num_recycle', 3)}")
    print(f"{'=' * 70}")

    for name, model_data in models.items():
        ckpt_info = CHECKPOINTS.get(name, {})
        print(f"\n--- {ckpt_info.get('label', name)} ---")

        af2_input = model_data['af2_input']
        af2_results = []

        for item in af2_input:
            full_ab = item['full_ab']
            print(f"  [{item['rank']}] {len(full_ab)}+{len(epi_seq)} AA ... ",
                  end='', flush=True)
            t0 = time.time()
            r = run_multimer_prediction(
                full_ab, epi_seq,
                num_recycle=af2_config.get('num_recycle', 3),
            )
            elapsed = time.time() - t0

            if r['success']:
                af2_results.append({
                    **item,
                    'af2_plddt': r['plddt'],
                    'af2_iptm': r['iptm'],
                    'af2_ptm': r['ptm'],
                    'af2_max_pae': r['max_pae'],
                    'af2_interface_pae': r.get('interface_pae'),
                    'af2_elapsed': elapsed,
                })
                print(f"pLDDT={r['plddt']:.3f} ipTM={r['iptm']:.3f} "
                      f"({elapsed:.0f}s)")
            else:
                af2_results.append({**item, 'af2_error': r.get('error', '?')})
                print(f"FAILED: {r.get('error', '?')[:80]}")

        # Compute per-model stats
        import numpy as np
        ratios_plddt = []
        ratios_iptm = []
        for item in af2_results:
            if 'af2_plddt' in item and item['bfn_plddt'] > 0:
                ratios_plddt.append(
                    item['bfn_plddt'] / max(item['af2_plddt'], 0.001)
                )
            if 'af2_iptm' in item and item['bfn_iptm'] > 0:
                ratios_iptm.append(
                    item['bfn_iptm'] / max(item['af2_iptm'], 0.001)
                )

        valid_results = [r for r in af2_results if 'af2_plddt' in r]

        model_data['af2_results'] = af2_results
        model_data['af2_plddt_mean'] = float(
            np.mean([r['af2_plddt'] for r in valid_results])
        ) if valid_results else None
        model_data['af2_iptm_mean'] = float(
            np.mean([r['af2_iptm'] for r in valid_results])
        ) if valid_results else None
        model_data['oc_ratio_plddt_mean'] = (
            float(np.mean(ratios_plddt)) if ratios_plddt else None
        )
        model_data['oc_ratio_plddt_std'] = (
            float(np.std(ratios_plddt)) if ratios_plddt else None
        )
        model_data['oc_ratio_iptm_mean'] = (
            float(np.mean(ratios_iptm)) if ratios_iptm else None
        )
        model_data['oc_ratio_iptm_std'] = (
            float(np.std(ratios_iptm)) if ratios_iptm else None
        )
        model_data['af2_success_count'] = len(valid_results)

        if valid_results:
            print(f"  AF2 pLDDT mean: {model_data['af2_plddt_mean']:.4f}")
            print(f"  AF2 ipTM mean:  {model_data['af2_iptm_mean']:.4f}")
            print(f"  OC ratio pLDDT: {model_data['oc_ratio_plddt_mean']:.2f}x "
                  f"(BFN/AF2)")
            print(f"  OC ratio ipTM:  {model_data['oc_ratio_iptm_mean']:.2f}x "
                  f"(BFN/AF2)")

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\n  Phase 2 results saved to: {output_path}")

    return data


# ── Phase 3: Analysis & Report ───────────────────────────────────────────────

def run_phase3(input_path):
    """Analyze overconfidence ratios and produce a comparison report."""
    with open(input_path) as f:
        data = json.load(f)

    models = data['models']
    if not models:
        print("No model data found.")
        return

    model_names = list(models.keys())
    print(f"\n{'=' * 80}")
    print(f"  Phase 3: Overconfidence Analysis Report")
    print(f"{'=' * 80}")

    # Check which phase this data came from
    has_af2 = any(
        'af2_results' in m for m in models.values()
    )

    if not has_af2:
        print("\n  ⚠ Phase 2 not yet run — showing BFN self-confidence only.")
        print(f"\n  {'Model':<25} {'BFN pLDDT':<14} {'BFN ipTM':<14}")
        print(f"  {'─' * 53}")
        for name in model_names:
            m = models[name]
            print(f"  {m.get('label', name):<25} "
                  f"{m['bfn_plddt_mean']:<14.4f} "
                  f"{m['bfn_iptm_mean']:<14.4f}")
        print(f"\n  Run --phase2 to get AF2 validation and OC ratios.")
        return

    # Full AF2 comparison
    print(f"\n  BFN Self-Confidence vs AF2 Ground Truth")
    print(f"  {'─' * 70}")

    # Table header
    header = (
        f"  {'Model':<22} {'BFN pLDDT':>10} {'AF2 pLDDT':>10} "
        f"{'OC Ratio':>10} {'BFN ipTM':>10} {'AF2 ipTM':>10} "
        f"{'OC Ratio':>10}"
    )
    print(header)
    print(f"  {'─' * 74}")

    winner = None
    best_oc = float('inf')

    for name in model_names:
        m = models[name]
        af2_ok = m.get('af2_success_count', 0) > 0

        if af2_ok:
            oc_plddt = m.get('oc_ratio_plddt_mean', float('nan'))
            oc_iptm = m.get('oc_ratio_iptm_mean', float('nan'))

            bfn_p = m['bfn_plddt_mean']
            af2_p = m.get('af2_plddt_mean', 0)
            bfn_i = m['bfn_iptm_mean']
            af2_i = m.get('af2_iptm_mean', 0)

            print(f"  {m.get('label', name):<22} "
                  f"{bfn_p:>10.4f} {af2_p:>10.4f} {oc_plddt:>9.2f}x "
                  f"{bfn_i:>10.4f} {af2_i:>10.4f} {oc_iptm:>9.2f}x")

            # Best = closest to 1.0 (perfect calibration)
            oc_avg = abs(oc_plddt - 1.0) + abs(oc_iptm - 1.0)
            if oc_avg < best_oc:
                best_oc = oc_avg
                winner = name
        else:
            print(f"  {m.get('label', name):<22} {'(no AF2 data)':>55}")

    # Winner
    if winner:
        w = models[winner]
        print(f"\n  ✓ Best calibrated: {w.get('label', winner)}")
        print(f"    OC pLDDT: {w['oc_ratio_plddt_mean']:.2f}x   "
              f"OC ipTM: {w['oc_ratio_iptm_mean']:.2f}x   "
              f"(1.0x = perfect)")

    # Pairwise comparison
    if len(model_names) >= 2:
        print(f"\n  Pairwise Comparison:")
        for i, a in enumerate(model_names):
            for b in model_names[i + 1:]:
                ma, mb = models[a], models[b]
                if (ma.get('oc_ratio_plddt_mean') and
                        mb.get('oc_ratio_plddt_mean')):
                    delta_plddt = (mb['oc_ratio_plddt_mean'] -
                                   ma['oc_ratio_plddt_mean'])
                    delta_iptm = (mb['oc_ratio_iptm_mean'] -
                                  ma['oc_ratio_iptm_mean'])
                    pct_plddt = (delta_plddt / ma['oc_ratio_plddt_mean']
                                 * 100) if ma['oc_ratio_plddt_mean'] else 0
                    pct_iptm = (delta_iptm / ma['oc_ratio_iptm_mean']
                                * 100) if ma['oc_ratio_iptm_mean'] else 0
                    print(f"  {b} vs {a}:")
                    print(f"    OC pLDDT: {ma['oc_ratio_plddt_mean']:.2f}x → "
                          f"{mb['oc_ratio_plddt_mean']:.2f}x  "
                          f"({delta_plddt:+.2f}x, {pct_plddt:+.1f}%)")
                    print(f"    OC ipTM:  {ma['oc_ratio_iptm_mean']:.2f}x → "
                          f"{mb['oc_ratio_iptm_mean']:.2f}x  "
                          f"({delta_iptm:+.2f}x, {pct_iptm:+.1f}%)")

    # Per-design detail
    print(f"\n  Per-Design Detail (top design per model):")
    for name in model_names:
        m = models[name]
        af2_results = m.get('af2_results', [])
        if af2_results:
            best_d = af2_results[0]
            if 'af2_plddt' in best_d:
                bfn_p = best_d['bfn_plddt']
                af2_p = best_d['af2_plddt']
                oc = bfn_p / max(af2_p, 0.001)
                print(f"  [{name}] BFN={bfn_p:.3f} AF2={af2_p:.3f} "
                      f"OC={oc:.2f}x | CDR: {best_d['sequence'][:40]}...")

    print(f"\n{'=' * 80}")
    print(f"  Report complete.")
    print(f"{'=' * 80}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Overconfidence Validation: BFN vs AF2 calibration'
    )
    parser.add_argument('--all', action='store_true',
                        help='Run full pipeline (Phase 1 + 2 + 3)')
    parser.add_argument('--phase1', action='store_true',
                        help='Phase 1: BFN design only')
    parser.add_argument('--phase2', action='store_true',
                        help='Phase 2: AF2 validation only')
    parser.add_argument('--phase3', action='store_true',
                        help='Phase 3: Analysis report only')
    parser.add_argument('--models', type=str,
                        default='phase3,phase4,idp_combined',
                        help='Comma-separated model list to compare')
    parser.add_argument('--samples', type=int, default=10,
                        help='Number of BFN design samples per model')
    parser.add_argument('--af2-recycle', type=int,
                        default=AF2_CONFIG['num_recycle'],
                        help='AF2 recycling steps')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Torch device for BFN (cuda/xpu/cpu)')
    parser.add_argument('--output-dir', type=str,
                        default=str(PROJECT_ROOT / 'oc_validation_results'),
                        help='Output directory for results')
    parser.add_argument('--no-af2-jax', action='store_true',
                        help='Use colabfold CLI instead of JAX runner')

    args = parser.parse_args()

    # Validate models
    models = [m.strip() for m in args.models.split(',') if m.strip()]
    for m in models:
        if m not in CHECKPOINTS:
            print(f"ERROR: Unknown model '{m}'. Choices: {list(CHECKPOINTS)}")
            sys.exit(1)

    missing = verify_checkpoints(models)
    if missing:
        print("ERROR: Missing checkpoints:")
        for name, path in missing:
            print(f"  {name}: {path}")
        sys.exit(1)

    # Update global configs
    DESIGN_CONFIG['n_samples'] = args.samples
    AF2_CONFIG['num_recycle'] = args.af2_recycle
    AF2_CONFIG['use_jax'] = not args.no_af2_jax

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase1_path = output_dir / 'oc_phase1_designs.json'
    phase2_path = output_dir / 'oc_phase2_af2_validated.json'

    # Determine what to run
    run_all = args.all
    run_p1 = args.phase1 or run_all
    run_p2 = args.phase2 or run_all
    run_p3 = args.phase3 or run_all

    # Default: if no flags, run Phase 3 on existing data
    if not (run_p1 or run_p2 or run_p3):
        if phase2_path.exists():
            run_p3 = True
            print("No flags specified. Running Phase 3 on existing results.")
        elif phase1_path.exists():
            run_p2 = True
            run_p3 = True
            print("No flags specified. Resuming from Phase 2.")
        else:
            run_p1 = True
            run_p2 = True
            run_p3 = True
            print("No flags specified. Running full pipeline.")

    os.chdir(str(PROJECT_ROOT))

    # Phase 1
    if run_p1:
        run_phase1(models, str(phase1_path), device=args.device)
    else:
        if not phase1_path.exists() and (run_p2 or run_p3):
            print(f"Phase 1 results not found: {phase1_path}")
            print("Run --phase1 first, or --all for full pipeline.")
            sys.exit(1)

    # Phase 2
    if run_p2:
        p1_path = phase1_path if phase1_path.exists() else phase1_path
        run_phase2(str(p1_path), str(phase2_path))
    else:
        if run_p3 and not phase2_path.exists():
            print(f"Phase 2 results not found: {phase2_path}")
            print("Run --phase2 first, or --all for full pipeline.")
            print("(Can also run --phase3 on Phase 1 data for BFN-only analysis.)")
            # Fall back to Phase 1 data for Phase 3
            if phase1_path.exists():
                phase2_path = phase1_path

    # Phase 3
    if run_p3:
        p2_path = phase2_path if phase2_path.exists() else phase1_path
        run_phase3(str(p2_path))


if __name__ == '__main__':
    main()
