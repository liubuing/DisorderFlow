#!/usr/bin/env python3
"""Quick OC test: V11 SeqConf only (Phase 1 + save for Phase 2).

Usage:
  python run_oc_v11seqconf_p1.py          # Phase 1: BFN design
  conda run -n bfn-msa python run_oc_v11seqconf_p2.py   # Phase 2: AF2
  python run_oc_v11seqconf_p3.py          # Phase 3: analysis
"""

import os, sys, json, time, argparse, copy

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, '.')
sys.path.insert(0, 'modules')

import numpy as np
import yaml

# ── Config ────────────────────────────────────────────────────────────────
V11_SEQCONF_CKPT = 'logs/bfn_v11_seqconf_xpu_2026_06_19__23_01_18/checkpoints/best.pt'

DESIGN_CONFIG = {
    'scaffold_pdb': 'data/misfolding_targets/5IMK.pdb',
    'scaffold_chain': 'B',
    'target_pdb': 'data/misfolding_targets/2NAO_model1_A_1-42.pdb',
    'target_chain': 'A',
    'cdr_spec': 'B:26-33,51-58,97-113',
    'context_chains': None,  # Complex mode
    'n_samples': 10,
    'design_top_n': 5,
}

OUTPUT_PHASE1 = 'oc_validation_results/oc_v11seqconf_phase1.json'
OUTPUT_PHASE2 = 'oc_validation_results/oc_v11seqconf_phase2.json'


def auto_detect_device():
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return 'xpu'
    return 'cpu'


def run_phase1(output_path=None, device=None):
    """Run BFN design with V11 SeqConf checkpoint."""
    if output_path is None:
        output_path = OUTPUT_PHASE1
    if device is None:
        device = auto_detect_device()

    from bfn_loader import run_bfn_design, load_bfn, has_disorder_head
    from idp_antibody_design import (
        _extract_sequence_from_pdb, _parse_cdr_ranges, graft_cdrs,
    )

    # ── Setup ──
    scaffold_pdb = DESIGN_CONFIG['scaffold_pdb']
    target_pdb = DESIGN_CONFIG['target_pdb']
    n_samples = DESIGN_CONFIG['n_samples']
    cdr_spec = DESIGN_CONFIG['cdr_spec']

    epi_seq = _extract_sequence_from_pdb(target_pdb, DESIGN_CONFIG['target_chain'])
    scaffold_seq = _extract_sequence_from_pdb(scaffold_pdb, DESIGN_CONFIG['scaffold_chain'])
    cdr_ranges = _parse_cdr_ranges(cdr_spec)

    # Save & set checkpoint
    cfg_path = os.path.join(PROJECT_ROOT, 'app_config.yaml')
    with open(cfg_path) as f:
        app_cfg = yaml.safe_load(f)
    orig_ckpt = app_cfg['models']['bfn']['checkpoint']
    app_cfg['models']['bfn']['checkpoint'] = V11_SEQCONF_CKPT
    with open(cfg_path, 'w') as f:
        yaml.dump(app_cfg, f, default_flow_style=False)

    results = {
        'config': DESIGN_CONFIG,
        'af2_config': {
            'num_recycle': 3, 'use_jax': True,
            'colabfold_exe': 'colabfold_batch', 'timeout_per_design': 1800,
        },
        'epi_seq': epi_seq,
        'scaffold_seq': scaffold_seq,
        'cdr_ranges': [[s, e, l] for s, e, l in cdr_ranges],
        'models': {},
    }

    try:
        # Force reload
        import bfn_loader
        bfn_loader._bfn_model = None
        bfn_loader._bfn_config = None

        print(f'[Phase 1] Loading V11 SeqConf model: {V11_SEQCONF_CKPT}')
        model, config = load_bfn(device)
        _has_disorder = has_disorder_head(model)

        t0 = time.time()
        designs = run_bfn_design(
            scaffold_pdb, cdr_spec,
            num_samples=n_samples, stochastic=True,
            context_chains=DESIGN_CONFIG['context_chains'],
            device=device,
        )
        elapsed = time.time() - t0

        bfn_plddts = [d['plddt'] for d in designs]
        bfn_iptms = [d['iptm'] for d in designs]
        bfn_ppls = [d.get('ppl', 0) for d in designs]
        bfn_entropies = [d.get('entropy', 0) for d in designs]

        print(f'  Generated {len(designs)} designs in {elapsed:.0f}s')
        print(f'  BFN pLDDT mean={np.mean(bfn_plddts):.4f} std={np.std(bfn_plddts):.6f}')
        print(f'  BFN ipTM  mean={np.mean(bfn_iptms):.4f} std={np.std(bfn_iptms):.6f}')
        print(f'  BFN PPL   mean={np.mean(bfn_ppls):.1f} std={np.std(bfn_ppls):.1f}')
        print(f'  BFN Entropy mean={np.mean(bfn_entropies):.4f}')
        print(f'  Disorder head: {_has_disorder}')

        # Graft CDRs
        af2_input = []
        for i, d in enumerate(designs[:DESIGN_CONFIG['design_top_n']]):
            full_ab, mutations = graft_cdrs(scaffold_seq, d['sequence'], cdr_ranges)
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
            print(f'  [{i+1}] pLDDT={d["plddt"]:.4f} ipTM={d["iptm"]:.4f} '
                  f'PPL={d.get("ppl", 0):.1f} E={d.get("entropy", 0):.3f} '
                  f'| {d["sequence"][:35]}...')

        results['models']['v11_seqconf'] = {
            'checkpoint': V11_SEQCONF_CKPT,
            'label': 'V11 SeqConf (sequence-aware bypass)',
            'desc': 'Entropy+max_prob bypass on confidence heads',
            'n_designs': len(designs),
            'bfn_plddt_mean': float(np.mean(bfn_plddts)),
            'bfn_plddt_std': float(np.std(bfn_plddts)),
            'bfn_iptm_mean': float(np.mean(bfn_iptms)),
            'bfn_iptm_std': float(np.std(bfn_iptms)),
            'af2_input': af2_input,
        }

    finally:
        # Restore original checkpoint
        app_cfg['models']['bfn']['checkpoint'] = orig_ckpt
        with open(cfg_path, 'w') as f:
            yaml.dump(app_cfg, f, default_flow_style=False)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\n  Phase 1 output saved to: {output_path}')
    print(f'  Next: conda run -n bfn-msa python run_oc_v11seqconf_p2.py')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output', type=str, default=OUTPUT_PHASE1)
    args = parser.parse_args()
    run_phase1(output_path=args.output, device=args.device)
