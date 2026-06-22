#!/usr/bin/env python
"""Unified Design Pipeline — AF2-validated antibody CDR design.

Replaces BFN self-reported confidence with AF2 ground-truth validation.
BFN generates designs (current env), AF2 validates them (bfn-msa conda env).

Phase 1 (this env):  BFN design → save designs JSON
Phase 2 (bfn-msa):   conda run -n bfn-msa python af2_batch_validate.py
Phase 3 (this env):  Load AF2 scores, rank by AF2 ipTM, report

Usage:
  python design_pipeline_af2.py --scaffold_chain B --samples 10 --af2-recycle 1
  python design_pipeline_af2.py --scaffold_chain B --fixbb --samples 5
  python design_pipeline_af2.py --scaffold_chain B --compare --samples 5
"""

import os, sys, json, time, subprocess, argparse, shutil

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')
sys.path.insert(0, 'modules')

import numpy as np

from bfn_loader import run_bfn_design
from idp_antibody_design import _extract_sequence_from_pdb, _parse_cdr_ranges, graft_cdrs

# ---- Config ----
SCAFFOLD_PDB = 'data/misfolding_targets/5IMK.pdb'
TARGET_PDB = 'data/misfolding_targets/2NAO_model1_A_1-42.pdb'
TARGET_CHAIN = 'A'
DEFAULT_CDR = '26-33,51-58,97-113'
BFN_MSA_CONDA = 'bfn-msa'
CONDA_EXE = '/home/liubuzi/miniforge3/bin/conda'
AF2_VALIDATE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'af2_batch_validate.py')


def auto_detect_device():
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return 'xpu'
    return 'cpu'


def run_phase2_af2(input_json_path, output_json_path, conda_env='bfn-msa'):
    """Run AF2 validation in bfn-msa conda environment.

    Returns True if successful, False otherwise.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(script_dir, 'af2_batch_validate.py')

    cmd = [
        CONDA_EXE, 'run', '-n', conda_env,
        'python', script, input_json_path, output_json_path
    ]

    print(f'\n{"="*80}')
    print(f'  Phase 2: AF2 Validation (conda env: {conda_env})')
    print(f'{"="*80}')
    print(f'  Command: conda run -n {conda_env} python af2_batch_validate.py ...')
    print()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        print(result.stdout)
        if result.stderr:
            # conda run produces stderr for activation messages — only show errors
            stderr_lines = result.stderr.strip().split('\n')
            error_lines = [l for l in stderr_lines if 'Error' in l or 'Traceback' in l or 'Exception' in l]
            if error_lines:
                print('STDERR:', '\n'.join(error_lines[-20:]))

        if result.returncode != 0:
            print(f'\nAF2 validation FAILED (exit code {result.returncode})')
            if not error_lines:
                print('STDERR:', result.stderr[-500:])
            return False

        if not os.path.exists(output_json_path):
            print(f'\nAF2 output file not found: {output_json_path}')
            return False

        return True

    except subprocess.TimeoutExpired:
        print('\nAF2 validation TIMED OUT (1 hour limit)')
        return False
    except FileNotFoundError:
        print('\nconda not found. Make sure conda is installed and in PATH.')
        return False


def parse_cdr_spec(cdr_arg, scaffold_chain):
    if cdr_arg:
        if ':' in cdr_arg:
            return cdr_arg
        return f'{scaffold_chain}:{cdr_arg}'
    return f'{scaffold_chain}:{DEFAULT_CDR}'


def main():
    parser = argparse.ArgumentParser(
        description='Unified Antibody Design — AF2-Validated')

    parser.add_argument('--target', default=TARGET_PDB)
    parser.add_argument('--target_chain', default=TARGET_CHAIN)
    parser.add_argument('--scaffold', default=SCAFFOLD_PDB)
    parser.add_argument('--scaffold_chain', required=True,
                        help='Scaffold chain (e.g., B)')
    parser.add_argument('--cdr', default=None,
                        help=f'CDR ranges (default: {DEFAULT_CDR})')

    parser.add_argument('--samples', type=int, default=10,
                        help='Number of BFN design samples')
    parser.add_argument('--fixbb', action='store_true',
                        help='FixBB mode (default: Complex mode)')
    parser.add_argument('--compare', action='store_true',
                        help='Run BOTH FixBB and Complex, compare AF2 scores')
    parser.add_argument('--deterministic', action='store_true')

    parser.add_argument('--af2-recycle', type=int, default=3,
                        help='AF2 recycling steps (default: 3)')
    parser.add_argument('--af2-top-n', type=int, default=5,
                        help='Top-N designs to validate with AF2 (default: 5, rest BFN-only)')
    parser.add_argument('--af2-all', action='store_true',
                        help='Validate ALL designs with AF2 (slow)')
    parser.add_argument('--af2-data-dir', default=None)

    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--skip-af2', action='store_true',
                        help='Skip AF2 validation (BFN-only, for debugging)')

    args = parser.parse_args()

    device = args.device or auto_detect_device()
    cdr_spec_str = parse_cdr_spec(args.cdr, args.scaffold_chain)
    mode = 'fixbb' if args.fixbb else 'complex'

    if args.output_dir is None:
        tag = f'{mode}{"_compare" if args.compare else ""}'
        args.output_dir = f'design_results/af2_{tag}_{int(time.time())}'
    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 80)
    print('  Unified Antibody Design — AF2-Validated Pipeline')
    print('=' * 80)
    print(f'  Target:   {args.target} (chain {args.target_chain})')
    print(f'  Scaffold: {args.scaffold} (chain {args.scaffold_chain})')
    print(f'  CDRs:     {cdr_spec_str}')
    print(f'  Mode:     {mode.upper()}')
    print(f'  Samples:  {args.samples}')
    print(f'  Device:   {device}')
    print(f'  Output:   {args.output_dir}')
    print()

    epi_seq = _extract_sequence_from_pdb(args.target, args.target_chain)
    context_chains = [] if args.fixbb else None

    # ── Phase 1: BFN Design ──
    print('=' * 80)
    print('  Phase 1: BFN Design')
    print('=' * 80)

    modes_to_run = [('fixbb', 'FixBB', []), ('complex', 'Complex', None)] if args.compare else [(mode, mode.upper(), context_chains)]

    all_phase1 = {}

    for m, label, ctx in modes_to_run:
        print(f'\n--- {label} Mode ---')
        t0 = time.time()
        designs = run_bfn_design(
            args.scaffold, cdr_spec_str, num_samples=args.samples,
            stochastic=not args.deterministic, context_chains=ctx, device=device)
        elapsed = time.time() - t0

        bfn_plddts = [d['plddt'] for d in designs]
        bfn_iptms = [d['iptm'] for d in designs]

        print(f'  Generated {len(designs)} designs in {elapsed:.0f}s')
        print(f'  BFN pLDDT: {np.mean(bfn_plddts):.4f} +/- {np.std(bfn_plddts):.4f}')
        print(f'  BFN ipTM:  {np.mean(bfn_iptms):.4f} +/- {np.std(bfn_iptms):.4f}')

        # Rank by BFN ipTM (for comparison later)
        ranked_bfn = sorted(designs, key=lambda d: d['iptm'], reverse=True)
        for i, d in enumerate(ranked_bfn):
            d['rank_bfn'] = i + 1

        all_phase1[m] = {
            'designs': designs,
            'bfn_plddt_mean': float(np.mean(bfn_plddts)),
            'bfn_iptm_mean': float(np.mean(bfn_iptms)),
            'bfn_plddt_std': float(np.std(bfn_plddts)),
            'bfn_iptm_std': float(np.std(bfn_iptms)),
            'elapsed': elapsed,
        }

        # Show top designs
        print(f'\n  {"─"*60}')
        print(f'  Top-5 by BFN ipTM:')
        for i, d in enumerate(ranked_bfn[:5]):
            print(f'  [{i+1}] BFN pLDDT={d["plddt"]:.4f} ipTM={d["iptm"]:.4f} '
                  f'PPL={d.get("ppl",0):.1f} | CDR={d["sequence"][:30]}...')

    # Save Phase 1 results
    phase1_path = os.path.join(args.output_dir, 'phase1_bfn_designs.json')
    with open(phase1_path, 'w') as f:
        json.dump({
            'config': {
                'target': args.target, 'target_chain': args.target_chain,
                'scaffold': args.scaffold, 'scaffold_chain': args.scaffold_chain,
                'cdr_spec': cdr_spec_str, 'mode': mode,
                'samples': args.samples, 'deterministic': args.deterministic,
            },
            'epi_seq': epi_seq,
            'modes': all_phase1,
        }, f, indent=2)
    print(f'\nPhase 1 saved to: {phase1_path}')

    # ── Phase 2: AF2 Validation ──
    if args.skip_af2:
        print('\nSkipping AF2 validation (--skip-af2). Done.')
        return

    # Determine which designs to validate
    af2_input = {}
    for m, label, ctx in modes_to_run:
        designs = all_phase1[m]['designs']
        ranked = sorted(designs, key=lambda d: d['iptm'], reverse=True)
        n_validate = len(ranked) if args.af2_all else min(args.af2_top_n, len(ranked))
        af2_designs = ranked[:n_validate]

        # Graft CDRs into full antibody for AF2 (done here in venv; avoids torch dep in bfn-msa)
        scaffold_seq = _extract_sequence_from_pdb(args.scaffold, args.scaffold_chain)
        cdr_ranges = _parse_cdr_ranges(cdr_spec_str)
        af2_designs_prepped = []
        for d in af2_designs:
            full_ab, _ = graft_cdrs(scaffold_seq, d['sequence'], cdr_ranges)
            af2_designs_prepped.append({
                'rank_bfn': d['rank_bfn'], 'sequence': d['sequence'],
                'full_ab': full_ab,
            })
        af2_input[m] = {
            'designs': af2_designs_prepped,
            'label': label,
            'n_validate': n_validate,
        }

    all_af2_results = {}

    for m, info in af2_input.items():
        label = info['label']
        n_val = info['n_validate']
        print(f'\nPhase 2 [{label}]: validating top-{n_val} designs...')

        # Prepare input JSON for AF2 validator
        tmp_input = os.path.join(args.output_dir, f'af2_input_{m}.json')
        tmp_output = os.path.join(args.output_dir, f'af2_output_{m}.json')

        with open(tmp_input, 'w') as f:
            json.dump({
                'epi_seq': epi_seq,
                'scaffold_pdb': args.scaffold,
                'scaffold_chain': args.scaffold_chain,
                'cdr_spec': cdr_spec_str,
                'designs': info['designs'],
                'num_recycle': args.af2_recycle,
                'data_dir': args.af2_data_dir,
            }, f)

        success = run_phase2_af2(tmp_input, tmp_output, BFN_MSA_CONDA)
        if success:
            with open(tmp_output) as f:
                all_af2_results[m] = json.load(f)
        else:
            print(f'  AF2 validation failed for {label} mode.')
            all_af2_results[m] = None

    # ── Phase 3: Analysis & Ranking ──
    print(f'\n{"="*80}')
    print(f'  Phase 3: AF2-Based Ranking & Analysis')
    print(f'{"="*80}')

    final_results = {}

    for m, label, ctx in modes_to_run:
        print(f'\n--- {label} ---')

        bfn_data = all_phase1[m]
        af2_data = all_af2_results.get(m)

        if af2_data is None or not af2_data.get('results'):
            print('  No AF2 results available. Using BFN ranking.')
            final_results[m] = {
                'ranked_by': 'bfn',
                'designs': sorted(bfn_data['designs'], key=lambda d: d['iptm'], reverse=True),
                'bfn_plddt_mean': bfn_data['bfn_plddt_mean'],
                'bfn_iptm_mean': bfn_data['bfn_iptm_mean'],
            }
            continue

        af2_results = af2_data['results']
        success_results = [r for r in af2_results if r.get('success')]
        summary = af2_data.get('summary', {})

        # Merge AF2 scores back into designs
        af2_map = {r['sequence']: r for r in success_results}
        for d in bfn_data['designs']:
            if d['sequence'] in af2_map:
                d.update({k: v for k, v in af2_map[d['sequence']].items()
                         if k not in ('sequence', 'rank_bfn')})

        # Rank by AF2 ipTM (primary), fall back to AF2 pLDDT
        validated = [d for d in bfn_data['designs'] if 'af2_iptm' in d]
        validated_af2 = sorted(validated, key=lambda d: d['af2_iptm'], reverse=True)
        for i, d in enumerate(validated_af2):
            d['rank_af2'] = i + 1

        # Rank change analysis
        rank_shifts = []
        for d in validated:
            shift = d['rank_bfn'] - d.get('rank_af2', d['rank_bfn'])
            rank_shifts.append(shift)

        print(f'  AF2-validated: {len(validated)}/{len(bfn_data["designs"])} designs')
        if summary:
            print(f'  AF2 pLDDT mean: {summary.get("af2_plddt_mean", "N/A"):.4f}')
            print(f'  AF2 ipTM  mean: {summary.get("af2_iptm_mean", "N/A"):.4f}')

        # Overconfidence ratios
        if 'af2_plddt_mean' in summary:
            oc_plddt = bfn_data['bfn_plddt_mean'] / max(summary['af2_plddt_mean'], 0.001)
            oc_iptm = bfn_data['bfn_iptm_mean'] / max(summary['af2_iptm_mean'], 0.001)
            print(f'  BFN/AF2 pLDDT ratio: {oc_plddt:.2f}x (overconfidence)')
            print(f'  BFN/AF2 ipTM  ratio: {oc_iptm:.2f}x (overconfidence)')

        print(f'\n  Rank correlation (BFN vs AF2):')
        print(f'    Mean rank shift: {np.mean(np.abs(rank_shifts)):.1f} positions')
        print(f'    Max rank shift:  {np.max(np.abs(rank_shifts)):.0f} positions')

        # Top-5 by AF2
        print(f'\n  {"─"*60}')
        print(f'  Top-5 by AF2 ipTM:')
        for i, d in enumerate(validated_af2[:5]):
            bfn_r = d['rank_bfn']
            shift = bfn_r - (i + 1)
            shift_str = f'↑{shift}' if shift > 0 else f'↓{abs(shift)}' if shift < 0 else '='
            print(f'  [{i+1}] AF2 ipTM={d["af2_iptm"]:.4f} pLDDT={d["af2_plddt"]:.4f} '
                  f'(BFN rank #{bfn_r} {shift_str}) | CDR={d["sequence"][:30]}...')

        # Was BFN's #1 also AF2's #1?
        bfn_top = sorted(bfn_data['designs'], key=lambda d: d['iptm'], reverse=True)[0]
        af2_top = validated_af2[0] if validated_af2 else None
        if af2_top and bfn_top['sequence'] == af2_top['sequence']:
            print(f'\n  BFN and AF2 agree on #1 design.')
        elif af2_top:
            print(f'\n  BFN #1 → AF2 rank #{af2_top.get("rank_af2", "?")}')
            print(f'  AF2 #1 → BFN rank #{af2_top["rank_bfn"]}')

        final_results[m] = {
            'ranked_by': 'af2',
            'designs': validated_af2 + [d for d in bfn_data['designs']
                                        if 'af2_iptm' not in d],
            'bfn_plddt_mean': bfn_data['bfn_plddt_mean'],
            'bfn_iptm_mean': bfn_data['bfn_iptm_mean'],
            'af2_summary': summary,
            'rank_shifts_mean': float(np.mean(np.abs(rank_shifts))),
            'rank_shifts_max': int(np.max(np.abs(rank_shifts))),
            'overconfidence_plddt': oc_plddt if 'af2_plddt_mean' in summary else None,
            'overconfidence_iptm': oc_iptm if 'af2_iptm_mean' in summary else None,
        }

    # ── Compare mode summary ──
    if args.compare and 'fixbb' in final_results and 'complex' in final_results:
        print(f'\n{"="*80}')
        print(f'  FixBB vs Complex — AF2-Validated Comparison')
        print(f'{"="*80}')
        for metric in ['af2_iptm_mean', 'af2_plddt_mean']:
            fb_val = final_results['fixbb'].get('af2_summary', {}).get(metric)
            cb_val = final_results['complex'].get('af2_summary', {}).get(metric)
            if fb_val is not None and cb_val is not None:
                delta = cb_val - fb_val
                better = 'Complex' if delta > 0 else 'FixBB' if delta < 0 else 'Tie'
                print(f'  {metric:<20} FixBB={fb_val:.4f}  Complex={cb_val:.4f}  '
                      f'Delta={delta:+.4f}  ({better} better)')

    # ── Save final results ──
    final_path = os.path.join(args.output_dir, 'final_results.json')
    with open(final_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f'\n{"="*80}')
    print(f'  Pipeline complete.')
    print(f'  Results saved to: {args.output_dir}/')
    print(f'{"="*80}')


if __name__ == '__main__':
    main()
