#!/usr/bin/env python
"""Unified Antibody Design CLI — BFN Phase 3 + Disorder Head.

Three-stage pipeline:
  1. Disorder Analysis — identify ordered epitope regions on target
  2. BFN CDR Design — Complex mode (default) or FixBB for comparison
  3. AF2 Multimer Validation — JAX-native (default) or colabfold

Usage:
  # Complex mode (default): antigen visible during design
  python run_design.py --scaffold_chain B

  # FixBB mode: scaffold only, no antigen context
  python run_design.py --scaffold_chain B --fixbb

  # Full pipeline with AF2 validation
  python run_design.py --scaffold_chain B --af2 --samples 20

  # Compare FixBB vs Complex (runs both, reports ipTM gap)
  python run_design.py --scaffold_chain B --compare

  # Disorder analysis only
  python run_design.py --analyze-only
"""

import os, sys, json, time, argparse, tempfile
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))

from design_wrappers import (
    create_bfn_design_fn,
    create_jax_af2_validator,
    create_physics_scorer_safe,
)
from iterative_refiner import IterativeRefiner, format_refinement_report
from cascade_filter import apply_cascade_af2

AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'
DEFAULT_CDR = '26-33,51-58,97-113'


def parse_cdr_spec(cdr_str, scaffold_chain):
    """Parse CDR spec string or use default for scaffold chain."""
    if cdr_str:
        if ':' in cdr_str:
            return cdr_str
        return f"{scaffold_chain}:{cdr_str}"
    return f"{scaffold_chain}:{DEFAULT_CDR}"


def auto_detect_device():
    """Auto-detect available torch device."""
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return 'xpu'
    except ImportError:
        pass
    return 'cpu'


def run_disorder_analysis(target_pdb, target_chain, device, threshold):
    """Stage 1: Predict disorder and find ordered segments."""
    from idp_disorder_analysis import predict_disorder, find_ordered_segments, format_disorder_report

    from bfn_loader import load_bfn
    model, config = load_bfn(device)

    result = predict_disorder(model, config, target_pdb, target_chain, device)
    segments = find_ordered_segments(
        result['disorder_scores'], result['residue_ids'], threshold=threshold)

    report = format_disorder_report(
        result['disorder_scores'], result['sequence'],
        result['residue_ids'], threshold)

    return {
        'disorder_scores': result['disorder_scores'],
        'sequence': result['sequence'],
        'residue_ids': result['residue_ids'],
        'segments': segments,
        'report': report,
    }


def run_single_design(scaffold_pdb, cdr_spec, num_samples, stochastic,
                      device, context_chains):
    """Stage 2: Run BFN CDR design."""
    from bfn_loader import run_bfn_design

    t0 = time.time()
    results = run_bfn_design(
        scaffold_pdb, cdr_spec,
        num_samples=num_samples,
        stochastic=stochastic,
        context_chains=context_chains,
        device=device,
    )
    elapsed = time.time() - t0
    return results, elapsed


def compute_stats(results):
    """Compute aggregate stats from design results."""
    if not results:
        return {}
    keys = ['plddt', 'iptm', 'pae', 'ppl', 'entropy']
    stats = {}
    for k in keys:
        vals = [r[k] for r in results if k in r]
        if vals:
            stats[f'{k}_mean'] = np.mean(vals)
            stats[f'{k}_std'] = np.std(vals)
    if 'iptm_mean' in stats and 'plddt_mean' in stats:
        stats['composite'] = (
            0.35 * stats['iptm_mean'] +
            0.25 * stats['plddt_mean'] +
            0.15 * (1 - min(stats.get('ppl_mean', 100), 200) / 200) +
            0.10 * (1 - min(stats.get('entropy_mean', 3), 3) / 3) +
            0.10 * (1 - min(stats.get('pae_mean', 15), 15) / 15)
        )
    return stats


def run_af2_validation(designs, scaffold_pdb, scaffold_chain, epi_seq, cdr_spec,
                       output_dir, use_jax=True, num_recycle=3):
    """Stage 3: AF2 Multimer validation."""
    from idp_antibody_design import validate_designs_with_af2

    colabfold_exe = None
    if not use_jax:
        venv_bin = os.path.join(os.path.dirname(SCRIPT_DIR), 'venv', 'bin', 'colabfold_batch')
        if os.path.exists(venv_bin):
            colabfold_exe = venv_bin
        else:
            colabfold_exe = 'colabfold_batch'

    return validate_designs_with_af2(
        designs, scaffold_pdb, scaffold_chain,
        epi_seq, cdr_spec, output_dir,
        colabfold_exe=colabfold_exe, num_recycle=num_recycle,
        verbose=True, use_jax=use_jax,
    )


# ── Iterative Refinement & Closed-Loop ──

def _iter_progress(round_idx, max_rounds, status):
    """Terminal progress callback for iterative refinement."""
    print(f"  [Round {round_idx}/{max_rounds}] {status}")


def run_iterative_refinement(args, disorder, segments, epi_seq):
    """Iterative refinement: design -> AF2 -> cascade filter -> top-K -> redesign."""
    import closed_loop_orchestrator as clo

    cdr_spec_str = parse_cdr_spec(args.cdr, args.scaffold_chain)
    mode = 'fixbb' if args.fixbb else 'complex'

    design_fn = create_bfn_design_fn(
        scaffold_pdb=args.scaffold,
        scaffold_chain=args.scaffold_chain,
        mode=mode,
        device=args.device or auto_detect_device(),
    )
    af2_validator = create_jax_af2_validator(
        scaffold_pdb=args.scaffold,
        scaffold_chain=args.scaffold_chain,
        epi_seq=epi_seq,
        cdr_spec=cdr_spec_str,
        num_recycle=args.af2_recycle,
        data_dir=args.af2_data_dir,
    )

    print(f"\n{'='*80}")
    print(f"  Iterative Refinement ({mode.upper()} mode)")
    print(f"  Rounds: {args.iter_max_rounds}  Samples/round: {args.iter_samples}  Top-K: {args.iter_top_k}")
    print(f"  Convergence delta: {args.iter_convergence}")
    print(f"{'='*80}\n")

    refiner = IterativeRefiner(
        design_fn=design_fn,
        af2_validator=af2_validator,
        cascade_fn=apply_cascade_af2,
        n_samples=args.iter_samples,
        top_k=args.iter_top_k,
        max_rounds=args.iter_max_rounds,
        convergence_delta=args.iter_convergence,
        log_cb=print,
        progress_cb=_iter_progress,
    )

    result = refiner.run(args.scaffold, cdr_spec_str)

    report = format_refinement_report(result)
    print(f"\n{report}")

    # Save per-round results
    iter_dir = os.path.join(args.output_dir, 'iterative')
    os.makedirs(iter_dir, exist_ok=True)
    for r in result.rounds:
        rd = os.path.join(iter_dir, f'round_{r.round_idx}')
        os.makedirs(rd, exist_ok=True)
        with open(os.path.join(rd, f'round_{r.round_idx}_results.json'), 'w') as f:
            json.dump(r.all_results, f, indent=2)
        with open(os.path.join(rd, f'round_{r.round_idx}_report.txt'), 'w') as f:
            f.write(r.filter_report)

    # Save summary
    summary = {
        'config': {
            'target': args.target, 'target_chain': args.target_chain,
            'scaffold': args.scaffold, 'scaffold_chain': args.scaffold_chain,
            'cdr_spec': cdr_spec_str, 'mode': f'iterative_{mode}',
            'n_samples': args.iter_samples, 'top_k': args.iter_top_k,
            'max_rounds': args.iter_max_rounds, 'convergence_delta': args.iter_convergence,
        },
        'disorder': {
            'threshold': args.threshold,
            'segments': [{'start': s['start'], 'end': s['end'],
                          'length': s['length'], 'mean_disorder': s['mean_disorder']}
                         for s in segments],
        },
        'total_rounds': len(result.rounds),
        'converged': result.converged,
        'convergence_reason': result.convergence_reason,
        'best_iptm': result.best_overall_iptm,
        'best_ptm': result.best_overall_ptm,
        'best_sequence': result.best_overall_sequence,
        'best_pdb': result.best_overall_pdb,
        'rounds': [{
            'round_idx': r.round_idx,
            'n_input': r.n_input,
            'n_passed': r.n_passed,
            'best_iptm': r.best_iptm,
            'best_plddt': r.best_plddt,
            'avg_iptm': r.avg_iptm,
            'elapsed': r.elapsed_seconds,
        } for r in result.rounds],
    }
    with open(os.path.join(iter_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to: {iter_dir}/")
    return summary


def run_closed_loop(args, disorder, segments, epi_seq):
    """Closed-loop multi-objective optimization with optional physics scoring."""
    import closed_loop_orchestrator as clo

    cdr_spec_str = parse_cdr_spec(args.cdr, args.scaffold_chain)
    mode = 'fixbb' if args.fixbb else 'complex'

    design_fn = create_bfn_design_fn(
        scaffold_pdb=args.scaffold,
        scaffold_chain=args.scaffold_chain,
        mode=mode,
        device=args.device or auto_detect_device(),
    )
    af2_validator = create_jax_af2_validator(
        scaffold_pdb=args.scaffold,
        scaffold_chain=args.scaffold_chain,
        epi_seq=epi_seq,
        cdr_spec=cdr_spec_str,
        num_recycle=args.af2_recycle,
        data_dir=args.af2_data_dir,
    )

    physics_scorer = None
    if not args.cl_no_physics:
        physics_scorer = create_physics_scorer_safe(
            target_pdb=args.target,
            antibody_chains=args.scaffold_chain,
            antigen_chain=args.target_chain,
        )
        if physics_scorer is None:
            print("  Note: PyRosetta not available — physics scoring disabled\n")

    print(f"\n{'='*80}")
    print(f"  Closed-Loop Optimization ({mode.upper()} mode)")
    print(f"  Cycles: {args.cl_cycles}  Samples/cycle: {args.samples}  Top-K: {args.cl_top_k}")
    print(f"  Physics: {'disabled' if physics_scorer is None else 'enabled (PyRosetta)'}")
    print(f"  Convergence delta: {args.cl_convergence}")
    print(f"{'='*80}\n")

    orchestrator = clo.ClosedLoopOrchestrator(
        design_fn=design_fn,
        af2_validator=af2_validator,
        physics_scorer=physics_scorer,
        n_samples=args.samples,
        top_k=args.cl_top_k,
        max_cycles=args.cl_cycles,
        convergence_delta=args.cl_convergence,
        selection_strategy='pareto_first',
        log_cb=print,
        progress_cb=_iter_progress,
    )

    result = orchestrator.run(args.scaffold, cdr_spec_str)

    report = clo.format_closed_loop_report(result)
    print(f"\n{report}")

    # Save per-cycle results
    cl_dir = os.path.join(args.output_dir, 'closed_loop')
    os.makedirs(cl_dir, exist_ok=True)
    for c in result.cycles:
        cd = os.path.join(cl_dir, f'cycle_{c.cycle_idx}')
        os.makedirs(cd, exist_ok=True)
        with open(os.path.join(cd, f'cycle_{c.cycle_idx}_ranked.json'), 'w') as f:
            json.dump(c.all_ranked, f, indent=2)
        with open(os.path.join(cd, f'cycle_{c.cycle_idx}_report.txt'), 'w') as f:
            f.write(c.report)

    # Save summary
    summary = {
        'config': {
            'target': args.target, 'target_chain': args.target_chain,
            'scaffold': args.scaffold, 'scaffold_chain': args.scaffold_chain,
            'cdr_spec': cdr_spec_str, 'mode': f'closed_loop_{mode}',
            'n_samples': args.samples, 'top_k': args.cl_top_k,
            'max_cycles': args.cl_cycles, 'convergence_delta': args.cl_convergence,
            'physics_enabled': physics_scorer is not None,
        },
        'disorder': {
            'threshold': args.threshold,
            'segments': [{'start': s['start'], 'end': s['end'],
                          'length': s['length'], 'mean_disorder': s['mean_disorder']}
                         for s in segments],
        },
        'total_cycles': len(result.cycles),
        'converged': result.converged,
        'convergence_reason': result.convergence_reason,
        'best_composite': result.best_overall_composite,
        'best_iptm': result.best_overall_iptm,
        'best_plddt': result.best_overall_plddt,
        'best_dG': result.best_overall_dG,
        'best_sequence': result.best_overall_sequence,
        'best_pdb': result.best_overall_pdb,
        'cycles': [{
            'cycle_idx': c.cycle_idx,
            'n_generated': c.n_generated,
            'n_final': c.n_final,
            'best_composite': c.best_composite,
            'best_iptm': c.best_iptm,
            'best_dG': c.best_dG,
            'pareto_front_size': c.pareto_front_size,
            'hypervolume': c.hypervolume,
            'elapsed': c.elapsed_seconds,
        } for c in result.cycles],
    }
    with open(os.path.join(cl_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to: {cl_dir}/")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Unified Antibody Design — BFN Phase 3 + Disorder Head')

    # Target
    parser.add_argument('--target', default='data/misfolding_targets/2NAO_model1_A_1-42.pdb',
                        help='Target protein PDB')
    parser.add_argument('--target_chain', default='A', help='Target chain ID')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Disorder threshold for ordered classification')

    # Scaffold (required)
    parser.add_argument('--scaffold', default='data/misfolding_targets/5IMK.pdb',
                        help='Scaffold PDB')
    parser.add_argument('--scaffold_chain', required=True,
                        help='Scaffold chain ID (required, e.g. B for 5IMK nanobody)')

    # CDR spec
    parser.add_argument('--cdr', default=None,
                        help=f'CDR ranges (default: <scaffold_chain>:{DEFAULT_CDR})')

    # Design params
    parser.add_argument('--samples', type=int, default=10, help='Design samples per segment')
    parser.add_argument('--deterministic', action='store_true', help='Deterministic sampling')
    parser.add_argument('--segments', type=int, default=3, help='Max ordered segments')

    # Modes
    parser.add_argument('--analyze-only', action='store_true', help='Disorder analysis only')
    parser.add_argument('--fixbb', action='store_true',
                        help='FixBB mode (no antigen context). Default: Complex mode.')
    parser.add_argument('--compare', action='store_true',
                        help='Run BOTH FixBB and Complex, compare ipTM')
    parser.add_argument('--af2', action='store_true', help='Enable AF2 validation')
    parser.add_argument('--af2-recycle', type=int, default=3, help='AF2 recycle steps')
    parser.add_argument('--af2-cli', action='store_true', help='Use colabfold CLI instead of JAX')

    # Iterative refinement
    parser.add_argument('--iterate', action='store_true',
                        help='Enable iterative refinement (design -> AF2 -> filter -> redesign)')
    parser.add_argument('--iter-samples', type=int, default=10,
                        help='Design samples per round (default: 10)')
    parser.add_argument('--iter-top-k', type=int, default=3,
                        help='Top-K sequences carried to next round (default: 3)')
    parser.add_argument('--iter-max-rounds', type=int, default=3,
                        help='Maximum refinement rounds (default: 3)')
    parser.add_argument('--iter-convergence', type=float, default=0.02,
                        help='ipTM improvement threshold for convergence (default: 0.02)')

    # Closed-loop optimization
    parser.add_argument('--closed-loop', action='store_true',
                        help='Enable closed-loop multi-objective optimization')
    parser.add_argument('--cl-cycles', type=int, default=3,
                        help='Maximum closed-loop cycles (default: 3)')
    parser.add_argument('--cl-top-k', type=int, default=5,
                        help='Top-K designs per cycle (default: 5)')
    parser.add_argument('--cl-convergence', type=float, default=0.02,
                        help='Hypervolume delta convergence threshold (default: 0.02)')
    parser.add_argument('--cl-no-physics', action='store_true',
                        help='Disable PyRosetta physics scoring')
    parser.add_argument('--af2-data-dir', default=None,
                        help='AlphaFold params directory (default: ~/.cache/colabfold)')

    # Output
    parser.add_argument('--output_dir', default=None, help='Output directory')
    parser.add_argument('--device', default=None, help='Torch device (auto-detect)')

    args = parser.parse_args()

    device = args.device or auto_detect_device()
    cdr_spec_str = parse_cdr_spec(args.cdr, args.scaffold_chain)

    if args.output_dir is None:
        args.output_dir = f'design_results/{int(time.time())}'
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("  Unified Antibody Design — BFN Phase 3 + Disorder Head")
    print("=" * 80)
    print(f"  Target:       {args.target} (chain {args.target_chain})")
    print(f"  Scaffold:     {args.scaffold} (chain {args.scaffold_chain})")
    print(f"  CDRs:         {cdr_spec_str}")
    print(f"  Samples:      {args.samples}")
    print(f"  Mode:         {'Analyze-only' if args.analyze_only else 'Compare' if args.compare else 'FixBB' if args.fixbb else 'Complex'}")
    print(f"  Device:       {device}")
    print(f"  Output:       {args.output_dir}")
    print()

    # ── Stage 1: Disorder Analysis ──
    print("─" * 80)
    print("  Stage 1: Disorder Analysis")
    print("─" * 80)

    disorder = run_disorder_analysis(
        args.target, args.target_chain, device, args.threshold)

    print(disorder['report'])

    if args.analyze_only:
        np.save(os.path.join(args.output_dir, 'disorder_scores.npy'),
                disorder['disorder_scores'])
        with open(os.path.join(args.output_dir, 'disorder_report.txt'), 'w') as f:
            f.write(disorder['report'])
        print(f"\nSaved to: {args.output_dir}/")
        return

    if not disorder['segments']:
        print("\nNo ordered segments found. Aborting design.")
        return

    # ── Stage 2: BFN CDR Design ──
    segments = disorder['segments'][:args.segments]
    epi_seq = disorder['sequence']

    # ── Iterative / Closed-Loop modes ──
    if args.iterate and args.closed_loop:
        print("ERROR: Cannot use both --iterate and --closed-loop simultaneously.")
        return

    if args.closed_loop:
        if args.compare:
            print("Warning: --compare is ignored in closed-loop mode.")
        run_closed_loop(args, disorder, segments, epi_seq)
        return

    if args.iterate:
        if args.compare:
            print("Warning: --compare is ignored in iterative mode.")
        run_iterative_refinement(args, disorder, segments, epi_seq)
        return

    # ── One-shot design (original path) ──
    if args.compare:
        # Run both FixBB and Complex for comparison
        print(f"\n{'='*80}")
        print("  Stage 2: FixBB vs Complex Comparison")
        print(f"{'='*80}")

        all_results = {'fixbb': [], 'complex': []}

        for mode, label, ctx in [('fixbb', 'FixBB', []), ('complex', 'Complex', None)]:
            print(f"\n  --- {label} Mode ---")
            t0 = time.time()
            results, _ = run_single_design(
                args.scaffold, cdr_spec_str, args.samples,
                not args.deterministic, device, ctx)
            elapsed = time.time() - t0

            stats = compute_stats(results)
            all_results[mode] = {'results': results, 'stats': stats, 'elapsed': elapsed}

        # Comparison summary
        fb = all_results['fixbb']['stats']
        cb = all_results['complex']['stats']
        gap = fb.get('iptm_mean', 0) - cb.get('iptm_mean', 0)

        print(f"\n{'='*80}")
        print("  FixBB vs Complex — Comparison")
        print(f"{'='*80}")
        print(f"  {'Metric':<12} {'FixBB':<12} {'Complex':<12} {'Delta':<12}")
        print(f"  {'-'*48}")
        for k in ['plddt_mean', 'iptm_mean', 'pae_mean', 'ppl_mean']:
            fv = fb.get(k, 0)
            cv = cb.get(k, 0)
            d = fv - cv
            print(f"  {k:<12} {fv:<12.4f} {cv:<12.4f} {d:+.4f}")
        print(f"\n  ipTM gap (FixBB - Complex): {gap:+.4f}")
        print(f"  BFN overconfidence: {'Significant' if gap > 0.01 else 'Minimal'}")
        print(f"  FixBB elapsed: {all_results['fixbb']['elapsed']:.0f}s")
        print(f"  Complex elapsed: {all_results['complex']['elapsed']:.0f}s")

        # Rank by Complex ipTM
        all_results['ranked'] = sorted(
            all_results['complex']['results'],
            key=lambda r: r.get('iptm', 0), reverse=True)

    else:
        # Single mode
        ctx = [] if args.fixbb else None
        mode_label = 'FixBB' if args.fixbb else 'Complex'

        print(f"\n{'='*80}")
        print(f"  Stage 2: BFN CDR Design ({mode_label} mode)")
        print(f"{'='*80}")

        results, elapsed = run_single_design(
            args.scaffold, cdr_spec_str, args.samples,
            not args.deterministic, device, ctx)
        stats = compute_stats(results)

        print(f"\n  Generated {len(results)} designs in {elapsed:.0f}s")
        print(f"  pLDDT: {stats.get('plddt_mean', 0):.4f} +/- {stats.get('plddt_std', 0):.4f}")
        print(f"  ipTM:  {stats.get('iptm_mean', 0):.4f} +/- {stats.get('iptm_std', 0):.4f}")
        print(f"  PAE:   {stats.get('pae_mean', 0):.2f} +/- {stats.get('pae_std', 0):.2f}")
        print(f"  PPL:   {stats.get('ppl_mean', 0):.1f} +/- {stats.get('ppl_std', 0):.1f}")
        if 'composite' in stats:
            print(f"  Score: {stats['composite']:.4f}")

        all_results = {'results': results, 'stats': stats, 'mode': mode_label}
        all_results['ranked'] = sorted(
            results, key=lambda r: r.get('iptm', 0), reverse=True)

    # ── Stage 3: AF2 Validation (optional) ──
    if args.af2 and all_results.get('ranked'):
        print(f"\n{'='*80}")
        print("  Stage 3: AF2 Multimer Validation")
        print(f"{'='*80}")

        top_n = min(args.samples, 5)
        validated = run_af2_validation(
            all_results['ranked'][:top_n],
            args.scaffold, args.scaffold_chain,
            epi_seq, cdr_spec_str, args.output_dir,
            use_jax=not args.af2_cli, num_recycle=args.af2_recycle)
        all_results['af2_validated'] = validated

    # ── Save Results ──
    top3 = all_results.get('ranked', [])[:3]
    if top3:
        print(f"\n{'='*80}")
        print("  Top 3 Designs")
        print(f"{'='*80}")
        for i, d in enumerate(top3):
            disorder_str = ''
            if 'disorder_score' in d:
                disorder_str = f" | disorder={d['disorder_score']:.3f}"
            print(f"  [{i+1}] pLDDT={d.get('plddt', 0):.4f}  "
                  f"ipTM={d.get('iptm', 0):.4f}  PPL={d.get('ppl', 0):.1f}"
                  f"{disorder_str}")
            print(f"      {d.get('sequence', '')}")

    save_path = os.path.join(args.output_dir, 'design_results.json')
    with open(save_path, 'w') as f:
        json.dump({
            'config': {
                'target': args.target, 'target_chain': args.target_chain,
                'scaffold': args.scaffold, 'scaffold_chain': args.scaffold_chain,
                'cdr_spec': cdr_spec_str, 'samples': args.samples,
                'mode': 'compare' if args.compare else ('fixbb' if args.fixbb else 'complex'),
            },
            'disorder': {
                'threshold': args.threshold,
                'segments': [{'start': s['start'], 'end': s['end'],
                              'length': s['length'], 'mean_disorder': s['mean_disorder']}
                             for s in segments],
            },
            'stats': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                      for k, v in all_results.get('stats', {}).items()},
            'top_designs': [{
                'rank': i + 1,
                'sequence': d.get('sequence', ''),
                'plddt': d.get('plddt'), 'iptm': d.get('iptm'),
                'pae': d.get('pae'), 'ppl': d.get('ppl'),
                'entropy': d.get('entropy'),
                'disorder_score': d.get('disorder_score'),
            } for i, d in enumerate(top3)],
        }, f, indent=2)

    print(f"\n{'='*80}")
    print(f"  DESIGN COMPLETE")
    print(f"  Results saved to: {save_path}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
