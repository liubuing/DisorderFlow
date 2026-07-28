#!/usr/bin/env python
"""IDP-Aware Antibody Design — CLI Orchestrator.

Full pipeline: Disorder Analysis → Ordered Segment Identification → Epitope Structure Building → BFN CDR Design → Confidence Ranking.

Usage:
  # 5IMK nanobody (chain B) + Abeta42 target (chain A)
  python run_idp_design.py --scaffold_chain B

  # Custom target + scaffold, Complex mode (default)
  python run_idp_design.py --target data/misfolding_targets/2NAO_model1_A_1-42.pdb \\
                           --target_chain A \\
                           --scaffold data/misfolding_targets/5IMK.pdb \\
                           --scaffold_chain B \\
                           --threshold 0.3 --segments 3 --samples 10

  # FixBB mode (no antigen context)
  python run_idp_design.py --scaffold_chain B --fixbb

  # Disorder-only analysis (no design)
  python run_idp_design.py --analyze-only

  # Deterministic mode (faster, single design per segment)
  python run_idp_design.py --scaffold_chain B --deterministic --samples 5
"""

import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='IDP-Aware Antibody CDR Design using BFN Disorder Head')

    # Target
    parser.add_argument('--target', type=str,
                        default='data/misfolding_targets/2NAO_model1_A_1-42.pdb',
                        help='Path to target protein PDB')
    parser.add_argument('--target_chain', type=str, default='A',
                        help='Chain ID of target protein')

    # Scaffold
    parser.add_argument('--scaffold', type=str,
                        default='data/misfolding_targets/5IMK.pdb',
                        help='Path to antibody scaffold PDB')
    parser.add_argument('--scaffold_chain', type=str, required=True,
                        help='Chain ID of antibody scaffold (required, e.g. B for 5IMK nanobody)')

    # CDR specification (auto-derived from scaffold_chain if not provided)
    parser.add_argument('--cdr_spec', type=str, default=None,
                        help='CDR region spec (format: chain:start-end,...). '
                             'Default: <scaffold_chain>:26-33,51-58,97-113')

    # Disorder
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Disorder threshold (< threshold = ordered)')
    parser.add_argument('--segments', type=int, default=3,
                        help='Max number of ordered segments to target')

    # Design
    parser.add_argument('--samples', type=int, default=10,
                        help='Number of BFN design samples per segment')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic sampling (faster, less diverse)')

    # Modes
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only run disorder analysis (skip design)')
    parser.add_argument('--fixbb', action='store_true',
                        help='FixBB mode: design without antigen context (default: Complex mode)')
    parser.add_argument('--af2', action='store_true', default=False,
                        help='Enable AF2 multimer validation for design-specific confidence')
    parser.add_argument('--af2-recycle', type=int, default=3,
                        help='AF2 recycle steps (default: 3)')
    parser.add_argument('--af2-cli', action='store_true', default=False,
                        help='Use colabfold CLI instead of JAX-native AF2 runner')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Torch device')

    # Config
    parser.add_argument('--config', type=str, default=None,
                        help='BFN config file (uses default if not specified)')

    args = parser.parse_args()

    # Set up paths relative to script dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))

    # Auto-detect colabfold binary
    colabfold_exe = 'colabfold_batch'
    venv_bin = os.path.join(os.path.dirname(script_dir), 'venv', 'bin', 'colabfold_batch')
    if os.path.exists(venv_bin):
        colabfold_exe = venv_bin
    elif os.path.exists('/home/liubuzi/disorderflow-main/venv/bin/colabfold_batch'):
        colabfold_exe = '/home/liubuzi/disorderflow-main/venv/bin/colabfold_batch'

    print("=" * 80)
    print("  IDP-Aware Antibody Design — BFN Disorder Head + CDR Design")
    print("=" * 80)
    print(f"  Target:     {args.target}")
    print(f"  Scaffold:   {args.scaffold}")
    cdr_spec = args.cdr_spec or f"{args.scaffold_chain}:26-33,51-58,97-113"
    print(f"  CDRs:       {cdr_spec}")
    print(f"  Threshold:  {args.threshold}")
    print(f"  Segments:   {args.segments}")
    print(f"  Samples:    {args.samples}")
    print(f"  Stochastic: {not args.deterministic}")
    print(f"  AF2 Valid:  {args.af2}")
    print()

    # Verify files exist
    for path, name in [(args.target, 'Target PDB'), (args.scaffold, 'Scaffold PDB')]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found: {path}")
            sys.exit(1)

    if args.analyze_only:
        # ── Disorder Analysis Only ──
        from idp_disorder_analysis import predict_disorder, format_disorder_report

        print("[Mode] Disorder Analysis Only")
        # Lazy-load BFN
        from bfn_loader import load_bfn, has_disorder_head
        model, config = load_bfn()

        result = predict_disorder(
            model, config, args.target, args.target_chain, args.device
        )
        report = format_disorder_report(
            result['disorder_scores'], result['sequence'],
            result['residue_ids'], args.threshold
        )
        print(report)

        # Save
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
        import numpy as np
        out = args.output_dir or '.'
        np.save(os.path.join(out, 'disorder_scores.npy'),
                result['disorder_scores'])
        with open(os.path.join(out, 'disorder_analysis.txt'), 'w') as f:
            f.write(report)
        print(f"\nDisorder scores saved to: {out}/disorder_scores.npy")
        print(f"Report saved to: {out}/disorder_analysis.txt")

    else:
        # ── Full Design Pipeline ──
        from idp_antibody_design import run_idp_antibody_design

        # context_chains: None = Complex (default), [] = FixBB
        _context = [] if args.fixbb else None

        result = run_idp_antibody_design(
            target_pdb=args.target,
            target_chain=args.target_chain,
            scaffold_pdb=args.scaffold,
            scaffold_chain=args.scaffold_chain,
            disorder_threshold=args.threshold,
            num_segments=args.segments,
            num_samples=args.samples,
            stochastic=not args.deterministic,
            output_dir=args.output_dir,
            device=args.device,
            verbose=True,
            use_af2=args.af2,
            af2_num_recycle=args.af2_recycle,
            use_af2_jax=not args.af2_cli,
            colabfold_exe=colabfold_exe,
            context_chains=_context,
        )

        # Print final summary
        ranked = result['ranked_designs']
        if ranked:
            print("\n" + "=" * 80)
            print("  DESIGN COMPLETE")
            print("=" * 80)
            print(f"  Total designs: {len(ranked)}")
            best = ranked[0]
            print(f"  Best composite score: {best['composite_score']:.3f} "
                  f"[{best['quality_label']}]")
            print(f"  Best CDR sequence: {best['sequence']}")
            print(f"  Confidence: pLDDT={best['plddt'] or 0:.3f}  "
                  f"ipTM={best['iptm'] or 0:.3f}  PAE={best['pae'] or 99:.1f}")
            # Show disorder calibration if active
            cal = best.get('_af2_calibrated')
            if cal:
                print(f"  Disorder Calibration: AF2 reliability={cal['af2_reliability']:.2f} "
                      f"[{cal['af2_reliability_label']}]")
                print(f"    Epitope disorder={cal['epitope_disorder']:.3f} | "
                      f"Calibrated ipTM={cal['iptm']:.3f} pLDDT={cal['plddt']:.3f} "
                      f"(raw AF2: ipTM={cal['af2_raw_iptm']:.3f} pLDDT={cal['af2_raw_plddt']:.3f})")
            print(f"\n  Results saved to: {result['output_dir']}")
            print(f"  JSON: {result['output_dir']}/idp_design_results.json")

            # Show top 3
            print(f"\n  Top 3 Designs:")
            for i, d in enumerate(ranked[:3]):
                cal_str = ''
                d_cal = d.get('_af2_calibrated')
                if d_cal:
                    cal_str = f" | D_cal={d_cal['af2_reliability_label']}"
                print(f"  [{i+1}] score={d['composite_score']:.3f} | "
                      f"pLDDT={d['plddt'] or 0:.3f} | ipTM={d['iptm'] or 0:.3f} | "
                      f"PPL={d['ppl'] or 99:.1f}{cal_str}")
                print(f"      seg={d['segment_start']}-{d['segment_end']} | "
                      f"{d['sequence']}")
        else:
            print("\nNo designs were generated. Check disorder analysis results.")


if __name__ == '__main__':
    main()
