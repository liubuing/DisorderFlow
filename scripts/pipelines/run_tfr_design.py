#!/usr/bin/env python
"""TfR Nanobody Design — CLI entry point.

One-click BFN design of a TfR-binding nanobody with negative design against
Transferrin. Thin wrapper over modules/tfr_design_pipeline.py.

Usage:
  # First, fetch TfR/Tf reference structures (one-time)
  python scripts/download_tfr_targets.py

  # Full design (Complex mode + negative design, no AF2 — fast)
  python run_tfr_design.py --scaffold_chain B --samples 20

  # With AF2 multimer validation (slow but design-specific confidence)
  python run_tfr_design.py --scaffold_chain B --samples 10 --af2

  # Disorder-only analysis of the TfR target
  python run_tfr_design.py --analyze-only
"""
from __future__ import annotations

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="TfR Nanobody Design via BFN (with negative design vs Transferrin)")
    parser.add_argument("--target", default=None,
                        help="TfR PDB (default: data/tfr_targets/6GZV.pdb)")
    parser.add_argument("--target_chain", default="A")
    parser.add_argument("--scaffold", default=None,
                        help="Antibody scaffold PDB (default: data/misfolding_targets/5IMK.pdb)")
    parser.add_argument("--scaffold_chain", default="B", help="Scaffold chain (required)")
    parser.add_argument("--cdr_spec", default=None,
                        help="CDR spec (default: <scaffold_chain>:26-33,51-58,97-113)")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only run disorder analysis on the TfR target")
    parser.add_argument("--fixbb", action="store_true",
                        help="FixBB mode (no antigen context). Default: Complex mode.")
    parser.add_argument("--af2", action="store_true", help="Enable AF2 multimer validation")
    parser.add_argument("--af2-recycle", type=int, default=3)
    parser.add_argument("--af2-cli", action="store_true", help="Use colabfold CLI instead of JAX AF2")
    parser.add_argument("--no-negative-design", action="store_true", default=False,
                        help="Disable off-target (Transferrin) negative design")
    parser.add_argument("--off-target", default=None,
                        help="Off-target PDB for negative design (default: data/tfr_targets/1A8E.pdb)")
    parser.add_argument("--off-target-chain", default="A")
    parser.add_argument("--off-target-max", type=float, default=0.3,
                        help="Hard-reject designs with off-target ipTM above this")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    sys.path.insert(0, ".")
    sys.path.insert(0, "modules")

    from tfr_design_pipeline import (
        run_tfr_design, DEFAULT_TFR_PDB, DEFAULT_SCAFFOLD_PDB, DEFAULT_TF_PDB,
    )
    from idp_disorder_analysis import predict_disorder, format_disorder_report

    target = args.target or DEFAULT_TFR_PDB
    scaffold = args.scaffold or DEFAULT_SCAFFOLD_PDB
    off_target = args.off_target or DEFAULT_TF_PDB

    print("=" * 80)
    print("  TfR Nanobody Design — BFN + Negative Design")
    print("=" * 80)
    print(f"  Target (TfR):   {target} [chain {args.target_chain}]")
    print(f"  Scaffold:       {scaffold} [chain {args.scaffold_chain}]")
    print(f"  Off-target (Tf):{off_target} [chain {args.off_target_chain}]")
    print()

    # Verify on-target exists (give a helpful pointer if not).
    if not os.path.exists(target):
        print(f"ERROR: TfR target PDB not found: {target}")
        print("  Run first:  python scripts/download_tfr_targets.py")
        sys.exit(1)
    if not os.path.exists(scaffold):
        print(f"ERROR: scaffold PDB not found: {scaffold}")
        sys.exit(1)

    if args.analyze_only:
        from bfn_loader import load_bfn
        model, config = load_bfn()
        result = predict_disorder(model, config, target, args.target_chain, args.device)
        print(format_disorder_report(
            result["disorder_scores"], result["sequence"],
            result["residue_ids"], args.threshold))
        return

    colabfold_exe = "colabfold_batch"
    venv_bin = os.path.join(os.path.dirname(script_dir), "venv", "bin", "colabfold_batch")
    if os.path.exists(venv_bin):
        colabfold_exe = venv_bin

    run_tfr_design(
        target_pdb=target,
        target_chain=args.target_chain,
        scaffold_pdb=scaffold,
        scaffold_chain=args.scaffold_chain,
        cdr_spec=args.cdr_spec or f"{args.scaffold_chain}:26-33,51-58,97-113",
        disorder_threshold=args.threshold,
        num_segments=args.segments,
        num_samples=args.samples,
        stochastic=not args.deterministic,
        output_dir=args.output_dir,
        device=args.device,
        use_af2=args.af2,
        af2_num_recycle=args.af2_recycle,
        use_af2_jax=not args.af2_cli,
        colabfold_exe=colabfold_exe,
        fixbb=args.fixbb,
        negative_design=not args.no_negative_design,
        off_target_pdb=off_target if os.path.exists(off_target) else None,
        off_target_chain=args.off_target_chain,
        off_target_iptm_max=args.off_target_max,
        top_n=args.top,
        verbose=True,
    )


if __name__ == "__main__":
    main()
