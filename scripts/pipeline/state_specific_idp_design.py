#!/usr/bin/env python
"""Standalone state-specific IDP antibody design MVP.

Example:
  python scripts/pipeline/state_specific_idp_design.py \
      --preset abeta_core --samples 200 --top 30 \
      --out outputs/state_specific_abeta_core
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "modules"))

from state_specificity_scorer import (  # noqa: E402
    DesignSpec,
    generate_paratope_candidates,
    load_preset,
    rank_candidates,
)


def parse_args():
    parser = argparse.ArgumentParser(description="State-specific IDP antibody design MVP")
    parser.add_argument("--preset", default="abeta_core",
                        help="Built-in preset: abeta_core, abeta_nterm, abeta_cterm")
    parser.add_argument("--target", default=None, help="Custom target name")
    parser.add_argument("--epitope-id", default=None, help="Custom epitope identifier")
    parser.add_argument("--epitope-seq", default=None, help="Custom positive-state epitope sequence")
    parser.add_argument("--positive-state", default=None, help="Custom positive disease state label")
    parser.add_argument("--negative-epitope", action="append", default=[],
                        help="Negative/off-state epitope sequence. Can be repeated.")
    parser.add_argument("--samples", type=int, default=200, help="Number of heuristic candidates")
    parser.add_argument("--length", type=int, default=13, help="CDR-H3 candidate length")
    parser.add_argument("--top", type=int, default=30, help="Top candidates to report")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scaffold", action="append", default=[],
                        help="Scaffold/carrier label. Can be repeated.")
    parser.add_argument("--candidates-csv", default=None,
                        help="Optional CSV with sequence or cdr_h3 column; skips heuristic generation")
    parser.add_argument("--out", required=True, help="Output directory")
    return parser.parse_args()


def build_spec(args) -> DesignSpec:
    if args.epitope_seq:
        return DesignSpec(
            target=args.target or "custom_idp",
            epitope_id=args.epitope_id or "custom_epitope",
            epitope_seq=args.epitope_seq,
            positive_state=args.positive_state or "disease_state",
            negative_epitopes=args.negative_epitope or [],
            goal="custom state-specific IDP antibody design",
            prior="custom_state_specific_paratope",
        )
    spec = load_preset(args.preset)
    if args.negative_epitope:
        spec = DesignSpec(
            target=spec.target,
            epitope_id=spec.epitope_id,
            epitope_seq=spec.epitope_seq,
            positive_state=spec.positive_state,
            negative_epitopes=args.negative_epitope,
            goal=spec.goal,
            prior=spec.prior,
        )
    return spec


def load_candidates_csv(path: str):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            seq = row.get("cdr_h3") or row.get("sequence")
            if not seq:
                continue
            rows.append({
                **row,
                "candidate_id": row.get("candidate_id") or f"input_{i:04d}",
                "cdr_h3": seq,
                "generator": row.get("generator") or "input_csv",
            })
    return rows


def write_outputs(out_dir: Path, spec: DesignSpec, ranked, top_n: int, config: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    top = ranked[:top_n]

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"spec": spec.__dict__, "run": config}, f, indent=2)

    fieldnames = [
        "rank", "candidate_id", "sequence", "cdr_h3", "scaffold", "generator",
        "target", "epitope_id", "epitope_seq", "positive_state",
        "positive_score", "max_negative_score", "specificity_gap",
        "complexity", "immunogenicity_proxy", "state_specific_score",
        "passes_filters", "filter_failures", "rationale",
    ]
    with open(out_dir / "candidate_library.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(top)

    with open(out_dir / "candidate_library.fasta", "w", encoding="utf-8") as f:
        for r in top:
            f.write(
                f">{r['candidate_id']}|rank={r['rank']}|score={r['state_specific_score']}"
                f"|gap={r['specificity_gap']}|pass={r['passes_filters']}\n{r['sequence']}\n"
            )

    with open(out_dir / "candidate_report.md", "w", encoding="utf-8") as f:
        n_pass = sum(1 for r in ranked if r["passes_filters"])
        f.write(f"# State-Specific IDP Antibody Design Report\n\n")
        f.write(f"Target: `{spec.target}`\n\n")
        f.write(f"Therapeutic goal: {spec.goal}\n\n")
        f.write(f"Positive epitope: `{spec.epitope_id}` / `{spec.epitope_seq}`\n\n")
        f.write(f"Positive state: `{spec.positive_state}`\n\n")
        f.write("Negative/off-state epitopes: " + ", ".join(f"`{x}`" for x in spec.negative_epitopes) + "\n\n")
        f.write(f"Generated/scored candidates: {len(ranked)}; passing filters: {n_pass}; reported top: {len(top)}\n\n")
        f.write("| Rank | Candidate | CDR-H3 | Score | Gap | Pos | Max Neg | Complexity | Immuno | Pass | Rationale |\n")
        f.write("|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|\n")
        for r in top:
            f.write(
                f"| {r['rank']} | {r['candidate_id']} | `{r['sequence']}` | "
                f"{r['state_specific_score']} | {r['specificity_gap']} | "
                f"{r['positive_score']} | {r['max_negative_score']} | "
                f"{r['complexity']} | {r['immunogenicity_proxy']} | "
                f"{r['passes_filters']} | {r['rationale']} |\n"
            )
        f.write("\n## Interpretation\n\n")
        f.write("This MVP ranks paratope candidates by positive disease-state chemistry minus negative/off-state cross-reactivity. ")
        f.write("It is intended as a pure-computation design screen and does not claim experimental binding accuracy. ")
        f.write("The scaffold field is a carrier label, not a fixed binding template.\n")


def main():
    args = parse_args()
    spec = build_spec(args)
    if args.candidates_csv:
        candidates = load_candidates_csv(args.candidates_csv)
    else:
        candidates = generate_paratope_candidates(
            spec,
            n=args.samples,
            length=args.length,
            seed=args.seed,
            scaffold_pool=args.scaffold or None,
        )
    ranked = rank_candidates(candidates, spec)
    write_outputs(Path(args.out), spec, ranked, args.top, vars(args))
    print(f"Wrote {min(args.top, len(ranked))} candidates to {args.out}")
    if ranked:
        best = ranked[0]
        print(
            f"Best: {best['candidate_id']} {best['sequence']} "
            f"score={best['state_specific_score']} gap={best['specificity_gap']} "
            f"pass={best['passes_filters']}"
        )


if __name__ == "__main__":
    main()
