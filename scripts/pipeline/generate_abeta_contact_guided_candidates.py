#!/usr/bin/env python
"""Generate contact-guided paratope variants from A-beta whitelist references."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "modules"))

from state_contact_scorer import (  # noqa: E402
    extract_contact_map,
    generate_contact_guided_variants,
    score_contact_guided_variants,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate A-beta contact-guided paratope candidates")
    parser.add_argument("--whitelist", default=str(PROJECT_ROOT / "configs" / "idp" / "abeta_reference_whitelist.yml"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples-per-ref", type=int, default=200)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--max-mutations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contact-cutoff", type=float, default=8.0)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.whitelist, encoding="utf-8") as f:
        refs = (yaml.safe_load(f) or {}).get("abeta_references", [])
    if not refs:
        raise SystemExit(f"No references in {args.whitelist}")

    all_rows = []
    details = []
    for ref_idx, ref in enumerate(refs):
        cmap = extract_contact_map(ref["path"], cutoff=args.contact_cutoff, peptide_chain=ref.get("peptide_chain"))
        variants = generate_contact_guided_variants(
            cmap,
            n=args.samples_per_ref,
            max_mutations=args.max_mutations,
            seed=args.seed + ref_idx,
        )
        ranked = score_contact_guided_variants(cmap, variants)
        for r in ranked:
            all_rows.append({
                "reference_pdb": ref["pdb"],
                "peptide_sequence": cmap["peptide_sequence"],
                "native_paratope": cmap["paratope_sequence"],
                **r,
            })
        details.append({
            "reference": ref,
            "contact_map": {
                **{k: v for k, v in cmap.items() if k != "contacts"},
                "contacts": [asdict(c) for c in cmap["contacts"]],
            },
            "ranked": ranked,
        })

    all_rows.sort(key=lambda r: (r["passes_filters"], r["candidate_score"], r["contact_retention"]), reverse=True)
    for i, r in enumerate(all_rows, 1):
        r["global_rank"] = i
    top = all_rows[:args.top]

    fields = [
        "global_rank", "rank", "reference_pdb", "candidate_id", "sequence", "mutations",
        "n_mutations", "peptide_sequence", "native_paratope", "state_contact_score",
        "hotspot_score", "contact_retention", "complexity", "immunogenicity_proxy",
        "candidate_score", "passes_filters", "filter_failures",
    ]
    with open(out_dir / "contact_guided_candidates.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(top)
    with open(out_dir / "contact_guided_candidates_all.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    with open(out_dir / "contact_guided_candidates.fasta", "w", encoding="utf-8") as f:
        for r in top:
            f.write(
                f">{r['reference_pdb']}|{r['candidate_id']}|rank={r['global_rank']}"
                f"|score={r['candidate_score']}|retention={r['contact_retention']}|pass={r['passes_filters']}\n"
                f"{r['sequence']}\n"
            )
    with open(out_dir / "contact_guided_candidate_details.json", "w", encoding="utf-8") as f:
        json.dump({"results": details}, f, indent=2)
    with open(out_dir / "contact_guided_candidate_report.md", "w", encoding="utf-8") as f:
        n_pass = sum(1 for r in all_rows if r["passes_filters"])
        f.write("# A-beta Contact-Guided Candidate Report\n\n")
        f.write(f"References: {len(refs)}; generated: {len(all_rows)}; passing filters: {n_pass}; reported top: {len(top)}\n\n")
        f.write("| Global Rank | Ref | Candidate | Sequence | Mutations | Score | Retention | Hotspot | Complexity | Immuno | Pass |\n")
        f.write("|---:|---|---|---|---|---:|---:|---:|---:|---:|---|\n")
        for r in top:
            f.write(
                f"| {r['global_rank']} | {r['reference_pdb']} | {r['candidate_id']} | `{r['sequence']}` | "
                f"{r['mutations']} | {r['candidate_score']} | {r['contact_retention']} | "
                f"{r['hotspot_score']} | {r['complexity']} | {r['immunogenicity_proxy']} | {r['passes_filters']} |\n"
            )
        f.write("\nThese are paratope variants scored on known contact topology, not final scaffold-grafted antibodies.\n")

    print(f"Wrote contact-guided candidates to {out_dir}")
    print(f"Generated={len(all_rows)} pass={sum(1 for r in all_rows if r['passes_filters'])} top={len(top)}")
    if top:
        b = top[0]
        print(f"Best: {b['reference_pdb']} {b['candidate_id']} {b['sequence']} score={b['candidate_score']} retention={b['contact_retention']}")


if __name__ == "__main__":
    main()
