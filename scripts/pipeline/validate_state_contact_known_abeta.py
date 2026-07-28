#!/usr/bin/env python
"""Validate v2 contact-aware scorer on known anti-A-beta PDBs."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "modules"))

from state_contact_scorer import benchmark_native_vs_scrambled, extract_contact_map  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Contact-aware known A-beta sanity check")
    parser.add_argument("--pdb", action="append", default=[], help="Reference PDB path. Can repeat.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--contact-cutoff", type=float, default=8.0)
    parser.add_argument("--n-scramble", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def serializable_contact_map(contact_map):
    return {
        **{k: v for k, v in contact_map.items() if k != "contacts"},
        "contacts": [asdict(c) for c in contact_map["contacts"]],
    }


def main():
    args = parse_args()
    pdbs = [Path(p) for p in args.pdb] or [
        PROJECT_ROOT / "data" / "anti_abeta_refs" / "4HIX.pdb",
        PROJECT_ROOT / "data" / "anti_abeta_refs" / "5CSZ.pdb",
    ]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    details = []
    for pdb in pdbs:
        cmap = extract_contact_map(str(pdb), cutoff=args.contact_cutoff)
        bench = benchmark_native_vs_scrambled(cmap, n_scramble=args.n_scramble, seed=args.seed)
        s = bench["summary"]
        row = {
            "pdb": pdb.stem,
            "peptide_chain": cmap["peptide_chain"],
            "peptide_sequence": cmap["peptide_sequence"],
            "paratope_sequence": cmap["paratope_sequence"],
            "n_paratope_positions": len(cmap["paratope_sequence"]),
            "n_contacts": len(cmap["contacts"]),
            **s,
        }
        rows.append(row)
        details.append({
            "pdb": str(pdb),
            "contact_map": serializable_contact_map(cmap),
            "benchmark": bench,
        })

    with open(out_dir / "state_contact_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "state_contact_details.json", "w", encoding="utf-8") as f:
        json.dump({"results": details}, f, indent=2)
    with open(out_dir / "state_contact_report.md", "w", encoding="utf-8") as f:
        f.write("# Contact-Aware Known Anti-A-beta Sanity Check\n\n")
        f.write("| PDB | Peptide | Paratope | Native Score | Scramble Mean | Native Percentile | Native Hotspot | Scramble Hotspot Mean | Hotspot Percentile | Contacts |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['pdb']} | `{r['peptide_sequence']}` | `{r['paratope_sequence']}` | "
                f"{r['native_score']} | {r['scramble_mean_score']} | {r['native_percentile']} | "
                f"{r['native_hotspot']} | {r['scramble_mean_hotspot']} | {r['native_hotspot_percentile']} | {r['n_contacts']} |\n"
            )
        f.write("\nPercentile near 1.0 means native ranks above nearly all scrambled controls.\n")

    print(f"Wrote contact-aware sanity-check results to {out_dir}")
    for r in rows:
        print(
            f"{r['pdb']}: native={r['native_score']} scramble_mean={r['scramble_mean_score']} "
            f"percentile={r['native_percentile']} hotspot={r['native_hotspot']} "
            f"hotspot_percentile={r['native_hotspot_percentile']} contacts={r['n_contacts']}"
        )


if __name__ == "__main__":
    main()
