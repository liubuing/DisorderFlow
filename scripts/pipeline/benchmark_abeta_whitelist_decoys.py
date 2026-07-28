#!/usr/bin/env python
"""Run standard decoy benchmark on the A-beta reference whitelist."""
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

from state_contact_scorer import benchmark_native_vs_decoys, extract_contact_map  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="A-beta whitelist decoy benchmark")
    parser.add_argument("--whitelist", default=str(PROJECT_ROOT / "configs" / "idp" / "abeta_reference_whitelist.yml"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-scramble", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contact-cutoff", type=float, default=8.0)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.whitelist, encoding="utf-8") as f:
        whitelist = yaml.safe_load(f) or {}
    refs = whitelist.get("abeta_references", [])
    if not refs:
        raise SystemExit(f"No abeta_references found in {args.whitelist}")

    rows = []
    details = []
    decoy_rows = []
    for ref in refs:
        cmap = extract_contact_map(ref["path"], cutoff=args.contact_cutoff,
                                   peptide_chain=ref.get("peptide_chain"))
        bench = benchmark_native_vs_decoys(cmap, n_scramble=args.n_scramble, seed=args.seed)
        s = bench["summary"]
        row = {
            "pdb": ref["pdb"],
            "peptide_chain": ref.get("peptide_chain"),
            "peptide_sequence": cmap["peptide_sequence"],
            "abeta_range": ref.get("abeta_range"),
            "paratope_sequence": cmap["paratope_sequence"],
            "n_contacts": len(cmap["contacts"]),
            "native_score": s["native_score"],
            "native_hotspot": s["native_hotspot"],
            "decoy_mean_score": s["decoy_mean_score"],
            "decoy_max_score": s["decoy_max_score"],
            "native_percentile": s["native_percentile"],
            "native_hotspot_percentile": s["native_hotspot_percentile"],
            "n_decoys": s["n_decoys"],
            "type_summary": json.dumps(s["type_summary"], sort_keys=True),
        }
        rows.append(row)
        for d in bench["decoys"]:
            decoy_rows.append({
                "pdb": ref["pdb"],
                "decoy_id": d["decoy_id"],
                "decoy_type": d["decoy_type"],
                "sequence": d["sequence"],
                "state_contact_score": d["state_contact_score"],
                "hotspot_score": d["hotspot_score"],
                "identity_to_native": d["identity_to_native"],
                "conservative_similarity": d["conservative_similarity"],
            })
        details.append({
            "reference": ref,
            "contact_map": {
                **{k: v for k, v in cmap.items() if k != "contacts"},
                "contacts": [asdict(c) for c in cmap["contacts"]],
            },
            "benchmark": bench,
        })

    with open(out_dir / "abeta_whitelist_decoy_benchmark_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "abeta_whitelist_decoys.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(decoy_rows[0].keys()))
        writer.writeheader()
        writer.writerows(decoy_rows)
    with open(out_dir / "abeta_whitelist_decoy_benchmark_details.json", "w", encoding="utf-8") as f:
        json.dump({"results": details}, f, indent=2)
    with open(out_dir / "abeta_whitelist_decoy_benchmark_report.md", "w", encoding="utf-8") as f:
        f.write("# A-beta Whitelist Decoy Benchmark\n\n")
        f.write("| PDB | Peptide | Native | Decoy Mean | Decoy Max | Native Percentile | Hotspot Percentile | Decoys |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['pdb']} | `{r['peptide_sequence']}` | {r['native_score']} | "
                f"{r['decoy_mean_score']} | {r['decoy_max_score']} | {r['native_percentile']} | "
                f"{r['native_hotspot_percentile']} | {r['n_decoys']} |\n"
            )
        f.write("\n## Decoy Types\n\n")
        for r in rows:
            f.write(f"### {r['pdb']}\n\n")
            type_summary = json.loads(r["type_summary"])
            f.write("| Type | N | Mean | Max | Hotspot Mean | Hotspot Max |\n")
            f.write("|---|---:|---:|---:|---:|---:|\n")
            for dtype, ts in sorted(type_summary.items()):
                f.write(
                    f"| {dtype} | {ts['n']} | {ts['mean_score']} | {ts['max_score']} | "
                    f"{ts['mean_hotspot']} | {ts['max_hotspot']} |\n"
                )
            f.write("\n")

    print(f"Wrote whitelist decoy benchmark to {out_dir}")
    for r in rows:
        print(
            f"{r['pdb']}: native={r['native_score']} decoy_mean={r['decoy_mean_score']} "
            f"decoy_max={r['decoy_max_score']} percentile={r['native_percentile']} "
            f"hotspot_percentile={r['native_hotspot_percentile']}"
        )


if __name__ == "__main__":
    main()
