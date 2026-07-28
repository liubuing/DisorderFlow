#!/usr/bin/env python
"""Audit anti-A-beta reference PDBs and benchmark contact-aware scorer.

The script checks data reliability before using a structure as a positive
control. It reports chain lengths, inferred peptide chain, whether the peptide
looks like a contiguous A-beta fragment, contact-map size, and native vs
scrambled separation for the v2 contact scorer.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "modules"))

from state_contact_scorer import (  # noqa: E402
    ABETA42,
    benchmark_native_vs_scrambled,
    extract_contact_map,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Audit anti-A-beta reference PDB contact maps")
    parser.add_argument("--pdb-dir", default=str(PROJECT_ROOT / "data" / "anti_abeta_refs"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--contact-cutoff", type=float, default=8.0)
    parser.add_argument("--n-scramble", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-contacts", type=int, default=8)
    parser.add_argument("--min-native-percentile", type=float, default=0.80)
    parser.add_argument("--include", action="append", default=[],
                        help="Optional explicit PDB path. Can repeat; overrides --pdb-dir glob.")
    return parser.parse_args()


def reliability_label(row, min_contacts: int, min_native_percentile: float):
    reasons = []
    if not row.get("is_abeta_like"):
        reasons.append("peptide_not_abeta_like")
    if row.get("n_contacts", 0) < min_contacts:
        reasons.append("too_few_contacts")
    if row.get("native_percentile", 0.0) < min_native_percentile:
        reasons.append("native_not_above_scrambled")
    if row.get("n_paratope_positions", 0) < 4:
        reasons.append("too_few_paratope_positions")
    if reasons:
        return "questionable", ";".join(reasons)
    return "reliable", ""


def serializable_contact_map(cmap):
    return {
        **{k: v for k, v in cmap.items() if k != "contacts"},
        "contacts": [asdict(c) for c in cmap["contacts"]],
    }


def portable_path(path):
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def main():
    args = parse_args()
    pdbs = [Path(p) for p in args.include] if args.include else [
        Path(p) for p in sorted(glob.glob(str(Path(args.pdb_dir) / "*.pdb")))
    ]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    details = []
    for pdb in pdbs:
        try:
            cmap = extract_contact_map(str(pdb), cutoff=args.contact_cutoff)
            bench = benchmark_native_vs_scrambled(cmap, n_scramble=args.n_scramble, seed=args.seed)
            chosen = cmap.get("chain_selection", {}).get("chosen", {})
            frag = {
                "is_abeta_like": chosen.get("is_abeta_like", False),
                "abeta_start": chosen.get("abeta_start"),
                "abeta_end": chosen.get("abeta_end"),
                "abeta_coverage": chosen.get("abeta_coverage", 0.0),
                "best_identity": chosen.get("best_identity", 0.0),
            }
            row = {
                "pdb": pdb.stem,
                "path": portable_path(pdb),
                "status": "parsed",
                "chain_lengths": json.dumps(cmap["chain_lengths"], sort_keys=True),
                "chain_selection": json.dumps(cmap.get("chain_selection", {}), sort_keys=True),
                "peptide_chain": cmap["peptide_chain"],
                "peptide_sequence": cmap["peptide_sequence"],
                **frag,
                "paratope_sequence": cmap["paratope_sequence"],
                "n_paratope_positions": len(cmap["paratope_sequence"]),
                "n_contacts": len(cmap["contacts"]),
                **bench["summary"],
            }
            label, reasons = reliability_label(row, args.min_contacts, args.min_native_percentile)
            row["reliability"] = label
            row["reliability_reasons"] = reasons
            rows.append(row)
            details.append({
                "pdb": pdb.stem,
                "contact_map": serializable_contact_map(cmap),
                "benchmark": bench,
                "fragment_info": frag,
                "reliability": label,
                "reliability_reasons": reasons,
            })
        except Exception as e:  # noqa: BLE001 - audit must continue across bad files
            rows.append({
                "pdb": pdb.stem,
                "path": str(pdb),
                "status": "failed",
                "error": str(e),
                "reliability": "unusable",
                "reliability_reasons": "parse_or_contact_failure",
            })

    if not rows:
        raise SystemExit("No PDB files found")

    csv_fields = sorted({k for r in rows for k in r.keys()})
    with open(out_dir / "abeta_reference_contact_audit.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    with open(out_dir / "abeta_reference_contact_audit_details.json", "w", encoding="utf-8") as f:
        json.dump({"abeta42": ABETA42, "results": details}, f, indent=2)

    reliable = [r for r in rows if r.get("reliability") == "reliable"]
    questionable = [r for r in rows if r.get("reliability") == "questionable"]
    unusable = [r for r in rows if r.get("reliability") == "unusable"]
    with open(out_dir / "abeta_reference_contact_audit_report.md", "w", encoding="utf-8") as f:
        f.write("# A-beta Reference Contact Audit\n\n")
        f.write(f"Input PDBs: {len(rows)}; reliable: {len(reliable)}; questionable: {len(questionable)}; unusable: {len(unusable)}\n\n")
        f.write("| PDB | Reliability | Reasons | Peptide | A-beta pos | Contacts | Native | Scramble Mean | Percentile | Hotspot Percentile |\n")
        f.write("|---|---|---|---|---|---:|---:|---:|---:|---:|\n")
        for r in rows:
            pos = ""
            if r.get("abeta_start"):
                pos = f"{r.get('abeta_start')}-{r.get('abeta_end')}"
            f.write(
                f"| {r.get('pdb')} | {r.get('reliability')} | {r.get('reliability_reasons', '')} | "
                f"`{r.get('peptide_sequence', '')}` | {pos} | {r.get('n_contacts', '')} | "
                f"{r.get('native_score', '')} | {r.get('scramble_mean_score', '')} | "
                f"{r.get('native_percentile', '')} | {r.get('native_hotspot_percentile', '')} |\n"
            )
        f.write("\nReliable means: peptide is A-beta-like, contact map is non-trivial, and native ranks above scrambled controls.\n")
        f.write("\n## Initial Whitelist YAML\n\n")
        f.write("```yaml\n")
        f.write("abeta_references:\n")
        for r in reliable:
            f.write(f"  - pdb: {r.get('pdb')}\n")
            f.write(f"    path: {r.get('path')}\n")
            f.write(f"    peptide_chain: {r.get('peptide_chain')}\n")
            f.write(f"    peptide_sequence: {r.get('peptide_sequence')}\n")
            f.write(f"    abeta_range: [{r.get('abeta_start')}, {r.get('abeta_end')}]\n")
            f.write(f"    n_contacts: {r.get('n_contacts')}\n")
            f.write(f"    native_percentile: {r.get('native_percentile')}\n")
        f.write("```\n")

    whitelist_dir = PROJECT_ROOT / "configs" / "idp"
    whitelist_dir.mkdir(parents=True, exist_ok=True)
    with open(whitelist_dir / "abeta_reference_whitelist.yml", "w", encoding="utf-8") as f:
        f.write("# Auto-generated by scripts/pipeline/audit_abeta_reference_contacts.py\n")
        f.write("# Include only references whose peptide chain is A-beta-like and whose\n")
        f.write("# native contact map ranks above scrambled controls. Review manually before publication.\n")
        f.write("abeta_references:\n")
        for r in reliable:
            f.write(f"  - pdb: {r.get('pdb')}\n")
            f.write(f"    path: {r.get('path')}\n")
            f.write(f"    peptide_chain: {r.get('peptide_chain')}\n")
            f.write(f"    peptide_sequence: {r.get('peptide_sequence')}\n")
            f.write(f"    abeta_range: [{r.get('abeta_start')}, {r.get('abeta_end')}]\n")
            f.write(f"    n_contacts: {r.get('n_contacts')}\n")
            f.write(f"    native_percentile: {r.get('native_percentile')}\n")
            f.write(f"    native_hotspot_percentile: {r.get('native_hotspot_percentile')}\n")
            f.write("    reliability: reliable\n")

    print(f"Wrote audit to {out_dir}")
    print(f"PDBs={len(rows)} reliable={len(reliable)} questionable={len(questionable)} unusable={len(unusable)}")
    for r in rows:
        print(
            f"{r.get('pdb')}: {r.get('reliability')} {r.get('reliability_reasons', '')} "
            f"peptide={r.get('peptide_sequence', '')} contacts={r.get('n_contacts', '')} "
            f"percentile={r.get('native_percentile', '')}"
        )


if __name__ == "__main__":
    main()
