#!/usr/bin/env python
"""Summarize ANARCI Chothia numbering outputs for VH/VL shortlist."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize ANARCI Chothia numbering")
    parser.add_argument("--heavy", required=True, help="ANARCI *_H.csv")
    parser.add_argument("--light", required=True, help="ANARCI *_KL.csv")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def construct_id(anarci_id):
    return str(anarci_id).split("|", 1)[0]


def chain_label(anarci_id):
    parts = str(anarci_id).split("|")
    return parts[1] if len(parts) > 1 else ""


def summarize_row(r):
    return {
        "anarci_id": r["Id"],
        "construct_id": construct_id(r["Id"]),
        "input_chain_label": chain_label(r["Id"]),
        "anarci_chain_type": r.get("chain_type", ""),
        "hmm_species": r.get("hmm_species", ""),
        "e_value": r.get("e-value", ""),
        "score": r.get("score", ""),
        "seqstart_index": r.get("seqstart_index", ""),
        "seqend_index": r.get("seqend_index", ""),
        "numbering_status": "anarci_chothia_numbered",
    }


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    heavy_rows = [summarize_row(r) for r in load_csv(args.heavy)]
    light_rows = [summarize_row(r) for r in load_csv(args.light)]
    chain_rows = heavy_rows + light_rows
    by_construct = defaultdict(dict)
    for r in chain_rows:
        by_construct[r["construct_id"]][r["input_chain_label"]] = r

    construct_rows = []
    for cid in sorted(by_construct):
        h = by_construct[cid].get("heavy")
        l = by_construct[cid].get("light")
        status = "anarci_chothia_numbered" if h and l else "anarci_incomplete"
        construct_rows.append({
            "construct_id": cid,
            "numbering_status": status,
            "heavy_chain_type": h.get("anarci_chain_type", "") if h else "",
            "light_chain_type": l.get("anarci_chain_type", "") if l else "",
            "heavy_species": h.get("hmm_species", "") if h else "",
            "light_species": l.get("hmm_species", "") if l else "",
            "heavy_score": h.get("score", "") if h else "",
            "light_score": l.get("score", "") if l else "",
            "heavy_e_value": h.get("e_value", "") if h else "",
            "light_e_value": l.get("e_value", "") if l else "",
        })

    write_csv(out_dir / "anarci_chothia_chain_summary.csv", chain_rows)
    write_csv(out_dir / "anarci_chothia_construct_summary.csv", construct_rows)
    summary = {
        "heavy_chains": len(heavy_rows),
        "light_chains": len(light_rows),
        "constructs": len(construct_rows),
        "complete_constructs": sum(1 for r in construct_rows if r["numbering_status"] == "anarci_chothia_numbered"),
        "incomplete_constructs": sum(1 for r in construct_rows if r["numbering_status"] != "anarci_chothia_numbered"),
    }
    with open(out_dir / "anarci_chothia_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "anarci_chothia_report.md", "w", encoding="utf-8") as f:
        f.write("# ANARCI Chothia Numbering Summary\n\n")
        f.write(
            f"Constructs: {summary['constructs']}; complete: {summary['complete_constructs']}; "
            f"incomplete: {summary['incomplete_constructs']}\n\n"
        )
        f.write("| Construct | Status | Heavy | Light | Heavy Score | Light Score |\n")
        f.write("|---|---|---|---|---:|---:|\n")
        for r in construct_rows:
            f.write(
                f"| {r['construct_id']} | {r['numbering_status']} | {r['heavy_chain_type']} | "
                f"{r['light_chain_type']} | {r['heavy_score']} | {r['light_score']} |\n"
            )
    print(f"Wrote ANARCI Chothia summary to {out_dir}")
    print(f"Constructs={summary['constructs']} complete={summary['complete_constructs']} incomplete={summary['incomplete_constructs']}")


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
