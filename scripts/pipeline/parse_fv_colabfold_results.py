#!/usr/bin/env python
"""Parse ColabFold Fv side-check outputs into a fold evidence table."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Parse Fv ColabFold results")
    parser.add_argument("--shortlist", required=True, help="synthesis_candidate_shortlist_v2.csv")
    parser.add_argument("--result-dir", required=True, help="ColabFold output directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode-label", default="single_sequence_cpu_smoke")
    return parser.parse_args()


def load_shortlist(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def construct_from_score_path(path):
    stem = path.name
    if "_candidate_uid_" in stem:
        return stem.split("_candidate_uid_", 1)[0]
    if "_Fv_HL_pair" in stem:
        return stem.split("_Fv_HL_pair", 1)[0]
    if "_scores_rank" in stem:
        return stem.split("_scores_rank", 1)[0]
    return stem


def mean(values):
    return sum(values) / max(len(values), 1)


def fold_status(mean_plddt, iptm, mode_label):
    if mode_label.endswith("smoke") or "smoke" in mode_label:
        return "tool_smoke_not_fold_pass_evidence"
    if mean_plddt >= 70.0 and (iptm is None or iptm >= 0.5):
        return "fold_sidecheck_pass"
    return "fold_sidecheck_review"


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    shortlist_rows = load_shortlist(args.shortlist)
    shortlist = {
        (row.get("construct_id") or row.get("candidate_uid")): row
        for row in shortlist_rows
    }
    result_dir = Path(args.result_dir)
    score_files = sorted(result_dir.glob("*_scores_rank_001_*.json"))
    rows = []
    for sf in score_files:
        with open(sf, encoding="utf-8") as f:
            scores = json.load(f)
        cid = construct_from_score_path(sf)
        # ColabFold sanitizes metadata-rich FASTA headers into the filename.
        # Recover the authoritative construct ID from the shortlist prefix.
        matches = [construct_id for construct_id in shortlist if cid.startswith(construct_id)]
        if len(matches) == 1:
            cid = matches[0]
        plddt = scores.get("plddt") or []
        mean_plddt = round(mean(plddt), 2)
        iptm = scores.get("iptm")
        ptm = scores.get("ptm")
        rows.append({
            "construct_id": cid,
            "shortlist_rank": shortlist.get(cid, {}).get(
                "shortlist_rank", shortlist.get(cid, {}).get("family_panel_rank", "")
            ),
            "reference_pdb": shortlist.get(cid, {}).get("reference_pdb", ""),
            "candidate_id": shortlist.get(cid, {}).get(
                "candidate_id", shortlist.get(cid, {}).get("candidate_uid", "")
            ),
            "mode_label": args.mode_label,
            "mean_plddt": mean_plddt,
            "ptm": round(ptm, 4) if isinstance(ptm, (int, float)) else "",
            "iptm": round(iptm, 4) if isinstance(iptm, (int, float)) else "",
            "max_pae": scores.get("max_pae", ""),
            "fold_sidecheck_status": fold_status(mean_plddt, iptm, args.mode_label),
            "score_json": str(sf),
        })
    fields = [
        "construct_id", "shortlist_rank", "reference_pdb", "candidate_id", "mode_label",
        "mean_plddt", "ptm", "iptm", "max_pae", "fold_sidecheck_status", "score_json",
    ]
    with open(out_dir / "fv_colabfold_evidence.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "result_dir": str(result_dir),
        "parsed_results": len(rows),
        "mode_label": args.mode_label,
        "pass": sum(1 for r in rows if r["fold_sidecheck_status"] == "fold_sidecheck_pass"),
        "review_or_smoke": sum(1 for r in rows if r["fold_sidecheck_status"] != "fold_sidecheck_pass"),
    }
    with open(out_dir / "fv_colabfold_evidence_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "fv_colabfold_evidence_report.md", "w", encoding="utf-8") as f:
        f.write("# Fv ColabFold Side-Check Evidence\n\n")
        f.write(f"Parsed results: {len(rows)}; mode: `{args.mode_label}`\n\n")
        f.write("| Construct | pLDDT | pTM | ipTM | Status |\n")
        f.write("|---|---:|---:|---:|---|\n")
        for r in rows:
            f.write(f"| {r['construct_id']} | {r['mean_plddt']} | {r['ptm']} | {r['iptm']} | {r['fold_sidecheck_status']} |\n")
        f.write("\nSingle-sequence CPU smoke runs verify tooling only; they are not synthesis-grade fold evidence.\n")
    print(f"Wrote Fv ColabFold evidence to {out_dir}")
    print(f"Parsed={len(rows)} pass={summary['pass']} review_or_smoke={summary['review_or_smoke']}")


if __name__ == "__main__":
    main()
