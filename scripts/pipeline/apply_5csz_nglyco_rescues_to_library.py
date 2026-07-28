#!/usr/bin/env python
"""Apply top 5CSZ N-glyco rescue variants to the selected library."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Apply top 5CSZ N-glyco rescues")
    parser.add_argument("--selected", required=True)
    parser.add_argument("--rescues", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mutation_string(native, seq):
    return ";".join(
        f"{n}{i + 1}{s}" for i, (n, s) in enumerate(zip(native, seq)) if n != s
    )


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = load_csv(args.selected)
    rescues = [r for r in load_csv(args.rescues) if r.get("passes_rescue_filter") == "True"]
    best_by_source = {}
    for r in rescues:
        src = ("5CSZ", r["source_candidate_id"])
        if src not in best_by_source:
            best_by_source[src] = r

    rows = []
    applied = []
    for row in selected:
        rescue = best_by_source.get((row["reference_pdb"], row["candidate_id"]))
        if rescue:
            native = row["native_paratope"]
            seq = rescue["rescue_sequence"]
            row = dict(row)
            row["candidate_id"] = rescue["rescue_candidate_id"]
            row["sequence"] = seq
            row["mutations"] = mutation_string(native, seq)
            row["n_mutations"] = str(sum(1 for a, b in zip(native, seq) if a != b))
            row["state_contact_score"] = ""
            row["hotspot_score"] = rescue["rescue_hotspot_score"]
            row["contact_retention"] = rescue["rescue_contact_retention"]
            row["candidate_score"] = rescue["rescue_candidate_score"]
            row["strict_pass"] = "True"
            row["strict_failures"] = ""
            applied.append({
                "selection_rank": row["selection_rank"],
                "source_candidate_id": rescue["source_candidate_id"],
                "rescue_candidate_id": rescue["rescue_candidate_id"],
                "rescue_mutation": rescue["rescue_mutation"],
            })
        rows.append(row)

    fields = list(rows[0].keys()) if rows else []
    with open(out_dir / "selected_contact_guided_library.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "applied_5csz_nglyco_rescues.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["selection_rank", "source_candidate_id", "rescue_candidate_id", "rescue_mutation"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(applied)
    with open(out_dir / "selected_contact_guided_library.fasta", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(f">{r['reference_pdb']}|{r['candidate_id']}|rank={r['selection_rank']}|score={r['candidate_score']}|retention={r['contact_retention']}\n{r['sequence']}\n")
    print(f"Wrote rescued selected library to {out_dir}")
    print(f"Input={len(selected)} rescues_applied={len(applied)}")


if __name__ == "__main__":
    main()
