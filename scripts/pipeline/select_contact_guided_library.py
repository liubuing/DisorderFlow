#!/usr/bin/env python
"""Select a balanced, diverse contact-guided candidate mini-library."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Select contact-guided mini-library")
    parser.add_argument("--candidates", required=True,
                        help="contact_guided_candidates_all.csv from generate_abeta_contact_guided_candidates.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--top", type=int, default=24)
    parser.add_argument("--per-ref", type=int, default=12)
    parser.add_argument("--min-retention", type=float, default=0.90)
    parser.add_argument("--max-retention", type=float, default=1.12,
                        help="Reject overly optimized scorer artifacts above this retention")
    parser.add_argument("--min-complexity", type=float, default=0.70)
    parser.add_argument("--max-immuno", type=float, default=0.45)
    parser.add_argument("--max-mutations", type=int, default=4)
    parser.add_argument("--min-hamming", type=int, default=2,
                        help="Minimum Hamming distance within the same reference bucket")
    return parser.parse_args()


def load_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(_coerce(row))
    return rows


def _coerce(row):
    numeric = {
        "rank", "global_rank", "n_mutations", "state_contact_score", "hotspot_score",
        "contact_retention", "complexity", "immunogenicity_proxy", "candidate_score",
    }
    for k in numeric:
        if k in row and row[k] not in (None, ""):
            row[k] = float(row[k]) if "." in str(row[k]) else int(row[k])
    row["passes_filters"] = str(row.get("passes_filters", "")).lower() == "true"
    return row


def strict_filter(rows, args):
    out = []
    for r in rows:
        reasons = []
        if not r.get("passes_filters"):
            reasons.append("prior_filter_fail")
        if r.get("contact_retention", 0) < args.min_retention:
            reasons.append("low_retention")
        if r.get("contact_retention", 0) > args.max_retention:
            reasons.append("overoptimized_retention")
        if r.get("complexity", 0) < args.min_complexity:
            reasons.append("low_complexity")
        if r.get("immunogenicity_proxy", 1) > args.max_immuno:
            reasons.append("high_immunogenicity_proxy")
        if r.get("n_mutations", 99) > args.max_mutations:
            reasons.append("too_many_mutations")
        if _max_homopolymer(r.get("sequence", "")) >= 4:
            reasons.append("homopolymer_run")
        r = dict(r)
        r["strict_pass"] = not reasons
        r["strict_failures"] = ";".join(reasons)
        if not reasons:
            out.append(r)
    return out


def select_library(rows, args):
    by_ref = defaultdict(list)
    for r in rows:
        by_ref[r["reference_pdb"]].append(r)
    selected = []
    ref_summaries = {}
    for ref, ref_rows in sorted(by_ref.items()):
        ref_rows.sort(key=lambda r: (r["candidate_score"], r["contact_retention"], -r["immunogenicity_proxy"]), reverse=True)
        buckets = _mutation_buckets(ref_rows)
        per_ref_selected = []
        # Round-robin across mutation-strength buckets so the library is not all
        # 1-mutation near-native variants or all aggressive variants.
        while len(per_ref_selected) < args.per_ref:
            progressed = False
            for bucket in ("low", "medium", "high"):
                for cand in buckets[bucket]:
                    if cand.get("_used"):
                        continue
                    if _is_diverse_enough(cand, per_ref_selected, args.min_hamming):
                        cand["_used"] = True
                        cand["selection_bucket"] = bucket
                        per_ref_selected.append(cand)
                        progressed = True
                        break
                if len(per_ref_selected) >= args.per_ref:
                    break
            if not progressed:
                break
        selected.extend(per_ref_selected)
        ref_summaries[ref] = {
            "available_after_strict_filter": len(ref_rows),
            "selected": len(per_ref_selected),
            "bucket_counts": dict(Counter(r.get("selection_bucket", "none") for r in per_ref_selected)),
        }
    selected.sort(key=lambda r: (r["candidate_score"], r["contact_retention"]), reverse=True)
    selected = selected[:args.top]
    for i, r in enumerate(selected, 1):
        r["selection_rank"] = i
        r.pop("_used", None)
    return selected, ref_summaries


def _mutation_buckets(rows):
    buckets = {"low": [], "medium": [], "high": []}
    for r in rows:
        n = int(r.get("n_mutations", 0))
        if n <= 1:
            buckets["low"].append(r)
        elif n <= 3:
            buckets["medium"].append(r)
        else:
            buckets["high"].append(r)
    return buckets


def _is_diverse_enough(cand, selected, min_hamming):
    return all(_hamming(cand["sequence"], s["sequence"]) >= min_hamming for s in selected)


def _hamming(a, b):
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(x != y for x, y in zip(a, b))


def _max_homopolymer(seq):
    best = 0
    cur = 0
    last = None
    for c in seq:
        if c == last:
            cur += 1
        else:
            cur = 1
            last = c
        best = max(best, cur)
    return best


def write_outputs(selected, filtered, all_rows, ref_summaries, out_dir: Path, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "selection_rank", "reference_pdb", "candidate_id", "sequence", "mutations",
        "n_mutations", "selection_bucket", "peptide_sequence", "native_paratope",
        "state_contact_score", "hotspot_score", "contact_retention", "complexity",
        "immunogenicity_proxy", "candidate_score", "strict_pass", "strict_failures",
    ]
    with open(out_dir / "selected_contact_guided_library.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    with open(out_dir / "selected_contact_guided_library.fasta", "w", encoding="utf-8") as f:
        for r in selected:
            f.write(
                f">{r['reference_pdb']}|{r['candidate_id']}|selection_rank={r['selection_rank']}"
                f"|score={r['candidate_score']}|retention={r['contact_retention']}|bucket={r.get('selection_bucket')}\n"
                f"{r['sequence']}\n"
            )
    summary = {
        "input_candidates": len(all_rows),
        "strict_pass": len(filtered),
        "selected": len(selected),
        "selection_args": vars(args),
        "per_reference": ref_summaries,
        "selected_by_reference": dict(Counter(r["reference_pdb"] for r in selected)),
        "selected_by_bucket": dict(Counter(r.get("selection_bucket", "none") for r in selected)),
    }
    with open(out_dir / "selected_contact_guided_library_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "selected_contact_guided_library_report.md", "w", encoding="utf-8") as f:
        f.write("# Selected Contact-Guided Mini-Library\n\n")
        f.write(f"Input candidates: {len(all_rows)}; strict pass: {len(filtered)}; selected: {len(selected)}\n\n")
        f.write(f"Selected by reference: `{summary['selected_by_reference']}`\n\n")
        f.write(f"Selected by mutation bucket: `{summary['selected_by_bucket']}`\n\n")
        f.write("| Rank | Ref | Candidate | Sequence | Mutations | Bucket | Score | Retention | Complexity | Immuno |\n")
        f.write("|---:|---|---|---|---|---|---:|---:|---:|---:|\n")
        for r in selected:
            f.write(
                f"| {r['selection_rank']} | {r['reference_pdb']} | {r['candidate_id']} | `{r['sequence']}` | "
                f"{r['mutations']} | {r.get('selection_bucket')} | {r['candidate_score']} | "
                f"{r['contact_retention']} | {r['complexity']} | {r['immunogenicity_proxy']} |\n"
            )
        f.write("\nThis is a balanced paratope mini-library, not complete antibody constructs.\n")


def main():
    args = parse_args()
    rows = load_rows(args.candidates)
    filtered = strict_filter(rows, args)
    selected, ref_summaries = select_library(filtered, args)
    write_outputs(selected, filtered, rows, ref_summaries, Path(args.out), args)
    print(f"Wrote selected mini-library to {args.out}")
    print(f"Input={len(rows)} strict_pass={len(filtered)} selected={len(selected)}")
    print(f"Selected by reference: {dict(Counter(r['reference_pdb'] for r in selected))}")
    print(f"Selected by bucket: {dict(Counter(r.get('selection_bucket', 'none') for r in selected))}")


if __name__ == "__main__":
    main()
