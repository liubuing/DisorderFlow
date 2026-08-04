#!/usr/bin/env python
"""Developability pre-screen for full-chain shortlist drafts."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
POS = set("KRH")
NEG = set("DE")
AROM = set("FWY")
HYDRO = set("AILMFWYV")


def parse_args():
    parser = argparse.ArgumentParser(description="Score shortlist developability")
    parser.add_argument("--shortlist", required=True, help="synthesis_candidate_shortlist.csv")
    parser.add_argument("--fasta", required=True, help="synthesis_candidate_shortlist.fasta")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-risk", type=float, default=0.55)
    return parser.parse_args()


def load_shortlist(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_construct_fasta(path):
    records = defaultdict(dict)
    header = None
    seq = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header:
                    _store(records, header, "".join(seq))
                header = line[1:]
                seq = []
            else:
                seq.append(line)
    if header:
        _store(records, header, "".join(seq))
    return records


def _store(records, header, seq):
    parts = header.split("|")
    construct = None
    chain_type = None
    for p in parts:
        if p.endswith("_cg_0059") or "_cg_" in p:
            construct = p
        if p in ("heavy", "light"):
            chain_type = p
    if construct and chain_type:
        records[construct][chain_type] = seq


def score_sequence(seq):
    seq = "".join(a for a in seq.upper() if a in KD)
    n = max(1, len(seq))
    charge = sum(1 for a in seq if a in POS) - sum(1 for a in seq if a in NEG)
    hydro_frac = sum(1 for a in seq if a in HYDRO) / n
    arom_frac = sum(1 for a in seq if a in AROM) / n
    cys_count = seq.count("C")
    max_hydro9 = max_window_hydropathy(seq, 9)
    max_run = max_homopolymer(seq)
    n_glyco = len(re.findall(r"N[^P][ST]", seq))
    n_deamidation = len(re.findall(r"N[GSTNQ]", seq))
    n_oxidation = seq.count("M") + seq.count("W")
    risk = 0.0
    risk += 0.20 * min(1.0, max(0.0, (max_hydro9 + 0.5) / 4.0))
    risk += 0.15 * min(1.0, hydro_frac / 0.55)
    risk += 0.10 * min(1.0, arom_frac / 0.18)
    risk += 0.15 if cys_count not in (0, 2, 4, 6, 8) else 0.0
    risk += 0.10 * min(1.0, abs(charge) / 20.0)
    risk += 0.10 * min(1.0, max(0, max_run - 3) / 4.0)
    risk += 0.10 * min(1.0, n_glyco / 2.0)
    risk += 0.05 * min(1.0, n_deamidation / 5.0)
    risk += 0.05 * min(1.0, n_oxidation / 8.0)
    flags = []
    if max_hydro9 > 2.2:
        flags.append("hydrophobic_window")
    if hydro_frac > 0.55:
        flags.append("high_hydrophobic_fraction")
    if max_run >= 5:
        flags.append("homopolymer_run")
    if n_glyco:
        flags.append("n_glycosylation_motif")
    if cys_count not in (0, 2, 4, 6, 8):
        flags.append("odd_cys_count")
    if abs(charge) > 20:
        flags.append("high_net_charge")
    return {
        "length": len(seq),
        "net_charge": charge,
        "hydrophobic_fraction": round(hydro_frac, 4),
        "aromatic_fraction": round(arom_frac, 4),
        "cys_count": cys_count,
        "max_hydro9": round(max_hydro9, 4),
        "max_homopolymer": max_run,
        "n_glycosylation_motifs": n_glyco,
        "n_deamidation_motifs": n_deamidation,
        "n_oxidation_residues": n_oxidation,
        "sequence_risk": round(min(1.0, risk), 4),
        "flags": ";".join(flags),
    }


def max_window_hydropathy(seq, win):
    if not seq:
        return 0.0
    if len(seq) < win:
        return sum(KD[a] for a in seq) / len(seq)
    return max(sum(KD[a] for a in seq[i:i + win]) / win for i in range(len(seq) - win + 1))


def max_homopolymer(seq):
    best = 0
    cur = 0
    last = None
    for a in seq:
        if a == last:
            cur += 1
        else:
            cur = 1
            last = a
        best = max(best, cur)
    return best


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    shortlist = load_shortlist(args.shortlist)
    fasta = load_construct_fasta(args.fasta)
    rows = []
    for r in shortlist:
        cid = r["construct_id"]
        heavy = fasta.get(cid, {}).get("heavy", "")
        light = fasta.get(cid, {}).get("light", "")
        hs = score_sequence(heavy)
        ls = score_sequence(light)
        combined_risk = round(max(hs["sequence_risk"], ls["sequence_risk"]), 4)
        flags = ";".join(x for x in [hs["flags"], ls["flags"]] if x)
        rows.append({
            **r,
            "heavy_risk": hs["sequence_risk"],
            "light_risk": ls["sequence_risk"],
            "combined_developability_risk": combined_risk,
            "developability_status": "pending_inherited_flag_classification",
            "developability_flags": flags,
            "heavy_net_charge": hs["net_charge"],
            "light_net_charge": ls["net_charge"],
            "heavy_max_hydro9": hs["max_hydro9"],
            "light_max_hydro9": ls["max_hydro9"],
            "heavy_nglyco": hs["n_glycosylation_motifs"],
            "light_nglyco": ls["n_glycosylation_motifs"],
        })
    classify_inherited_flags(rows, args.max_risk)
    rows.sort(key=lambda r: (r["developability_status"] != "developability_pass", r["combined_developability_risk"], int(r["shortlist_rank"])))
    for i, r in enumerate(rows, 1):
        r["developability_rank"] = i

    fields = [
        "developability_rank", "shortlist_rank", "construct_id", "reference_pdb", "candidate_id",
        "n_mutations", "candidate_score", "contact_retention", "combined_developability_risk",
        "developability_status", "developability_flags", "heavy_risk", "light_risk",
        "heavy_net_charge", "light_net_charge", "heavy_max_hydro9", "light_max_hydro9",
        "heavy_nglyco", "light_nglyco", "inherited_flags", "introduced_or_candidate_specific_flags",
        "applied_mutations", "h3_mutations", "background_mutations", "h3_mutation_count",
        "total_mutation_count", "nas_status", "nglyco_rescue", "sequence_sha256",
    ]
    with open(out_dir / "shortlist_developability.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "input": len(rows),
        "pass": sum(1 for r in rows if r["developability_status"] == "developability_pass"),
        "review": sum(1 for r in rows if r["developability_status"] != "developability_pass"),
        "max_risk_threshold": args.max_risk,
    }
    with open(out_dir / "shortlist_developability_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "shortlist_developability_report.md", "w", encoding="utf-8") as f:
        f.write("# Shortlist Developability Pre-Screen\n\n")
        f.write(f"Input: {len(rows)}; pass: {summary['pass']}; review: {summary['review']}\n\n")
        f.write("| Dev Rank | Construct | Status | Risk | Inherited Flags | Candidate Flags | Contact Retention | Mutations |\n")
        f.write("|---:|---|---|---:|---|---|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['developability_rank']} | {r['construct_id']} | {r['developability_status']} | "
                f"{r['combined_developability_risk']} | {r['inherited_flags']} | {r['introduced_or_candidate_specific_flags']} | "
                f"{r['contact_retention']} | {r['n_mutations']} |\n"
            )
        f.write("\nThis is a heuristic pre-screen, not a substitute for expression, aggregation, or immunogenicity assays.\n")

    print(f"Wrote developability pre-screen to {out_dir}")
    print(f"Input={len(rows)} pass={summary['pass']} review={summary['review']}")


def classify_inherited_flags(rows, max_risk):
    """Split systematic reference-level flags from candidate-specific flags."""
    by_ref = defaultdict(list)
    for r in rows:
        by_ref[r["reference_pdb"]].append(r)
    inherited_by_ref = {}
    for ref, ref_rows in by_ref.items():
        flag_sets = [set(filter(None, str(r.get("developability_flags", "")).split(";"))) for r in ref_rows]
        inherited_by_ref[ref] = set.intersection(*flag_sets) if flag_sets else set()
    for r in rows:
        flags = set(filter(None, str(r.get("developability_flags", "")).split(";")))
        inherited = flags & inherited_by_ref.get(r["reference_pdb"], set())
        candidate_specific = flags - inherited
        r["inherited_flags"] = ";".join(sorted(inherited))
        r["introduced_or_candidate_specific_flags"] = ";".join(sorted(candidate_specific))
        nas_retained = str(r.get("nas_status", "")).startswith("retained")
        if r["combined_developability_risk"] <= max_risk and not candidate_specific and not nas_retained:
            r["developability_status"] = "developability_pass"
        else:
            r["developability_status"] = "developability_review"


if __name__ == "__main__":
    main()
