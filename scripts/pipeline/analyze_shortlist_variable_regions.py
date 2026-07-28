#!/usr/bin/env python
"""Heuristic VH/VL boundary and inherited developability flag localization.

This is an ANARCI substitute only for triage. It identifies common antibody J-region
motifs, separates variable regions from trailing constant/construct sequence, and
checks whether inherited sequence flags are actually in VH/VL.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_shortlist_developability import score_sequence  # noqa: E402


HEAVY_J_MOTIFS = (
    "WGQGTLVTVSS", "WGQGTLVTVSA", "WGQGTTVTVSS", "WGQGTSVTVSS",
)
LIGHT_J_MOTIFS = (
    "FGQGTKVEIK", "FGQGTKLEIK", "FGGGTKVEIK", "FGQGTRLEIK", "FGGGTKLDIK",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze shortlist VH/VL variable regions")
    parser.add_argument("--shortlist", required=True, help="synthesis_candidate_shortlist.csv")
    parser.add_argument("--fasta", required=True, help="synthesis_candidate_shortlist.fasta")
    parser.add_argument("--developability", required=True, help="shortlist_developability.csv")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_csv(path):
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
                    store_record(records, header, "".join(seq))
                header = line[1:]
                seq = []
            else:
                seq.append(line)
    if header:
        store_record(records, header, "".join(seq))
    return records


def store_record(records, header, seq):
    parts = header.split("|")
    construct = next((p for p in parts if "_cg_" in p), None)
    chain_type = next((p for p in parts if p in ("heavy", "light")), None)
    if construct and chain_type:
        records[construct][chain_type] = seq


def find_variable_region(seq, chain_type):
    motifs = HEAVY_J_MOTIFS if chain_type == "heavy" else LIGHT_J_MOTIFS
    best = None
    for motif in motifs:
        idx = seq.find(motif)
        if idx >= 0:
            end = idx + len(motif)
            if best is None or end > best[0]:
                best = (end, motif)
    if best:
        end, motif = best
        return {
            "variable_seq": seq[:end],
            "trailing_seq": seq[end:],
            "boundary_status": "j_motif_found",
            "boundary_motif": motif,
        }
    fallback = 120 if chain_type == "heavy" else 112
    return {
        "variable_seq": seq[:fallback],
        "trailing_seq": seq[fallback:],
        "boundary_status": "fallback_length_cut_low_confidence",
        "boundary_motif": "",
    }


def motif_sites(seq, pattern):
    return [f"{m.start() + 1}:{m.group(0)}" for m in re.finditer(pattern, seq)]


def cys_sites(seq):
    return [str(i + 1) for i, aa in enumerate(seq) if aa == "C"]


def flag_region(flag, full_seq, variable_seq, trailing_seq):
    if flag == "n_glycosylation_motif":
        in_var = bool(motif_sites(variable_seq, r"N[^P][ST]"))
        in_tail = bool(motif_sites(trailing_seq, r"N[^P][ST]"))
    elif flag == "odd_cys_count":
        in_var = score_sequence(variable_seq)["cys_count"] not in (0, 2, 4, 6, 8)
        in_tail = score_sequence(trailing_seq)["cys_count"] not in (0, 2, 4, 6, 8)
    elif flag == "hydrophobic_window":
        in_var = score_sequence(variable_seq)["max_hydro9"] > 2.2
        in_tail = score_sequence(trailing_seq)["max_hydro9"] > 2.2
    else:
        in_var = flag in score_sequence(variable_seq)["flags"].split(";")
        in_tail = flag in score_sequence(trailing_seq)["flags"].split(";")
    if in_var and in_tail:
        return "variable_and_trailing"
    if in_var:
        return "variable"
    if in_tail:
        return "trailing_or_constant"
    if flag in score_sequence(full_seq)["flags"].split(";"):
        return "full_chain_only_or_pairing_artifact"
    return "not_detected"


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    shortlist = load_csv(args.shortlist)
    dev_by_id = {r["construct_id"]: r for r in load_csv(args.developability)}
    fasta = load_construct_fasta(args.fasta)

    chain_rows = []
    construct_rows = []
    variable_fasta = []
    for r in shortlist:
        cid = r["construct_id"]
        dev = dev_by_id.get(cid, {})
        inherited_flags = [f for f in dev.get("inherited_flags", "").split(";") if f]
        unresolved = []
        per_chain_regions = []
        for chain_type in ("heavy", "light"):
            seq = fasta.get(cid, {}).get(chain_type, "")
            cut = find_variable_region(seq, chain_type)
            var = cut["variable_seq"]
            tail = cut["trailing_seq"]
            var_score = score_sequence(var)
            tail_score = score_sequence(tail)
            variable_fasta.append(f">{cid}|{chain_type}|heuristic_variable_region")
            variable_fasta.append(var)
            localized = []
            for flag in inherited_flags:
                region = flag_region(flag, seq, var, tail)
                localized.append(f"{flag}:{region}")
                if region in ("variable", "variable_and_trailing", "full_chain_only_or_pairing_artifact"):
                    unresolved.append(f"{chain_type}:{flag}:{region}")
            per_chain_regions.append(cut["boundary_status"])
            chain_rows.append({
                "construct_id": cid,
                "reference_pdb": r["reference_pdb"],
                "chain_type": chain_type,
                "full_length": len(seq),
                "variable_length": len(var),
                "trailing_length": len(tail),
                "boundary_status": cut["boundary_status"],
                "boundary_motif": cut["boundary_motif"],
                "variable_risk": var_score["sequence_risk"],
                "trailing_risk": tail_score["sequence_risk"],
                "variable_flags": var_score["flags"],
                "trailing_flags": tail_score["flags"],
                "variable_cys_count": var_score["cys_count"],
                "trailing_cys_count": tail_score["cys_count"],
                "variable_cys_sites": ";".join(cys_sites(var)),
                "trailing_cys_sites": ";".join(cys_sites(tail)),
                "variable_nglyco_sites": ";".join(motif_sites(var, r"N[^P][ST]")),
                "trailing_nglyco_sites": ";".join(motif_sites(tail, r"N[^P][ST]")),
                "inherited_flag_localization": ";".join(localized),
            })
        if unresolved:
            status = "variable_region_review"
        elif inherited_flags:
            status = "inherited_flags_outside_variable_region"
        else:
            status = "no_inherited_flags"
        if any(s != "j_motif_found" for s in per_chain_regions):
            status = "boundary_review"
        construct_rows.append({
            **r,
            "developability_status": dev.get("developability_status", ""),
            "inherited_flags": dev.get("inherited_flags", ""),
            "candidate_specific_flags": dev.get("introduced_or_candidate_specific_flags", ""),
            "variable_region_status": status,
            "variable_region_unresolved_flags": ";".join(unresolved),
        })

    write_csv(out_dir / "variable_region_chain_evidence.csv", chain_rows)
    write_csv(out_dir / "variable_region_construct_evidence.csv", construct_rows)
    with open(out_dir / "shortlist_variable_regions.fasta", "w", encoding="utf-8") as f:
        f.write("\n".join(variable_fasta) + "\n")
    summary = {
        "input_constructs": len(construct_rows),
        "j_motif_boundaries": sum(1 for r in chain_rows if r["boundary_status"] == "j_motif_found"),
        "chains": len(chain_rows),
        "construct_status_counts": dict(counts(r["variable_region_status"] for r in construct_rows)),
    }
    with open(out_dir / "variable_region_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_report(out_dir / "variable_region_report.md", summary, construct_rows)
    print(f"Wrote variable-region evidence to {out_dir}")
    print(f"Input={len(construct_rows)} status_counts={summary['construct_status_counts']}")


def counts(values):
    out = defaultdict(int)
    for v in values:
        out[v] += 1
    return out


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, summary, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Variable Region Boundary Evidence\n\n")
        f.write(f"Input constructs: {summary['input_constructs']}\n\n")
        f.write(f"J-motif chain boundaries: {summary['j_motif_boundaries']}/{summary['chains']}\n\n")
        f.write("| Shortlist Rank | Construct | Status | Inherited Flags | Unresolved Variable Flags |\n")
        f.write("|---:|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['shortlist_rank']} | {r['construct_id']} | {r['variable_region_status']} | "
                f"{r['inherited_flags']} | {r['variable_region_unresolved_flags']} |\n"
            )
        f.write("\nThis is a motif-based triage layer, not ANARCI/Chothia numbering.\n")


if __name__ == "__main__":
    main()
