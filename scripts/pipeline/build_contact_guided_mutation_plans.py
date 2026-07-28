#!/usr/bin/env python
"""Build residue-level mutation plans for selected contact-guided candidates."""
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

from state_contact_scorer import extract_contact_map  # noqa: E402


# Broad CDR-like residue windows for risk labeling only. These are not a
# replacement for ANARCI/Chothia numbering; they flag obvious framework hits.
CDR_WINDOWS = {
    "H": [(26, 35), (50, 65), (95, 105)],
    "L": [(24, 34), (50, 56), (89, 97)],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build mutation plans for selected paratope candidates")
    parser.add_argument("--selected", required=True,
                        help="selected_contact_guided_library.csv")
    parser.add_argument("--whitelist", default=str(PROJECT_ROOT / "configs" / "idp" / "abeta_reference_whitelist.yml"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--contact-cutoff", type=float, default=8.0)
    return parser.parse_args()


def load_selected(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_refs(path):
    with open(path, encoding="utf-8") as f:
        refs = (yaml.safe_load(f) or {}).get("abeta_references", [])
    return {r["pdb"]: r for r in refs}


def region_label(chain, resid):
    chain = str(chain).upper()
    windows = CDR_WINDOWS.get(chain, [])
    for idx, (start, end) in enumerate(windows, 1):
        if start <= int(resid) <= end:
            return f"CDR{idx}"
    return "framework_or_unknown"


def infer_chain_roles(contact_map):
    """Infer antibody heavy/light roles for chain IDs.

    Some reference PDBs use A/B instead of H/L for antibody chains. This simple
    sequence-pattern inference is only for risk labeling; it does not alter the
    underlying mutation plan coordinates.
    """
    roles = {}
    chain_info = contact_map.get("chain_selection", {}).get("chains", [])
    for c in chain_info:
        cid = str(c.get("chain", "")).upper()
        seq = str(c.get("sequence", "")).upper()
        if cid in ("H", "L"):
            roles[cid] = cid
            continue
        if len(seq) < 80:
            continue
        if seq.startswith(("QVQL", "EVQL", "VQL", "VELV", "SEVQ")) or "CAR" in seq[:130]:
            roles[cid] = "H"
        elif seq.startswith(("DIQ", "DIV", "YVV", "EIV", "QIV")) or "LTQ" in seq[:25] or "MTQ" in seq[:25]:
            roles[cid] = "L"
    return roles


def build_plan_for_candidate(row, contact_map):
    native = contact_map["paratope_sequence"]
    candidate = row["sequence"]
    residues = contact_map["paratope_residues"]
    chain_roles = infer_chain_roles(contact_map)
    if len(candidate) != len(native):
        raise ValueError(f"Candidate length mismatch for {row.get('candidate_id')}")
    plan = []
    for i, (n_aa, c_aa) in enumerate(zip(native, candidate)):
        if n_aa == c_aa:
            continue
        res = residues[i]
        inferred_role = chain_roles.get(str(res["chain"]).upper(), str(res["chain"]).upper())
        label = region_label(inferred_role, res["resid"])
        plan.append({
            "reference_pdb": row["reference_pdb"],
            "candidate_id": row["candidate_id"],
            "selection_rank": row.get("selection_rank"),
            "paratope_index": i + 1,
            "chain": res["chain"],
            "chain_role": inferred_role,
            "resid": res["resid"],
            "region_label": label,
            "native_aa": n_aa,
            "mutant_aa": c_aa,
            "mutation": f"{res['chain']}:{res['resid']}:{n_aa}>{c_aa}",
        })
    return plan


def risk_summary(plan):
    n = len(plan)
    framework = [p for p in plan if p["region_label"] == "framework_or_unknown"]
    cdr = [p for p in plan if p["region_label"] != "framework_or_unknown"]
    reasons = []
    if n == 0:
        reasons.append("no_mutations")
    if framework:
        reasons.append("framework_or_unknown_mutations")
    if n > 4:
        reasons.append("too_many_mutations")
    if any(p["mutant_aa"] == "C" for p in plan):
        reasons.append("introduces_cys")
    return {
        "n_mutations_mapped": n,
        "n_cdr_like_mutations": len(cdr),
        "n_framework_or_unknown_mutations": len(framework),
        "construct_risk": "high" if framework else "medium" if n > 3 else "low",
        "risk_reasons": ";".join(reasons),
    }


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_selected(args.selected)
    refs = load_refs(args.whitelist)
    contact_maps = {}
    for pdb, ref in refs.items():
        contact_maps[pdb] = extract_contact_map(ref["path"], cutoff=args.contact_cutoff,
                                                peptide_chain=ref.get("peptide_chain"))

    plan_rows = []
    summary_rows = []
    details = []
    for row in rows:
        cmap = contact_maps[row["reference_pdb"]]
        plan = build_plan_for_candidate(row, cmap)
        summary = risk_summary(plan)
        summary_row = {
            "selection_rank": row.get("selection_rank"),
            "reference_pdb": row["reference_pdb"],
            "candidate_id": row["candidate_id"],
            "sequence": row["sequence"],
            "native_paratope": row.get("native_paratope"),
            "contact_retention": row.get("contact_retention"),
            "candidate_score": row.get("candidate_score"),
            **summary,
        }
        summary_rows.append(summary_row)
        plan_rows.extend(plan)
        details.append({"candidate": row, "mutation_plan": plan, "risk_summary": summary})

    with open(out_dir / "mutation_plan_residues.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["selection_rank", "reference_pdb", "candidate_id", "paratope_index", "chain", "chain_role", "resid",
                  "region_label", "native_aa", "mutant_aa", "mutation"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(plan_rows)
    with open(out_dir / "mutation_plan_summary.csv", "w", newline="", encoding="utf-8") as f:
        fields = list(summary_rows[0].keys()) if summary_rows else []
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)
    with open(out_dir / "mutation_plan_details.json", "w", encoding="utf-8") as f:
        json.dump({
            "cdr_windows_for_risk_labeling": CDR_WINDOWS,
            "contact_maps": {k: {**{kk: vv for kk, vv in v.items() if kk != "contacts"},
                                  "contacts": [asdict(c) for c in v["contacts"]]}
                             for k, v in contact_maps.items()},
            "candidates": details,
        }, f, indent=2)
    with open(out_dir / "mutation_plan_report.md", "w", encoding="utf-8") as f:
        high = sum(1 for r in summary_rows if r["construct_risk"] == "high")
        medium = sum(1 for r in summary_rows if r["construct_risk"] == "medium")
        low = sum(1 for r in summary_rows if r["construct_risk"] == "low")
        f.write("# Contact-Guided Mutation Plan Report\n\n")
        f.write(f"Candidates: {len(summary_rows)}; low risk: {low}; medium risk: {medium}; high risk: {high}\n\n")
        f.write("| Rank | Ref | Candidate | Mutations | CDR-like | Framework/Unknown | Risk | Reasons |\n")
        f.write("|---:|---|---|---:|---:|---:|---|---|\n")
        for r in summary_rows:
            f.write(
                f"| {r['selection_rank']} | {r['reference_pdb']} | {r['candidate_id']} | "
                f"{r['n_mutations_mapped']} | {r['n_cdr_like_mutations']} | "
                f"{r['n_framework_or_unknown_mutations']} | {r['construct_risk']} | {r['risk_reasons']} |\n"
            )
        f.write("\nFramework/unknown labels use broad residue windows only; confirm with ANARCI/Chothia before synthesis.\n")

    print(f"Wrote mutation plans to {out_dir}")
    print(f"Candidates={len(summary_rows)} low={sum(1 for r in summary_rows if r['construct_risk']=='low')} medium={sum(1 for r in summary_rows if r['construct_risk']=='medium')} high={sum(1 for r in summary_rows if r['construct_risk']=='high')}")


if __name__ == "__main__":
    main()
