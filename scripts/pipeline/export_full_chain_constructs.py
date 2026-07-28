#!/usr/bin/env python
"""Export full-chain mutated constructs from mutation plans.

This uses PDB residue order and residue numbers. It does not perform ANARCI or
Chothia renumbering; the output is a construct-level sequence draft for further
numbering, side-chain rebuilding, and fold/stability side checks.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "modules"))

from state_contact_scorer import AA3_TO_1, extract_contact_map  # noqa: E402
from build_contact_guided_mutation_plans import infer_chain_roles  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Export full-chain constructs from mutation plans")
    parser.add_argument("--selected", required=True, help="selected_contact_guided_library.csv")
    parser.add_argument("--mutation-plan", required=True, help="mutation_plan_residues.csv")
    parser.add_argument("--whitelist", default=str(PROJECT_ROOT / "configs" / "idp" / "abeta_reference_whitelist.yml"))
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_refs(path):
    with open(path, encoding="utf-8") as f:
        return {r["pdb"]: r for r in (yaml.safe_load(f) or {}).get("abeta_references", [])}


def pdb_chain_sequences(pdb_path):
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(pdb_path))
    out = {}
    for model in structure:
        for chain in model:
            residues = []
            seen = set()
            for res in chain:
                aa = AA3_TO_1.get(res.resname.strip())
                if not aa:
                    continue
                resid = res.id[1]
                icode = res.id[2].strip() if len(res.id) > 2 else ""
                key = (resid, icode)
                if key in seen:
                    continue
                seen.add(key)
                residues.append({"resid": resid, "icode": icode, "aa": aa})
            if residues:
                out[chain.id.strip() or "_"] = residues
        break
    return out


def apply_candidate_mutations(chain_res, mutations):
    seq_by_chain = {cid: [r["aa"] for r in residues] for cid, residues in chain_res.items()}
    resid_to_idx = {
        cid: {(r["resid"], r["icode"]): i for i, r in enumerate(residues)}
        for cid, residues in chain_res.items()
    }
    applied = []
    warnings = []
    for m in mutations:
        cid = m["chain"]
        resid = int(m["resid"])
        native = m["native_aa"]
        mutant = m["mutant_aa"]
        if cid not in seq_by_chain:
            warnings.append(f"missing_chain:{cid}")
            continue
        key = None
        for k in resid_to_idx[cid]:
            if k[0] == resid:
                key = k
                break
        if key is None:
            warnings.append(f"missing_residue:{cid}:{resid}")
            continue
        idx = resid_to_idx[cid][key]
        observed = seq_by_chain[cid][idx]
        if observed != native:
            warnings.append(f"native_mismatch:{cid}:{resid}:{observed}!={native}")
        seq_by_chain[cid][idx] = mutant
        applied.append(m["mutation"])
    return {cid: "".join(seq) for cid, seq in seq_by_chain.items()}, applied, warnings


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = load_csv(args.selected)
    mutation_rows = load_csv(args.mutation_plan)
    refs = load_refs(args.whitelist)

    mut_by_candidate = defaultdict(list)
    for m in mutation_rows:
        mut_by_candidate[(m["reference_pdb"], m["candidate_id"])].append(m)

    chain_cache = {}
    role_cache = {}
    contact_role_cache = {}
    for pdb, ref in refs.items():
        chain_cache[pdb] = pdb_chain_sequences(ref["path"])
        cmap = extract_contact_map(ref["path"], peptide_chain=ref.get("peptide_chain"))
        role_cache[pdb] = infer_chain_roles(cmap)
        contact_role_cache[pdb] = contact_chain_roles(cmap, role_cache[pdb])

    construct_rows = []
    fasta_records = []
    details = []
    for row in selected:
        key = (row["reference_pdb"], row["candidate_id"])
        ref = refs[row["reference_pdb"]]
        mutated, applied, warnings = apply_candidate_mutations(chain_cache[row["reference_pdb"]], mut_by_candidate[key])
        roles = role_cache[row["reference_pdb"]]
        role_to_chain = dict(contact_role_cache[row["reference_pdb"]])
        for cid, role in roles.items():
            if role in ("H", "L") and cid in mutated and role not in role_to_chain:
                role_to_chain[role] = cid
        heavy_chain = role_to_chain.get("H")
        light_chain = role_to_chain.get("L")
        heavy_seq = mutated.get(heavy_chain, "") if heavy_chain else ""
        light_seq = mutated.get(light_chain, "") if light_chain else ""
        risk = []
        if not heavy_seq:
            risk.append("missing_heavy_chain")
        if not light_seq:
            risk.append("missing_light_chain")
        if warnings:
            risk.append("mutation_application_warnings")
        if any(m.get("region_label") == "framework_or_unknown" for m in mut_by_candidate[key]):
            risk.append("framework_or_unknown_mutation")
        construct_id = f"{row['reference_pdb']}_{row['candidate_id']}"
        construct_rows.append({
            "construct_id": construct_id,
            "selection_rank": row.get("selection_rank"),
            "reference_pdb": row["reference_pdb"],
            "candidate_id": row["candidate_id"],
            "heavy_chain": heavy_chain or "",
            "light_chain": light_chain or "",
            "heavy_length": len(heavy_seq),
            "light_length": len(light_seq),
            "n_mutations": len(applied),
            "applied_mutations": ";".join(applied),
            "candidate_score": row.get("candidate_score"),
            "contact_retention": row.get("contact_retention"),
            "construct_risk_flags": ";".join(risk),
            "warnings": ";".join(warnings),
            "anarci_numbered": False,
        })
        if heavy_seq:
            fasta_records.append((f">{construct_id}|heavy|chain={heavy_chain}|rank={row.get('selection_rank')}", heavy_seq))
        if light_seq:
            fasta_records.append((f">{construct_id}|light|chain={light_chain}|rank={row.get('selection_rank')}", light_seq))
        details.append({
            "construct_id": construct_id,
            "reference": ref,
            "candidate": row,
            "roles": roles,
            "applied_mutations": applied,
            "warnings": warnings,
            "heavy_chain": heavy_chain,
            "light_chain": light_chain,
            "heavy_sequence": heavy_seq,
            "light_sequence": light_seq,
        })

    with open(out_dir / "full_chain_constructs.csv", "w", newline="", encoding="utf-8") as f:
        fields = list(construct_rows[0].keys()) if construct_rows else []
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(construct_rows)
    with open(out_dir / "full_chain_constructs.fasta", "w", encoding="utf-8") as f:
        for header, seq in fasta_records:
            f.write(f"{header}\n{seq}\n")
    with open(out_dir / "full_chain_construct_details.json", "w", encoding="utf-8") as f:
        json.dump({"anarci_available": False, "constructs": details}, f, indent=2)
    with open(out_dir / "full_chain_construct_report.md", "w", encoding="utf-8") as f:
        flagged = [r for r in construct_rows if r["construct_risk_flags"]]
        f.write("# Full-Chain Construct Drafts\n\n")
        f.write(f"Constructs: {len(construct_rows)}; flagged: {len(flagged)}\n\n")
        f.write("ANARCI was not available; chain roles and CDR labels are inferred heuristically. Confirm numbering before synthesis.\n\n")
        f.write("| Rank | Construct | Heavy | Light | Mutations | Flags |\n")
        f.write("|---:|---|---:|---:|---:|---|\n")
        for r in construct_rows:
            f.write(
                f"| {r['selection_rank']} | {r['construct_id']} | {r['heavy_length']} | {r['light_length']} | "
                f"{r['n_mutations']} | {r['construct_risk_flags']} |\n"
            )

    print(f"Wrote full-chain constructs to {out_dir}")
    print(f"Constructs={len(construct_rows)} flagged={sum(1 for r in construct_rows if r['construct_risk_flags'])}")


def contact_chain_roles(contact_map, roles):
    """Choose heavy/light chains from actual paratope-contacting residues first.

    Some PDBs contain duplicate antibody copies. The contact map defines which copy
    was scored and mutated, so full-chain export must use those chains rather than
    the first H/L-like chains encountered in the PDB.
    """
    out = {}
    for res in contact_map.get("paratope_residues", []):
        cid = str(res.get("chain", ""))
        role = roles.get(cid.upper(), roles.get(cid, cid.upper()))
        if role in ("H", "L") and role not in out:
            out[role] = cid
    return out


if __name__ == "__main__":
    main()
