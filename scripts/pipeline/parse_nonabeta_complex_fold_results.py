#!/usr/bin/env python
"""Parse VH:VL:IDP complex folds with antigen-specific interface metrics."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel", default="outputs/non_abeta_idp_complex_fold_panel_v1/complex_fold_panel.csv"
    )
    parser.add_argument(
        "--result-dir",
        default="outputs/non_abeta_idp_complex_fold_panel_v1/colabfold_complex_gpu",
    )
    parser.add_argument(
        "--out", default="outputs/non_abeta_idp_complex_fold_evidence_v1"
    )
    return parser.parse_args()


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def construct_id(path, panel):
    matches = [construct for construct in panel if path.name.startswith(construct + "_")]
    if len(matches) != 1:
        raise ValueError(f"Cannot resolve construct ID for {path.name}: {matches}")
    return matches[0]


def seed_from_name(path):
    marker = "_seed_"
    return int(path.name.split(marker, 1)[1].split(".", 1)[0])


def model_for_score(score_path):
    prefix, ranked = score_path.name.split("_scores_rank_", 1)
    rank, suffix = ranked.split("_", 1)
    model = score_path.parent / f"{prefix}_unrelaxed_rank_{rank}_{suffix.replace('.json', '.pdb')}"
    if not model.exists():
        raise FileNotFoundError(model)
    return model


def interface_geometry(pdb_path):
    model = next(iter(PDBParser(QUIET=True).get_structure("complex", str(pdb_path))))
    chains = list(model)
    if len(chains) != 3:
        raise ValueError(f"Expected three chains in {pdb_path}, found {len(chains)}")
    antibody_atoms = [
        atom for chain in chains[:2] for residue in chain for atom in residue
        if atom.element != "H"
    ]
    antigen_atoms = [
        atom for residue in chains[2] for atom in residue if atom.element != "H"
    ]
    distances = np.linalg.norm(
        np.asarray([atom.coord for atom in antibody_atoms])[:, None, :]
        - np.asarray([atom.coord for atom in antigen_atoms])[None, :, :], axis=-1,
    )
    contacting_antigen_residues = set()
    for antigen_index in np.where((distances < 5.0).any(axis=0))[0]:
        residue = antigen_atoms[int(antigen_index)].get_parent()
        contacting_antigen_residues.add((residue.get_parent().id, residue.id[1]))
    return {
        "minimum_antibody_antigen_distance": float(distances.min()),
        "severe_clash_pairs_lt_1_5A": int((distances < 1.5).sum()),
        "close_atom_pairs_lt_5A": int((distances < 5.0).sum()),
        "contact_atom_pairs_lt_8A": int((distances < 8.0).sum()),
        "contacting_antigen_residues": len(contacting_antigen_residues),
    }


def main():
    args = parse_args()
    panel_rows = load_csv(args.panel)
    panel = {row["construct_id"]: row for row in panel_rows}
    result_dir = Path(args.result_dir)
    rows = []
    latest_by_construct_seed = {}
    for score_path in result_dir.glob("*_scores_rank_*_*.json"):
        cid = construct_id(score_path, panel)
        key = (cid, seed_from_name(score_path))
        current = latest_by_construct_seed.get(key)
        if current is None or score_path.stat().st_mtime > current.stat().st_mtime:
            latest_by_construct_seed[key] = score_path
    for score_path in sorted(latest_by_construct_seed.values()):
        cid = construct_id(score_path, panel)
        source = panel[cid]
        with open(score_path, encoding="utf-8") as handle:
            scores = json.load(handle)
        antigen_length = len(source["antigen_sequence"])
        antibody_length = len(scores["plddt"]) - antigen_length
        pae = np.asarray(scores["pae"], dtype=float)
        ab_to_antigen = pae[:antibody_length, antibody_length:]
        antigen_to_ab = pae[antibody_length:, :antibody_length]
        geometry = interface_geometry(model_for_score(score_path))
        rows.append({
            "construct_id": cid,
            "target": source["target"],
            "antibody_family": source["antibody_family"],
            "construct_type": source["construct_type"],
            "seed": seed_from_name(score_path),
            "mean_plddt": round(float(np.mean(scores["plddt"])), 2),
            "antigen_mean_plddt": round(float(np.mean(scores["plddt"][-antigen_length:])), 2),
            "ptm": round(float(scores["ptm"]), 4),
            "iptm": round(float(scores["iptm"]), 4),
            "antibody_to_antigen_mean_pae": round(float(ab_to_antigen.mean()), 2),
            "antigen_to_antibody_mean_pae": round(float(antigen_to_ab.mean()), 2),
            **geometry,
            "score_json": str(score_path),
        })

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with open(out_dir / "complex_fold_evidence.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_construct = defaultdict(list)
    for row in rows:
        by_construct[row["construct_id"]].append(row)
    aggregates = []
    for cid, records in sorted(by_construct.items()):
        source = panel[cid]
        aggregates.append({
            "construct_id": cid,
            "target": source["target"],
            "antibody_family": source["antibody_family"],
            "construct_type": source["construct_type"],
            "seeds": len(records),
            "antigen_mean_plddt_mean": round(float(np.mean([r["antigen_mean_plddt"] for r in records])), 2),
            "antigen_mean_plddt_min": round(float(min(r["antigen_mean_plddt"] for r in records)), 2),
            "antibody_to_antigen_pae_mean": round(float(np.mean([r["antibody_to_antigen_mean_pae"] for r in records])), 2),
            "antibody_to_antigen_pae_max": round(float(max(r["antibody_to_antigen_mean_pae"] for r in records)), 2),
            "contacting_antigen_residues_mean": round(float(np.mean([r["contacting_antigen_residues"] for r in records])), 2),
            "zero_clash_seeds": sum(r["severe_clash_pairs_lt_1_5A"] == 0 for r in records),
        })
    native = {
        row["target"]: row for row in aggregates if row["construct_type"] == "native_control"
    }
    for row in aggregates:
        baseline = native[row["target"]]
        row["delta_antigen_plddt_vs_native"] = round(
            row["antigen_mean_plddt_mean"] - baseline["antigen_mean_plddt_mean"], 2
        )
        row["delta_antibody_to_antigen_pae_vs_native"] = round(
            row["antibody_to_antigen_pae_mean"] - baseline["antibody_to_antigen_pae_mean"], 2
        )
        if row["construct_type"] == "native_control":
            row["complex_interface_status"] = "native_control"
        elif row["seeds"] < 3:
            row["complex_interface_status"] = "insufficient_seeds"
        elif (
            row["zero_clash_seeds"] == row["seeds"]
            and row["delta_antigen_plddt_vs_native"] >= 0.0
            and row["delta_antibody_to_antigen_pae_vs_native"] <= 0.0
        ):
            row["complex_interface_status"] = "complex_interface_pass"
        else:
            row["complex_interface_status"] = "complex_interface_review"
    with open(out_dir / "complex_fold_aggregate.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    priority = []
    for target in sorted(native):
        priority.append(panel[native[target]["construct_id"]])
        candidates = [
            row for row in aggregates
            if row["target"] == target and row["construct_type"] == "designed_candidate"
        ]
        winner = max(candidates, key=lambda row: (
            row["delta_antigen_plddt_vs_native"],
            -row["delta_antibody_to_antigen_pae_vs_native"],
            row["contacting_antigen_residues_mean"],
        ))
        priority.append(panel[winner["construct_id"]])
    with open(out_dir / "priority_multiseed.fasta", "w", encoding="ascii") as handle:
        for row in priority:
            handle.write(
                f">{row['construct_id']}|target={row['target']}|family={row['antibody_family']}|VH_VL_IDP\n"
                f"{row['vh_sequence']}:{row['vl_sequence']}:{row['antigen_sequence']}\n"
            )

    passed_by_target = {
        target: sum(
            row["target"] == target and row["complex_interface_status"] == "complex_interface_pass"
            for row in aggregates
        )
        for target in sorted(native)
    }
    computation_complete = len(rows) >= len(panel) and all(
        any(row["target"] == target and row["seeds"] >= 3 for row in aggregates)
        for target in native
    )
    summary = {
        "schema_version": "nonabeta.complex_fold_evidence.v1",
        "status": "pass" if computation_complete and all(passed_by_target.values()) else "partial",
        "computation_status": "complete" if computation_complete else "partial",
        "models": len(rows),
        "constructs": len(by_construct),
        "complex_interface_pass_by_target": passed_by_target,
        "priority_multiseed_constructs": [row["construct_id"] for row in priority],
        "claim_boundary": (
            "Global multimer confidence is not treated as antibody-IDP evidence. "
            "Antigen pLDDT, cross-chain PAE, geometry, native controls, and seed stability are required."
        ),
    }
    with open(out_dir / "complex_fold_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Complex fold evidence: status={summary['status']} models={len(rows)}")


if __name__ == "__main__":
    main()
