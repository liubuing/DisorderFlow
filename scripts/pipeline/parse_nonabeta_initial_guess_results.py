#!/usr/bin/env python
"""Audit pose-conditioned VH:VL:IDP folds against native controls and input poses."""
from __future__ import annotations

import csv
import json
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANEL = PROJECT_ROOT / "outputs/non_abeta_idp_initial_guess_panel_v1/initial_guess_panel.csv"
DEFAULT_RESULTS = PROJECT_ROOT / "outputs/non_abeta_idp_initial_guess_panel_v1/colabfold_initial_guess_gpu"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/non_abeta_idp_initial_guess_evidence_v1"


def io_path(path):
    resolved = str(Path(path).resolve())
    return f"\\\\?\\{resolved}" if len(resolved) >= 248 else resolved


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def model_for_score(score_path):
    prefix, ranked = score_path.name.split("_scores_rank_", 1)
    rank, suffix = ranked.split("_", 1)
    return score_path.parent / f"{prefix}_unrelaxed_rank_{rank}_{suffix.replace('.json', '.pdb')}"


def parsed_model(path):
    return next(iter(PDBParser(QUIET=True).get_structure("structure", io_path(path))))


def ca_coordinates(model, chain_index):
    chain = list(model)[chain_index]
    return np.asarray([residue["CA"].coord for residue in chain if "CA" in residue])


def representative_coordinates(model, chain_indices):
    coordinates = []
    labels = []
    for chain_index in chain_indices:
        chain = list(model)[chain_index]
        for residue_index, residue in enumerate(chain):
            atom = residue["CB"] if "CB" in residue else residue["CA"] if "CA" in residue else None
            if atom is not None:
                coordinates.append(atom.coord)
                labels.append((chain_index, residue_index))
    return np.asarray(coordinates), labels


def contact_pairs(model, cutoff=8.0):
    antibody, antibody_labels = representative_coordinates(model, (0, 1))
    antigen, antigen_labels = representative_coordinates(model, (2,))
    distances = np.linalg.norm(antibody[:, None, :] - antigen[None, :, :], axis=-1)
    return {
        (antibody_labels[i], antigen_labels[j])
        for i, j in zip(*np.where(distances < cutoff))
    }


def kabsch(mobile, target):
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (mobile - mobile_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - mobile_center @ rotation.T
    return rotation, translation


def rmsd(mobile, target, rotation=None, translation=None):
    if rotation is not None:
        mobile = mobile @ rotation.T + translation
    return float(np.sqrt(np.mean(np.sum((mobile - target) ** 2, axis=1))))


def severe_clashes(model):
    chains = list(model)
    antibody = np.asarray([
        atom.coord for chain in chains[:2] for residue in chain for atom in residue
        if atom.element != "H"
    ])
    antigen = np.asarray([
        atom.coord for residue in chains[2] for atom in residue if atom.element != "H"
    ])
    distances = np.linalg.norm(antibody[:, None, :] - antigen[None, :, :], axis=-1)
    return int((distances < 1.5).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--layout", choices=["construct_first", "conformer_first"], default="construct_first")
    args = parser.parse_args()
    results = args.results if args.results.is_absolute() else PROJECT_ROOT / args.results
    output = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    panel_path = args.panel if args.panel.is_absolute() else PROJECT_ROOT / args.panel
    panel = load_csv(panel_path)
    rows = []
    for source in panel:
        if args.layout == "construct_first":
            job_dir = results / source["construct_id"] / f"conformer{source['conformer']}"
            score_paths = sorted(job_dir.glob("*_scores_rank_*_*_seed_*.json"))
        else:
            job_dir = results / f"conformer{source['conformer']}"
            score_paths = sorted(job_dir.glob(
                f"{source['construct_id']}_scores_rank_*_*_seed_*.json"
            ))
        for score_path in score_paths:
            with open(io_path(score_path), encoding="utf-8") as handle:
                scores = json.load(handle)
            prediction = parsed_model(model_for_score(score_path))
            initial = parsed_model(PROJECT_ROOT / source["initial_guess_pdb"])
            initial_antibody = np.vstack([ca_coordinates(initial, 0), ca_coordinates(initial, 1)])
            predicted_antibody = np.vstack([ca_coordinates(prediction, 0), ca_coordinates(prediction, 1)])
            rotation, translation = kabsch(predicted_antibody, initial_antibody)
            initial_contacts = contact_pairs(initial)
            predicted_contacts = contact_pairs(prediction)
            retained = len(initial_contacts & predicted_contacts) / max(len(initial_contacts), 1)
            antigen_length = len(source["antigen_sequence"])
            antibody_length = len(scores["plddt"]) - antigen_length
            pae = np.asarray(scores["pae"], dtype=float)
            rows.append({
                "construct_id": source["construct_id"],
                "target": source["target"],
                "construct_type": source["construct_type"],
                "conformer": int(source["conformer"]),
                "seed": int(score_path.stem.rsplit("_seed_", 1)[1]),
                "antigen_mean_plddt": round(float(np.mean(scores["plddt"][-antigen_length:])), 2),
                "antibody_to_antigen_mean_pae": round(float(pae[:antibody_length, antibody_length:].mean()), 2),
                "iptm": round(float(scores["iptm"]), 4),
                "initial_contacts": len(initial_contacts),
                "retained_initial_contacts": len(initial_contacts & predicted_contacts),
                "contact_retention": round(retained, 4),
                "antibody_aligned_antigen_ca_rmsd": round(rmsd(
                    ca_coordinates(prediction, 2), ca_coordinates(initial, 2), rotation, translation
                ), 3),
                "heavy_aligned_light_ca_rmsd": round(rmsd(
                    ca_coordinates(prediction, 1), ca_coordinates(initial, 1),
                    *kabsch(ca_coordinates(prediction, 0), ca_coordinates(initial, 0))
                ), 3),
                "severe_clash_pairs_lt_1_5A": severe_clashes(prediction),
                "score_json": score_path.relative_to(PROJECT_ROOT).as_posix(),
            })
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "initial_guess_evidence.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["construct_id"]].append(row)
    aggregates = []
    for construct_id, records in sorted(grouped.items()):
        aggregates.append({
            "construct_id": construct_id,
            "target": records[0]["target"],
            "construct_type": records[0]["construct_type"],
            "conformers": len({row["conformer"] for row in records}),
            "models": len(records),
            "antigen_plddt_mean": round(float(np.mean([row["antigen_mean_plddt"] for row in records])), 2),
            "ab_to_antigen_pae_mean": round(float(np.mean([row["antibody_to_antigen_mean_pae"] for row in records])), 2),
            "contact_retention_mean": round(float(np.mean([row["contact_retention"] for row in records])), 4),
            "contact_retention_min": round(float(min(row["contact_retention"] for row in records)), 4),
            "antigen_ca_rmsd_mean": round(float(np.mean([row["antibody_aligned_antigen_ca_rmsd"] for row in records])), 3),
            "antigen_ca_rmsd_max": round(float(max(row["antibody_aligned_antigen_ca_rmsd"] for row in records)), 3),
            "vh_vl_orientation_rmsd_mean": round(float(np.mean([row["heavy_aligned_light_ca_rmsd"] for row in records])), 3),
            "zero_clash_models": sum(row["severe_clash_pairs_lt_1_5A"] == 0 for row in records),
        })
    native = {row["target"]: row for row in aggregates if row["construct_type"] == "native_control"}
    for row in aggregates:
        baseline = native[row["target"]]
        row["delta_antigen_plddt_vs_native"] = round(row["antigen_plddt_mean"] - baseline["antigen_plddt_mean"], 2)
        row["delta_pae_vs_native"] = round(row["ab_to_antigen_pae_mean"] - baseline["ab_to_antigen_pae_mean"], 2)
        row["delta_contact_retention_vs_native"] = round(row["contact_retention_mean"] - baseline["contact_retention_mean"], 4)
        row["delta_antigen_rmsd_vs_native"] = round(row["antigen_ca_rmsd_mean"] - baseline["antigen_ca_rmsd_mean"], 3)
        if row["construct_type"] == "native_control":
            row["status"] = "native_control"
        elif (
            row["conformers"] == 5
            and row["zero_clash_models"] == row["models"]
            and row["delta_pae_vs_native"] <= 0.0
            and row["delta_contact_retention_vs_native"] >= 0.0
            and row["delta_antigen_rmsd_vs_native"] <= 0.0
        ):
            row["status"] = "initial_guess_interface_pass"
        else:
            row["status"] = "initial_guess_interface_review"
    with open(output / "initial_guess_aggregate.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    passed = {
        target: sum(row["target"] == target and row["status"] == "initial_guess_interface_pass" for row in aggregates)
        for target in native
    }
    summary = {
        "schema_version": "nonabeta.initial_guess_evidence.v1",
        "status": "pass" if all(passed.values()) else "partial",
        "models": len(rows),
        "constructs": len(aggregates),
        "conformers_per_construct": 5,
        "interface_pass_by_target": passed,
        "seeds_per_conformer": max(
            len({row["seed"] for row in rows if row["construct_id"] == construct_id and row["conformer"] == conformer})
            for construct_id in grouped for conformer in {row["conformer"] for row in grouped[construct_id]}
        ),
        "claim_boundary": (
            "Pose-conditioned structural screen. Confidence, contact retention, and geometry are computational "
            "evidence and do not establish binding or state specificity."
        ),
    }
    with open(output / "initial_guess_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Initial-guess evidence: status={summary['status']} models={len(rows)} pass={passed}")


if __name__ == "__main__":
    main()
