#!/usr/bin/env python
"""Run frozen structural and BFN-path sanity checks for the Tau/5MP3 target."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from Bio.PDB import PDBParser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def chain_sequence(path, chain_id):
    model = PDBParser(QUIET=True).get_structure(Path(path).stem, str(path))[0]
    return "".join(
        AA3[residue.resname]
        for residue in model[chain_id]
        if residue.resname in AA3 and "CA" in residue
    )


def validate_inputs(config):
    for name in (
        "ensemble_manifest", "pose_panel_audit", "historical_ensemble_summary"
    ):
        path = ROOT / config["inputs"][name]
        if sha256(path) != config["inputs"][f"{name}_sha256"]:
            raise ValueError(f"Frozen input hash mismatch: {name}")
    checkpoint = ROOT / config["model"]["checkpoint"]
    if sha256(checkpoint) != config["model"]["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")


def selected_tau_poses(config, audit):
    rows = [
        row for row in audit["entries"]
        if row["target"] == "tau" and row.get("selected_for_panel")
    ]
    rows.sort(key=lambda row: row["conformer"])
    if len(rows) != int(config["target"]["selected_poses"]):
        raise ValueError("Selected Tau pose count differs from frozen contract")
    return rows


def structural_checks(config, pose_rows):
    target = config["target"]
    checks = []
    for row in pose_rows:
        path = ROOT / row["pose_pdb"]
        heavy = chain_sequence(path, target["heavy_chain"])
        antigen = chain_sequence(path, target["pose_antigen_chain"])
        h3 = heavy[95:108]
        checks.append({
            "conformer": row["conformer"],
            "pose_pdb": row["pose_pdb"],
            "pose_sha256": sha256(path),
            "geometry_status": row["status"],
            "n_contacts": row["n_contacts"],
            "severe_clashes": row["severe_clash_pairs_lt_1_5A"],
            "heavy_length": len(heavy),
            "antigen_sequence": antigen,
            "h3_sequence": h3,
            "exact_antigen": antigen == target["antigen_sequence"],
            "exact_h3": h3 == target["h3_sequence"],
        })
    return checks


def score_poses(config, pose_rows, device):
    from modules import bfn_loader
    from modules.bfn_prospective import score_bfn_candidates

    os.environ["DISORDERFLOW_CHECKPOINT"] = str(
        ROOT / config["model"]["checkpoint"]
    )
    bfn_loader._bfn_model = None
    bfn_loader._bfn_config = None
    model, _ = bfn_loader.load_bfn(device)
    target = config["target"]
    rows = []
    for pose in pose_rows:
        path = str(ROOT / pose["pose_pdb"])
        complex_result = score_bfn_candidates(
            path,
            target["h3_region_spec"],
            [target["h3_sequence"]],
            context_chains=[target["light_chain"], target["pose_antigen_chain"]],
            antigen_chains=[target["pose_antigen_chain"]],
            device=device,
            model=model,
            fixed_t=float(config["model"]["fixed_t"]),
        )
        stripped_result = score_bfn_candidates(
            path,
            target["h3_region_spec"],
            [target["h3_sequence"]],
            context_chains=[target["light_chain"]],
            device=device,
            model=model,
            fixed_t=float(config["model"]["fixed_t"]),
        )
        complex_score = float(complex_result["state_compatibility"][0].cpu())
        stripped_score = float(stripped_result["state_compatibility"][0].cpu())
        rows.append({
            "conformer": pose["conformer"],
            "complex_state_compatibility": complex_score,
            "stripped_state_compatibility": stripped_score,
            "complex_minus_stripped": complex_score - stripped_score,
            "complex_iptm_diagnostic": float(complex_result["iptm"][0].cpu()),
            "stripped_iptm_diagnostic": float(stripped_result["iptm"][0].cpu()),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmarks/ibec_tau_sanity_v1.yml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_inputs(config)
    manifest = json.loads((ROOT / config["inputs"]["ensemble_manifest"]).read_text(
        encoding="utf-8"))
    audit = json.loads((ROOT / config["inputs"]["pose_panel_audit"]).read_text(
        encoding="utf-8"))
    historical = json.loads((
        ROOT / config["inputs"]["historical_ensemble_summary"]
    ).read_text(encoding="utf-8"))
    tau_manifest = next(row for row in manifest["targets"] if row["target"] == "tau")
    pose_rows = selected_tau_poses(config, audit)
    structure_rows = structural_checks(config, pose_rows)
    score_rows = score_poses(config, pose_rows, args.device)
    deltas = [row["complex_minus_stripped"] for row in score_rows]
    complex_values = [row["complex_state_compatibility"] for row in score_rows]
    gates = config["gates"]
    historical_tau = historical["by_target"]["tau"]
    checks = {
        "manifest_pass": manifest["status"] == "pass",
        "pose_audit_pass": audit["status"] == "pass",
        "all_pose_geometry_pass": all(row["geometry_status"] == "pass" for row in structure_rows),
        "zero_severe_clashes": all(row["severe_clashes"] == 0 for row in structure_rows),
        "exact_antigen_sequence": all(row["exact_antigen"] for row in structure_rows),
        "exact_h3_sequence": all(row["exact_h3"] for row in structure_rows),
        "coordinate_sensitive_poses": sum(
            abs(value) >= float(gates["minimum_abs_complex_minus_stripped"])
            for value in deltas
        ) >= int(gates["minimum_coordinate_sensitive_poses"]),
        "pose_output_range": max(complex_values) - min(complex_values)
        >= float(gates["minimum_pose_output_range"]),
        "historical_heldout_mean_delta": historical_tau["heldout_mean_delta_native"]
        >= float(gates["historical_heldout_mean_delta_minimum"]),
        "historical_heldout_beat_native_rate": historical_tau["heldout_beat_native_rate"]
        >= float(gates["historical_heldout_beat_native_rate_minimum"]),
    }
    decision = "go_tau_second_target_extension" if all(checks.values()) else "no_go_tau_extension"
    output = {
        "schema_version": 1,
        "status": "ibec_tau_sanity_complete",
        "decision": decision,
        "config": args.config,
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__)),
        "target": config["target"],
        "reference_complex": tau_manifest["trimmed_reference_pdb"],
        "reference_complex_sha256": tau_manifest["trimmed_reference_sha256"].lower(),
        "structure_checks": structure_rows,
        "bfn_pose_scores": score_rows,
        "aggregate": {
            "mean_complex_minus_stripped": float(np.mean(deltas)),
            "minimum_abs_complex_minus_stripped": float(np.min(np.abs(deltas))),
            "coordinate_sensitive_poses": sum(
                abs(value) >= float(gates["minimum_abs_complex_minus_stripped"])
                for value in deltas
            ),
            "complex_state_compatibility_range": max(complex_values) - min(complex_values),
            "historical_heldout_mean_delta_native": historical_tau[
                "heldout_mean_delta_native"],
            "historical_heldout_beat_native_rate": historical_tau[
                "heldout_beat_native_rate"],
        },
        "checks": checks,
        "historical_candidate_policy": config["policy"],
        "next_step": (
            "freeze_correct_13_residue_conservative_H3_candidate_protocol"
            if decision.startswith("go_") else "stop_tau_and_report_boundary"
        ),
        "claim_boundary": config["claim_boundary"],
    }
    output_path = ROOT / (args.out or config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(json.dumps({
        "status": output["status"],
        "decision": decision,
        "h3": config["target"]["h3_sequence"],
        "aggregate": output["aggregate"],
        "checks": checks,
        "next_step": output["next_step"],
    }, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
