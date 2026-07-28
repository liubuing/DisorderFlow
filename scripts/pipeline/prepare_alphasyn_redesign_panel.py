#!/usr/bin/env python
"""Prepare native plus 20 conservative alpha-synuclein redesigns for paired folding."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from prepare_nonabeta_initial_guess_panel import build_guess


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/alpha_synuclein_redesign_validation_v2"


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    candidates = [
        row for row in load_csv(ROOT / "outputs/non_abeta_idp_ensemble_redesign_v2/ensemble_designed_candidates.csv")
        if row["target"] == "alpha_synuclein"
    ]
    native_panel = next(
        row for row in load_csv(ROOT / "outputs/non_abeta_idp_complex_fold_panel_v1/complex_fold_panel.csv")
        if row["construct_id"] == "8B9V_native"
    )
    constructs = [{**native_panel, "n_mutations": "0", "applied_mutations": ""}]
    for row in candidates:
        constructs.append({
            "construct_id": row["candidate_uid"], "target": "alpha_synuclein",
            "antibody_family": "8B9V", "construct_type": "designed_candidate",
            "vh_sequence": row["heavy_sequence"][:len(native_panel["vh_sequence"])],
            "vl_sequence": row["light_sequence"][:len(native_panel["vl_sequence"])],
            "antigen_sequence": native_panel["antigen_sequence"],
            "source_candidate_uid": row["candidate_uid"],
            "n_mutations": row["n_mutations"], "applied_mutations": row["applied_mutations"],
        })
    poses = json.load(open(ROOT / "outputs/non_abeta_idp_pose_panels_v1/pose_panel_audit.json", encoding="utf-8"))
    pose_rows = sorted(
        [row for row in poses["entries"] if row["target"] == "alpha_synuclein" and row.get("selected_for_panel")],
        key=lambda row: row["conformer"],
    )
    panel = []
    for construct in constructs:
        for pose in pose_rows:
            output = OUT / "initial_guesses" / construct["construct_id"] / f"conformer{pose['conformer']}.pdb"
            build_guess(ROOT / pose["pose_pdb"], output, construct, "H", "L")
            panel.append({
                **construct, "conformer": pose["conformer"],
                "initial_guess_pdb": output.relative_to(ROOT).as_posix(),
                "source_pose_pdb": pose["pose_pdb"],
            })
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "redesign_constructs.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(constructs[0]), extrasaction="ignore")
        writer.writeheader(); writer.writerows(constructs)
    with open(OUT / "initial_guess_panel.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(panel[0]), extrasaction="ignore")
        writer.writeheader(); writer.writerows(panel)

    native_a3m = ROOT / "outputs/non_abeta_idp_complex_fold_panel_v1/colabfold_complex_gpu/8B9V_native_target_alpha_synuclein_family_8B9V_VH_VL_IDP.a3m"
    lines = native_a3m.read_text(encoding="utf-8").splitlines()
    for construct in constructs:
        query = construct["vh_sequence"] + construct["vl_sequence"] + construct["antigen_sequence"]
        candidate_lines = lines[:]
        candidate_lines[1] = f">{construct['construct_id']}"
        candidate_lines[2] = query
        path = OUT / "a3m" / f"{construct['construct_id']}.a3m"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(candidate_lines) + "\n", encoding="utf-8")
    summary = {
        "schema_version": "alphasyn.redesign_validation_panel.v2", "status": "pass",
        "constructs": len(constructs), "designed_candidates": len(candidates),
        "conformers": len(pose_rows), "planned_seeds": 3, "planned_models": len(panel) * 3,
        "pre_registered_gates": {
            "delta_pae_max": 0.0, "delta_contact_retention_min": 0.0,
            "delta_antigen_pose_rmsd_max": 0.0, "delta_refined_contact_retention_min": 0.0,
            "required_refined_zero_clash_models": 15,
        },
    }
    json.dump(summary, open(OUT / "panel_summary.json", "w", encoding="utf-8"), indent=2)
    print(f"Alpha-syn redesign panel: {summary}")


if __name__ == "__main__":
    main()
