#!/usr/bin/env python3
"""Parse calibrated full-MSA ColabFold controls and candidate structures."""

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/v5_1_candidates"


def load_score(directory, name):
    matches = list(directory.glob(f"{name}_scores_rank_001_*.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected one score file for {name}, found {len(matches)}")
    return json.loads(matches[0].read_text(encoding="utf-8")), matches[0]


def metrics(score, lengths):
    antibody_length = sum(lengths[:-1])
    plddt = np.asarray(score["plddt"], dtype=np.float64)
    pae = np.asarray(score["pae"], dtype=np.float64)
    forward = pae[:antibody_length, antibody_length:]
    reverse = pae[antibody_length:, :antibody_length]
    return {
        "plddt": float(plddt.mean()),
        "antibody_plddt": float(plddt[:antibody_length].mean()),
        "epitope_plddt": float(plddt[antibody_length:].mean()),
        "ptm": float(score["ptm"]),
        "iptm": float(score["iptm"]),
        "interface_pae": float(np.concatenate([forward.ravel(), reverse.ravel()]).mean()),
        "max_pae": float(score["max_pae"]),
    }


def main():
    manifest = json.loads((
        BASE / "abeta42_colabfold_candidates/input_manifest.json"
    ).read_text(encoding="utf-8"))["candidates"]
    decision = json.loads((
        BASE / "abeta42_final/candidate_decision.json"
    ).read_text(encoding="utf-8"))
    candidate_meta = {row["construct_id"]: row for row in decision["hypothesis_queue"]}
    candidate_dir = BASE / "abeta42_colabfold_candidates/msa64_r1"
    controls = {
        "4HIX": (
            "4HIX_native_DAEFRH",
            BASE / "abeta42_colabfold_calibration/4hix_msa64_r1",
            [213, 214, 6],
        ),
        "5CSZ": (
            "5CSZ_native_DAEFRHDSGY",
            BASE / "abeta42_colabfold_calibration/5csz_msa64_r1",
            [212, 214, 10],
        ),
    }
    control_metrics = {}
    for reference, (name, directory, lengths) in controls.items():
        score, path = load_score(directory, name)
        control_metrics[reference] = {
            **metrics(score, lengths),
            "score_json": str(path.relative_to(ROOT)),
        }
    calibrated = all(
        row["antibody_plddt"] >= 80.0
        and row["iptm"] >= 0.7
        and row["interface_pae"] <= 20.0
        for row in control_metrics.values()
    )

    records = []
    for entry in manifest:
        construct_id = entry["construct_id"]
        score, score_path = load_score(candidate_dir, construct_id)
        structure_matches = list(candidate_dir.glob(f"{construct_id}_unrelaxed_rank_001_*.pdb"))
        if len(structure_matches) != 1:
            raise ValueError(f"Expected one structure for {construct_id}, found {len(structure_matches)}")
        current = metrics(score, entry["chain_lengths"])
        parent = control_metrics[entry["reference_pdb"]]
        meta = candidate_meta[construct_id]
        record = {
            "construct_id": construct_id,
            "reference_pdb": entry["reference_pdb"],
            "mutations": entry["mutations"],
            "n_mutations": int(meta["n_mutations"]),
            "condition_advantage": float(meta["condition_advantage"]),
            "contact_retention": float(meta["contact_retention"]),
            "corrected_rank_score": float(meta["corrected_rank_score"]),
            "full_construct_sequence_exact": (
                entry["chain_lengths"][:2]
                == [len(meta["heavy_sequence"]), len(meta["light_sequence"])]
            ),
            **current,
            "iptm_delta_vs_parent": current["iptm"] - parent["iptm"],
            "interface_pae_delta_vs_parent": current["interface_pae"] - parent["interface_pae"],
            "antibody_plddt_delta_vs_parent": current["antibody_plddt"] - parent["antibody_plddt"],
            "score_json": str(score_path.relative_to(ROOT)),
            "structure_pdb": str(structure_matches[0].relative_to(ROOT)),
        }
        record["structure_pass"] = (
            calibrated
            and record["antibody_plddt"] >= 80.0
            and record["iptm"] >= 0.7
            and record["interface_pae"] <= 20.0
        )
        record["panel_score"] = (
            record["corrected_rank_score"]
            + 0.30 * record["iptm_delta_vs_parent"]
            - 0.02 * record["interface_pae_delta_vs_parent"]
            - 0.005 * record["n_mutations"]
        )
        records.append(record)
    records.sort(key=lambda row: (row["structure_pass"], row["panel_score"]), reverse=True)
    for rank, record in enumerate(records, 1):
        record["panel_rank"] = rank

    report = {
        "backend": "ColabFold 1.6.1, AF2-Multimer v3, full MSA, no templates",
        "msa_contract": {"max_seq": 64, "max_extra_seq": 128},
        "num_recycle": 1,
        "control_gates": {
            "antibody_plddt_min": 80.0,
            "iptm_min": 0.7,
            "interface_pae_max": 20.0,
        },
        "calibrated": calibrated,
        "controls": control_metrics,
        "n_candidates": len(records),
        "n_structure_pass": sum(record["structure_pass"] for record in records),
        "records": records,
    }
    output_dir = BASE / "abeta42_colabfold_candidates"
    (output_dir / "calibrated_screen.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    fields = [
        "panel_rank", "construct_id", "reference_pdb", "n_mutations", "mutations",
        "condition_advantage", "contact_retention", "antibody_plddt", "iptm",
        "interface_pae", "antibody_plddt_delta_vs_parent", "iptm_delta_vs_parent",
        "interface_pae_delta_vs_parent", "structure_pass", "panel_score", "score_json",
        "structure_pdb", "full_construct_sequence_exact",
    ]
    with (output_dir / "calibrated_screen.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({
        "calibrated": calibrated,
        "n_candidates": len(records),
        "n_structure_pass": report["n_structure_pass"],
        "top": records[0]["construct_id"] if records else None,
    }, indent=2))


if __name__ == "__main__":
    main()
