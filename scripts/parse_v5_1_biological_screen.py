#!/usr/bin/env python3
"""Parse complete-SEQRES parent and candidate ColabFold screens."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/v5_1_candidates/abeta42_biological_constructs"
LENGTHS = [228, 215, 11]


def score_file(directory, name):
    matches = list(directory.glob(f"{name}_scores_rank_001_*.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected one score file for {name}, found {len(matches)}")
    return json.loads(matches[0].read_text(encoding="utf-8")), matches[0]


def metrics(score):
    antibody_length = sum(LENGTHS[:2])
    plddt = np.asarray(score["plddt"], dtype=np.float64)
    pae = np.asarray(score["pae"], dtype=np.float64)
    interface = np.concatenate([
        pae[:antibody_length, antibody_length:].ravel(),
        pae[antibody_length:, :antibody_length].ravel(),
    ])
    return {
        "plddt": float(plddt.mean()),
        "antibody_plddt": float(plddt[:antibody_length].mean()),
        "epitope_plddt": float(plddt[antibody_length:].mean()),
        "ptm": float(score["ptm"]),
        "iptm": float(score["iptm"]),
        "interface_pae": float(interface.mean()),
        "max_pae": float(score["max_pae"]),
    }


def main():
    construct_report = json.loads((BASE / "biological_construct_report.json").read_text(encoding="utf-8"))
    parent_score, parent_score_path = score_file(BASE / "parent_msa64_r1", "5CSZ_complete_parent")
    parent = {
        **metrics(parent_score),
        "score_json": str(parent_score_path.relative_to(ROOT)),
    }
    parent_pass = (
        parent["antibody_plddt"] >= 80.0
        and parent["iptm"] >= 0.7
    )
    candidates = []
    candidate_dir = BASE / "candidate_msa64_r1"
    for construct in construct_report["records"]:
        name = construct["construct_id"]
        score, score_path = score_file(candidate_dir, name)
        current = metrics(score)
        pdb_matches = list(candidate_dir.glob(f"{name}_unrelaxed_rank_001_*.pdb"))
        if len(pdb_matches) != 1:
            raise ValueError(f"Expected one PDB for {name}")
        record = {
            "construct_id": name,
            "mutations": construct["mutations"],
            "heavy_sequence": construct["heavy_sequence"],
            "light_sequence": construct["light_sequence"],
            "epitope_sequence": construct["epitope_sequence"],
            **current,
            "iptm_delta_vs_parent": current["iptm"] - parent["iptm"],
            "antibody_plddt_delta_vs_parent": current["antibody_plddt"] - parent["antibody_plddt"],
            "interface_pae_delta_vs_parent": current["interface_pae"] - parent["interface_pae"],
            "score_json": str(score_path.relative_to(ROOT)),
            "structure_pdb": str(pdb_matches[0].relative_to(ROOT)),
        }
        record["structure_pass"] = (
            parent_pass
            and current["antibody_plddt"] >= 80.0
            and current["iptm"] >= 0.7
            and current["interface_pae"] <= 20.0
        )
        candidates.append(record)
    candidates.sort(key=lambda row: (row["structure_pass"], row["iptm"]), reverse=True)
    for rank, record in enumerate(candidates, 1):
        record["biological_structure_rank"] = rank
    report = {
        "sequence_contract": "5CSZ SEQRES heavy=228, light=215, A-beta1-11=11",
        "backend": "ColabFold 1.6.1, AF2-Multimer v3, full MSA, no templates",
        "msa_contract": {"max_seq": 64, "max_extra_seq": 128},
        "num_recycle": 1,
        "parent_calibrated": parent_pass,
        "parent": parent,
        "n_candidates": len(candidates),
        "n_structure_pass": sum(row["structure_pass"] for row in candidates),
        "records": candidates,
    }
    (BASE / "biological_screen.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "parent_calibrated": parent_pass,
        "parent_iptm": parent["iptm"],
        "n_structure_pass": report["n_structure_pass"],
        "top": candidates[0]["construct_id"],
    }, indent=2))


if __name__ == "__main__":
    main()
