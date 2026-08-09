#!/usr/bin/env python
"""Summarize matched 4HIX generation, AF2 uncertainty, and developability."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NATIVE_H3 = "VRYDHYSGSSDY"
AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def chain_sequences():
    path = ROOT / "data/anti_abeta_refs/4HIX.pdb"
    model = PDBParser(QUIET=True).get_structure("4hix", str(path))[0]
    return {
        chain.id: "".join(
            AA3_TO_1[residue.resname] for residue in chain if residue.resname in AA3_TO_1
        )
        for chain in model
        if chain.id in {"A", "H", "L"}
    }


def hamming(left, right):
    if len(left) != len(right):
        raise ValueError("Cannot compare unequal H3 sequences")
    return sum(a != b for a, b in zip(left, right, strict=True)) / len(left)


def generation_summary(bfn, mpnn):
    arms = {arm["name"]: arm["results"] for arm in bfn["arms"]}
    arms["proteinmpnn"] = mpnn["results"]
    output = {}
    for name, rows in arms.items():
        output[name] = {
            "n": len(rows),
            "n_unique": len({row["sequence"] for row in rows}),
            "mean_native_recovery": float(np.mean([row["native_recovery"] for row in rows])),
            "wall_seconds": (
                next(arm["wall_seconds"] for arm in bfn["arms"] if arm["name"] == name)
                if name != "proteinmpnn"
                else mpnn["wall_seconds"]
            ),
        }
    keyed_on = {(row["seed"], row["sample_index"]): row for row in arms["current_disorder_on"]}
    keyed_off = {(row["seed"], row["sample_index"]): row for row in arms["current_disorder_off"]}
    differences = [
        hamming(keyed_on[key]["sequence"], keyed_off[key]["sequence"]) for key in sorted(keyed_on)
    ]
    output["current_on_vs_off_paired"] = {
        "n_pairs": len(differences),
        "mean_hamming_fraction": float(np.mean(differences)),
        "identical_fraction": float(np.mean(np.asarray(differences) == 0)),
        "pairs_changed": int(np.sum(np.asarray(differences) > 0)),
    }
    return output, arms


def af2_summary(data):
    grouped = defaultdict(list)
    metadata = {}
    for row in data["results"]:
        if row["success"]:
            grouped[row["id"]].append(row)
            metadata[row["id"]] = {key: row[key] for key in ("arm", "h3", "pre_af2_rank")}
    candidates = {}
    by_arm = defaultdict(list)
    for candidate_id, rows in grouped.items():
        iptm = [row["iptm"] for row in rows]
        summary = {
            **metadata[candidate_id],
            "n_success": len(rows),
            "median_iptm": float(np.median(iptm)),
            "min_iptm": float(np.min(iptm)),
            "max_iptm": float(np.max(iptm)),
            "median_interface_pae": float(np.median([row["interface_pae"] for row in rows])),
            "median_plddt": float(np.median([row["plddt"] for row in rows])),
        }
        candidates[candidate_id] = summary
        by_arm[summary["arm"]].append(summary["median_iptm"])
    arms = {
        arm: {
            "n_candidates": len(values),
            "mean_candidate_median_iptm": float(np.mean(values)),
            "min_candidate_median_iptm": float(np.min(values)),
            "max_candidate_median_iptm": float(np.max(values)),
        }
        for arm, values in by_arm.items()
    }
    return {"candidates": candidates, "arms": arms}


def developability_summary(arms):
    from scripts.pipeline.score_shortlist_developability import score_sequence

    sequences = chain_sequences()
    light_score = score_sequence(sequences["L"])
    output = {}
    for arm, rows in arms.items():
        risks = []
        passes = 0
        flags = defaultdict(int)
        for row in rows:
            heavy = sequences["H"][:95] + row["sequence"] + sequences["H"][107:]
            heavy_score = score_sequence(heavy)
            risk = max(heavy_score["sequence_risk"], light_score["sequence_risk"])
            risks.append(risk)
            passes += risk <= 0.55
            for flag in heavy_score["flags"].split(";"):
                if flag:
                    flags[flag] += 1
        output[arm] = {
            "n": len(rows),
            "pass_fraction_at_0_55": passes / len(rows),
            "mean_combined_risk": float(np.mean(risks)),
            "candidate_heavy_flags": dict(sorted(flags.items())),
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfn", default=str(ROOT / "results/ablation/4hix_bfn_matched_v1.json"))
    parser.add_argument("--mpnn", default=str(ROOT / "results/ablation/4hix_mpnn_matched_v1.json"))
    parser.add_argument(
        "--af2", default=str(ROOT / "results/ablation/4hix_matched_af2_v1/results.json")
    )
    parser.add_argument(
        "--out", default=str(ROOT / "results/ablation/4hix_matched_summary_v1.json")
    )
    args = parser.parse_args()
    bfn = json.loads(Path(args.bfn).read_text())
    mpnn = json.loads(Path(args.mpnn).read_text())
    af2 = json.loads(Path(args.af2).read_text())
    generation, arms = generation_summary(bfn, mpnn)
    output = {
        "schema_version": 1,
        "status": "single_scaffold_diagnostic_not_design_validation",
        "generation": generation,
        "af2": af2_summary(af2),
        "developability": developability_summary(arms),
        "claim_boundary": (
            "One short-peptide scaffold cannot establish BFN design success, "
            "IDP preference, affinity, or external generalization."
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
