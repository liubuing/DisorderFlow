#!/usr/bin/env python
"""Rescore fixed v3 candidates on experimental, sampled, and mixed poses."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
PROTEIN_MPNN = ROOT / "ProteinMPNN"
for entry in (str(ROOT), str(PROTEIN_MPNN)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from scripts.generate_idp_ensemble_v3_candidates import (  # noqa: E402
    ALPHABET,
    featurize_arm,
)
from scripts.protein_mpnn_prefix_adapter import build_model  # noqa: E402


def score_sequence(adapter, sequence):
    score = 0.0
    for position, amino_acid in enumerate(sequence):
        values = adapter.next_log_probs(sequence[:position], position)
        score += float(values[ALPHABET.index(amino_acid)])
    return score / len(sequence)


def panel_rows(poses):
    experimental = [
        row for row in poses
        if row["source"] != "restrained_local_interface_sampling"
    ]
    sampled = [
        row for row in poses
        if row["source"] == "restrained_local_interface_sampling"
    ]
    if not experimental or not sampled:
        raise RuntimeError("Every component requires experimental and sampled poses")
    return {
        "experimental_only": experimental,
        "sampled_only": sampled,
        "mixed": poses,
    }


def score(config_path, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite source sensitivity: {output}")
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    candidates = json.loads(
        (ROOT / config["inputs"]["candidates"]).read_text(encoding="ascii")
    )
    poses = json.loads(
        (ROOT / config["inputs"]["poses"]).read_text(encoding="ascii")
    )
    checkpoint = ROOT / config["inputs"]["checkpoint"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint, device=device)
    poses_by_component = {
        row["component_id"]: row["poses"] for row in poses["components"]
    }
    records, paired = [], []
    panel_scores = {name: [] for name in config["sensitivity"]["source_panels"]}
    for component_id, component in candidates["components"].items():
        native = component["native_h3"]
        fixed_rows = [
            {
                "candidate_id": f"{arm_name}_{row['substitution_bucket']}",
                "origin_arm": arm_name,
                **row,
            }
            for arm_name, arm in component["arms"].items()
            for row in arm["candidates"]
        ]
        component_scores = {}
        for panel_name, rows in panel_rows(poses_by_component[component_id]).items():
            adapter, provenance = featurize_arm(
                model, rows, panel_name, native, device
            )
            cache = {}
            for fixed in fixed_rows:
                sequence = fixed["sequence"]
                cache.setdefault(sequence, score_sequence(adapter, sequence))
                value = cache[sequence]
                component_scores[(fixed["candidate_id"], panel_name)] = value
                panel_scores[panel_name].append(value)
                records.append({
                    "component_id": component_id,
                    "target": component["target"],
                    "candidate_id": fixed["candidate_id"],
                    "origin_arm": fixed["origin_arm"],
                    "substitution_bucket": fixed["substitution_bucket"],
                    "sequence": sequence,
                    "source_panel": panel_name,
                    "pose_count": len(rows),
                    "mean_h3_log_probability": value,
                    "pose_ids": provenance["pose_ids"],
                })
        for bucket in component["substitution_buckets"]:
            differences = {}
            for panel_name in panel_rows(poses_by_component[component_id]):
                differences[panel_name] = (
                    component_scores[(f"ensemble_{bucket}", panel_name)]
                    - component_scores[(f"single_state_{bucket}", panel_name)]
                )
            signs = {
                1 if value > 0 else -1 if value < 0 else 0
                for value in differences.values()
            }
            paired.append({
                "component_id": component_id,
                "substitution_bucket": bucket,
                "ensemble_minus_single_state_log_probability": differences,
                "source_panel_sign_consistent": len(signs) == 1,
            })
    correlations = {}
    for left, right in (
        ("experimental_only", "mixed"),
        ("sampled_only", "mixed"),
        ("experimental_only", "sampled_only"),
    ):
        correlation = spearmanr(panel_scores[left], panel_scores[right]).statistic
        correlations[f"{left}_vs_{right}"] = float(correlation)
    payload = {
        "schema_version": 1,
        "status": "source_sensitivity_complete",
        "classification": config["classification"],
        "config": str(config_path),
        "device": str(device),
        "fixed_candidate_count": len(records) // 3,
        "score_record_count": len(records),
        "matched_pair_count": len(paired),
        "source_panel_sign_consistent_pair_count": sum(
            row["source_panel_sign_consistent"] for row in paired
        ),
        "ensemble_favored_pair_count": {
            panel: sum(
                row["ensemble_minus_single_state_log_probability"][panel] > 0
                for row in paired
            )
            for panel in config["sensitivity"]["source_panels"]
        },
        "median_absolute_experimental_sampled_score_difference": statistics.median(
            abs(experimental - sampled)
            for experimental, sampled in zip(
                panel_scores["experimental_only"],
                panel_scores["sampled_only"],
                strict=True,
            )
        ),
        "spearman_correlations": correlations,
        "records": records,
        "matched_bucket_contrasts": paired,
        "decision": "report_source_dependence_without_candidate_regeneration",
        "claim_boundary": config["claim_boundary"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/benchmarks/idp_ensemble_v3_computational_robustness.yml"),
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    result = score(config_path, ROOT / config["outputs"]["source_sensitivity"])
    print(json.dumps({
        "status": result["status"],
        "fixed_candidates": result["fixed_candidate_count"],
        "matched_pairs": result["matched_pair_count"],
        "sign_consistent_pairs": result["source_panel_sign_consistent_pair_count"],
        "ensemble_favored": result["ensemble_favored_pair_count"],
        "spearman_correlations": result["spearman_correlations"],
    }, indent=2))


if __name__ == "__main__":
    main()
