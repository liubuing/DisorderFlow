#!/usr/bin/env python
"""Evaluate a StateContrast-v2 checkpoint on a frozen manifest split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import disorderflow.datasets.statecontrast_v2_pose_manifest  # noqa: E402,F401
import disorderflow.models.statecontrast_v2  # noqa: E402,F401
from disorderflow.datasets import get_dataset  # noqa: E402
from disorderflow.models import get_model  # noqa: E402
from disorderflow.modules.statecontrast_v2 import (  # noqa: E402
    grouped_source_constrained_pose_weights,
)
from disorderflow.utils.data import PaddingCollate  # noqa: E402
from disorderflow.utils.misc import load_config  # noqa: E402
from disorderflow.utils.train import recursive_to  # noqa: E402


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bootstrap_mean(values, trials, seed):
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    estimates = generator.choice(
        values, size=(int(trials), len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(estimates, 0.025)),
            float(np.quantile(estimates, 0.975))]


def hard_developability_failure(sequence):
    hydrophobic = set("AILMFWVY")
    run = longest = 0
    for amino_acid in sequence:
        run = run + 1 if amino_acid in hydrophobic else 0
        longest = max(longest, run)
    return (
        any(sequence[index] == "N" and sequence[index + 1] != "P"
            and sequence[index + 2] in "ST"
            for index in range(max(0, len(sequence) - 2)))
        or longest > 4
    )


def aggregate_state_score(rows, state_scores, pose_logits, model_cfg, panel=None):
    indices = [
        index for index, row in enumerate(rows)
        if panel is None or row["state"].get("source_panel") == panel
    ]
    if not indices:
        return None
    scores = state_scores[indices]
    logits = pose_logits[indices]
    states = torch.tensor([
        {"target": 0, "apo": 1, "off_target": 2}[rows[index]["state"]["type"]]
        for index in indices], device=scores.device)
    sources = torch.tensor([
        stable_int(rows[index]["state"]["source"]) for index in indices
    ], device=scores.device)
    weights, _ = grouped_source_constrained_pose_weights(
        logits, states, sources,
        minimum_source_mass=float(model_cfg.get("minimum_source_mass", 0.10)),
        maximum_pose_weight=float(model_cfg.get("maximum_pose_weight", 0.60)),
    )
    state_values = {}
    for state_type, name in ((0, "target"), (1, "apo"), (2, "off_target")):
        mask = states == state_type
        if mask.any():
            state_values[name] = float(
                (scores[mask] * weights[mask]).sum()
                / weights[mask].sum().clamp(min=1e-8))
    negatives = [state_values[name] for name in ("apo", "off_target")
                 if name in state_values]
    state_values["state_gap"] = (
        state_values["target"] - max(negatives) if negatives else None)
    return state_values


def stable_int(value):
    return int(hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:15], 16)


def load_checkpoint(config_path, checkpoint_path, device):
    config, _ = load_config(str(config_path))
    model = get_model(config.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, config


def evaluate(config_path, checkpoint_path, gate_path, split, output_path,
             device="cpu", bootstrap_trials=10000, bootstrap_seed=6201):
    device = torch.device(device)
    model, train_config = load_checkpoint(config_path, checkpoint_path, device)
    dataset_cfg = train_config.dataset[split]
    dataset = get_dataset(dataset_cfg)
    manifest = json.loads(Path(dataset_cfg.manifest_path).read_text(encoding="ascii"))
    # In CV configs ``split=val`` means held-out-fold role, not the original
    # manifest split label. Index all records and let the dataset own filtering.
    manifest_records = {row["record_id"]: row for row in manifest["records"]}
    candidate_rows = []
    for group in dataset.group_indices:
        samples = [dataset[index] for index in group]
        batch = recursive_to(PaddingCollate()(samples), device)
        with torch.no_grad():
            scores = model.score_states(batch)
        records = [manifest_records[dataset.records[index]["record_id"]] for index in group]
        by_candidate = defaultdict(list)
        for index, record in enumerate(records):
            by_candidate[record["group_id"]].append(index)
        for group_id, indices in by_candidate.items():
            rows = [records[index] for index in indices]
            state_scores = scores["state_score"][indices]
            pose_logits = scores["pose_quality_logit"][indices]
            panels = {
                "mixed": aggregate_state_score(
                    rows, state_scores, pose_logits,
                    train_config.model.statecontrast_v2),
                "experimental": aggregate_state_score(
                    rows, state_scores, pose_logits,
                    train_config.model.statecontrast_v2, "experimental"),
                "sampled": aggregate_state_score(
                    rows, state_scores, pose_logits,
                    train_config.model.statecontrast_v2, "sampled"),
            }
            row = rows[0]
            teacher_values = [
                item["targets"].get("independent_teacher_score")
                for item in rows
                if item["targets"].get("independent_teacher_score") is not None
            ]
            candidate_rows.append({
                "group_id": group_id,
                "teacher_group_id": row["teacher_group_id"],
                "component_id": row["component_id"],
                "target": row["target"],
                "origin_arm": row["origin_arm"],
                "substitution_bucket": row["substitution_bucket"],
                "sequence": row["candidate_sequence"],
                "hard_developability_failure": hard_developability_failure(
                    row["candidate_sequence"]),
                "independent_teacher_score": (
                    float(np.mean(teacher_values)) if teacher_values else None),
                "panels": panels,
            })

    pairs = []
    by_teacher_group = defaultdict(list)
    for row in candidate_rows:
        by_teacher_group[row["teacher_group_id"]].append(row)
    for teacher_group, rows in by_teacher_group.items():
        by_arm = {row["origin_arm"]: row for row in rows}
        if set(by_arm) != {"ensemble", "single_state"}:
            continue
        differences = {}
        for panel in ("mixed", "experimental", "sampled"):
            left = by_arm["ensemble"]["panels"][panel]
            right = by_arm["single_state"]["panels"][panel]
            differences[panel] = (
                None if left is None or right is None
                else left["state_gap"] - right["state_gap"])
        observed = [value for value in differences.values() if value is not None]
        signs = {1 if value > 0 else -1 if value < 0 else 0 for value in observed}
        pairs.append({
            "teacher_group_id": teacher_group,
            "component_id": rows[0]["component_id"],
            "target": rows[0]["target"],
            "substitution_bucket": rows[0]["substitution_bucket"],
            "state_gap_difference": differences,
            "source_direction_consistent": len(signs) == 1 and len(observed) >= 2,
            "ensemble_target_noninferior": (
                by_arm["ensemble"]["panels"]["mixed"]["target"]
                >= by_arm["single_state"]["panels"]["mixed"]["target"]),
            "ensemble_hard_developability_failure": by_arm["ensemble"][
                "hard_developability_failure"],
            "independent_teacher_difference": (
                by_arm["ensemble"]["independent_teacher_score"]
                - by_arm["single_state"]["independent_teacher_score"]
                if by_arm["ensemble"]["independent_teacher_score"] is not None
                and by_arm["single_state"]["independent_teacher_score"] is not None
                else None),
        })

    by_component = defaultdict(list)
    for row in pairs:
        if row["state_gap_difference"]["mixed"] is not None:
            by_component[row["component_id"]].append(
                row["state_gap_difference"]["mixed"])
    component_effects = {
        component: float(np.mean(values)) for component, values in by_component.items()
    }
    values = list(component_effects.values())
    ci = bootstrap_mean(values, bootstrap_trials, bootstrap_seed) if values else None
    gate = yaml.safe_load(gate_path.read_text(encoding="ascii"))
    thresholds = gate["gates"]
    source_fraction = sum(row["source_direction_consistent"] for row in pairs) / max(1, len(pairs))
    teacher_pairs = [
        row for row in pairs if row["independent_teacher_difference"] is not None
    ]
    independent_fraction = sum(
        row["independent_teacher_difference"] > 0 for row in teacher_pairs
    ) / max(1, len(teacher_pairs))
    target_noninferior = sum(row["ensemble_target_noninferior"] for row in pairs) / max(1, len(pairs))
    hard_failure = sum(row["ensemble_hard_developability_failure"] for row in pairs) / max(1, len(pairs))
    gate_results = {
        "minimum_components": len(component_effects) >= int(
            gate["required_panels"]["expanded_replication"]["minimum_components"]),
        "minimum_positive_component_fraction": (
            sum(value > 0 for value in values) / max(1, len(values))
            >= float(thresholds["minimum_positive_component_fraction"])),
        "bootstrap_ci95_lower_above_zero": bool(ci and ci[0] > 0),
        "source_panel_direction_consistency": source_fraction >= float(
            thresholds["minimum_source_panel_direction_consistency"]),
        "independent_scorer_favored_fraction": independent_fraction >= float(
            thresholds["minimum_independent_scorer_favored_fraction"]),
        "absolute_target_state_noninferiority": target_noninferior >= 0.5,
        "developability_hard_failure_fraction": hard_failure <= float(
            thresholds["maximum_developability_hard_failure_fraction"]),
    }
    payload = {
        "schema_version": 1,
        "status": (
            "statecontrast_v2_expanded_gate_passed"
            if all(gate_results.values()) else
            "statecontrast_v2_expanded_gate_failed"),
        "classification": "exposed_expanded_idp_checkpoint_evaluation",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "training_config_sha256": sha256(config_path),
        "gate_config_sha256": sha256(gate_path),
        "split": split,
        "candidate_count": len(candidate_rows),
        "matched_pair_count": len(pairs),
        "component_count": len(component_effects),
        "component_effects": component_effects,
        "component_bootstrap_ci95": ci,
        "source_direction_consistent_fraction": source_fraction,
        "ensemble_state_gap_favored_fraction": independent_fraction,
        "target_noninferior_fraction": target_noninferior,
        "developability_hard_failure_fraction": hard_failure,
        "gate_results": gate_results,
        "candidate_records": candidate_rows,
        "matched_pairs": pairs,
        "decision": (
            "freeze_for_new_untouched_panel" if all(gate_results.values())
            else "continue_exposed_development_without_threshold_changes"),
        "claim_boundary": gate["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gate", type=Path, default=Path(
        "configs/benchmarks/statecontrast_v2_expanded_replication_gate.yml"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    result = evaluate(
        ROOT / args.config, ROOT / args.checkpoint, ROOT / args.gate,
        args.split, ROOT / args.output, device=args.device)
    print(json.dumps({
        "status": result["status"],
        "components": result["component_count"],
        "pairs": result["matched_pair_count"],
        "ci95": result["component_bootstrap_ci95"],
        "gate_results": result["gate_results"],
        "decision": result["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
