#!/usr/bin/env python
"""Generate and state-rank versioned H3 candidates from a trained v2 model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import disorderflow.datasets.statecontrast_v2_pose_manifest  # noqa: E402,F401
import disorderflow.models.statecontrast_v2  # noqa: E402,F401
from disorderflow.datasets import get_dataset  # noqa: E402
from disorderflow.models import get_model  # noqa: E402
from disorderflow.utils.data import PaddingCollate  # noqa: E402
from disorderflow.utils.misc import load_config, seed_all  # noqa: E402
from disorderflow.utils.train import recursive_to  # noqa: E402
from scripts.evaluate_statecontrast_v2_checkpoint import (  # noqa: E402
    aggregate_state_score,
)

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def substitutions(left, right):
    return sum(a != b for a, b in zip(left, right, strict=True))


def inject_h3(batch, sequence):
    output = {key: value.clone() if isinstance(value, torch.Tensor) else value
              for key, value in batch.items()}
    mask = output["design_region_flag"].bool()
    encoded = torch.tensor(
        [ALPHABET.index(aa) for aa in sequence], device=output["aa"].device)
    if int(mask.sum()) != len(sequence):
        raise ValueError("Candidate length differs from design region")
    output["aa"][mask] = encoded
    return output


def load_model(config_path, checkpoint_path, device):
    config, _ = load_config(str(config_path))
    model = get_model(config.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, config


def score_candidate(model, dataset, indices, manifest_rows, sequence, model_cfg, device):
    scores, logits, rows = [], [], []
    for index in indices:
        sample = inject_h3(dataset[index], sequence)
        batch = recursive_to(PaddingCollate()([sample]), device)
        result = model.score_states(batch)
        scores.append(result["state_score"][0])
        logits.append(result["pose_quality_logit"][0])
        rows.append(manifest_rows[dataset.records[index]["record_id"]])
    values = aggregate_state_score(
        rows, torch.stack(scores), torch.stack(logits), model_cfg)
    return values


def generate(config_path, checkpoint_path, output_path, device="cuda",
             attempts_per_arm=40, seeds=(8301, 8311, 8321, 8331),
             component_filter=None):
    device = torch.device(device)
    model, config = load_model(config_path, checkpoint_path, device)
    dataset_cfg = copy.deepcopy(config.dataset.train)
    dataset_cfg.split = "all"
    dataset_cfg.pop("heldout_fold", None)
    dataset_cfg.pop("fold_role", None)
    dataset = get_dataset(dataset_cfg)
    manifest = json.loads(Path(dataset_cfg.manifest_path).read_text(encoding="ascii"))
    manifest_rows = {row["record_id"]: row for row in manifest["records"]}
    by_component_group = defaultdict(lambda: defaultdict(list))
    for index, record in enumerate(dataset.records):
        by_component_group[record["component_id"]][record["group_id"]].append(index)
    outputs = {}
    for component_id in sorted(by_component_group):
        if component_filter and component_id not in component_filter:
            continue
        candidate_groups = by_component_group[component_id]
        template_group = sorted(candidate_groups)[0]
        state_indices = candidate_groups[template_group]
        native = dataset.records[state_indices[0]]["native_h3"]
        target_indices = [
            index for index in state_indices
            if dataset.records[index]["state"]["type"] == "target"
        ]
        experimental = [
            index for index in target_indices
            if dataset.records[index]["state"].get("source_panel") == "experimental"
        ]
        if not target_indices or not experimental:
            raise RuntimeError(f"{component_id} lacks target/experimental states")
        arm_pose_indices = {
            "ensemble": target_indices,
            "single_state": experimental[:1],
        }
        arm_results = {}
        for arm, pose_indices in arm_pose_indices.items():
            generated = []
            for attempt in range(int(attempts_per_arm)):
                pose_index = pose_indices[attempt % len(pose_indices)]
                sample = dataset[pose_index]
                sample = inject_h3(sample, native)
                batch = recursive_to(PaddingCollate()([sample]), device)
                seed = int(seeds[attempt % len(seeds)]) + attempt
                seed_all(seed)
                trajectory = model.sample(batch, sample_opt={
                    "deterministic": False, "num_recycles": 1,
                    "sample_structure": False, "sample_sequence": True,
                })
                mask = batch["generate_flag"][0].bool()
                amino_acids = trajectory[0][2][0][mask]
                if any(int(value) >= len(ALPHABET) for value in amino_acids):
                    continue
                sequence = "".join(ALPHABET[int(value)] for value in amino_acids.cpu())
                if sequence not in {row["sequence"] for row in generated}:
                    generated.append({
                        "sequence": sequence,
                        "substitutions": substitutions(sequence, native),
                        "seed": seed,
                        "source_pose_record": dataset.records[pose_index]["record_id"],
                    })
            for row in generated:
                row["state_scores"] = score_candidate(
                    model, dataset, state_indices, manifest_rows, row["sequence"],
                    config.model.statecontrast_v2, device)
            selected = []
            for bucket in (2, 4, 6, 8):
                eligible = [row for row in generated if row["substitutions"] == bucket]
                if eligible:
                    selected.append(max(
                        eligible,
                        key=lambda row: (row["state_scores"]["state_gap"], row["sequence"])))
            arm_results[arm] = {
                "attempted": attempts_per_arm,
                "unique_candidates": len(generated),
                "selected_buckets": [row["substitutions"] for row in selected],
                "failed_buckets": [bucket for bucket in (2, 4, 6, 8)
                                   if bucket not in {row["substitutions"] for row in selected}],
                "selected": selected,
                "all_candidates": generated,
            }
        outputs[component_id] = {
            "native_h3": native,
            "arms": arm_results,
        }
    payload = {
        "schema_version": 1,
        "status": "statecontrast_v2_candidate_generation_complete",
        "classification": "exposed_expanded_idp_development_candidates",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "config_sha256": sha256(config_path),
        "attempts_per_arm": attempts_per_arm,
        "seeds": list(seeds),
        "components": outputs,
        "claim_boundary": (
            "Development candidate generation only; missing mutation buckets "
            "remain explicit failures and are never imputed"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attempts-per-arm", type=int, default=40)
    parser.add_argument("--component", action="append", default=[])
    args = parser.parse_args()
    result = generate(
        ROOT / args.config, ROOT / args.checkpoint, ROOT / args.output,
        device=args.device, attempts_per_arm=args.attempts_per_arm,
        component_filter=set(args.component))
    print(json.dumps({
        "status": result["status"],
        "components": len(result["components"]),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
