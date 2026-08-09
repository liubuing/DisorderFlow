#!/usr/bin/env python
"""Select a diverse mutation panel for prospective SPR/BLI validation."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import nct, t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from disorderflow.datasets import get_dataset  # noqa: E402
from disorderflow.models import get_model  # noqa: E402
from disorderflow.utils.data import PaddingCollate  # noqa: E402
from disorderflow.utils.misc import load_config  # noqa: E402
from disorderflow.utils.train import recursive_to  # noqa: E402


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_t_power(n, effect_log10_kd, sd_log10_kd, alpha=0.05):
    df = n - 1
    critical = t.ppf(1 - alpha / 2, df)
    noncentrality = effect_log10_kd / sd_log10_kd * np.sqrt(n)
    return float(nct.sf(critical, df, noncentrality) + nct.cdf(-critical, df, noncentrality))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--structural-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir
    generated_outputs = (output_dir / "constructs.csv", output_dir / "panel.json")
    if any(path.exists() for path in generated_outputs):
        raise FileExistsError(
            f"Refusing to overwrite generated experiment panel in {output_dir}")
    dataset_manifest_path = ROOT / args.dataset_manifest
    structure_path = ROOT / args.structural_manifest
    checkpoint_path = ROOT / args.checkpoint
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    structures = json.loads(structure_path.read_text(encoding="utf-8"))
    by_id = {row["instance"]: row for row in structures["records"]}
    eval_rows = sorted(
        [row for row in dataset_manifest["records"]
         if row["fold"] == dataset_manifest["eval_fold"]],
        key=lambda row: row["instance"])
    config, _ = load_config("configs/train/bfn_successor_v3_contact_v2_exposed.yml")
    dataset = get_dataset(copy.deepcopy(config.dataset.val))
    model = get_model(copy.deepcopy(config.model)).to(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    candidates = []
    for meta, item in zip(eval_rows, dataset, strict=True):
        batch = recursive_to(PaddingCollate()([item]), args.device)
        with torch.inference_mode():
            result = model.score(batch, fixed_t=0.5)
        generated = batch["generate_flag"][0].bool()
        probabilities = torch.sigmoid(result["contact"][0][generated]).cpu().numpy()
        if len(probabilities) < 4:
            continue
        order = np.argsort(probabilities)
        high = order[-2:][::-1]
        low = int(order[0])
        candidates.append({
            "meta": meta, "record": by_id[meta["instance"]],
            "probabilities": probabilities, "high": high.tolist(), "low": low,
            "score": float(probabilities[high].mean() - probabilities[low]),
        })
    candidates.sort(key=lambda row: (-row["score"], row["meta"]["instance"]))
    selected, components = [], set()
    for candidate in candidates:
        component = candidate["meta"]["component_id"]
        if component not in components:
            selected.append(candidate)
            components.add(component)
        if len(selected) == args.targets:
            break
    if len(selected) < args.targets:
        raise RuntimeError(f"Fewer than {args.targets} eligible independent components")

    constructs, panel = [], []
    for candidate in selected:
        sequence = candidate["record"]["cdr_h3_sequence"]
        target_id = candidate["meta"]["instance"]
        target_constructs = [{
            "construct_id": f"{target_id}__WT", "target_instance": target_id,
            "component_id": candidate["meta"]["component_id"], "role": "wild_type",
            "h3_index_1based": "", "native_aa": "", "mutant_aa": "",
            "h3_sequence": sequence, "predicted_contact_probability": "",
        }]
        for rank, index in enumerate(candidate["high"], 1):
            mutant = "A" if sequence[index] != "A" else "G"
            mutated = sequence[:index] + mutant + sequence[index + 1:]
            target_constructs.append({
                "construct_id": f"{target_id}__HOT{rank}_{sequence[index]}{index + 1}{mutant}",
                "target_instance": target_id, "component_id": candidate["meta"]["component_id"],
                "role": "predicted_contact_disruption", "h3_index_1based": index + 1,
                "native_aa": sequence[index], "mutant_aa": mutant, "h3_sequence": mutated,
                "predicted_contact_probability": float(candidate["probabilities"][index]),
            })
        index = candidate["low"]
        mutant = "A" if sequence[index] != "A" else "G"
        target_constructs.append({
            "construct_id": f"{target_id}__LOW_{sequence[index]}{index + 1}{mutant}",
            "target_instance": target_id, "component_id": candidate["meta"]["component_id"],
            "role": "low_contact_mutation_control", "h3_index_1based": index + 1,
            "native_aa": sequence[index], "mutant_aa": mutant,
            "h3_sequence": sequence[:index] + mutant + sequence[index + 1:],
            "predicted_contact_probability": float(candidate["probabilities"][index]),
        })
        constructs.extend(target_constructs)
        panel.append({
            "target_instance": target_id, "component_id": candidate["meta"]["component_id"],
            "pdb_id": candidate["record"]["pdb_id"], "h3_sequence": sequence,
            "selection_score": candidate["score"],
            "construct_ids": [row["construct_id"] for row in target_constructs],
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(constructs[0])
    with (output_dir / "constructs.csv").open("x", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(constructs)
    payload = {
        "schema_version": 1,
        "status": "planned_not_executed",
        "classification": "prospective wet-experiment design; contains no measurements",
        "inputs": {
            "checkpoint": {"path": args.checkpoint.as_posix(), "sha256": sha256(checkpoint_path)},
            "dataset_manifest": {"path": args.dataset_manifest.as_posix(), "sha256": sha256(dataset_manifest_path)},
            "structural_manifest": {"path": args.structural_manifest.as_posix(), "sha256": sha256(structure_path)},
        },
        "panel": panel,
        "power_analysis": {
            "primary_comparison": (
                "target-level mean(high-contact mutant minus WT log10(KD)) "
                "minus low-contact mutant minus WT log10(KD)"),
            "independent_targets": args.targets,
            "high_contact_mutations": args.targets * 2,
            "low_contact_controls": args.targets,
            "two_sided_alpha": 0.05,
            "target_effect_log10_kd": 0.5, "assumed_sd_log10_kd": 0.5,
            "target_level_paired_t_test_power": paired_t_power(
                args.targets, 0.5, 0.5),
            "technical_replicates_per_construct": 3,
            "note": "Technical replicates assess assay precision and do not increase biological n.",
        },
        "claim_boundary": "Constructs are model-selected hypotheses. No binding result is asserted.",
    }
    (output_dir / "panel.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="ascii")
    print(json.dumps({"targets": len(panel), "constructs": len(constructs),
                      "power": payload["power_analysis"]
                      ["target_level_paired_t_test_power"]}, indent=2))


if __name__ == "__main__":
    main()
