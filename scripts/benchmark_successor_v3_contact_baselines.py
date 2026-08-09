#!/usr/bin/env python
"""Run geometry, sequence, and frozen-encoder contact baselines by component fold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from disorderflow.utils.data import PaddingCollate  # noqa: E402
from disorderflow.utils.misc import load_config  # noqa: E402
from disorderflow.utils.train import recursive_to  # noqa: E402
from scripts.evaluate_successor_v3_confirmatory import (  # noqa: E402
    load_frozen_model,
    record_batch,
    sha256,
    write_json_once,
)
from scripts.evaluate_successor_v3_development import contact_labels  # noqa: E402


AA = "ACDEFGHIKLMNPQRSTVWY"


def sequence_features(record):
    sequence = record["cdr_h3_sequence"]
    antigen = record["antigen_sequence"]
    composition = np.asarray([antigen.count(aa) / len(antigen) for aa in AA])
    rows = []
    for index, aa in enumerate(sequence):
        current = np.eye(20)[AA.index(aa)]
        previous = np.zeros(20) if index == 0 else np.eye(20)[AA.index(sequence[index - 1])]
        following = (np.zeros(20) if index + 1 == len(sequence)
                     else np.eye(20)[AA.index(sequence[index + 1])])
        numeric = np.asarray([
            index / max(len(sequence) - 1, 1), len(sequence) / 30.0,
            len(antigen) / 50.0])
        rows.append(np.concatenate([current, previous, following, composition, numeric]))
    return np.stack(rows)


def metrics(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    probabilities = np.clip(scores, 1e-7, 1 - 1e-7)
    return {
        "n": int(len(labels)), "positive_fraction": float(labels.mean()),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }


def cross_validated_logistic(features, labels, folds, c_value=1.0):
    predictions = np.zeros(len(labels), dtype=np.float64)
    for fold in sorted(set(folds)):
        train, test = folds != fold, folds == fold
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c_value, max_iter=2000, class_weight="balanced"))
        model.fit(features[train], labels[train])
        predictions[test] = model.predict_proba(features[test])[:, 1]
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--structural-manifest", type=Path, required=True)
    parser.add_argument("--isolation-audit", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    paths = {name: ROOT / value for name, value in {
        "dataset": args.dataset_manifest, "structural": args.structural_manifest,
        "isolation": args.isolation_audit, "protocol": args.protocol}.items()}
    dataset = json.loads(paths["dataset"].read_text(encoding="utf-8"))
    structural = json.loads(paths["structural"].read_text(encoding="utf-8"))
    isolation = json.loads(paths["isolation"].read_text(encoding="utf-8"))
    protocol = yaml.safe_load(paths["protocol"].read_text(encoding="utf-8"))
    source = {row["instance"]: row for row in structural["records"]}
    rows = sorted(dataset["records"], key=lambda row: row["instance"])
    isolated = {row["instance"] for row in isolation["audit"] if row["independent"]}

    config, _ = load_config("configs/train/bfn_successor_v3_h3_specificity.yml")
    initializer = protocol["initializer"]
    model = load_frozen_model(
        ROOT / initializer["path"], initializer["sha256"], config.model, args.device)
    captured = []
    hook = model.bfn.receiver.encoder.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output.detach()))

    labels_all, folds_all, instances_all = [], [], []
    geometry_scores, sequence_rows, encoder_rows = [], [], []
    try:
        for index, item in enumerate(rows, 1):
            record = source[item["instance"]]
            raw = record_batch(record)
            original_aa = raw["aa"].clone()
            raw["aa"][raw["generate_flag"].bool()] = 0
            batch = recursive_to(PaddingCollate()([raw]), args.device)
            captured.clear()
            with torch.inference_mode():
                model.score(batch, fixed_t=0.5)
            labels, generated = contact_labels(batch)
            generated_mask = generated[0]
            target_labels = labels[0][generated_mask].cpu().int().numpy()
            target_features = captured[-1][0][generated_mask].cpu().numpy()

            cb = batch["pos_heavyatom"][0, :, 4]
            cb_mask = batch["mask_heavyatom"][0, :, 4].bool()
            ca = batch["pos_heavyatom"][0, :, 1]
            positions = torch.where(cb_mask.unsqueeze(-1), cb, ca)
            antigen = batch["mask_antigen"][0].bool()
            distances = torch.cdist(positions[generated_mask], positions[antigen]).min(dim=1).values
            geometry_scores.extend(torch.sigmoid(8.0 - distances).cpu().tolist())
            labels_all.extend(target_labels.tolist())
            folds_all.extend([item["fold"]] * len(target_labels))
            instances_all.extend([item["instance"]] * len(target_labels))
            sequence_rows.append(sequence_features(record))
            encoder_rows.append(target_features)
            raw["aa"] = original_aa
            print(f"Extracted {index}/{len(rows)}", flush=True)
    finally:
        hook.remove()

    labels_array = np.asarray(labels_all, dtype=np.int64)
    folds_array = np.asarray(folds_all, dtype=np.int64)
    sequence_array = np.concatenate(sequence_rows)
    encoder_array = np.concatenate(encoder_rows)
    sequence_scores = cross_validated_logistic(
        sequence_array, labels_array, folds_array)
    encoder_scores = cross_validated_logistic(
        encoder_array, labels_array, folds_array)
    geometry_array = np.asarray(geometry_scores)
    isolated_mask = np.asarray([value in isolated for value in instances_all])

    arms = {
        "geometry_distance_ceiling": geometry_array,
        "sequence_logistic": sequence_scores,
        "frozen_encoder_logistic_probe": encoder_scores,
    }
    payload = {
        "schema_version": 1,
        "status": "exposed_contact_baselines_complete",
        "classification": "retrospective exposed development; not confirmatory or external",
        "inputs": {name: {"path": value.relative_to(ROOT).as_posix(), "sha256": sha256(value)}
                   for name, value in paths.items()},
        "counts": {
            "records": len(rows), "residues": len(labels_array),
            "reference_isolated_records": len(isolated),
            "reference_isolated_residues": int(isolated_mask.sum()),
        },
        "arms": {
            name: {
                "all_exposed_records": metrics(labels_array, scores),
                "matched_held_out_fold_4": metrics(
                    labels_array[folds_array == 4], scores[folds_array == 4]),
                "reference_isolated_subset": metrics(
                    labels_array[isolated_mask], scores[isolated_mask]),
            } for name, scores in arms.items()
        },
        "claim_boundary": (
            "Geometry is a label-derived ceiling. Sequence and encoder results are "
            "component-fold development estimates only."),
    }
    write_json_once(ROOT / args.output, payload)
    print(json.dumps(payload["arms"], indent=2))


if __name__ == "__main__":
    main()
