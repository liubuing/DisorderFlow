"""Fit dev-only disorder probability calibration and an operating threshold."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from disorderflow.datasets import get_dataset  # noqa: E402
from disorderflow.datasets.disorder_augmented import (  # noqa: E402
    DisorderAugmentedDataset,
    load_disorder_lookup,
)
from disorderflow.disorder_calibration import (  # noqa: E402
    apply_platt,
    calibration_metrics,
    fit_platt,
    mcc_at_threshold,
    select_mcc_threshold,
)
from disorderflow.disorder_metrics import pooled_disorder_metrics  # noqa: E402
from disorderflow.models import get_model  # noqa: E402
from disorderflow.utils.data import PaddingCollate  # noqa: E402
from disorderflow.utils.train import recursive_to  # noqa: E402


def load_dev_dataset(config):
    dataset = get_dataset(config.dataset.val)
    ids = getattr(dataset, "ids", getattr(dataset, "all_ids", None))
    if ids is None and hasattr(dataset, "_valid_indices"):
        ids = [f"{index:08d}" for index in dataset._valid_indices]
    if ids is None:
        raise RuntimeError("Validation sample IDs are unavailable")
    lookup_path = config.dataset.get("disorder_lookup_val")
    if not lookup_path:
        raise RuntimeError("Checkpoint config has no disorder_lookup_val")
    lookup = load_disorder_lookup(
        lookup_path,
        expected_split=config.dataset.get("disorder_lookup_val_split", "dev"),
        expected_ids=ids,
        require_envelope=True,
    )
    valid = {str(key).casefold() for key, value in lookup.items() if value is not None}
    filtered_ids = [value for value in ids if str(value).casefold() in valid]
    if hasattr(dataset, "_valid_indices"):
        keep = {str(value).casefold() for value in filtered_ids}
        dataset._valid_indices = [
            index for index in dataset._valid_indices
            if f"{index:08d}".casefold() in keep
        ]
    else:
        dataset.ids = filtered_ids
    return DisorderAugmentedDataset(dataset, lookup, filtered_ids), filtered_ids, lookup_path


def collect_logits(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("weights_kind") != "ema":
        raise RuntimeError("Disorder calibration requires an EMA best checkpoint")
    config = checkpoint["config"]
    model = get_model(config.model).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    dataset, ids, lookup_path = load_dev_dataset(config)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=PaddingCollate(), num_workers=0)
    logits, labels, weights, groups = [], [], [], []
    with torch.no_grad():
        for index, batch in enumerate(tqdm(loader, desc="Collect dev logits")):
            batch = recursive_to(batch, device)
            torch.manual_seed(42)
            if device == "cuda":
                torch.cuda.manual_seed_all(42)
            trajectory = model.sample(batch, sample_opt={
                "deterministic": True,
                "return_disorder": True,
                "num_recycles": 1,
            })
            raw = trajectory["disorder"]
            target = batch["disorder_label"].float()
            mask = (batch["mask"].bool()
                    & batch["disorder_supervision_mask"].bool()
                    & torch.isfinite(target))
            count = int(mask.sum())
            logits.extend(raw.float()[mask].cpu().tolist())
            labels.extend(target[mask].cpu().tolist())
            weights.extend(batch["disorder_confidence"].float()[mask].cpu().tolist())
            groups.extend([str(ids[index])] * count)
    return logits, labels, weights, groups, lookup_path


def grouped_cross_validation(logits, labels, weights, groups, folds=5):
    fold_ids = [int(hashlib.sha256(group.encode()).hexdigest(), 16) % folds for group in groups]
    oof_probabilities = [None] * len(logits)
    oof_predictions = [None] * len(logits)
    fold_reports = []
    for fold in range(folds):
        train_indices = [i for i, value in enumerate(fold_ids) if value != fold]
        test_indices = [i for i, value in enumerate(fold_ids) if value == fold]
        parameters = fit_platt(
            [logits[i] for i in train_indices],
            [labels[i] for i in train_indices],
            [weights[i] for i in train_indices],
        )
        train_probabilities = apply_platt(
            [logits[i] for i in train_indices], **parameters).tolist()
        operating_point = select_mcc_threshold(
            train_probabilities, [labels[i] for i in train_indices])
        test_probabilities = apply_platt(
            [logits[i] for i in test_indices], **parameters).tolist()
        for index, probability in zip(test_indices, test_probabilities, strict=True):
            oof_probabilities[index] = probability
            oof_predictions[index] = probability >= operating_point["threshold"]
        fold_reports.append({
            "fold": fold,
            "n_proteins": len({groups[i] for i in test_indices}),
            "n_residues": len(test_indices),
            **parameters,
            "threshold": operating_point["threshold"],
        })
    oof_metrics = calibration_metrics(oof_probabilities, labels, weights)
    # Convert fold-specific predictions to a common binary score for pooled OOF MCC.
    oof_mcc = mcc_at_threshold([float(value) for value in oof_predictions], labels, 0.5)
    return {"folds": fold_reports, "oof_weighted_calibration": oof_metrics,
            "oof_operating_point": oof_mcc}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    logits, labels, weights, groups, lookup_path = collect_logits(args.checkpoint, args.device)
    uncalibrated = torch.sigmoid(torch.tensor(logits, dtype=torch.float64)).tolist()
    parameters = fit_platt(logits, labels, weights)
    calibrated = apply_platt(logits, **parameters).tolist()
    operating_point = select_mcc_threshold(calibrated, labels)
    report = {
        "schema_version": 1,
        "method": "weighted_platt_scaling",
        "inference_path": "model.sample deterministic num_recycles=1",
        "checkpoint": os.path.normpath(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "calibration_split": "uniref50_disjoint_dev",
        "lookup": os.path.normpath(lookup_path),
        "negative_evidence": "AFDB pLDDT >= 0.90 ordered proxy",
        "warning": "Absolute probability calibration depends on proxy ordered labels.",
        "n_proteins": len(set(groups)),
        "n_residues": len(labels),
        "positive_rate": sum(float(value) >= 0.5 for value in labels) / len(labels),
        "evidence_weighted_positive_rate": (
            sum(weight * (float(label) >= 0.5) for label, weight in zip(
                labels, weights, strict=True)) / sum(weights)),
        "parameters": parameters,
        "threshold": operating_point["threshold"],
        "discrimination": pooled_disorder_metrics(uncalibrated, labels),
        "uncalibrated_weighted": calibration_metrics(uncalibrated, labels, weights),
        "uncalibrated_operating_point_at_0_5": mcc_at_threshold(
            uncalibrated, labels, 0.5),
        "calibrated_weighted": calibration_metrics(calibrated, labels, weights),
        "calibrated_operating_point": operating_point,
        "grouped_cross_validation": grouped_cross_validation(
            logits, labels, weights, groups, args.folds),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
