#!/usr/bin/env python
"""Audit successor-v3 exploratory contact labels, masks, and score separation."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from disorderflow.utils.data import PaddingCollate  # noqa: E402
from disorderflow.utils.protein.constants import Fragment  # noqa: E402
from disorderflow.utils.train import recursive_to  # noqa: E402
from scripts.evaluate_successor_v3_confirmatory import (  # noqa: E402
    load_frozen_model,
    record_batch,
    sha256,
    write_json_once,
)
from scripts.evaluate_successor_v3_development import contact_labels  # noqa: E402


def binary_metrics(labels, logits):
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    result = {
        "n": int(labels.size),
        "positives": int(labels.sum()),
        "positive_fraction": float(labels.mean()) if labels.size else None,
        "mean_positive_logit": float(logits[labels == 1].mean()) if labels.any() else None,
        "mean_negative_logit": float(logits[labels == 0].mean()) if (~labels.astype(bool)).any() else None,
        "logit_standard_deviation": float(logits.std()) if logits.size else None,
        "brier_score": float(brier_score_loss(labels, probabilities)) if labels.size else None,
        "auroc": None,
        "average_precision": None,
    }
    if labels.size and len(set(labels.tolist())) == 2:
        result["auroc"] = float(roc_auc_score(labels, logits))
        result["average_precision"] = float(average_precision_score(labels, logits))
    return result


def length_bin(value, boundaries):
    for upper, label in boundaries:
        if value <= upper:
            return label
    return boundaries[-1][1]


def resolution_bin(value):
    if value is None:
        return "missing"
    if value <= 2.0:
        return "le_2.0"
    if value <= 3.0:
        return "2.0_to_3.0"
    return "gt_3.0"


def score_model(model, pairs, device, paired):
    rows, labels_all, logits_all = [], [], []
    strata = defaultdict(lambda: ([], []))
    mask_mismatches = 0
    raw_logits = {}
    for record, donor in pairs:
        source = [record_batch(record)]
        if paired:
            source.append(record_batch(donor))
        batch = PaddingCollate()(source)
        batch = recursive_to(copy.deepcopy(batch), device)
        with torch.inference_mode():
            scores = model.score(batch, fixed_t=0.5)
        expected_antigen = (
            batch["mask"].bool()
            & (batch["fragment_type"] == int(Fragment.Antigen)))
        mask_mismatches += int(not torch.equal(batch["mask_antigen"], expected_antigen))
        labels, generated = contact_labels(batch)
        selected_labels = labels[0][generated[0]].cpu().int().numpy()
        selected_logits = scores["contact"][0][generated[0]].cpu().numpy()
        raw_logits[record["instance"]] = selected_logits
        labels_all.extend(selected_labels.tolist())
        logits_all.extend(selected_logits.tolist())

        bins = {
            "h3_length": length_bin(
                len(record["cdr_h3_sequence"]),
                [(9, "4_to_9"), (14, "10_to_14"), (30, "15_to_30")]),
            "antigen_length": length_bin(
                len(record["antigen_sequence"]),
                [(10, "5_to_10"), (20, "11_to_20"), (50, "21_to_50")]),
            "resolution": resolution_bin(record.get("resolution")),
        }
        for axis, value in bins.items():
            key = f"{axis}:{value}"
            strata[key][0].extend(selected_labels.tolist())
            strata[key][1].extend(selected_logits.tolist())
        component_metrics = binary_metrics(selected_labels, selected_logits)
        rows.append({
            "instance": record["instance"],
            "pdb_id": record["pdb_id"],
            "h3_length": len(record["cdr_h3_sequence"]),
            "antigen_length": len(record["antigen_sequence"]),
            "resolution": record.get("resolution"),
            **component_metrics,
        })
    return {
        "aggregate": binary_metrics(labels_all, logits_all),
        "strata": {
            key: binary_metrics(values[0], values[1])
            for key, values in sorted(strata.items())
        },
        "components": rows,
        "mask_mismatch_components": mask_mismatches,
    }, raw_logits


def context_difference(singleton, paired):
    differences = []
    component_maxima = {}
    for identifier in sorted(singleton):
        delta = np.abs(singleton[identifier] - paired[identifier])
        differences.extend(delta.tolist())
        component_maxima[identifier] = float(delta.max()) if delta.size else 0.0
    values = np.asarray(differences, dtype=np.float64)
    return {
        "n_residues": int(values.size),
        "mean_absolute_logit_difference": float(values.mean()),
        "maximum_absolute_logit_difference": float(values.max()),
        "components_above_1e_6": sum(value > 1e-6 for value in component_maxima.values()),
        "components_above_1e_3": sum(value > 1e-3 for value in component_maxima.values()),
        "per_component_maximum_absolute_difference": component_maxima,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path,
        default=Path("configs/benchmarks/successor_v3_exploratory.yml"))
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol_path, panel_path, output_path = (
        ROOT / args.protocol, ROOT / args.panel, ROOT / args.output)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    records = {row["instance"]: row for row in panel["representatives"]}
    pairs = [
        (records[component["representative_id"]], records[component["donor_id"]])
        for component in panel["components"]
    ]
    checkpoint_contract = protocol["frozen_checkpoints"]
    from disorderflow.utils.misc import load_config
    train_config, _ = load_config("configs/train/bfn_successor_v3_h3_specificity.yml")

    model_reports = {}
    for name in ("successor", "stage_a"):
        contract = checkpoint_contract[name]
        model = load_frozen_model(
            ROOT / contract["path"], contract["sha256"], train_config.model, args.device)
        singleton, singleton_logits = score_model(model, pairs, args.device, paired=False)
        paired, paired_logits = score_model(model, pairs, args.device, paired=True)
        model_reports[name] = {
            "singleton": singleton,
            "evaluation_paired": paired,
            "batch_context_difference": context_difference(
                singleton_logits, paired_logits),
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "schema_version": 1,
        "status": "exploratory_contact_diagnostic_complete",
        "classification": "post_evaluation_diagnostic; exposed exploratory development",
        "claim_boundary": "diagnosis only; not a new evaluation or confirmatory result",
        "inputs": {
            "protocol": args.protocol.as_posix(),
            "protocol_sha256": sha256(protocol_path),
            "panel": args.panel.as_posix(),
            "panel_sha256": sha256(panel_path),
        },
        "label_contract": {
            "generated_positions": "H3 positions in canonical heavy chain H",
            "antigen_mask": "mask AND fragment_type equals Fragment.Antigen",
            "atom": "CB with CA fallback when CB is unresolved",
            "distance_cutoff_angstrom": 8.0,
            "comparison": "strictly less than",
            "training_evaluation_definition_match": True,
        },
        "counts": {"components": len(records)},
        "models": model_reports,
    }
    write_json_once(output_path, payload)
    print(json.dumps({name: report["evaluation_paired"]["aggregate"]
                      for name, report in model_reports.items()}, indent=2))


if __name__ == "__main__":
    main()
