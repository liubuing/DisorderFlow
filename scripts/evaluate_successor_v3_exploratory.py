#!/usr/bin/env python
"""Evaluate the frozen models on a retrospective exploratory successor-v3 panel."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from disorderflow.utils.data import PaddingCollate  # noqa: E402
from disorderflow.utils.misc import load_config, seed_all  # noqa: E402
from disorderflow.utils.train import recursive_to  # noqa: E402
from scripts.evaluate_successor_v3_development import contact_labels  # noqa: E402
from scripts.evaluate_successor_v3_confirmatory import (  # noqa: E402
    bootstrap_ci,
    load_frozen_model,
    record_batch,
    sha256,
    summarize,
    write_json_once,
)


def evaluate_exploratory(model, pairs, seeds, device):
    rows, contact_scores, contact_labels_all = [], [], []
    for index, (target, donor) in enumerate(pairs):
        source_batch = PaddingCollate()([record_batch(target), record_batch(donor)])
        gaps, factual_nll = [], []
        for base_seed in seeds:
            seed_all(int(base_seed) + index)
            batch = recursive_to(copy.deepcopy(source_batch), device)
            batch["fixed_t"] = 0.5
            batch["return_per_sample_metrics"] = True
            with torch.inference_mode():
                losses = model(batch)
            valid = losses.get("antigen_mismatch_valid_per_sample")
            if valid is None or not bool(valid[0]):
                continue
            gaps.append(float(losses["antigen_mismatch_gap_per_sample"][0]))
            factual_nll.append(float(losses["factual_nll_per_sample"][0]))

        score_batch = recursive_to(copy.deepcopy(source_batch), device)
        with torch.inference_mode():
            scores = model.score(score_batch, fixed_t=0.5)
        labels, generated = contact_labels(score_batch)
        contact_scores.extend(
            torch.sigmoid(scores["contact"][0][generated[0]]).cpu().tolist())
        contact_labels_all.extend(labels[0][generated[0]].cpu().int().tolist())
        rows.append({
            "component_id": target["component_id"],
            "id": target["instance"],
            "donor_id": donor["instance"],
            "valid_seeds": len(gaps),
            "factual_minus_mismatched_native_h3_nll": (
                float(np.mean(gaps)) if gaps else None),
            "factual_native_h3_nll": float(np.mean(factual_nll)) if factual_nll else None,
        })
    auroc = None
    if len(set(contact_labels_all)) == 2:
        auroc = float(roc_auc_score(contact_labels_all, contact_scores))
    return rows, {
        "auroc": auroc,
        "n_residues": len(contact_labels_all),
        "positive_fraction": float(np.mean(contact_labels_all)),
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
    if output_path.exists():
        raise FileExistsError("Exploratory output already exists")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    if panel.get("status") != "exploratory_model_free_isolation_ready":
        raise ValueError("Panel did not pass exploratory model-free isolation")
    if not panel.get("gate_passed") or panel["counts"]["components"] < 12:
        raise ValueError("Exploratory panel has fewer than 12 components")

    train_config, _ = load_config("configs/train/bfn_successor_v3_h3_specificity.yml")
    checkpoint_contract = protocol["frozen_checkpoints"]
    records = {row["instance"]: row for row in panel["representatives"]}
    pairs = []
    for component in panel["components"]:
        target = dict(records[component["representative_id"]])
        target["component_id"] = component["component_id"]
        pairs.append((target, records[component["donor_id"]]))

    endpoint = protocol["primary_endpoint"]
    successor = load_frozen_model(
        ROOT / checkpoint_contract["successor"]["path"],
        checkpoint_contract["successor"]["sha256"], train_config.model, args.device)
    successor_rows, successor_contact = evaluate_exploratory(
        successor, pairs, endpoint["seeds"], args.device)
    del successor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    stage_a = load_frozen_model(
        ROOT / checkpoint_contract["stage_a"]["path"],
        checkpoint_contract["stage_a"]["sha256"], train_config.model, args.device)
    stage_a_rows, stage_a_contact = evaluate_exploratory(
        stage_a, pairs, endpoint["seeds"], args.device)

    successor_summary = summarize(
        successor_rows, endpoint["seeds"], endpoint["bootstrap_seed"],
        endpoint["bootstrap_draws"])
    stage_a_summary = summarize(
        stage_a_rows, endpoint["seeds"], endpoint["bootstrap_seed"],
        endpoint["bootstrap_draws"])
    successor_by_id = {row["id"]: row for row in successor_rows}
    nll_deltas = [
        successor_by_id[row["id"]]["factual_native_h3_nll"]
        - row["factual_native_h3_nll"]
        for row in stage_a_rows
        if row["valid_seeds"] == len(endpoint["seeds"])
        and successor_by_id[row["id"]]["valid_seeds"] == len(endpoint["seeds"])
    ]
    report = {
        "schema_version": 1,
        "status": "retrospective_exploratory_evaluation_complete",
        "classification": "retrospective_exploratory; not confirmatory or external",
        "claim_boundary": protocol["claim_boundary"],
        "protocol_sha256": sha256(protocol_path),
        "panel_sha256": sha256(panel_path),
        "seeds": endpoint["seeds"],
        "fixed_t": endpoint["fixed_t"],
        "successor": {**successor_summary, "contact": successor_contact},
        "stage_a": {**stage_a_summary, "contact": stage_a_contact},
        "successor_minus_stage_a_factual_native_h3_nll": {
            "mean": float(np.mean(nll_deltas)) if nll_deltas else None,
            "component_bootstrap_ci95": bootstrap_ci(
                nll_deltas, endpoint["bootstrap_seed"] + 1,
                endpoint["bootstrap_draws"]) if nll_deltas else None,
        },
        "rows": successor_rows,
    }
    write_json_once(output_path, report)
    print(json.dumps({
        "status": report["status"],
        "successor": report["successor"],
        "successor_minus_stage_a_factual_native_h3_nll":
            report["successor_minus_stage_a_factual_native_h3_nll"],
    }, indent=2))


if __name__ == "__main__":
    main()
