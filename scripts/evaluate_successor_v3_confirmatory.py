#!/usr/bin/env python
"""Run the one-shot successor-v3 confirmatory evaluation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

from disorderflow.models import get_model
from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.misc import load_config, seed_all
from disorderflow.utils.train import recursive_to
from modules.bfn_loader import build_region_batch
from scripts.evaluate_successor_v3_development import contact_labels


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_once(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="ascii", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def unbatch(batch):
    result = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == 1:
            result[key] = value[0].cpu()
        elif isinstance(value, list) and value and all(
                isinstance(item, (list, tuple)) and len(item) == 1 for item in value):
            result[key] = [item[0] for item in value]
        else:
            result[key] = value
    return result


def record_batch(record):
    positions = [int(value) + 1 for value in record["h3_heavy_indices_zero_based"]]
    region = "H:" + ",".join(str(value) for value in positions)
    batch = build_region_batch(
        ROOT / record["pdb_path"], region,
        context_chains=["L", "P"], antigen_chains=["P"], device="cpu")
    return unbatch(batch)


def load_frozen_model(path, expected_hash, model_config, device):
    actual_hash = sha256(path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Checkpoint hash mismatch for {path}: {actual_hash} != {expected_hash}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = get_model(copy.deepcopy(model_config)).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint state mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
    model.eval()
    return model


def bootstrap_ci(values, seed, draws):
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def evaluate(model, pairs, seeds, device):
    rows, all_contact_scores, all_contact_labels = [], [], []
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
            if not bool(losses["antigen_mismatch_valid_per_sample"][0]):
                continue
            gaps.append(float(losses["antigen_mismatch_gap_per_sample"][0]))
            factual_nll.append(float(losses["factual_nll_per_sample"][0]))

        score_batch = recursive_to(copy.deepcopy(source_batch), device)
        with torch.inference_mode():
            scores = model.score(score_batch, fixed_t=0.5)
        labels, generated = contact_labels(score_batch)
        all_contact_scores.extend(
            torch.sigmoid(scores["contact"][0][generated[0]]).cpu().tolist())
        all_contact_labels.extend(labels[0][generated[0]].cpu().int().tolist())
        rows.append({
            "component_id": target["component_id"],
            "id": target["instance"],
            "donor_id": donor["instance"],
            "valid_seeds": len(gaps),
            "factual_minus_mismatched_native_h3_nll": (
                float(np.mean(gaps)) if gaps else None),
            "factual_native_h3_nll": (
                float(np.mean(factual_nll)) if factual_nll else None),
        })
    contact_auroc = None
    if len(set(all_contact_labels)) == 2:
        contact_auroc = float(roc_auc_score(all_contact_labels, all_contact_scores))
    return rows, {
        "auroc": contact_auroc,
        "n_residues": len(all_contact_labels),
        "positive_fraction": float(np.mean(all_contact_labels)),
    }


def summarize(rows, seeds, bootstrap_seed, draws):
    valid = [row for row in rows if row["valid_seeds"] == len(seeds)]
    gaps = [row["factual_minus_mismatched_native_h3_nll"] for row in valid]
    nll = [row["factual_native_h3_nll"] for row in valid]
    if not gaps:
        return {
            "valid_components": 0, "valid_fraction": 0.0,
            "mean_factual_minus_mismatched_native_h3_nll": None,
            "component_bootstrap_ci95": None,
            "negative_component_fraction": None,
            "mean_factual_native_h3_nll": None,
        }
    return {
        "valid_components": len(valid),
        "valid_fraction": len(valid) / len(rows),
        "mean_factual_minus_mismatched_native_h3_nll": float(np.mean(gaps)),
        "component_bootstrap_ci95": bootstrap_ci(gaps, bootstrap_seed, draws),
        "negative_component_fraction": float(np.mean(np.asarray(gaps) < 0)),
        "mean_factual_native_h3_nll": float(np.mean(nll)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path,
        default=Path("configs/benchmarks/successor_v3_confirmatory.yml"))
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol_path, panel_path, output_path = (
        ROOT / args.protocol, ROOT / args.panel, ROOT / args.output)
    attempt_path = output_path.with_suffix(".attempt.json")
    if output_path.exists() or attempt_path.exists():
        raise FileExistsError("Confirmatory evaluation is one-shot and was already attempted")

    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    if panel.get("status") != "frozen_model_free_successor_v3_isolation_audit":
        raise ValueError("Panel is not a frozen model-free isolation audit")
    if not panel.get("gate_passed") or panel["counts"]["components"] < 12:
        raise ValueError("Panel did not pass the frozen minimum-component gate")

    write_json_once(attempt_path, {
        "schema_version": 1,
        "status": "confirmatory_attempt_started",
        "started_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "protocol_sha256": sha256(protocol_path),
        "panel_sha256": sha256(panel_path),
    })

    train_config, _ = load_config("configs/train/bfn_successor_v3_h3_specificity.yml")
    checkpoint_contract = protocol["frozen_checkpoints"]
    records = {row["instance"]: row for row in panel["representatives"]}
    pairs = []
    for component in panel["components"]:
        target = dict(records[component["representative_id"]])
        target["component_id"] = component["component_id"]
        donor = records[component["donor_id"]]
        pairs.append((target, donor))

    successor_path = ROOT / checkpoint_contract["successor"]["path"]
    stage_a_path = ROOT / checkpoint_contract["stage_a"]["path"]
    successor = load_frozen_model(
        successor_path, checkpoint_contract["successor"]["sha256"],
        train_config.model, args.device)
    seeds = protocol["primary_endpoint"]["seeds"]
    successor_rows, successor_contact = evaluate(successor, pairs, seeds, args.device)
    del successor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    stage_a = load_frozen_model(
        stage_a_path, checkpoint_contract["stage_a"]["sha256"],
        train_config.model, args.device)
    stage_a_rows, stage_a_contact = evaluate(stage_a, pairs, seeds, args.device)

    endpoint = protocol["primary_endpoint"]
    successor_summary = summarize(
        successor_rows, seeds, endpoint["bootstrap_seed"], endpoint["bootstrap_draws"])
    stage_a_summary = summarize(
        stage_a_rows, seeds, endpoint["bootstrap_seed"], endpoint["bootstrap_draws"])
    successor_by_id = {row["id"]: row for row in successor_rows}
    nll_deltas = [
        successor_by_id[row["id"]]["factual_native_h3_nll"]
        - row["factual_native_h3_nll"]
        for row in stage_a_rows
        if row["valid_seeds"] == len(seeds)
        and successor_by_id[row["id"]]["valid_seeds"] == len(seeds)
    ]
    mean_nll_delta = float(np.mean(nll_deltas)) if nll_deltas else None
    gates_contract = protocol["gates"]
    ci = successor_summary["component_bootstrap_ci95"]
    gates = {
        "minimum_valid_components": (
            successor_summary["valid_components"] >= gates_contract["minimum_valid_components"]),
        "minimum_valid_fraction": (
            successor_summary["valid_fraction"] >= gates_contract["minimum_valid_fraction"]),
        "maximum_mean_factual_minus_mismatched_nll": (
            successor_summary["mean_factual_minus_mismatched_native_h3_nll"] is not None
            and successor_summary["mean_factual_minus_mismatched_native_h3_nll"]
            <= gates_contract["maximum_mean_factual_minus_mismatched_nll"]),
        "maximum_component_bootstrap_ci95_upper": (
            ci is not None and ci[1]
            < gates_contract["maximum_component_bootstrap_ci95_upper"]),
        "minimum_negative_component_fraction": (
            successor_summary["negative_component_fraction"] is not None
            and successor_summary["negative_component_fraction"]
            >= gates_contract["minimum_negative_component_fraction"]),
        "maximum_successor_minus_stage_a_factual_nll": (
            mean_nll_delta is not None
            and mean_nll_delta <= gates_contract["maximum_successor_minus_stage_a_factual_nll"]),
        "minimum_contact_auroc": (
            successor_contact["auroc"] is not None
            and successor_contact["auroc"] >= gates_contract["minimum_contact_auroc"]),
    }
    report = {
        "schema_version": 1,
        "status": "confirmatory_gate_passed" if all(gates.values()) else "confirmatory_gate_failed",
        "classification": protocol["classification"],
        "claim_boundary": protocol["claim_boundary"],
        "protocol_sha256": sha256(protocol_path),
        "panel_sha256": sha256(panel_path),
        "seeds": seeds,
        "fixed_t": endpoint["fixed_t"],
        "successor": {**successor_summary, "contact": successor_contact},
        "stage_a": {**stage_a_summary, "contact": stage_a_contact},
        "successor_minus_stage_a_factual_native_h3_nll": {
            "mean": mean_nll_delta,
            "component_bootstrap_ci95": (
                bootstrap_ci(nll_deltas, endpoint["bootstrap_seed"] + 1,
                             endpoint["bootstrap_draws"])
                if nll_deltas else None),
        },
        "gates": gates,
        "all_confirmatory_gates_passed": all(gates.values()),
        "rows": successor_rows,
    }
    write_json_once(output_path, report)
    print(json.dumps({key: report[key] for key in (
        "status", "successor", "successor_minus_stage_a_factual_native_h3_nll",
        "gates", "all_confirmatory_gates_passed")}, indent=2))


if __name__ == "__main__":
    main()
