#!/usr/bin/env python
"""Freeze the PAE single-axis deployment contract with per-seed calibration maps."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_candidate_interface_checkpoint import evaluate_checkpoint  # noqa: E402
from summarize_candidate_interface_multiseed import pair_statistics  # noqa: E402
from disorderflow.datasets.confidence_dataset import ConfidenceRegressionDataset  # noqa: E402
from disorderflow.utils.data import CompleteGroupBatchSampler, PaddingCollate  # noqa: E402
from disorderflow.utils.data import lmdb_records_sha256  # noqa: E402

RESELECTION = "results/candidate_interface_multiscaffold_calibration_ext/checkpoint_reselection.json"
PAE_MAE_GATE = 0.10
MIN_PAIRS = 30
MIN_SCAFFOLDS = 3
MAX_FRACTION = 0.50


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(left, right):
    left, right = rankdata(left), rankdata(right)
    if np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def fit_std_matching(pred, target):
    slope = float(np.std(target) / np.std(pred)) if np.std(pred) > 0 else 1.0
    intercept = float(np.mean(target) - slope * np.mean(pred))
    return slope, intercept


def scaffold_spearman(pred, target, scaffolds):
    groups = defaultdict(list)
    for p, t, s in zip(pred, target, scaffolds, strict=True):
        groups[s].append((p, t))
    return {
        scaffold: spearman([p for p, _ in rows], [t for _, t in rows])
        for scaffold, rows in groups.items() if len(rows) >= 2
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="data/confidence_candidate_interface_multiscaffold_calibration_ext_dual_sem_v1/calibration.lmdb")
    parser.add_argument(
        "--dataset-manifest",
        default="data/confidence_candidate_interface_multiscaffold_calibration_ext_dual_sem_v1/manifest.json")
    parser.add_argument(
        "--reselection", default=RESELECTION)
    parser.add_argument(
        "--output",
        default="publication/candidate_interface_pae_deployment_v1.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    manifest_path = ROOT / args.dataset_manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_path = ROOT / args.dataset
    if lmdb_records_sha256(dataset_path) != manifest["summary"]["calibration"]["lmdb_records_sha256"]:
        raise ValueError("Calibration LMDB does not match its manifest")

    reselection_path = ROOT / args.reselection
    reselection = json.loads(reselection_path.read_text(encoding="utf-8"))

    dataset = ConfidenceRegressionDataset({
        "db_path": str(dataset_path),
        "candidate_interface_v1": True,
        "max_residues": 0,
    })
    sampler = CompleteGroupBatchSampler(dataset, 20, shuffle=False)
    loader = DataLoader(
        dataset, batch_sampler=sampler, collate_fn=PaddingCollate(), num_workers=0)

    seeds = []
    for entry in reselection["selections"]:
        checkpoint_path = ROOT / entry["checkpoint"]
        evaluation = evaluate_checkpoint(
            checkpoint_path, loader, args.device, expected_manifest_sha256=None)
        records = evaluation["entity_aggregated_records"]
        pred = np.asarray([r["pred_pae"] for r in records])
        target = np.asarray([r["target_pae"] for r in records])
        scaffolds = [r["scaffold_family"] for r in records]

        slope, intercept = fit_std_matching(pred, target)
        corrected = slope * pred + intercept
        corrected_mae = float(np.mean(np.abs(corrected - target)))
        corrected_variance_ratio = float((np.std(corrected) / np.std(target)) ** 2)
        sc_corr = scaffold_spearman(pred, target, scaffolds)
        median_scaffold_spearman = float(np.median(
            [v for v in sc_corr.values() if v is not None]))

        pair_records = [{
            "scaffold_family": r["scaffold_family"],
            "pred_pae": float(corrected[i]),
            "target_pae": float(target[i]),
            "noise_pae": float(r["noise_pae"]),
        } for i, r in enumerate(records)]
        pairs = pair_statistics(pair_records, "pae", bootstrap_seed=2041 + len(seeds))

        gates = {
            "mae": corrected_mae <= PAE_MAE_GATE,
            "median_scaffold_spearman": median_scaffold_spearman >= 0.5,
            "variance_ratio": 0.5 <= corrected_variance_ratio <= 2.0,
            "pair_evidence": (
                pairs["pairs"] >= MIN_PAIRS
                and pairs["contributing_scaffolds"] >= MIN_SCAFFOLDS
                and pairs["max_scaffold_pair_fraction"] is not None
                and pairs["max_scaffold_pair_fraction"] <= MAX_FRACTION),
            "pair_accuracy": pairs["accuracy"] is not None and pairs["accuracy"] >= 0.65,
            "pair_bootstrap_lower": (
                pairs["bootstrap_95_lower"] is not None
                and pairs["bootstrap_95_lower"] > 0.50),
        }
        seeds.append({
            "seed": entry["seed"],
            "checkpoint": entry["checkpoint"],
            "checkpoint_sha256": sha256(checkpoint_path),
            "calibration": {"slope": slope, "intercept": intercept},
            "corrected_mae": corrected_mae,
            "corrected_variance_ratio": corrected_variance_ratio,
            "median_scaffold_spearman": median_scaffold_spearman,
            "pairwise": {
                "reliable_pairs": pairs["pairs"],
                "contributing_scaffolds": pairs["contributing_scaffolds"],
                "max_scaffold_pair_fraction": pairs["max_scaffold_pair_fraction"],
                "accuracy": pairs["accuracy"],
                "bootstrap_95_lower": pairs["bootstrap_95_lower"],
            },
            "gates": gates,
            "gates_passed": all(gates.values()),
        })

    contract = {
        "schema_version": "candidate_interface_pae_deployment_v1",
        "classification": "development_only",
        "status": "frozen_pae_single_axis_deployment",
        "scope": "PAE single-axis candidate-interface confidence (normalized candidate-to-antigen PAE)",
        "abstentions": {
            "plddt": "non-transferable across scaffolds; five head/loss ablations falsified",
            "iptm": "insufficient reliable pairs (2/6 contributing scaffolds)",
        },
        "calibration": {
            "method": "variance-matching affine target = slope*pred + intercept, slope = target_std/pred_std",
            "fit_domain": "6 independent external calibration scaffolds (84 entity records)",
            "monotonic": True,
            "ranking_invariant": True,
            "not_model_training": True,
        },
        "deployment_gates": {
            "mae": {"threshold": PAE_MAE_GATE, "metric": "corrected_mae"},
            "median_scaffold_spearman": {"threshold": 0.5},
            "variance_ratio": {"range": [0.5, 2.0], "post_calibration_value": 1.0},
            "pair_evidence": {"min_pairs": MIN_PAIRS, "min_scaffolds": MIN_SCAFFOLDS,
                              "max_scaffold_fraction": MAX_FRACTION},
            "pair_accuracy": {"threshold": 0.65},
            "pair_bootstrap_lower": {"threshold": 0.5},
        },
        "seeds": seeds,
        "deployment_gate_passed": all(s["gates_passed"] for s in seeds),
        "multi_seed_stable": all(s["gates_passed"] for s in seeds),
        "sealed_test_policy": (
            "test split remains sealed; this PAE single-axis deployment does not unseal it"),
        "dataset_manifest_sha256": sha256(manifest_path),
        "checkpoint_reselection_sha256": sha256(reselection_path),
        "provenance": {
            "reselection": args.reselection,
            "dataset_manifest": args.dataset_manifest,
        },
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(contract, indent=2) + "\n", encoding="ascii")
    print(json.dumps({
        "deployment_gate_passed": contract["deployment_gate_passed"],
        "seeds": [{s["seed"]: s["gates"]} for s in seeds],
    }, indent=2))


if __name__ == "__main__":
    main()
