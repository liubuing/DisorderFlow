"""One-time independent final-holdout evaluation of the frozen PAE surrogate.

Evaluates the three frozen deployment-contract checkpoints on the previously
unexported final-test components (V2C001/V2C002/V2C003, GP2 antigen family),
under the frozen split policy: test components may be evaluated once only
after training and calibration are frozen.

No checkpoint, threshold, or calibration parameter may change after this run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_candidate_interface_checkpoint import evaluate_checkpoint  # noqa: E402
from summarize_candidate_interface_multiseed import summarize_selection  # noqa: E402
from disorderflow.datasets.confidence_dataset import ConfidenceRegressionDataset  # noqa: E402
from disorderflow.utils.data import CompleteGroupBatchSampler, PaddingCollate  # noqa: E402
from disorderflow.utils.data import lmdb_records_sha256  # noqa: E402

# Frozen deployment-contract checkpoints (candidate_interface_pae_deployment_v1.json)
CHECKPOINTS = [
    "logs/bfn_candidate_interface_multiscaffold_dual_sem_v2_2026_09_01__12_13_52_dual_sem_v2_s2041/checkpoints/1000.pt",
    "logs/bfn_candidate_interface_multiscaffold_dual_sem_v2_2026_09_01__12_29_06_dual_sem_v2_s2053/checkpoints/800.pt",
    "logs/bfn_candidate_interface_multiscaffold_dual_sem_v2_2026_09_01__21_10_10_dual_sem_v2_s2069_retry/checkpoints/800.pt",
]
DEFAULT_DATASET = (
    "data/confidence_candidate_interface_multiscaffold_v1_holdout_dual_sem_v2/test.lmdb")
DEFAULT_MANIFEST = (
    "data/confidence_candidate_interface_multiscaffold_v1_holdout_dual_sem_v2/manifest.json")
DEPLOYMENT_CONTRACT = "publication/candidate_interface_pae_deployment_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def apply_frozen_calibration(evaluation, slope, intercept):
    """Apply a frozen monotonic affine PAE map to entity-aggregated records."""
    records = evaluation["entity_aggregated_records"]
    corrected = []
    for row in records:
        row = dict(row)
        row["pred_pae_uncorrected"] = row["pred_pae"]
        row["pred_pae"] = slope * row["pred_pae"] + intercept
        corrected.append(row)
    return corrected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", default=CHECKPOINTS)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-manifest", default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output",
        default="results/candidate_interface_final_holdout_v1/holdout_summary.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    manifest_path = ROOT / args.dataset_manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_path = ROOT / args.dataset
    expected_sha = manifest["summary"]["test"]["lmdb_records_sha256"]
    actual_sha = lmdb_records_sha256(dataset_path)
    if actual_sha != expected_sha:
        raise ValueError("Final-holdout LMDB does not match its manifest")

    contract = json.loads((ROOT / DEPLOYMENT_CONTRACT).read_text(encoding="ascii"))
    contract_by_seed = {entry["seed"]: entry for entry in contract["seeds"]}

    dataset = ConfidenceRegressionDataset({
        "db_path": str(dataset_path),
        "candidate_interface_v1": True,
        "max_residues": 0,
    })
    sampler = CompleteGroupBatchSampler(dataset, 20, shuffle=False)
    loader = DataLoader(
        dataset, batch_sampler=sampler, collate_fn=PaddingCollate(), num_workers=0)

    selections = []
    sources = []
    for index, relative in enumerate(args.checkpoints):
        checkpoint_path = ROOT / relative
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        if sha256(checkpoint_path) != contract_by_seed[
                ["2041", "2053", "2069"][index]]["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash drifted from frozen contract: {relative}")
        evaluation = evaluate_checkpoint(
            checkpoint_path, loader, args.device, expected_manifest_sha256=None)
        summary = summarize_selection(evaluation, bootstrap_seed=2041 + index)

        seed_entry = contract_by_seed[["2041", "2053", "2069"][index]]
        slope = seed_entry["calibration"]["slope"]
        intercept = seed_entry["calibration"]["intercept"]
        corrected_records = apply_frozen_calibration(evaluation, slope, intercept)
        pred = np.asarray([row["pred_pae"] for row in corrected_records])
        target = np.asarray([row["target_pae"] for row in corrected_records])
        raw_pred = np.asarray([row["pred_pae_uncorrected"] for row in corrected_records])
        summary["pae_frozen_map_calibration"] = {
            "slope": slope,
            "intercept": intercept,
            "source": "publication/candidate_interface_pae_deployment_v1.json",
            "raw_mae": float(np.mean(np.abs(raw_pred - target))),
            "corrected_mae": float(np.mean(np.abs(pred - target))),
            "raw_variance_ratio": float(
                (np.std(raw_pred) / np.std(target)) ** 2),
            "corrected_variance_ratio": float(
                (np.std(pred) / np.std(target)) ** 2),
            "corrected_median_scaffold_spearman": summary["metrics"]["pae"][
                "median_scaffold_spearman"],
        }
        selections.append(summary)
        sources.append({
            "path": relative,
            "checkpoint_sha256": sha256(checkpoint_path),
        })

    report = {
        "schema_version": "candidate_interface_final_holdout_v1",
        "classification": "final_holdout_single_evaluation",
        "evaluation_mode": (
            "one-time frozen evaluation: deployment-contract checkpoints, "
            "trained on the 15-component lineage, evaluated on the sealed "
            "3-component GP2-family final test; calibration maps frozen "
            "before this run on the 6 external scaffolds"),
        "split_manifest_sha256": manifest.get("split_manifest_sha256"),
        "dataset_manifest_sha256": sha256(manifest_path),
        "test_lmdb_records_sha256": expected_sha,
        "sources": sources,
        "selections": selections,
        "final_test_evaluated": True,
        "rerun_policy": "this final test may not be evaluated again",
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")

    print("=== one-time final-holdout evaluation ===")
    for source, summary in zip(sources, selections, strict=True):
        print(f"\n{Path(source['path']).parts[-4]}")
        for name in ("pae", "plddt", "iptm"):
            metric = summary["metrics"][name]
            print(f"  {name}: median_scaffold_spearman={metric['median_scaffold_spearman']}"
                  f" pair_acc={metric['pairwise']['accuracy']}"
                  f" pairs={metric['pairwise']['pairs']}"
                  f" boot_lower={metric['pairwise']['bootstrap_95_lower']}"
                  f" var_ratio={metric['entity_prediction_to_target_variance_ratio']}"
                  f" gates={all(metric['gates'].values())}")
        print(f"  frozen-map PAE: {json.dumps(summary['pae_frozen_map_calibration'])}")


if __name__ == "__main__":
    main()
