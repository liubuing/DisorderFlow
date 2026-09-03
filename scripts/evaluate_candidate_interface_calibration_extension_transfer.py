#!/usr/bin/env python
"""Transfer-evaluate v2 confidence checkpoints on independent calibration-extension scaffolds."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

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

CKPTS = [
    "logs/bfn_candidate_interface_multiscaffold_dual_sem_v2_2026_09_01__12_13_52_dual_sem_v2_s2041/checkpoints/best.pt",
    "logs/bfn_candidate_interface_multiscaffold_dual_sem_v2_2026_09_01__12_29_06_dual_sem_v2_s2053/checkpoints/best.pt",
    "logs/bfn_candidate_interface_multiscaffold_dual_sem_v2_2026_09_01__21_10_10_dual_sem_v2_s2069_retry/checkpoints/100.pt",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints", nargs="+", default=CKPTS,
        help="v2 checkpoint .pt paths evaluated in transfer mode")
    parser.add_argument(
        "--dataset",
        default="data/confidence_candidate_interface_multiscaffold_calibration_ext_dual_sem_v1/calibration.lmdb")
    parser.add_argument(
        "--dataset-manifest",
        default="data/confidence_candidate_interface_multiscaffold_calibration_ext_dual_sem_v1/manifest.json")
    parser.add_argument(
        "--output",
        default="results/candidate_interface_multiscaffold_calibration_ext/transfer_summary.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    manifest_path = ROOT / args.dataset_manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_path = ROOT / args.dataset
    expected_sha = manifest["summary"]["calibration"]["lmdb_records_sha256"]
    if lmdb_records_sha256(dataset_path) != expected_sha:
        raise ValueError("Calibration LMDB does not match its manifest")

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
        evaluation = evaluate_checkpoint(
            checkpoint_path, loader, args.device,
            expected_manifest_sha256=None)
        summary = summarize_selection(evaluation, bootstrap_seed=2041 + index)
        selections.append(summary)
        sources.append({
            "path": relative,
            "checkpoint_sha256": sha256(checkpoint_path),
        })

    report = {
        "schema_version": "candidate_interface_calibration_extension_transfer_v1",
        "classification": "development_only",
        "evaluation_mode": (
            "deliberate transfer: v2 checkpoints trained on the original 15-scaffold "
            "train set, evaluated on 6 independent external calibration scaffolds; "
            "lineage manifest check intentionally skipped"),
        "dataset_manifest_sha256": sha256(manifest_path),
        "sources": sources,
        "selections": selections,
        "plddt_pae_only_note": (
            "ipTM remains abstained (insufficient cross-scaffold reliable pairs); "
            "this summary reports pLDDT and PAE deployment gates only"),
        "final_test_evaluated": False,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")

    print("=== pLDDT / PAE transfer gates per seed ===")
    for source, summary in zip(sources, selections, strict=True):
        line = {"seed_ckpt": Path(source["path"]).parts[-4] + "/" + Path(source["path"]).name}
        for name in ("plddt", "pae"):
            gates = summary["metrics"][name]["gates"]
            line[name] = {
                "mae": round(summary["metrics"][name]["mae"], 4),
                "median_spearman": summary["metrics"][name]["median_scaffold_spearman"],
                "entity_var_ratio": summary["metrics"][name]["entity_prediction_to_target_variance_ratio"],
                "pair_accuracy": summary["metrics"][name]["pairwise"]["accuracy"],
                "all_gates": gates,
            }
        print(json.dumps(line, indent=2))


if __name__ == "__main__":
    main()
