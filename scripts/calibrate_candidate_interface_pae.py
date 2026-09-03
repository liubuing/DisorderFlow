#!/usr/bin/env python
"""Fit and evaluate a monotonic affine PAE scale correction on calibration scaffolds."""

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
from disorderflow.datasets.confidence_dataset import ConfidenceRegressionDataset  # noqa: E402
from disorderflow.utils.data import CompleteGroupBatchSampler, PaddingCollate  # noqa: E402
from disorderflow.utils.data import lmdb_records_sha256  # noqa: E402

CKPTS = [
    "logs/bfn_candidate_interface_multiscaffold_dual_sem_v2_2026_09_01__12_13_52_dual_sem_v2_s2041/checkpoints/best.pt",
    "logs/bfn_candidate_interface_multiscaffold_dual_sem_v2_2026_09_01__12_29_06_dual_sem_v2_s2053/checkpoints/best.pt",
    "logs/bfn_candidate_interface_multiscaffold_dual_sem_v2_2026_09_01__21_10_10_dual_sem_v2_s2069_retry/checkpoints/100.pt",
]
PAE_MAE_GATE = 0.10


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


def fit_affine(pred, target):
    design = np.vstack([np.ones_like(pred), pred]).T
    intercept, slope = np.linalg.lstsq(design, target, rcond=None)[0]
    return float(slope), float(intercept)


def fit_std_matching(pred, target):
    slope = float(np.std(target) / np.std(pred)) if np.std(pred) > 0 else 1.0
    intercept = float(np.mean(target) - slope * np.mean(pred))
    return slope, intercept


def affine_metrics(pred, target):
    slope, intercept = fit_affine(pred, target)
    corrected = slope * pred + intercept
    mae = float(np.mean(np.abs(corrected - target)))
    pred_std = float(np.std(corrected))
    target_std = float(np.std(target))
    variance_ratio = (pred_std / target_std) ** 2 if target_std > 0 else None
    corr = spearman(pred, target)
    return {
        "slope": slope,
        "intercept": intercept,
        "corrected_mae": mae,
        "corrected_variance_ratio": variance_ratio,
        "spearman": corr,
    }


def std_matching_metrics(pred, target):
    slope, intercept = fit_std_matching(pred, target)
    corrected = slope * pred + intercept
    mae = float(np.mean(np.abs(corrected - target)))
    pred_std = float(np.std(corrected))
    target_std = float(np.std(target))
    variance_ratio = (pred_std / target_std) ** 2 if target_std > 0 else None
    corr = spearman(pred, target)
    return {
        "slope": slope,
        "intercept": intercept,
        "corrected_mae": mae,
        "corrected_variance_ratio": variance_ratio,
        "spearman": corr,
    }


def scaffold_spearman(pred, target, scaffolds):
    values = {}
    for p, t, s in zip(pred, target, scaffolds, strict=True):
        values.setdefault(s, ([], []))
        values[s][0].append(p)
        values[s][1].append(t)
    return {
        scaffold: spearman(p, t)
        for scaffold, (p, t) in values.items()
        if len(p) >= 2
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", default=CKPTS)
    parser.add_argument(
        "--dataset",
        default="data/confidence_candidate_interface_multiscaffold_calibration_ext_dual_sem_v1/calibration.lmdb")
    parser.add_argument(
        "--dataset-manifest",
        default="data/confidence_candidate_interface_multiscaffold_calibration_ext_dual_sem_v1/manifest.json")
    parser.add_argument(
        "--output",
        default="results/candidate_interface_multiscaffold_calibration_ext/pae_scale_calibration.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    manifest_path = ROOT / args.dataset_manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_path = ROOT / args.dataset
    if lmdb_records_sha256(dataset_path) != manifest["summary"]["calibration"]["lmdb_records_sha256"]:
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
    for relative in args.checkpoints:
        checkpoint_path = ROOT / relative
        evaluation = evaluate_checkpoint(
            checkpoint_path, loader, args.device, expected_manifest_sha256=None)
        records = evaluation["entity_aggregated_records"]
        pred = np.asarray([r["pred_pae"] for r in records])
        target = np.asarray([r["target_pae"] for r in records])
        scaffolds = [r["scaffold_family"] for r in records]

        fitted = affine_metrics(pred, target)
        std_matched = std_matching_metrics(pred, target)
        raw_mae = float(np.mean(np.abs(pred - target)))
        raw_variance_ratio = (float(np.std(pred)) / float(np.std(target))) ** 2

        loso_slopes = []
        for held in sorted(set(scaffolds)):
            keep = [s != held for s in scaffolds]
            slope, intercept = fit_affine(pred[keep], target[keep])
            loso_slopes.append(slope)

        selections.append({
            "checkpoint": relative,
            "checkpoint_sha256": sha256(checkpoint_path),
            "n_entities": len(records),
            "raw_mae": raw_mae,
            "raw_variance_ratio": raw_variance_ratio,
            "raw_spearman": fitted["spearman"],
            "raw_median_scaffold_spearman": float(np.median([
                v for v in scaffold_spearman(pred, target, scaffolds).values()
                if v is not None])),
            "calibration": fitted,
            "std_matching_calibration": std_matched,
            "corrected_median_scaffold_spearman": float(np.median([
                v for v in scaffold_spearman(
                    fitted["slope"] * pred + fitted["intercept"], target, scaffolds).values()
                if v is not None])),
            "loso_slopes": {"min": float(min(loso_slopes)), "max": float(max(loso_slopes)),
                            "mean": float(np.mean(loso_slopes))},
            "gates_after_calibration": {
                "mae": fitted["corrected_mae"] <= PAE_MAE_GATE,
                "median_scaffold_spearman": (
                    float(np.median([v for v in scaffold_spearman(
                        fitted["slope"] * pred + fitted["intercept"],
                        target, scaffolds).values() if v is not None])) >= 0.5),
                "variance_ratio": (
                    fitted["corrected_variance_ratio"] is not None
                    and 0.5 <= fitted["corrected_variance_ratio"] <= 2.0),
            },
            "gates_after_std_matching": {
                "mae": std_matched["corrected_mae"] <= PAE_MAE_GATE,
                "median_scaffold_spearman": (
                    float(np.median([v for v in scaffold_spearman(
                        std_matched["slope"] * pred + std_matched["intercept"],
                        target, scaffolds).values() if v is not None])) >= 0.5),
                "variance_ratio": (
                    std_matched["corrected_variance_ratio"] is not None
                    and 0.5 <= std_matched["corrected_variance_ratio"] <= 2.0),
            },
        })

    report = {
        "schema_version": "candidate_interface_pae_scale_calibration_v1",
        "classification": "development_only",
        "calibration_scope": (
            "affine target~slope*pred+intercept fit on the 6 external calibration "
            "scaffolds; monotonic, ranking-invariant; not model training"),
        "dataset_manifest_sha256": sha256(manifest_path),
        "checkpoints": selections,
        "final_test_evaluated": False,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")

    print("=== PAE affine scale calibration ===")
    for s in selections:
        name = Path(s["checkpoint"]).parent.parent.name
        sm = s["std_matching_calibration"]
        print(f"\n{name}:")
        print(f"  raw MAE={s['raw_mae']:.4f} var_ratio={s['raw_variance_ratio']:.3f} "
              f"Spearman={s['raw_spearman']:.3f}")
        print(f"  LS fit: target = {s['calibration']['slope']:.3f}*pred + "
              f"{s['calibration']['intercept']:.4f} -> MAE={s['calibration']['corrected_mae']:.4f} "
              f"var_ratio={s['calibration']['corrected_variance_ratio']:.3f}")
        print(f"  std-match: target = {sm['slope']:.3f}*pred + {sm['intercept']:.4f} "
              f"-> MAE={sm['corrected_mae']:.4f} var_ratio={sm['corrected_variance_ratio']:.3f}")
        print(f"  LOSO slope range [{s['loso_slopes']['min']:.3f}, "
              f"{s['loso_slopes']['max']:.3f}]")
        print(f"  gates (LS): {s['gates_after_calibration']}")
        print(f"  gates (std-match): {s['gates_after_std_matching']}")


if __name__ == "__main__":
    main()
