#!/usr/bin/env python
"""Evaluate calibration-extension target pair evidence (no model required)."""

# 本脚本评估"候选界面多支架校准扩展"的靶标配对证据（不运行任何模型）：
# 从校准 LMDB 读取 AF2 教师指标（pLDDT/ipTM/界面 PAE），按构建体聚合重复，
# 在同一支架家族内做实体两两比较；仅当 |靶标差| 严格大于两实体 SEM 的
# 联合噪声（欧氏合成）时记为一个"可靠对"。门槛：可靠对 ≥30、贡献支架 ≥3、
# 单一支架占比 ≤0.50（防止证据被单一支架垄断）。输入哈希须与清单一致。

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from disorderflow.datasets.confidence_dataset import ConfidenceRegressionDataset
from disorderflow.utils.data import CompleteGroupBatchSampler, PaddingCollate
from disorderflow.utils.data import lmdb_records_sha256

OUTPUTS = ("plddt", "iptm", "pae")
# 可靠对证据门槛：最少对数、最少贡献支架数、单支架最大占比
MIN_RELIABLE_PAIRS = 30
MIN_PAIR_SCAFFOLDS = 3
MAX_SCAFFOLD_PAIR_FRACTION = 0.50


def sha256(path: Path) -> str:
    # 计算文件 SHA-256，用于清单溯源
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# 逐批次抽取靶标教师指标：候选区/抗原 PAE 区由掩码确定；
# 未归一化的 PAE 以 31.0 埃尺度归一（AFDB PAE 的常见上限）
def extract_targets(loader) -> list[dict]:
    records = []
    for batch in loader:
        candidate = batch["generate_flag"].bool() & batch["mask"].bool()
        pae_antigen = batch["pae_supervision_antigen_mask"].bool() & batch["mask"].bool()
        for index in range(batch["aa"].shape[0]):
            candidate_index = candidate[index]
            pae_antigen_index = pae_antigen[index]
            normalized = bool(batch["af2_pae_normalized"][index].item())
            target_pae = batch["af2_pae_matrix"][index]
            if not normalized:
                target_pae = target_pae / 31.0
            records.append({
                "construct_id": batch["construct_id"][index],
                "scaffold_family": batch["scaffold_family"][index],
                "target_plddt": float(
                    batch["af2_plddt"][index][candidate_index].mean()),
                "target_iptm": float(batch["af2_iptm"][index]),
                "target_pae": float(
                    target_pae[candidate_index][:, pae_antigen_index].mean()),
                "noise_plddt": float(batch["af2_candidate_plddt_sem"][index]),
                "noise_iptm": float(batch["af2_iptm_sem"][index]),
                "noise_pae": float(batch["af2_interface_pae_normalized_sem"][index]),
            })
    return records


# 同一构建体的重复测量取均值，得到实体级靶标值与噪声（SEM）
def aggregate_conditions(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in records:
        groups[row["construct_id"]].append(row)
    output = []
    for rows in groups.values():
        output.append({
            "construct_id": rows[0]["construct_id"],
            "scaffold_family": rows[0]["scaffold_family"],
            **{
                f"target_{name}": float(np.mean([r[f"target_{name}"] for r in rows]))
                for name in OUTPUTS
            },
            **{
                f"noise_{name}": float(np.mean([r[f"noise_{name}"] for r in rows]))
                for name in OUTPUTS
            },
        })
    return output


# 两两配对判据：|靶标差| 严格大于两侧噪声的欧氏合成才算"可分辨对"，
# 差异落在噪声内的对不计入（避免把噪声当信号）
def pair_outcomes(rows: list[dict], name: str) -> list[float]:
    outcomes = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            target_delta = rows[left][f"target_{name}"] - rows[right][f"target_{name}"]
            noise = float(np.hypot(
                rows[left][f"noise_{name}"], rows[right][f"noise_{name}"]))
            if abs(target_delta) <= noise:
                continue
            outcomes.append(1.0)
    return outcomes


# 按支架家族分组统计可靠对：总对数、贡献支架数、单支架最大占比
def pair_statistics(records: list[dict], name: str) -> dict:
    groups = defaultdict(list)
    for row in records:
        groups[row["scaffold_family"]].append(row)
    outcomes_by_scaffold = {
        scaffold: pair_outcomes(rows, name) for scaffold, rows in groups.items()
    }
    outcomes = [
        outcome for scaffold_outcomes in outcomes_by_scaffold.values()
        for outcome in scaffold_outcomes
    ]
    contributing = sum(bool(v) for v in outcomes_by_scaffold.values())
    max_fraction = (
        max(map(len, outcomes_by_scaffold.values()), default=0) / len(outcomes)
        if outcomes else None)
    return {
        "reliable_pairs": len(outcomes),
        "contributing_scaffolds": contributing,
        "max_scaffold_pair_fraction": max_fraction,
        "pairs_by_scaffold": {
            scaffold: len(values) for scaffold, values in outcomes_by_scaffold.items()
        },
    }


def main():
    # 主流程：校验 LMDB 记录哈希与清单一致 → 加载数据并抽取靶标 →
    # 聚合 → 对三个指标分别统计配对证据并逐项判门槛 → 写 JSON 报告
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="data/confidence_candidate_interface_multiscaffold_calibration_ext_dual_sem_v1/calibration.lmdb")
    parser.add_argument(
        "--dataset-manifest",
        default="data/confidence_candidate_interface_multiscaffold_calibration_ext_dual_sem_v1/manifest.json")
    parser.add_argument(
        "--output",
        default="results/candidate_interface_multiscaffold_calibration_ext/pair_evidence.json")
    args = parser.parse_args()

    manifest_path = ROOT / args.dataset_manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_path = ROOT / args.dataset
    expected_sha = manifest["summary"]["calibration"]["lmdb_records_sha256"]
    actual_sha = lmdb_records_sha256(dataset_path)
    if actual_sha != expected_sha:
        raise ValueError("Calibration LMDB does not match its manifest")

    dataset = ConfidenceRegressionDataset({
        "db_path": str(dataset_path),
        "candidate_interface_v1": True,
        "max_residues": 0,
    })
    sampler = CompleteGroupBatchSampler(dataset, 20, shuffle=False)
    loader = DataLoader(
        dataset, batch_sampler=sampler, collate_fn=PaddingCollate(), num_workers=0)

    records = extract_targets(loader)
    aggregated = aggregate_conditions(records)

    report = {
        "schema_version": "candidate_interface_calibration_extension_pair_evidence_v1",
        "classification": "development_only",
        "dataset_manifest_sha256": sha256(manifest_path),
        "reliable_pair_rule": {
            "comparison_unit": "entities_within_scaffold",
            "criterion": "absolute_target_delta_strictly_exceeds_joint_sem",
            "minimum_total_pairs": MIN_RELIABLE_PAIRS,
            "minimum_contributing_scaffolds": MIN_PAIR_SCAFFOLDS,
            "maximum_fraction_from_one_scaffold": MAX_SCAFFOLD_PAIR_FRACTION,
        },
        "counts": {
            "replicate_records": len(records),
            "entity_aggregated_records": len(aggregated),
            "scaffolds": len({r["scaffold_family"] for r in aggregated}),
        },
        "metrics": {},
    }
    gate_all = True
    for name in OUTPUTS:
        stats = pair_statistics(aggregated, name)
        gate = (
            stats["reliable_pairs"] >= MIN_RELIABLE_PAIRS
            and stats["contributing_scaffolds"] >= MIN_PAIR_SCAFFOLDS
            and stats["max_scaffold_pair_fraction"] is not None
            and stats["max_scaffold_pair_fraction"] <= MAX_SCAFFOLD_PAIR_FRACTION)
        report["metrics"][name] = {**stats, "gate_passed": gate}
        gate_all &= gate
    report["pair_evidence_gate_passed"] = gate_all

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
