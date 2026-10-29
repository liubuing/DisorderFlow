#!/usr/bin/env python
"""Select frozen v2 pre-AF2 candidates using seed coverage and H3 diversity only."""

# 多骨架确认实验 v2 的 AF2 前候选筛选：只依据"seed 覆盖度 + H3 序列多样性"
# 这两类与结构无关的准则，从原始生成尝试中为每个 (component, arm) 组合
# 挑选固定数量的候选槽位。刻意不使用任何结构/打分信息，避免引入
# AF2 之前的选择偏差；不足时槽位记为 failed 并注明原因。
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    # 文件 sha256：冻结输入完整性校验。
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hamming(first, second):
    # 两条 H3 序列的汉明距离；仅支持等长比较（本实验中 H3 长度固定）。
    if len(first) != len(second):
        raise ValueError("Cannot compare H3 sequences with different lengths")
    return sum(a != b for a, b in zip(first, second, strict=True))


def combination_score(rows):
    # 组合打分的字典序比较键（元组越大越优）：
    # 1) 覆盖的 seed 数越多越好；
    # 2) 最小两两汉明距离越大越好（避免选到近似重复序列）；
    # 3) 总两两汉明距离越大越好（整体多样性）；
    # 4) sample_index 之和取负：优先选择更早采样的候选（确定性偏好）。
    distances = [
        hamming(first["sequence"], second["sequence"])
        for first, second in combinations(rows, 2)]
    return (
        len({row["seed"] for row in rows}),
        min(distances) if distances else 0,
        sum(distances),
        -sum(row["sample_index"] for row in rows),
    )


def choose_rows(rows, target):
    # 穷举所有 size=target 的组合（唯一序列数不足时按唯一数截断），
    # 挑选打分最优的组合；完全平局时按 attempt_id 字典序取最小者，
    # 保证筛选过程完全确定、可复现。
    unique_count = len({row["sequence"] for row in rows})
    size = min(target, unique_count)
    best_score = None
    best_ids = None
    best = None
    for candidate_rows in combinations(rows, size):
        if len({row["sequence"] for row in candidate_rows}) != size:
            continue
        score = combination_score(candidate_rows)
        attempt_ids = tuple(sorted(row["attempt_id"] for row in candidate_rows))
        if best_score is None or score > best_score or (
            score == best_score and attempt_ids < best_ids
        ):
            best_score = score
            best_ids = attempt_ids
            best = candidate_rows
    return list(best or []), best_score, unique_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/benchmarks/multiscaffold_confirmatory_v2_selection.yml")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    # 三个冻结输入（生成结果、完整性审计、生成配置）必须全部通过哈希校验。
    raw_path = ROOT / config["raw_generation"]
    audit_path = ROOT / config["raw_generation_audit"]
    generation_config_path = ROOT / config["generation_config"]
    for path, expected in [
        (raw_path, config["raw_generation_sha256"]),
        (audit_path, config["raw_generation_audit_sha256"]),
        (generation_config_path, config["generation_config_sha256"]),
    ]:
        if sha256(path) != expected:
            raise ValueError(f"Frozen input hash mismatch: {path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["status"] != "raw_generation_integrity_passed":
        raise ValueError("Raw generation integrity audit did not pass")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    target = int(config["selection"]["slots_per_arm_component"])
    # seed_order 固定 seed 的排序依据，使最终选择的槽位顺序确定。
    seed_order = {seed: index for index, seed in enumerate(raw["requested"]["seeds"])}
    grouped = defaultdict(list)
    for row in raw["attempts"]:
        if row["status"] == "success":
            grouped[(row["component_id"], row["arm"])].append(row)

    selections = []
    group_audit = []
    for component in raw["requested"]["components"]:
        for arm in raw["requested"]["arms"]:
            # 每个 (component, arm) 独立挑选 target 个槽位；
            # 候选不足时剩余槽位显式记为 failed（而非静默缩减）。
            rows = grouped[(component, arm)]
            chosen, objective, unique_count = choose_rows(rows, target)
            chosen.sort(key=lambda row: (
                seed_order[row["seed"]], row["sample_index"], row["attempt_id"]))
            for slot in range(target):
                selection_id = f"{component}|{arm}|selection_{slot + 1:02d}"
                if slot >= len(chosen):
                    selections.append({
                        "selection_id": selection_id,
                        "component_id": component,
                        "arm": arm,
                        "selection_slot": slot + 1,
                        "status": "failed",
                        "reason": "insufficient_unique_candidates",
                        "source_attempt_id": None,
                        "sequence": None,
                        "full_heavy_sequence": None,
                        "seed": None,
                        "sample_index": None,
                    })
                    continue
                row = chosen[slot]
                selections.append({
                    "selection_id": selection_id,
                    "component_id": component,
                    "arm": arm,
                    "selection_slot": slot + 1,
                    "status": "selected",
                    "reason": None,
                    "source_attempt_id": row["attempt_id"],
                    "sequence": row["sequence"],
                    "full_heavy_sequence": row["full_heavy_sequence"],
                    "seed": row["seed"],
                    "sample_index": row["sample_index"],
                })
            # 审计信息：记录原始尝试数、唯一序列数、选择目标函数各分量。
            group_audit.append({
                "component_id": component,
                "arm": arm,
                "raw_attempts": len(rows),
                "unique_h3_sequences": unique_count,
                "selected": len(chosen),
                "failed_slots": target - len(chosen),
                "objective": {
                    "distinct_seeds": objective[0] if objective else 0,
                    "minimum_pairwise_hamming": objective[1] if objective else None,
                    "total_pairwise_hamming": objective[2] if objective else None,
                    "negative_sum_sample_indices": objective[3] if objective else None,
                },
            })

    selected = [row for row in selections if row["status"] == "selected"]
    failed = [row for row in selections if row["status"] == "failed"]
    output = {
        "schema_version": 1,
        "status": "pre_af2_selection_complete",
        "config": args.config,
        "config_sha256": sha256(config_path),
        "raw_generation_sha256": sha256(raw_path),
        "summary": {
            "expected_slots": int(config["expected_slots"]),
            "recorded_slots": len(selections),
            "selected_candidates": len(selected),
            "failed_slots": len(failed),
            "selected_source_attempts_unique": len({
                row["source_attempt_id"] for row in selected}),
        },
        "selections": selections,
        "group_audit": group_audit,
        "claim_boundary": config["claim_boundary"],
    }
    # 槽位总数必须与冻结契约一致，否则筛选流程本身出了问题。
    if len(selections) != int(config["expected_slots"]):
        raise ValueError("Selection slot count differs from frozen contract")
    out_path = ROOT / (args.out or config["output"])
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
