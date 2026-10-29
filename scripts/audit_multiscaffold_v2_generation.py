#!/usr/bin/env python
"""Audit slot accounting and sequence invariants in frozen v2 raw generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
# 20 种标准氨基酸字母表，用于校验生成序列中不含非法字符。
AA = set("ACDEFGHIKLMNPQRSTVWY")


def sha256(path):
    # 计算文件 SHA-256；用于核对原始生成工件与冻结生成配置的一致性。
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    # 审计冻结 v2 原始生成的完整性与描述性多样性。
    # 检查项:
    #   - 槽位记账：attempt_id 应为 component|arm|seed|sample 的笛卡尔积，
    #     不得重复、缺失或多余。
    #   - 序列不变量：H3 序列长度须等于 H3 索引数且只含标准氨基酸；
    #     重链全长序列在非 H3 固定位置必须与代表序列完全一致（只允许 H3 区突变）。
    #   - 描述性统计：各臂唯一 H3 序列数、与原生 H3 相同的候选数。
    # 全部通过才写 passed 状态，否则以退出码 1 失败（CI 可感知）。
    # 注意：本审计只覆盖生成完整性与多样性，不涉及选择或性能端点。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw", default="results/multiscaffold_confirmatory_v2/raw_generation.json")
    parser.add_argument(
        "--config", default="configs/benchmarks/multiscaffold_confirmatory_v2_generation.yml")
    parser.add_argument(
        "--out", default="results/multiscaffold_confirmatory_v2/raw_generation_audit.json")
    args = parser.parse_args()

    raw_path = ROOT / args.raw
    config_path = ROOT / args.config
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    holdout = json.loads((ROOT / config["holdout_manifest"]).read_text(encoding="utf-8"))
    if raw["config_sha256"] != sha256(config_path):
        raise ValueError("Raw artifact does not match the frozen generation config")
    representatives = {
        component["component_id"]: component["representative"]
        for component in holdout["components"]}
    expected_ids = {
        f"{component}|{arm}|{seed}|{sample:02d}"
        for component in raw["requested"]["components"]
        for arm in raw["requested"]["arms"]
        for seed in raw["requested"]["seeds"]
        for sample in range(raw["requested"]["samples_per_seed"])}
    observed_ids = [row["attempt_id"] for row in raw["attempts"]]
    duplicate_ids = sorted(
        attempt_id for attempt_id, count in Counter(observed_ids).items() if count > 1)
    missing_ids = sorted(expected_ids - set(observed_ids))
    unexpected_ids = sorted(set(observed_ids) - expected_ids)

    invalid_sequences = []
    fixed_position_violations = []
    unique_by_arm = defaultdict(set)
    unique_by_arm_component = defaultdict(set)
    status_by_arm = Counter()
    native_by_arm = Counter()
    for row in raw["attempts"]:
        status_by_arm[(row["arm"], row["status"])] += 1
        if row["status"] != "success":
            continue
        representative = representatives[row["component_id"]]
        sequence = row["sequence"]
        full_heavy = row["full_heavy_sequence"]
        indices = set(representative["h3_heavy_indices_zero_based"])
        if len(sequence) != len(indices) or set(sequence) - AA:
            invalid_sequences.append(row["attempt_id"])
        if len(full_heavy) != len(representative["heavy_sequence"]) or any(
            full_heavy[index] != amino_acid
            for index, amino_acid in enumerate(representative["heavy_sequence"])
            if index not in indices
        ):
            fixed_position_violations.append(row["attempt_id"])
        unique_by_arm[row["arm"]].add(sequence)
        unique_by_arm_component[(row["arm"], row["component_id"])].add(sequence)
        if sequence == representative["cdr_h3_sequence"]:
            native_by_arm[row["arm"]] += 1

    passed = not any([
        duplicate_ids, missing_ids, unexpected_ids,
        invalid_sequences, fixed_position_violations,
    ]) and all(row["status"] == "success" for row in raw["attempts"])
    audit = {
        "schema_version": 1,
        "status": "raw_generation_integrity_passed" if passed else "raw_generation_integrity_failed",
        "raw_generation": args.raw,
        "raw_generation_sha256": sha256(raw_path),
        "config_sha256": sha256(config_path),
        "expected_attempts": len(expected_ids),
        "observed_attempts": len(observed_ids),
        "unique_attempt_ids": len(set(observed_ids)),
        "duplicate_attempt_ids": duplicate_ids,
        "missing_attempt_ids": missing_ids,
        "unexpected_attempt_ids": unexpected_ids,
        "invalid_sequences": invalid_sequences,
        "fixed_position_violations": fixed_position_violations,
        "status_by_arm": {
            arm: {
                status: status_by_arm[(arm, status)]
                for status in ("success", "failed")}
            for arm in raw["requested"]["arms"]},
        "unique_h3_sequences_by_arm": {
            arm: len(unique_by_arm[arm]) for arm in raw["requested"]["arms"]},
        "component_arm_unique_h3": {
            f"{component}|{arm}": len(unique_by_arm_component[(arm, component)])
            for component in raw["requested"]["components"]
            for arm in raw["requested"]["arms"]},
        "native_candidates_by_arm": dict(native_by_arm),
        "claim_boundary": "raw generation integrity and descriptive diversity only; no selection or performance endpoint",
    }
    out_path = ROOT / args.out
    out_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="ascii")
    print(json.dumps({
        "status": audit["status"],
        "raw_generation_sha256": audit["raw_generation_sha256"],
        "expected_attempts": audit["expected_attempts"],
        "observed_attempts": audit["observed_attempts"],
        "unique_h3_sequences_by_arm": audit["unique_h3_sequences_by_arm"],
    }, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
