#!/usr/bin/env python
"""Build leakage-controlled explicit-state records from predeclared inputs."""

# 本脚本从"训练前已冻结"的显式状态清单构建防泄漏的 statecontrast v2 训练清单：
# 以 (谱系簇, 全局序列簇) 的稳定哈希为全局簇，split 与 CV 折均按全局簇的
# 哈希确定性分配（同一簇内所有记录共享划分与折，杜绝跨折泄漏）；
# 每条记录附带坐标文件并逐一校验 SHA-256，随后调用合同校验器检查
# teacher 覆盖率等约束。默认 80/10/10 划分、5 折、种子 5201。

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from disorderflow.statecontrast_v2_contract import validate_statecontrast_v2_records



# 生成稳定短 ID：多字段以 "|" 连接、小写归一后取 SHA-256 前 20 个十六进制位
def stable_id(*values):
    payload = "|".join(str(value).strip().casefold() for value in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


# 确定性划分：对 (种子, 全局簇) 取 SHA-256 前 16 位映射到 [0,1)，
# 依次落入 train/val/test 区间；同一簇内所有记录划分一致
def assign_split(global_cluster_id, split_seed, train_fraction, val_fraction):
    value = int(hashlib.sha256(
        f"{split_seed}|{global_cluster_id}".encode("ascii")).hexdigest()[:16], 16
    ) / float(16 ** 16)
    if value < train_fraction:
        return "train"
    if value < train_fraction + val_fraction:
        return "val"
    return "test"


# 确定性折分配：SHA-256 前 16 位对折数取模；同样以全局簇为单位
def assign_fold(global_cluster_id, split_seed, folds):
    digest = hashlib.sha256(
        f"fold|{split_seed}|{global_cluster_id}".encode("ascii")).hexdigest()
    return int(digest[:16], 16) % int(folds)


# 清单构建主流程：输入状态必须已冻结 → 逐成分逐状态展开记录 →
# 坐标文件存在性与哈希校验 → 合同校验（teacher 覆盖率等）→ 写出 JSON
def build(input_path, output_path, split_seed=5201, train_fraction=0.8,
          val_fraction=0.1, minimum_teacher_coverage=0.8, folds=5):
    source = json.loads(input_path.read_text(encoding="utf-8"))
    if source.get("status") != "explicit_states_frozen_before_training":
        raise ValueError("Input states must be frozen before training")
    records = []
    cluster_splits = {}
    for component in source["components"]:
        lineage = component["antibody_lineage_cluster"]
        sequence_cluster = component["global_sequence_cluster"]
        global_cluster = stable_id(lineage, sequence_cluster)
        split = cluster_splits.setdefault(
            global_cluster,
            assign_split(global_cluster, split_seed, train_fraction, val_fraction),
        )
        fold_id = assign_fold(global_cluster, split_seed, folds)
        group_id = stable_id(lineage, component["candidate_sequence"])
        for state in component["states"]:
            record = {
                "schema_version": "statecontrast_v2_explicit_states_v1",
                "record_id": stable_id(group_id, state["state_id"], state["pose_id"]),
                "group_id": group_id,
                "teacher_group_id": stable_id(
                    component.get("component_id", lineage),
                    component.get("substitution_bucket", "all")),
                "split": split,
                "fold_id": fold_id,
                "candidate_sequence": component["candidate_sequence"],
                "component_id": component["component_id"],
                "target": component["target"],
                "origin_arm": component.get("origin_arm"),
                "substitution_bucket": component.get("substitution_bucket"),
                "native_h3": component["native_h3"],
                "antibody_lineage_cluster": lineage,
                "global_sequence_cluster": sequence_cluster,
                "coordinate": {
                    "path": state["coordinate_path"],
                    "sha256": state["coordinate_sha256"],
                },
                "state": {
                    "type": state["state_type"],
                    "source": state["source"],
                    "source_panel": state.get("source_panel", "all"),
                    "prior_weight": float(state["prior_weight"]),
                    "admitted": bool(state.get("admitted", True)),
                    "state_id": state["state_id"],
                    "pose_id": state["pose_id"],
                    "antibody_chains": state["antibody_chains"],
                    "antigen_chains": state["antigen_chains"],
                    "antigen_sequence_override": state.get(
                        "antigen_sequence_override"),
                },
                "targets": {
                    "state_training_weight": float(
                        state.get("state_training_weight", 1.0)),
                    "independent_teacher_score": state.get(
                        "independent_teacher_score"),
                    "independent_teacher_required": bool(state.get(
                        "independent_teacher_required", True)),
                },
                "provenance": {
                    "source": component["dataset_source"],
                    "input_manifest": str(input_path),
                },
            }
            coordinate = ROOT / state["coordinate_path"]
            if not coordinate.is_file():
                raise FileNotFoundError(f"Missing state coordinate: {coordinate}")
            if hashlib.sha256(coordinate.read_bytes()).hexdigest() != state["coordinate_sha256"]:
                raise ValueError(f"State coordinate hash mismatch: {coordinate}")
            records.append(record)

    validation = validate_statecontrast_v2_records(
        records, minimum_teacher_coverage=minimum_teacher_coverage)
    if not validation["valid"]:
        raise ValueError("Invalid StateContrast-v2 manifest: " + "; ".join(
            validation["errors"][:10]))
    output = {
        "schema_version": "statecontrast_v2_explicit_states_v1",
        "status": "statecontrast_v2_manifest_ready",
        "split_seed": split_seed,
        "split_policy": "global_lineage_plus_sequence_cluster_hash",
        "cross_validation_folds": int(folds),
        "fold_policy": "global_lineage_plus_sequence_cluster_hash_modulo",
        "records": records,
        "validation": validation,
        "claim_boundary": "Exposed development training data only",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    return output


# 主入口：输入/输出路径必填，划分种子、teacher 覆盖率与折数可覆盖默认值
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=5201)
    parser.add_argument("--minimum-teacher-coverage", type=float, default=0.8)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    result = build(
        ROOT / args.input, ROOT / args.output, split_seed=args.split_seed,
        minimum_teacher_coverage=args.minimum_teacher_coverage, folds=args.folds,
    )
    print(json.dumps({
        "status": result["status"],
        "records": len(result["records"]),
        "groups": result["validation"]["group_count"],
        "teacher_coverage": result["validation"]["teacher_coverage"],
    }, indent=2))


if __name__ == "__main__":
    main()
