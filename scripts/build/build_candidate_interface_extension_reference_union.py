#!/usr/bin/env python
"""Build the calibration-extension reference union from the frozen successor-v3 union plus round-1 exposures."""

# 本脚本构建"校准扩展"参考并集：在冻结的 successor-v3 确认性参考并集之上，
# 并入第一轮（round-1）暴露条目，形成更大的排除参考库。
# 五轴序列字段（VH/VL/成对 CDR/CDR-H3/抗原）必须全部非空；round-1 快照的
# 全部 PDB 条目进入 exact 排除集，结构合格序列进入五轴排除集。
# 输出拒绝覆盖已有文件，全部输入记录 SHA-256 溯源。

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
UNION_STATUS = "frozen_successor_v3_confirmatory_reference_union"
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
AXES_FIELDS = ("vh_sequence", "vl_sequence", "paired_cdr_sequence",
               "cdr_h3_sequence", "antigen_sequence")


def sha256_file(path: Path) -> str:
    # 计算文件 SHA-256，用于参考并集的输入溯源
    return hashlib.sha256(path.read_bytes()).hexdigest()


# PDB ID 归一化：小写、去 pdb_ 前缀、去除多余前导零（保留 4 位）、
# 截掉 "_" 或 "-" 之后的后缀，保证不同来源的同一 entry 可以对齐
def normalize_pdb(value) -> str:
    text = str(value or "").casefold().strip()
    if text.startswith("pdb_"):
        text = text[4:]
    if len(text) > 4 and text.startswith("0"):
        text = text.lstrip("0")
    if len(text) > 4:
        text = text.split("_", 1)[0].split("-", 1)[0]
    return text


# 序列清洗：转大写并仅保留 20 种标准氨基酸字符
def clean_sequence(value) -> str:
    return "".join(char for char in str(value).upper() if char in AA)


# 读取 round-1 RCSB 搜索快照的 entry ID 列表；缺失 result_set 或
# 出现重复 ID 均视为快照不合规
def snapshot_entry_ids(snapshot_dir: Path) -> list[str]:
    response = json.loads((snapshot_dir / "search_response.json").read_text(encoding="ascii"))
    result_set = response.get("result_set")
    if not isinstance(result_set, list):
        raise ValueError("Round-1 snapshot search response lacks result_set")
    ids = [
        normalize_pdb(item["identifier"] if isinstance(item, dict) else item)
        for item in result_set
    ]
    if len(ids) != len(set(ids)):
        raise ValueError("Round-1 snapshot contains duplicate entry IDs")
    return sorted(ids)


# 从结构清单生成扩展参考记录：按顺序分配 EXTR##### 编号，
# 五轴序列字段任一清洗后为空即报错，PDB ID 统一归一化
def reference_records(structural_manifest: Path) -> list[dict[str, str]]:
    manifest = json.loads(structural_manifest.read_text(encoding="utf-8"))
    records = []
    seen: set[str] = set()
    for row in manifest["records"]:
        reference_id = f"EXTR{len(records) + 1:05d}"
        for field in AXES_FIELDS:
            if not clean_sequence(row[field]):
                raise ValueError(f"{row['instance']} has empty {field}")
        if reference_id in seen:
            raise ValueError(f"Duplicate reference id {reference_id}")
        seen.add(reference_id)
        records.append({
            "reference_id": reference_id,
            "id": reference_id,
            "pdb_id": normalize_pdb(row["pdb_id"]),
            "instance": row["instance"],
            **{field: clean_sequence(row[field]) for field in AXES_FIELDS},
        })
    return records


# 并集构建主流程：校验基础并集状态 → 合并参考记录（ID 冲突即报错）→
# 合并 exact 暴露 PDB 集 → 写出带完整溯源的新冻结并集（独占创建，不覆盖）
def build(base_manifest: Path, round1_snapshot: Path, round1_structural: Path,
          output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite reference union: {output}")
    base = json.loads(base_manifest.read_text(encoding="utf-8"))
    if base.get("status") != UNION_STATUS:
        raise ValueError("Base manifest is not a frozen successor-v3 reference union")

    added_records = reference_records(round1_structural)
    base_records = base["records"]
    base_ids = {row["reference_id"] for row in base_records}
    if base_ids & {row["reference_id"] for row in added_records}:
        raise ValueError("Extension reference ids collide with base union ids")

    added_pdb_ids = snapshot_entry_ids(round1_snapshot)
    exact_exposed = sorted(set(
        normalize_pdb(value) for value in base.get("exact_exposed_pdb_ids", [])) | set(added_pdb_ids))

    union = {
        "schema_version": 1,
        "status": UNION_STATUS,
        "classification": "reference_union",
        "created_by": "scripts/build/build_candidate_interface_extension_reference_union.py",
        "base_reference_union": {
            "path": base_manifest.resolve().as_posix(),
            "sha256": sha256_file(base_manifest),
        },
        "extension_round_inputs": {
            "round1_snapshot": {
                "path": round1_snapshot.resolve().as_posix(),
                "search_response_sha256": sha256_file(round1_snapshot / "search_response.json"),
                "entry_ids": len(added_pdb_ids),
            },
            "round1_structural_manifest": {
                "path": round1_structural.resolve().as_posix(),
                "sha256": sha256_file(round1_structural),
                "records": len(added_records),
            },
        },
        "exposure_policy": {
            "all_round1_snapshot_entry_ids": "exact_exclude",
            "all_round1_structurally_eligible_sequences": "five_axis_exclude",
        },
        "counts": {
            "base_reference_records": len(base_records),
            "added_extension_records": len(added_records),
            "deduplicated_reference_records": len(base_records) + len(added_records),
            "base_exact_exposed_pdb_ids": len(base.get("exact_exposed_pdb_ids", [])),
            "round1_snapshot_pdb_ids": len(added_pdb_ids),
            "exact_exposed_pdb_ids": len(exact_exposed),
        },
        "exact_exposed_pdb_ids": exact_exposed,
        "records": base_records + added_records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="ascii", newline="\n") as handle:
        json.dump(union, handle)
        handle.write("\n")
    return union


# 主入口：四个路径均为必填参数，完成后打印并集计数摘要
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--round1-snapshot", type=Path, required=True)
    parser.add_argument("--round1-structural", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    union = build(
        ROOT / args.base_manifest,
        ROOT / args.round1_snapshot,
        ROOT / args.round1_structural,
        ROOT / args.out,
    )
    print(json.dumps(union["counts"], indent=2))


if __name__ == "__main__":
    main()
