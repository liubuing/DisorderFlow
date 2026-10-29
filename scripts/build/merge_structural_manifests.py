#!/usr/bin/env python
"""Merge multiple structural manifests into one consolidated candidate manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


# 合并时每条记录必须具备的最小字段集：序列、H3/CDR、抗原与接触计数、分辨率，
# 缺任一字段即整体失败，保证下游候选生成拿到的清单字段完备。
REQUIRED_FIELDS = {
    "instance", "pdb_id", "vh_sequence", "vl_sequence", "paired_cdr_sequence",
    "cdr_h3_sequence", "antigen_sequence", "n_contacting_h3_positions",
    "n_h3_antigen_residue_contacts", "resolution",
}


def sha256_file(path: Path) -> str:
    # 计算文件 SHA-256，登记每个来源清单的哈希以保证合并结果可溯源。
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge(manifests: list[Path], output: Path) -> dict:
    # 把多个结构清单合并为一份候选清单。
    # 参数:
    #   manifests: 来源清单 JSON 路径列表（至少一个）。
    #   output: 输出路径；已存在则拒绝写入（fail-closed，防止覆盖既有工件）。
    # 返回: 合并后的字典（含 sources 溯源、counts 与按 instance 排序的 records）。
    # 约束: instance 在所有清单间必须全局唯一，重复即抛错——下游以 instance
    # 为主键做哈希/去重，键冲突会静默污染候选池。合并仅为结构资格元数据，
    # 不引入任何模型或结果信息（见 claim_boundary）。
    manifests = [Path(path) for path in manifests]
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite merged manifest: {output}")
    if len(manifests) < 1:
        raise ValueError("At least one manifest is required")

    records: list[dict] = []
    sources = []
    seen_instances: set[str] = set()
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        sources.append({
            "path": str(path),
            "sha256": sha256_file(path),
            "records": len(payload["records"]),
        })
        for row in payload["records"]:
            missing = sorted(REQUIRED_FIELDS - row.keys())
            if missing:
                raise ValueError(f"{path} record {row.get('instance')} lacks {missing}")
            instance = str(row["instance"])
            if instance in seen_instances:
                raise ValueError(f"Duplicate instance across manifests: {instance}")
            seen_instances.add(instance)
            records.append(row)

    merged = {
        "schema_version": "candidate_interface_extension_merged_manifest_v1",
        "status": "merged_extension_structural_manifest",
        "classification": "model_free_structural_eligibility",
        "claim_boundary": "sequence and structure metadata only; no model or result imports",
        "sources": sources,
        "counts": {
            "source_manifests": len(manifests),
            "total_records": len(records),
        },
        "records": sorted(records, key=lambda row: str(row["instance"])),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="ascii", newline="\n") as handle:
        json.dump(merged, handle, indent=2)
        handle.write("\n")
    return merged


def main():
    # 命令行入口：--manifests 传入一个或多个清单，打印合并计数摘要。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.manifests, args.output)["counts"], indent=2))


if __name__ == "__main__":
    main()
