#!/usr/bin/env python
"""Extract the viral-antigen candidate subset from a discovery artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def filter_viral(discovery_path: Path, output: Path) -> dict:
    discovery_path = Path(discovery_path)
    output = Path(output)
    # write-once：过滤产物绝不允许被覆盖重写，保证证据产物不可变性
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite filtered discovery: {output}")
    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    # 只接受元数据级（metadata_only）的发现产物——本脚本不得接触任何含
    # 模型分数/标签的数据，这是"新数据源仅做资格审查"边界的一部分
    if discovery.get("classification") != "metadata_only":
        raise ValueError("Discovery artifact is not metadata-only")
    viral_entries = [
        entry for entry in discovery["entries"]
        if entry.get("status") == "candidate" and entry.get("viral_antigen") is True
    ]
    artifact = {
        "schema_version": discovery.get("schema_version"),
        "classification": "metadata_only",
        "status": "write_once_viral_filtered_discovery",
        "claim_boundary": "metadata-only; viral-antigen candidate subset only",
        "source_discovery": str(discovery_path),
        # 源文件哈希入档：审计时可验证过滤输入未被替换
        "source_discovery_sha256": sha256_file(discovery_path),
        "snapshot_dir": discovery["snapshot_dir"],
        "snapshot_manifest_sha256": discovery["snapshot_manifest_sha256"],
        "eligibility": discovery.get("eligibility", {}),
        "counts": {
            "viral_antigen_candidates": len(viral_entries),
        },
        "entries": viral_entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # 'x' 模式打开：文件已存在时直接抛异常，与 write-once 约定双保险
    with output.open("x", encoding="ascii", newline="\n") as handle:
        json.dump(artifact, handle, indent=2)
        handle.write("\n")
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(filter_viral(args.discovery, args.output)["counts"], indent=2))


if __name__ == "__main__":
    main()
