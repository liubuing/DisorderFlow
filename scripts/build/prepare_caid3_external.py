#!/usr/bin/env python
"""Download and freeze official CAID3 reference files before evaluation."""

# 本脚本在正式评估之前一次性下载并冻结 CAID3 官方参考文件
# （无序/binding/linker 等五套基准），同时把每个文件的 SHA-256、字节数与
# 记录数写入 manifest.json，声明"方法在下载前已冻结"以保证前瞻性。
# manifest 已存在时拒绝覆盖（--force 除外），防止参考集被悄悄替换。

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "https://caid.idpcentral.org/assets/sections/challenge/static/references/3"
REFERENCES = (
    "disorder_nox.fasta",
    "disorder_pdb.fasta",
    "binding.fasta",
    "binding_idr.fasta",
    "linker.fasta",
)


def sha256_bytes(data):
    # 计算字节串 SHA-256，记录参考文件的冻结指纹
    return hashlib.sha256(data).hexdigest()


def count_records(data):
    # 统计 FASTA 记录数（以 ">" 开头的行数）
    return sum(line.startswith(b">") for line in data.splitlines())


def main():
    # 主流程：逐个下载参考文件（校验响应确实以 FASTA header 开头）→
    # 写入冻结目录并记录哈希 → 生成 manifest（内含方法冻结声明与
    # 检查点/校准文件的预置哈希），manifest 一经写出不可覆盖。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "data/caid3/frozen_raw"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"Frozen manifest already exists: {manifest_path}")

    files = []
    for name in REFERENCES:
        url = f"{BASE_URL}/{name}"
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        if not data.startswith(b">"):
            raise ValueError(f"Unexpected CAID3 payload: {url}")
        path = out_dir / name
        path.write_bytes(data)
        files.append(
            {
                "name": name,
                "url": url,
                "sha256": sha256_bytes(data),
                "bytes": len(data),
                "records": count_records(data),
            }
        )

    manifest = {
        "schema_version": 1,
        "status": "frozen_raw_before_label_analysis",
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "method_frozen_before_download": True,
        "checkpoint_sha256": "ed9984504d799187b7660c1d40c3dc68c01a0ce9b3cf92e25af1273195931252",
        "calibration_sha256": "166cb02ae8eca4da4d1f041b9e2b0020457188a1a7b6e0141041213985d1da5a",
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
