#!/usr/bin/env python
"""Download or verify the reviewed A-beta antibody reference structures."""
from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="configs/idp/abeta_reference_sources.yml")
    parser.add_argument("--output", default="data/anti_abeta_refs")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def sha256(path):
    # 分块读取计算 SHA-256（1 MiB 块），避免大文件一次性载入内存；大写十六进制
    # 与注册表中的记录格式保持一致
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main():
    args = parse_args()
    registry_path = PROJECT_ROOT / args.registry
    with open(registry_path, encoding="utf-8") as handle:
        registry = yaml.safe_load(handle)
    output = PROJECT_ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)

    failures = []
    for reference in registry["references"]:
        pdb = reference["pdb"].upper()
        path = output / f"{pdb}.pdb"
        # --verify-only 模式跳过下载，只校验已有文件；下载模板来自注册表
        # （版本固定的 RCSB URL），文件来源可追溯
        if not path.exists() and not args.verify_only:
            url = registry["download_template"].format(pdb=pdb)
            with urllib.request.urlopen(url, timeout=60) as response, open(path, "wb") as handle:
                handle.write(response.read())
        actual = sha256(path) if path.exists() else "missing"
        # 下载后必须与注册表中预录的 sha256 一致；任何不一致都会以非零退出码失败，
        # 防止被替换/截断的参考结构进入下游设计管线
        status = "pass" if actual == reference["sha256"] else "fail"
        print(f"{pdb}: {status} {actual}")
        if status == "fail":
            failures.append(pdb)
    if failures:
        raise SystemExit(f"Reference verification failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
