#!/usr/bin/env python
"""Verify the extracted DisorderFlow iBEC evidence package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(root, manifest_path):
    rows = json.loads(manifest_path.read_text(encoding="ascii"))
    failures = []
    for row in rows:
        path = root / row["path"]
        if not path.is_file():
            failures.append(f"missing: {row['path']}")
            continue
        if path.stat().st_size != row["bytes"]:
            failures.append(f"size mismatch: {row['path']}")
        if sha256(path) != row["sha256"]:
            failures.append(f"hash mismatch: {row['path']}")
    return failures


def main():
    root = Path(__file__).resolve().parent
    failures = verify_manifest(root, root / "PACKAGE_MANIFEST.json")
    supplement = root / "Supplementary_Materials"
    failures.extend(verify_manifest(supplement, supplement / "MANIFEST.json"))
    result = {
        "status": "package_integrity_pass" if not failures else "package_integrity_fail",
        "failures": failures,
    }
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
