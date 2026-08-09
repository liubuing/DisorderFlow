#!/usr/bin/env python
"""Build a deterministic ZIP and checksummed external manifest for a release."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def expand_paths(values):
    files = []
    for value in values:
        path = ROOT / value
        if path.is_dir():
            files.extend(sorted(item for item in path.rglob("*") if item.is_file()))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(value)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec", type=Path,
        default=Path("release/ecls_v1/artifact_bundle_spec.yml"))
    parser.add_argument("--output-dir", type=Path, default=Path("dist"))
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("release/ecls_v1/artifact_bundle_manifest.json"))
    args = parser.parse_args()
    spec_path = ROOT / args.spec
    spec = yaml.safe_load(spec_path.read_text(encoding="ascii"))
    files = expand_paths(spec["files"])
    output = ROOT / args.output_dir / spec["bundle_name"]
    manifest_path = ROOT / args.manifest
    if output.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite release bundle or manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(relative, ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=6)
            entries.append({
                "path": relative, "bytes": path.stat().st_size,
                "sha256": digest(path),
            })
    payload = {
        "schema_version": 1,
        "status": spec["distribution"]["status"],
        "bundle": {
            "name": output.name, "bytes": output.stat().st_size,
            "sha256": digest(output),
        },
        "distribution": spec["distribution"],
        "spec": {"path": args.spec.as_posix(), "sha256": digest(spec_path)},
        "files": entries,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    print(json.dumps({
        "bundle": output.relative_to(ROOT).as_posix(),
        "bytes": output.stat().st_size, "sha256": payload["bundle"]["sha256"],
        "files": len(entries), "status": payload["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
