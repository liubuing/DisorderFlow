#!/usr/bin/env python
"""Fetch or restore the immutable release artifact bundle and verify every file."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("release/ecls_v1/artifact_bundle_manifest.json"))
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / args.manifest).read_text(encoding="ascii"))
    if args.verify_only:
        failures = []
        for entry in manifest["files"]:
            path = ROOT / entry["path"]
            if not path.is_file() or path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
                failures.append(entry["path"])
        if failures:
            raise RuntimeError(f"Artifact verification failed: {failures}")
        print(f"artifact-files-valid count={len(manifest['files'])}")
        return
    temporary = None
    bundle = ROOT / args.bundle if args.bundle else None
    if bundle is None:
        url = manifest["distribution"].get("immutable_url")
        revision = manifest["distribution"].get("immutable_revision")
        if not url or not revision or manifest["status"] != "published":
            raise RuntimeError(
                "Artifact bundle is not remotely published; use --bundle with the local ZIP")
        temporary = Path(tempfile.mkdtemp(prefix="disorderflow_release_"))
        bundle = temporary / manifest["bundle"]["name"]
        urllib.request.urlretrieve(url, bundle)
    try:
        if bundle.stat().st_size != manifest["bundle"]["bytes"] or digest(bundle) != manifest["bundle"]["sha256"]:
            raise RuntimeError("Bundle size or SHA-256 mismatch")
        with zipfile.ZipFile(bundle) as archive:
            archive.extractall(ROOT)
        failures = [entry["path"] for entry in manifest["files"]
                    if digest(ROOT / entry["path"]) != entry["sha256"]]
        if failures:
            raise RuntimeError(f"Extracted artifact verification failed: {failures}")
        print(f"artifact-bundle-restored count={len(manifest['files'])}")
    finally:
        if temporary:
            shutil.rmtree(temporary)


if __name__ == "__main__":
    main()
