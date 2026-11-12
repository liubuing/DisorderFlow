#!/usr/bin/env python
"""Refresh the ECLS release manifests after tracked source files change.

Two manifests are kept in sync:
  release/ecls_v1/source_manifest.json  -- per-file bytes/sha256, honoring hash_mode
  release/zenodo_v1/UPLOAD_MANIFEST.json -- upload copies, source hashes and
                                            publication_line/metadata digests

The upload directory is local-only (gitignored); on a clean checkout the
manifest refresh still updates the source hashes, and the upload-copy sync is
skipped when the directory is absent.

Usage:
  python scripts/refresh_release_manifests.py [--check]

--check exits 1 if anything would change, without writing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "release/ecls_v1/source_manifest.json"
UPLOAD = ROOT / "release/zenodo_v1/UPLOAD_MANIFEST.json"


def digest(path: Path, mode: str = "raw") -> tuple[str, int]:
    content = path.read_bytes()
    if mode == "canonical_lf":
        content = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest(), len(content)


def refresh_source(check: bool) -> list[str]:
    doc = json.loads(SOURCE.read_text(encoding="utf-8"))
    changed = []
    for entry in doc["files"]:
        path = ROOT / entry["path"]
        if not path.is_file():
            continue
        sha, nbytes = digest(path, entry.get("hash_mode", "raw"))
        if sha != entry.get("sha256") or nbytes != entry.get("bytes"):
            changed.append(entry["path"])
            entry["sha256"] = sha
            entry["bytes"] = nbytes
    if changed and not check:
        SOURCE.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return changed


def refresh_upload(check: bool) -> list[str]:
    doc = json.loads(UPLOAD.read_text(encoding="utf-8"))
    upload_dir = ROOT / doc["upload_directory"]
    changed = []
    for entry in doc["files"]:
        source = ROOT / entry["source"]
        if not source.is_file():
            continue
        sha, nbytes = digest(source)
        if sha != entry.get("sha256") or nbytes != entry.get("bytes"):
            changed.append(entry["source"])
            entry["sha256"] = sha
            entry["bytes"] = nbytes
        if upload_dir.is_dir():
            target = upload_dir / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file() or target.read_bytes() != source.read_bytes():
                if not check:
                    shutil.copy2(source, target)
                changed.append(entry["path"])
    for key, path in (
            ("publication_line_sha256", ROOT / "release/ecls_v1/publication_line.json"),
            ("metadata_sha256", ROOT / "release/zenodo_v1/zenodo_metadata.json")):
        if path.is_file():
            sha, _ = digest(path)
            if doc.get(key) != sha:
                doc[key] = sha
                changed.append(path.name)
    if changed and not check:
        UPLOAD.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return sorted(set(changed))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source_changed = refresh_source(args.check)
    upload_changed = refresh_upload(args.check)
    if args.check:
        if source_changed or upload_changed:
            print("stale:", *source_changed, *upload_changed, sep="\n  ")
            raise SystemExit(1)
        print("manifests current")
        return
    print(f"source_manifest updated: {len(source_changed)}")
    for path in source_changed:
        print(f"  {path}")
    print(f"upload_manifest updated: {len(upload_changed)}")
    for path in upload_changed:
        print(f"  {path}")


if __name__ == "__main__":
    main()
