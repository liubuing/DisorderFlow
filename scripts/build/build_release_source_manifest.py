#!/usr/bin/env python
"""Generate the checksummed source and lightweight-evidence release manifest."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path, PurePath

import yaml

ROOT = Path(__file__).resolve().parents[2]
TEXT_SUFFIXES = {
    ".csv", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml",
}


def digest(path, hash_mode="raw"):
    if hash_mode == "canonical_lf":
        content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        return hashlib.sha256(content).hexdigest()
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def matches_any(path, patterns):
    value = PurePath(path.as_posix())
    return any(value.match(pattern) for pattern in patterns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec", type=Path, default=Path("release/ecls_v1/source_release_spec.yml"))
    parser.add_argument(
        "--output", type=Path, default=Path("release/ecls_v1/source_manifest.json"))
    args = parser.parse_args()
    spec_path, output = ROOT / args.spec, ROOT / args.output
    if output.exists():
        raise FileExistsError(output)
    spec = yaml.safe_load(spec_path.read_text(encoding="ascii"))
    files = set()
    for pattern in spec["include"]:
        files.update(path for path in ROOT.glob(pattern) if path.is_file())
    relative_files = sorted(
        (path.relative_to(ROOT) for path in files
         if not matches_any(path.relative_to(ROOT), spec.get("exclude", []))),
        key=lambda path: path.as_posix())
    tracked = set(subprocess.check_output(
        ["git", "ls-files", "--cached"], cwd=ROOT, text=True).splitlines())
    untracked = [value.as_posix() for value in relative_files
                 if value.as_posix() not in tracked]
    if untracked:
        raise RuntimeError(
            f"Source release contains files absent from the Git index: {untracked}")
    payload = {
        "schema_version": 1,
        "release_id": spec["release_id"],
        "status": "source_manifest_complete",
        "classification": spec["classification"],
        "spec": {"path": args.spec.as_posix(), "sha256": digest(spec_path)},
        "files": [{
            "path": value.as_posix(),
            "bytes": (ROOT / value).stat().st_size,
            "hash_mode": (
                "canonical_lf" if value.suffix.lower() in TEXT_SUFFIXES else "raw"),
            "sha256": digest(
                ROOT / value,
                "canonical_lf" if value.suffix.lower() in TEXT_SUFFIXES else "raw"),
        } for value in relative_files],
    }
    output.write_bytes((json.dumps(payload, indent=2) + "\n").encode("ascii"))
    print(json.dumps({"release_id": payload["release_id"], "files": len(relative_files)}, indent=2))


if __name__ == "__main__":
    main()
