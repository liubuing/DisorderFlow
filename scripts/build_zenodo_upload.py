"""Assemble the Zenodo upload bundle for the ecls_v1 release.

Copies the release artifacts that should accompany the code deposit into
release/zenodo_v1/upload/ and writes a checksummed UPLOAD_MANIFEST.json.
Run after the release manifests are (re)generated:

    python scripts/build/build_release_source_manifest.py --output <new-manifest>
    python scripts/build/build_release_artifact_bundle.py  # initial bundle only
    python scripts/build_zenodo_upload.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "release" / "zenodo_v1" / "upload"
LINE = "release/ecls_v1/publication_line.json"

INCLUDE = [
    "dist/disorderflow-ecls-v1-artifacts.zip",
    "release/ecls_v1/source_manifest.json",
    "release/ecls_v1/artifact_bundle_manifest.json",
    "release/ecls_v1/REPRODUCIBILITY.md",
    "release/ecls_v1/software_environment.yml",
    LINE,
    "README.md",
    "PUBLICATION_PROTOCOL.md",
    "docs/PUBLICATION_MAP.md",
    "publication/ECLS_SCOPE_FREEZE.yml",
    "publication/MANUSCRIPT_DRAFT.md",
    "results/publication/h3_ecls_temporal_final_v1/final_decision.json",
    "results/publication/h3_ecls_temporal_final_v1/results.json",
    "results/publication/h3_submission_package_v1/ECLS_GCLC_reviewer_package_v1.zip",
    "results/publication/h3_submission_package_v1/main_results.csv",
]


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def build(root=ROOT):
    root = Path(root).resolve()
    base = root / "release/zenodo_v1"
    out = base / "upload"
    line = json.loads((root / LINE).read_text(encoding="utf-8"))
    metadata = json.loads((base / "zenodo_metadata.json").read_text(encoding="utf-8"))
    title = line["manuscript_title"]
    if (line["primary_line"] != "ecls" or line["manuscript"] != "publication/MANUSCRIPT_DRAFT.md"
            or metadata["title"] != title + " (ECLS v1 reproducibility package)"):
        raise ValueError("Primary manuscript and deposit identity must agree on ECLS")
    manuscript_title = (root / line["manuscript"]).read_text(encoding="utf-8-sig").splitlines()[0].removeprefix("# ")
    if manuscript_title != title:
        raise ValueError("Manuscript title differs from publication_line.json")
    names = [Path(relative).name for relative in INCLUDE]
    if len(names) != len(set(names)):
        raise ValueError("Upload filenames collide")
    # Validate every source before changing staging. No research result is edited.
    for relative in INCLUDE:
        if not (root / relative).is_file():
            raise FileNotFoundError(relative)
    out.mkdir(parents=True, exist_ok=True)
    unexpected = [path for path in out.iterdir() if path.name not in names]
    if any(not path.is_file() or path.is_symlink() for path in unexpected):
        raise ValueError("Unexpected directory/symlink in upload; refusing automatic archive")
    if unexpected:
        archive = base / "archive" / ("mixed-upload-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        # Resolve and constrain both endpoints before any move on Windows.
        if not archive.resolve().is_relative_to(base.resolve()):
            raise ValueError("Archive escapes release directory")
        for path in unexpected:
            if not path.resolve().is_relative_to(out.resolve()):
                raise ValueError("Staging path escapes upload directory")
        archive.mkdir(parents=True, exist_ok=False)
        previous_manifest = base / "UPLOAD_MANIFEST.json"
        if previous_manifest.is_file():
            shutil.copy2(previous_manifest, archive / "UPLOAD_MANIFEST.json")
        for path in unexpected:
            shutil.move(str(path), str(archive / path.name))
    entries = []
    for relative in INCLUDE:
        source = root / relative
        target = out / source.name
        if target.is_symlink() or not target.resolve().is_relative_to(out.resolve()):
            raise ValueError("Upload target escapes staging directory")
        shutil.copy2(source, target)
        entries.append({
            "path": source.name,
            "source": relative,
            "bytes": target.stat().st_size,
            "sha256": digest(target),
        })
    manifest = {
        "schema_version": "zenodo_upload_v2",
        "primary_line": line["primary_line"],
        "release_id": line["release_id"],
        "stage": "local_preparation_not_published",
        "manuscript": line["manuscript"],
        "publication_line_sha256": digest(root / LINE),
        "metadata": "release/zenodo_v1/zenodo_metadata.json",
        "metadata_sha256": digest(base / "zenodo_metadata.json"),
        "upload_directory": "release/zenodo_v1/upload",
        "legacy_bundle_scope": "contact-v2 supporting provenance plus ECLS reviewer archive; not primary contact/binding evidence",
        "files": entries,
    }
    (base / "UPLOAD_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    for entry in entries:
        print(f"{entry['path']:55s} {entry['bytes']:>12,d}  {entry['sha256'][:16]}…")
    print(f"\n{len(entries)} ECLS files prepared locally in {out}")
    return manifest


def main():
    build()


if __name__ == "__main__":
    main()
