#!/usr/bin/env python
"""Discover calibration-extension candidates from a frozen RCSB metadata snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ANTIBODY_KEYWORDS = ("antibody", "fab", "nanobody", "vhh")
HEAVY_KEYWORDS = ("heavy chain", "heavy", "vh", "vhh", "nanobody", "heavy-chain")
LIGHT_KEYWORDS = ("light chain", "light", "vl", "kappa", "lambda", "light-chain")
VIRAL_PATTERNS = (
    "virus", "viral", "sars-cov", "influenza", "hiv", "ebola", "marburg",
    "rabies", "hepatitis", "herpes", "dengue", "zika", "norovirus",
    "rotavirus", "coronavirus", "spike", "glycoprotein g", "hemagglutinin",
    "neuraminidase", "capsid", "envelope protein", "lp", "lassa", "hantavirus",
    "alphavirus", "vesiculovirus", "measles", "mumps", "rubella", "poliovirus",
)
MIN_PEPTIDE_LENGTH = 5
MAX_PEPTIDE_LENGTH = 50
MAX_RESOLUTION = 4.0
EXPERIMENTAL_METHODS = ("X-RAY DIFFRACTION", "ELECTRON MICROSCOPY")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iso_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("Discovery timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def description_of(entity: dict[str, object]) -> str:
    value = (entity.get("rcsb_polymer_entity") or {}).get("pdbx_description") or ""
    return str(value).strip()


def length_of(entity: dict[str, object]) -> int | None:
    value = (entity.get("entity_poly") or {}).get("rcsb_sample_sequence_length")
    return int(value) if value is not None else None


def polymer_type_of(entity: dict[str, object]) -> str | None:
    value = (entity.get("entity_poly") or {}).get("rcsb_entity_polymer_type")
    return str(value) if value is not None else None


def is_viral_description(text: str) -> bool:
    lowered = text.casefold()
    return any(pattern in lowered for pattern in VIRAL_PATTERNS)


def classify_entity(entity: dict[str, object], title: str) -> str:
    description = description_of(entity).casefold()
    title_lower = title.casefold()
    if polymer_type_of(entity) not in {None, "Protein"}:
        return "nonprotein"
    has_antibody_evidence = any(keyword in description for keyword in ANTIBODY_KEYWORDS)
    if not has_antibody_evidence and not description:
        has_antibody_evidence = any(keyword in title_lower for keyword in ANTIBODY_KEYWORDS)
    if not has_antibody_evidence:
        return "unclassified"
    if any(keyword in description for keyword in LIGHT_KEYWORDS):
        if any(keyword in description for keyword in HEAVY_KEYWORDS):
            return "heavy"
        return "light"
    if any(keyword in description for keyword in HEAVY_KEYWORDS):
        return "heavy"
    length = length_of(entity)
    if length is not None and 100 <= length <= 140:
        return "heavy"
    if length is not None and 95 <= length <= 135:
        return "light"
    return "unclassified"


def entry_resolution(entry: dict[str, object]) -> float | None:
    values = (entry.get("rcsb_entry_info") or {}).get("resolution_combined")
    if not values:
        return None
    return float(values[0])


def discover(snapshot_dir: Path, output: Path,
             now=lambda: datetime.now(timezone.utc),
             include_viral: bool = False) -> dict[str, object]:
    snapshot_dir = Path(snapshot_dir)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite discovery artifact: {output}")

    manifest_path = snapshot_dir / "acquisition.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    if manifest.get("classification") != "metadata_only":
        raise ValueError("Snapshot is not classified as metadata_only")
    for name, recorded in manifest["files"].items():
        path = snapshot_dir / name
        if sha256_file(path) != recorded["sha256"]:
            raise ValueError(f"Snapshot file hash mismatch: {name}")
    entries = json.loads((snapshot_dir / "entries.json").read_text(encoding="ascii"))["entries"]

    results: list[dict[str, object]] = []
    counts = {
        "total_entries": len(entries),
        "antibody_peptide_candidates": 0,
        "deferred_viral_antigen": 0,
        "viral_antigen_candidates": 0,
        "excluded": {},
    }

    for entry in entries:
        title = str((entry.get("struct") or {}).get("title") or "")
        methods = {str(item.get("method")) for item in entry.get("exptl") or []}
        resolution = entry_resolution(entry)
        reasons: list[str] = []

        if not methods or not methods.issubset(set(EXPERIMENTAL_METHODS)):
            reasons.append("nonconforming_method")
        if resolution is None or resolution > MAX_RESOLUTION:
            reasons.append("resolution_above_threshold")

        roles = [classify_entity(entity, title) for entity in entry.get("polymer_entities") or []]
        heavies = sum(1 for role in roles if role == "heavy")
        lights = sum(1 for role in roles if role == "light")

        peptide_antigens = []
        protein_antigens = 0
        viral = False
        for entity, role in zip(entry.get("polymer_entities") or [], roles):
            if role not in {"unclassified", "nonprotein"}:
                continue
            if role == "nonprotein":
                continue
            length = length_of(entity)
            description = description_of(entity)
            if is_viral_description(f"{description} {title}"):
                viral = True
            if length is not None and MIN_PEPTIDE_LENGTH <= length <= MAX_PEPTIDE_LENGTH:
                peptide_antigens.append({
                    "entity_id": entity["rcsb_id"],
                    "description": description,
                    "length": length,
                })
            else:
                protein_antigens += 1

        if heavies == 0:
            reasons.append("no_heavy_chain")
        if lights > 0 and heavies != lights:
            reasons.append("unpaired_heavy_light")
        if heavies > 0 and lights == 0 and "nanobody" not in title.casefold():
            heavy_named = any(
                "nanobody" in description_of(e).casefold() or "vhhs" in description_of(e).casefold()
                for e in entry.get("polymer_entities") or [])
            if not heavy_named:
                reasons.append("single_chain_without_nanobody_evidence")
        if len(peptide_antigens) == 0:
            reasons.append("no_peptide_antigen_5_50")
        elif len(peptide_antigens) > 1:
            reasons.append("multiple_peptide_antigens")

        record = {
            "pdb_id": entry["rcsb_id"],
            "title": title,
            "release_date": (entry.get("rcsb_accession_info") or {}).get("initial_release_date"),
            "methods": sorted(methods),
            "resolution": resolution,
            "heavy_chains": heavies,
            "light_chains": lights,
            "peptide_antigens": peptide_antigens,
            "protein_antigen_entities": protein_antigens,
        }
        if reasons:
            record["status"] = "excluded"
            record["reasons"] = reasons
            for reason in reasons:
                counts["excluded"][reason] = counts["excluded"].get(reason, 0) + 1
        elif viral:
            if include_viral:
                record["status"] = "candidate"
                record["viral_antigen"] = True
                counts["viral_antigen_candidates"] += 1
            else:
                record["status"] = "deferred_viral_antigen"
                counts["deferred_viral_antigen"] += 1
        else:
            record["status"] = "candidate"
            record["viral_antigen"] = False
            counts["antibody_peptide_candidates"] += 1
        results.append(record)

    artifact = {
        "schema_version": "candidate_interface_extension_discovery_v1",
        "classification": "metadata_only",
        "status": "write_once_discovery_complete",
        "claim_boundary": (
            "metadata-only discovery; no coordinate access, homology claim, "
            "model inference, or target access; viral-antigen entries are "
            "retained with deferred analysis status unless the viral pool is "
            "explicitly enabled"
        ),
        "include_viral": include_viral,
        "discovered_at_utc": iso_utc(now()),
        "snapshot_dir": snapshot_dir.as_posix(),
        "snapshot_manifest_sha256": sha256_file(manifest_path),
        "eligibility": {
            "min_peptide_length": MIN_PEPTIDE_LENGTH,
            "max_peptide_length": MAX_PEPTIDE_LENGTH,
            "max_resolution_angstrom": MAX_RESOLUTION,
            "experimental_methods": list(EXPERIMENTAL_METHODS),
        },
        "counts": counts,
        "entries": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes((json.dumps(artifact, indent=2) + "\n").encode("ascii"))
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-viral", action="store_true",
        help="Promote deferred viral-antigen entries to candidates",
    )
    args = parser.parse_args()
    print(json.dumps(discover(args.snapshot_dir, args.output,
                             include_viral=args.include_viral), indent=2))


if __name__ == "__main__":
    main()
