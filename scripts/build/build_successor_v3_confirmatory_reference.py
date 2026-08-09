#!/usr/bin/env python
"""Build the frozen reference union for successor-v3 confirmatory isolation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.build.materialize_multiscaffold_candidates import (  # noqa: E402
    number_variable_domains,
)
from scripts.prepare_sabdab2_external import (  # noqa: E402
    clean_sequence,
    load_training_records,
)


DEFAULT_INPUTS = {
    "reference_cdr_annotations": Path(
        "data/multiscaffold_confirmatory_v2/reference_cdr_annotations.json"),
    "phase3_validation": Path("data/phase3_v5_1_pair_clustered/val.lmdb"),
    "v1_structural": Path(
        "data/multiscaffold_confirmatory_v1/structural_manifest.json"),
    "peptide_h3_v4_audit": Path("data/peptide_h3_publication_split_v4/audit.json"),
    "peptide_h3_temporal_audit": Path(
        "data/peptide_h3_temporal_split_v3/audit.json"),
    "v1_discovery": Path("data/multiscaffold_confirmatory_v1/discovery.json"),
    "v2_holdout": Path("data/multiscaffold_confirmatory_v2/holdout_manifest.json"),
    "successor_v3_development": Path("data/successor_v3_development/manifest.json"),
}
CDR_ORDER = ("H1", "H2", "H3", "L1", "L2", "L3")
PDB_IN_ID = re.compile(r"^pdb_0*([0-9][a-z0-9]{3})(?:[_-]|$)", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_sha256(values) -> str:
    normalized = sorted({str(value).casefold().strip() for value in values if str(value).strip()})
    return hashlib.sha256("\n".join(normalized).encode("ascii")).hexdigest()


def normalize_pdb(value) -> str:
    text = str(value or "").casefold().strip()
    match = PDB_IN_ID.match(text)
    if match:
        return match.group(1)
    if text.startswith("pdb_"):
        text = text[4:]
    text = text.lstrip("0") or "0"
    if re.fullmatch(r"[0-9][a-z0-9]{3}", text):
        return text
    raise ValueError(f"Cannot normalize PDB ID: {value!r}")


def pdb_from_record(record: dict) -> str:
    for field in ("pdb_id", "pdb_code", "id", "instance", "official_instance"):
        value = record.get(field)
        if value:
            try:
                return normalize_pdb(value)
            except ValueError:
                continue
    raise ValueError(f"Record has no normalizable PDB ID: {record.get('id', record)!r}")


def audit_exposed_pdbs(payload: dict) -> set[str]:
    result = set()
    for records in payload.get("records", {}).values():
        for record in records:
            for value in record.get("axis_values", {}).get("pdb_id", []):
                result.add(normalize_pdb(value))
    return result


def records_pdbs(records) -> set[str]:
    return {pdb_from_record(record) for record in records}


def normalize_reference(record: dict, source: str, identifier: str | None = None) -> dict:
    record_id = str(identifier or record.get("id") or record.get("instance")).strip()
    if not record_id:
        raise ValueError(f"Reference from {source} has no ID")
    cdrs = record.get("cdr_sequences", {})
    paired_cdr = record.get("paired_cdr_sequence") or "".join(
        str(cdrs.get(name, "")) for name in CDR_ORDER)
    return {
        "id": record_id,
        "pdb_id": pdb_from_record(record),
        "vh_sequence": clean_sequence(record.get("vh_sequence", "")),
        "vl_sequence": clean_sequence(record.get("vl_sequence", "")),
        "paired_cdr_sequence": clean_sequence(paired_cdr),
        "cdr_h3_sequence": clean_sequence(
            record.get("cdr_h3_sequence", "") or cdrs.get("H3", "")),
        "antigen_sequence": clean_sequence(record.get("antigen_sequence", "")),
        "sources": [source],
    }


def annotate_phase3(records: list[dict]) -> list[dict]:
    inputs = [{
        "instance": str(record["id"]),
        "heavy_sequence": clean_sequence(record.get("vh_sequence", "")),
        "light_sequence": clean_sequence(record.get("vl_sequence", "")),
    } for record in records if record.get("vh_sequence") and record.get("vl_sequence")]
    domains = number_variable_domains(inputs)
    annotated = []
    for record in records:
        item = domains.get(str(record["id"]), {})
        cdrs = {**item.get("H", {}).get("cdrs", {}),
                **item.get("L", {}).get("cdrs", {})}
        annotated.append({
            **record,
            "paired_cdr_sequence": "".join(cdrs.get(name, "") for name in CDR_ORDER),
        })
    return annotated


def deduplicate_records(records: list[dict]) -> list[dict]:
    """Deduplicate by normalized ID with deterministic source precedence."""
    selected = {}
    for record in sorted(records, key=lambda row: (
            row["id"].casefold(), json.dumps(row, sort_keys=True, separators=(",", ":")))):
        key = record["id"].casefold()
        if key not in selected:
            selected[key] = dict(record)
        else:
            selected[key]["sources"] = sorted(set(
                selected[key]["sources"] + record["sources"]))
    result = sorted(selected.values(), key=lambda row: (row["id"].casefold(), row["id"]))
    for index, record in enumerate(result, 1):
        record["reference_id"] = f"SV3R{index:05d}"
    return result


def write_json_once(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="ascii", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path,
        default=Path("data/successor_v3_confirmatory/reference_union_manifest.json"))
    args = parser.parse_args()
    out_path = ROOT / args.out
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen reference: {out_path}")
    paths = {name: ROOT / value for name, value in DEFAULT_INPUTS.items()}
    paths["phase3_validation_ids"] = Path(f"{paths['phase3_validation']}-ids")
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} input: {path}")

    cdr_payload = json.loads(paths["reference_cdr_annotations"].read_text(encoding="utf-8"))
    phase3_raw = load_training_records(paths["phase3_validation"])
    structural_payload = json.loads(paths["v1_structural"].read_text(encoding="utf-8"))
    if cdr_payload.get("n_records") != 4129 or len(cdr_payload.get("records", [])) != 4129:
        raise RuntimeError("Expected exactly 4,129 v2 annotated references")
    if len(phase3_raw) != 328:
        raise RuntimeError(f"Expected exactly 328 phase3 validation records, got {len(phase3_raw)}")
    if len(structural_payload.get("records", [])) != 91:
        raise RuntimeError("Expected exactly 91 v1 structural records")

    phase3_annotated = annotate_phase3(phase3_raw)
    source_records = {
        "v2_reference_cdr_annotations": [
            normalize_reference(row, "v2_reference_cdr_annotations")
            for row in cdr_payload["records"]],
        "phase3_validation": [
            normalize_reference(row, "phase3_validation") for row in phase3_annotated],
        "v1_structural": [
            normalize_reference(row, "v1_structural", row["instance"])
            for row in structural_payload["records"]],
    }
    references = deduplicate_records([
        row for source in source_records.values() for row in source])

    json_inputs = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
        if path.suffix == ".json"
    }
    exposure_sets = {
        "peptide_h3_v4_audit": audit_exposed_pdbs(json_inputs["peptide_h3_v4_audit"]),
        "peptide_h3_temporal_audit": audit_exposed_pdbs(
            json_inputs["peptide_h3_temporal_audit"]),
        "v1_discovery": (
            records_pdbs(json_inputs["v1_discovery"].get("candidates", []))
            | {normalize_pdb(value) for value in
               json_inputs["v1_discovery"].get("exclusion", {}).get("fixed_extra_pdbs", [])}),
        "v2_holdout": {
            normalize_pdb(member)
            for component in json_inputs["v2_holdout"].get("components", [])
            for member in component.get("members", [])
        },
        "successor_v3_development": records_pdbs(
            json_inputs["successor_v3_development"].get("records", [])),
    }
    exact_pdbs = set().union(*exposure_sets.values())
    for required_source in ("v2_holdout", "successor_v3_development"):
        missing = exposure_sets[required_source] - exact_pdbs
        if missing:
            raise RuntimeError(
                f"{required_source} PDBs absent from exact exposure set: {sorted(missing)}")

    source_set_hashes = {
        name: {
            "ids_sha256": set_sha256(row["id"] for row in rows),
            "pdb_ids_sha256": set_sha256(row["pdb_id"] for row in rows),
            "n_records": len(rows),
            "n_unique_ids": len({row["id"].casefold() for row in rows}),
            "n_unique_pdb_ids": len({row["pdb_id"] for row in rows}),
        } for name, rows in source_records.items()
    }
    source_set_hashes.update({
        name: {"pdb_ids_sha256": set_sha256(values), "n_unique_pdb_ids": len(values)}
        for name, values in exposure_sets.items()
    })
    payload = {
        "schema_version": 1,
        "status": "frozen_successor_v3_confirmatory_reference_union",
        "created_by": "scripts/build/build_successor_v3_confirmatory_reference.py",
        "numbering": {
            "tool": "ANARCII",
            "version": importlib.metadata.version("anarcii"),
            "scheme": "chothia",
        },
        "input_sha256": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in sorted(paths.values())
        },
        "source_set_hashes": source_set_hashes,
        "counts": {
            "input_records": sum(len(rows) for rows in source_records.values()),
            "deduplicated_reference_records": len(references),
            "exact_exposed_pdb_ids": len(exact_pdbs),
            "phase3_paired_cdr_annotated": sum(
                bool(row["paired_cdr_sequence"]) for row in source_records["phase3_validation"]),
        },
        "normalized_sets_sha256": {
            "reference_ids": set_sha256(row["id"] for row in references),
            "reference_pdb_ids": set_sha256(row["pdb_id"] for row in references),
            "exact_exposed_pdb_ids": set_sha256(exact_pdbs),
        },
        "exact_exposed_pdb_ids": sorted(exact_pdbs),
        "records": references,
    }
    write_json_once(out_path, payload)
    print(json.dumps(payload["counts"], indent=2))


if __name__ == "__main__":
    main()
