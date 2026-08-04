#!/usr/bin/env python
"""Build a metadata-only strict split audit for the flexible-peptide H3 benchmark."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import pickle
import re
import tarfile
from pathlib import Path

import lmdb
import numpy as np
import yaml
from Bio.Align import PairwiseAligner


ROOT = Path(__file__).resolve().parents[2]
AXES = (
    "pdb_id",
    "official_ab_cluster",
    "official_cdrh3_cluster",
    "official_antigen_cluster",
    "vh_sequence_exact",
    "vl_sequence_exact",
    "cdr_h3_sequence_exact",
    "antigen_sequence_exact",
)


def clean_sequence(value):
    return "".join(char for char in str(value).upper() if char in "ACDEFGHIKLMNPQRSTVWY")


def cluster_tokens(value):
    """Normalize scalar, list-like, and slash-delimited official cluster fields."""
    if value is None:
        return ()
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "[]"}:
        return ()
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = text
    if isinstance(parsed, (list, tuple, set)):
        values = parsed
    else:
        values = re.split(r"[/,;|\s]+", str(parsed))
    return tuple(sorted({str(item).strip() for item in values if str(item).strip()}))


def read_lmdb(path):
    path = Path(path)
    ids = pickle.loads(Path(f"{path}-ids").read_bytes())
    env = lmdb.open(str(path), subdir=False, readonly=True, lock=False, readahead=False)
    records = []
    with env.begin() as transaction:
        for sample_id in ids:
            payload = transaction.get(str(sample_id).encode())
            if payload is None:
                raise RuntimeError(f"Missing LMDB key {sample_id} in {path}")
            records.append(pickle.loads(payload))
    env.close()
    return records


def read_official_rows(archive_path, member):
    with tarfile.open(archive_path, "r:gz") as archive:
        source = archive.extractfile(member)
        if source is None:
            raise FileNotFoundError(f"Missing {member} in {archive_path}")
        rows = list(csv.DictReader(io.TextIOWrapper(source, encoding="utf-8")))
    return {row["INSTANCE"]: row for row in rows}


def exact_alignment_map(official_sequence, parsed_sequence):
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -2.0
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -0.5
    aligner.end_insertion_score = 0.0
    aligner.end_deletion_score = 0.0
    alignment = aligner.align(official_sequence, parsed_sequence)[0]
    mapping = {}
    for official_block, parsed_block in zip(*alignment.aligned, strict=True):
        official_start, official_end = map(int, official_block)
        parsed_start, parsed_end = map(int, parsed_block)
        if official_end - official_start != parsed_end - parsed_start:
            continue
        for offset in range(official_end - official_start):
            official_index = official_start + offset
            parsed_index = parsed_start + offset
            if official_sequence[official_index] == parsed_sequence[parsed_index]:
                mapping[official_index] = parsed_index
    return mapping


def map_official_h3(record, row):
    official_vh = clean_sequence(
        row.get("VH_numerable_seq", "") or row.get("Hseq", ""))
    official_h3 = clean_sequence(row.get("CDRH3", ""))
    parsed_vh = clean_sequence(record.get("vh_sequence", ""))
    if not official_vh or not official_h3 or not parsed_vh:
        raise ValueError("missing_official_h3_sequence")
    if official_vh.count(official_h3) != 1:
        raise ValueError("official_h3_not_unique_in_vh")
    start = official_vh.index(official_h3)
    official_indices = list(range(start, start + len(official_h3)))
    mapping = exact_alignment_map(official_vh, parsed_vh)
    if any(index not in mapping for index in official_indices):
        raise ValueError("official_h3_not_fully_resolved")
    parsed_indices = [mapping[index] for index in official_indices]
    if parsed_indices != list(range(parsed_indices[0], parsed_indices[0] + len(parsed_indices))):
        raise ValueError("official_h3_mapping_not_contiguous")
    if "".join(parsed_vh[index] for index in parsed_indices) != official_h3:
        raise ValueError("official_h3_coordinate_sequence_mismatch")
    return official_h3, parsed_indices, {
        "official_vh_length": len(official_vh),
        "parsed_vh_length": len(parsed_vh),
        "official_h3_span_zero_based": [start, start + len(official_h3)],
        "parsed_h3_indices": parsed_indices,
    }


def h3_contact_summary(record, h3_indices, cutoff):
    heavy = record["heavy"]
    antigen = record["antigen"]
    contacting_positions = []
    residue_pairs = 0
    minimum_distance = None
    for h3_index in h3_indices:
        h_mask = np.asarray(heavy["mask_heavyatom"][h3_index], dtype=bool)
        h_coords = np.asarray(heavy["pos_heavyatom"][h3_index], dtype=float)[h_mask]
        if not len(h_coords):
            continue
        position_contact = False
        for antigen_index in range(len(antigen["aa"])):
            a_mask = np.asarray(antigen["mask_heavyatom"][antigen_index], dtype=bool)
            a_coords = np.asarray(antigen["pos_heavyatom"][antigen_index], dtype=float)[a_mask]
            if not len(a_coords):
                continue
            distance = float(np.linalg.norm(
                h_coords[:, None, :] - a_coords[None, :, :], axis=-1).min())
            minimum_distance = distance if minimum_distance is None else min(minimum_distance, distance)
            if distance <= cutoff:
                position_contact = True
                residue_pairs += 1
        if position_contact:
            contacting_positions.append(int(h3_index))
    return {
        "n_h3_positions": int(len(h3_indices)),
        "n_contacting_h3_positions": len(contacting_positions),
        "contacting_h3_structure_indices": contacting_positions,
        "n_h3_antigen_residue_contacts": residue_pairs,
        "minimum_h3_antigen_distance": (
            round(minimum_distance, 4) if minimum_distance is not None else None),
    }


def annotate_record(record, official, eligibility):
    instance = str(record.get("official_instance") or record["id"])
    row = official.get(instance)
    if row is None:
        raise KeyError(f"Official metadata is missing for {instance}")
    antigen_sequence = clean_sequence(record.get("antigen_sequence", ""))
    h3_mapping_error = None
    try:
        h3_sequence, h3_indices, h3_mapping = map_official_h3(record, row)
    except ValueError as error:
        h3_sequence = clean_sequence(row.get("CDRH3", ""))
        h3_indices = []
        h3_mapping = None
        h3_mapping_error = str(error)
    antigen_types = str(row.get("agtypes", "")).upper().split("/")
    contact = h3_contact_summary(
        record, h3_indices, float(eligibility["heavy_atom_contact_cutoff_angstrom"]))
    reasons = []
    if h3_mapping_error:
        reasons.append(h3_mapping_error)
    if not int(eligibility["minimum_antigen_length"]) <= len(antigen_sequence) <= int(
            eligibility["maximum_antigen_length"]):
        reasons.append("antigen_length")
    if not int(eligibility["minimum_h3_length"]) <= len(h3_sequence) <= int(
            eligibility["maximum_h3_length"]):
        reasons.append("h3_length")
    if eligibility.get("require_peptide_antigen_type") and "PEPTIDE" not in antigen_types:
        reasons.append("not_peptide_type")
    if eligibility.get("exclude_protein_antigen_type") and "PROTEIN" in antigen_types:
        reasons.append("contains_protein_type")
    if contact["n_contacting_h3_positions"] < int(
            eligibility["minimum_contacting_h3_positions"]):
        reasons.append("no_h3_peptide_contact")
    return {
        "id": str(record["id"]),
        "official_instance": instance,
        "eligible": not reasons,
        "exclusion_reasons": reasons,
        "antigen_types": antigen_types,
        "antigen_length": len(antigen_sequence),
        "h3_length": len(h3_sequence),
        "h3_coordinate_mapping": h3_mapping,
        **contact,
        "axis_values": {
            "pdb_id": (str(record.get("pdb_id", "")).casefold(),),
            "official_ab_cluster": cluster_tokens(row.get("ab_cluster")),
            "official_cdrh3_cluster": cluster_tokens(row.get("cdrh3_cluster")),
            "official_antigen_cluster": cluster_tokens(row.get("agclusters")),
            "vh_sequence_exact": (clean_sequence(
                row.get("VH_numerable_seq", "") or record.get("vh_sequence", "")),),
            "vl_sequence_exact": (clean_sequence(
                row.get("VL_numerable_seq", "") or record.get("vl_sequence", "")),),
            "cdr_h3_sequence_exact": (h3_sequence,),
            "antigen_sequence_exact": (antigen_sequence,),
        },
    }


def axis_index(records):
    index = {axis: set() for axis in AXES}
    for record in records:
        for axis in AXES:
            index[axis].update(value for value in record["axis_values"][axis] if value)
    return index


def exclude_conflicts(records, earlier_records):
    occupied = axis_index(earlier_records)
    retained = []
    excluded = []
    for record in records:
        conflicts = {
            axis: sorted(set(record["axis_values"][axis]) & occupied[axis])
            for axis in AXES
        }
        conflicts = {axis: values for axis, values in conflicts.items() if values}
        if conflicts:
            excluded.append({"id": record["id"], "conflicts": conflicts})
        else:
            retained.append(record)
    return retained, excluded


def sequence_homology(left, right, minimum_identity, minimum_coverage, aligner=None):
    """Return symmetric local-overlap homology statistics for two sequences."""
    left = clean_sequence(left)
    right = clean_sequence(right)
    if not left or not right:
        return {"match": False, "identity": 0.0, "coverage": 0.0, "aligned": 0}
    aligner = aligner or build_homology_aligner()
    alignment = aligner.align(left, right)[0]
    aligned = 0
    matches = 0
    for left_block, right_block in zip(*alignment.aligned, strict=True):
        left_start, left_end = map(int, left_block)
        right_start, right_end = map(int, right_block)
        block_length = min(left_end - left_start, right_end - right_start)
        aligned += block_length
        matches += sum(
            left[left_start + offset] == right[right_start + offset]
            for offset in range(block_length))
    identity = matches / aligned if aligned else 0.0
    coverage = aligned / min(len(left), len(right))
    return {
        "match": identity >= minimum_identity and coverage >= minimum_coverage,
        "identity": round(identity, 6),
        "coverage": round(coverage, 6),
        "aligned": aligned,
    }


def build_homology_aligner():
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = -1.0
    aligner.extend_gap_score = -0.1
    aligner.end_insertion_score = 0.0
    aligner.end_deletion_score = 0.0
    return aligner


def exclude_homology_conflicts(records, earlier_records, axes):
    """Exclude later records homologous to any earlier record on any sequence axis."""
    aligner = build_homology_aligner()
    earlier_by_axis = {}
    for axis, settings in axes.items():
        source_axis = settings["source_axis"]
        unique = {}
        for record in earlier_records:
            for sequence in record["axis_values"][source_axis]:
                if sequence:
                    unique.setdefault(sequence, record["id"])
        earlier_by_axis[axis] = unique
    cache = {}
    retained = []
    excluded = []
    for record in records:
        conflicts = {}
        for axis, settings in axes.items():
            source_axis = settings["source_axis"]
            candidate_values = record["axis_values"][source_axis]
            hit = None
            for candidate_sequence in candidate_values:
                for reference_sequence, reference_id in earlier_by_axis[axis].items():
                    key = (
                        axis,
                        candidate_sequence if candidate_sequence <= reference_sequence else reference_sequence,
                        reference_sequence if candidate_sequence <= reference_sequence else candidate_sequence,
                    )
                    if key not in cache:
                        cache[key] = sequence_homology(
                            candidate_sequence,
                            reference_sequence,
                            float(settings["minimum_identity"]),
                            float(settings["minimum_coverage"]),
                            aligner,
                        )
                    comparison = cache[key]
                    if comparison["match"]:
                        hit = {
                            "reference_id": reference_id,
                            "identity": comparison["identity"],
                            "coverage": comparison["coverage"],
                        }
                        break
                if hit:
                    break
            if hit:
                conflicts[axis] = hit
        if conflicts:
            excluded.append({"id": record["id"], "conflicts": conflicts})
        else:
            retained.append(record)
    return retained, excluded


def digest_ids(records):
    payload = "\n".join(sorted(record["id"] for record in records))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def serializable_record(record):
    return {
        **record,
        "axis_values": {
            axis: list(values) for axis, values in record["axis_values"].items()
        },
    }


def write_manifest(path, records):
    fields = [
        "id", "official_instance", "antigen_types", "antigen_length", "h3_length",
        "n_h3_positions", "n_contacting_h3_positions", "n_h3_antigen_residue_contacts",
        "minimum_h3_antigen_distance",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field) for field in fields}
            row["antigen_types"] = "/".join(record["antigen_types"])
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "benchmarks" / "peptide_h3_publication_split_v1.yml"))
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "data" / "peptide_h3_publication_split_v1"))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = config["source"]
    official = read_official_rows(
        ROOT / source["archive"], source["official_csv_member"])
    raw = {
        "adaptation": read_lmdb(ROOT / source["adaptation_lmdb"]),
        "development": read_lmdb(ROOT / source["development_lmdb"]),
        "sealed_final": read_lmdb(ROOT / source["sealed_final_lmdb"]),
    }
    annotated = {
        split: [annotate_record(record, official, config["eligibility"]) for record in records]
        for split, records in raw.items()
    }
    eligible = {
        split: [record for record in records if record["eligible"]]
        for split, records in annotated.items()
    }
    exact_retained_dev, excluded_dev = exclude_conflicts(
        eligible["development"], eligible["adaptation"])
    homology_axes = config.get("homology_isolation", {})
    if homology_axes:
        retained_dev, homology_excluded_dev = exclude_homology_conflicts(
            exact_retained_dev, eligible["adaptation"], homology_axes)
    else:
        retained_dev, homology_excluded_dev = exact_retained_dev, []
    exact_retained_final, excluded_final = exclude_conflicts(
        eligible["sealed_final"], [*eligible["adaptation"], *retained_dev])
    if homology_axes:
        retained_final, homology_excluded_final = exclude_homology_conflicts(
            exact_retained_final, [*eligible["adaptation"], *retained_dev], homology_axes)
    else:
        retained_final, homology_excluded_final = exact_retained_final, []
    retained = {
        "adaptation": eligible["adaptation"],
        "development": retained_dev,
        "sealed_final": retained_final,
    }
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {out_dir}")
    out_dir.mkdir(parents=True)
    for split, records in retained.items():
        write_manifest(out_dir / f"{split}_manifest.csv", records)
    audit = {
        "schema_version": 1,
        "status": "metadata_only_audit; no model forward; final not yet publication-frozen",
        "config": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "counts": {
            split: {
                "source": len(raw[split]),
                "eligible_before_isolation": len(eligible[split]),
                "retained": len(retained[split]),
            }
            for split in raw
        },
        "ids_sha256": {split: digest_ids(records) for split, records in retained.items()},
        "development_conflicts": excluded_dev,
        "final_conflicts": excluded_final,
        "development_homology_conflicts": homology_excluded_dev,
        "final_homology_conflicts": homology_excluded_final,
        "eligibility_exclusions": {
            split: [
                {"id": record["id"], "reasons": record["exclusion_reasons"]}
                for record in records if not record["eligible"]
            ]
            for split, records in annotated.items()
        },
        "records": {
            split: [serializable_record(record) for record in records]
            for split, records in retained.items()
        },
    }
    (out_dir / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": audit["status"],
        "counts": audit["counts"],
        "development_conflicts": len(excluded_dev),
        "final_conflicts": len(excluded_final),
        "development_homology_conflicts": len(homology_excluded_dev),
        "final_homology_conflicts": len(homology_excluded_final),
        "ids_sha256": audit["ids_sha256"],
        "out_dir": str(out_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
