"""Deterministic, provenance-aware dataset factory for state-contrast design."""

from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import io
import json
import os
import pickle
import random
import tarfile
from collections import Counter
from pathlib import Path

import lmdb


SCHEMA_VERSION = "statecontrast_factory_v1"
CDR_NAMES = ("H1", "H2", "H3", "L1", "L2", "L3")
AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWYX")
EXTERNAL_EVALUATION_TOKENS = ("caid", "external_evaluation")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest().upper()


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def stable_id(*parts):
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def normalize_sequence(sequence):
    sequence = "".join(str(sequence or "").split()).upper()
    if not sequence or set(sequence) - AA_ALPHABET:
        return None
    return sequence


def _source_seed(seed, source_name, source_record_id):
    return seed ^ int(stable_id(source_name, source_record_id)[:8], 16)


def _changed_shuffle(sequence, rng):
    values = list(sequence)
    rng.shuffle(values)
    shuffled = "".join(values)
    if shuffled == sequence and len(set(sequence)) > 1:
        shuffled = sequence[1:] + sequence[:1]
    return shuffled if shuffled != sequence else None


def _single_mutation(sequence, rng):
    if not sequence:
        return None, None
    position = rng.randrange(len(sequence))
    alternatives = sorted(AA_ALPHABET - {"X", sequence[position]})
    replacement = alternatives[rng.randrange(len(alternatives))]
    mutated = sequence[:position] + replacement + sequence[position + 1:]
    return mutated, {
        "position_zero_based": position,
        "from": sequence[position],
        "to": replacement,
    }


def _map_cdr_spans(vh, vl, cdrs, require_all_six=False):
    if require_all_six and any(not cdrs.get(name) for name in CDR_NAMES):
        return None
    spans = {}
    for name, sequence in cdrs.items():
        chain_name = "vh" if name.startswith("H") else "vl"
        chain = vh if chain_name == "vh" else vl
        if not chain or chain.count(sequence) != 1:
            return None
        start = chain.index(sequence)
        spans[name] = {"chain": chain_name, "start": start, "end": start + len(sequence)}
    return spans


def make_observed_record(raw, source, split, schema_version=SCHEMA_VERSION):
    vh = normalize_sequence(raw.get("vh_sequence"))
    vl = normalize_sequence(raw.get("vl_sequence"))
    cdr_h3 = normalize_sequence(raw.get("cdr_h3_sequence"))
    antigen = normalize_sequence(raw.get("antigen_sequence"))
    if not vh or not cdr_h3 or not antigen:
        return None

    source_record_id = str(raw.get("pdb_id") or raw.get("id") or "").casefold()
    if not source_record_id:
        return None
    raw_cdrs = raw.get("cdrs") or {"H3": cdr_h3}
    cdrs = {name: normalize_sequence(raw_cdrs.get(name)) for name in CDR_NAMES}
    cdrs = {name: sequence for name, sequence in cdrs.items() if sequence}
    spans = _map_cdr_spans(vh, vl, cdrs, bool(source.get("require_all_six_cdrs", False)))
    if spans is None:
        return None

    group_id = stable_id(source["name"], source_record_id)
    record_id = stable_id(
        group_id, "observed_bound", vh, vl or "", canonical_json(cdrs), antigen)
    annotation = raw.get("cdr_annotation") or {
        "scheme": "legacy_source_annotation",
        "source": "phase3_cdr_h3_sequence",
    }
    return {
        "schema_version": schema_version,
        "record_id": record_id,
        "group_id": group_id,
        "split": split,
        "record_type": "antibody_antigen_pair",
        "antibody": {
            "vh": vh,
            "vl": vl,
            "cdr_h3": cdrs["H3"],
            "cdrs": cdrs,
            "cdr_spans": spans,
            "cdr_annotation": annotation,
        },
        "antigen": {"sequence": antigen, "state": "structure_observed_bound_state"},
        "targets": {
            "binding_label": 1,
            "contrastive_rank": 1,
            "training_weight": float(source.get("native_weight", 1.0)),
        },
        "evidence": {
            "tier": source.get("evidence_tier", "experimental_structure"),
            "label_origin": "observed_antibody_antigen_complex",
        },
        "provenance": {
            "source": source["name"],
            "source_record_id": source_record_id,
            "release": source["release"],
            "license": source["license"],
            "usage": source.get("usage", "training"),
            "official_split": raw.get("official_split"),
            "source_cluster_id": raw.get("source_cluster_id"),
        },
        "lineage": {"parent_record_id": None, "operation": "observed_bound"},
    }


def _replace_cdr(candidate, cdr_name, replacement):
    span = candidate["antibody"]["cdr_spans"][cdr_name]
    chain_name = span["chain"]
    chain = candidate["antibody"][chain_name]
    if len(replacement) != span["end"] - span["start"]:
        raise ValueError("CDR counterfactuals must preserve loop length")
    candidate["antibody"][chain_name] = (
        chain[:span["start"]] + replacement + chain[span["end"]:])
    candidate["antibody"]["cdrs"][cdr_name] = replacement
    if cdr_name == "H3":
        candidate["antibody"]["cdr_h3"] = replacement


def _finalize_counterfactual(candidate, parent, operation, weight, detail):
    candidate["record_id"] = stable_id(
        parent["record_id"], operation, canonical_json(candidate["antibody"]["cdrs"]),
        candidate["antigen"]["sequence"])
    candidate["targets"] = {
        "binding_label": None,
        "contrastive_rank": 0,
        "training_weight": weight,
    }
    candidate["evidence"] = {
        "tier": "synthetic_weak",
        "label_origin": "matched_counterfactual_not_experimental_nonbinder",
    }
    candidate["lineage"] = {
        "parent_record_id": parent["record_id"],
        "operation": operation,
        "operation_detail": detail,
    }
    return candidate


def generate_counterfactuals(record, operations, seed):
    """Generate matched weak counterfactuals without calling them true non-binders."""
    rng = random.Random(_source_seed(
        seed, record["provenance"]["source"], record["provenance"]["source_record_id"]))
    generated = []
    for operation in operations:
        if operation in ("cdr_single_mutation_each", "cdr_composition_shuffle_each"):
            for cdr_name in CDR_NAMES:
                native = record["antibody"]["cdrs"].get(cdr_name)
                if not native:
                    continue
                candidate = copy.deepcopy(record)
                if operation == "cdr_single_mutation_each":
                    sequence, mutation = _single_mutation(native, rng)
                    detail = {"cdr": cdr_name, **mutation}
                    weight = 0.35
                    lineage_operation = f"cdr_single_mutation:{cdr_name}"
                else:
                    sequence = _changed_shuffle(native, rng)
                    if sequence is None:
                        continue
                    detail = {"cdr": cdr_name}
                    weight = 0.15
                    lineage_operation = f"cdr_composition_shuffle:{cdr_name}"
                _replace_cdr(candidate, cdr_name, sequence)
                generated.append(_finalize_counterfactual(
                    candidate, record, lineage_operation, weight, detail))
            continue

        candidate = copy.deepcopy(record)
        detail = None
        if operation == "cdr_single_mutation":
            sequence, detail = _single_mutation(candidate["antibody"]["cdr_h3"], rng)
            _replace_cdr(candidate, "H3", sequence)
            weight = 0.35
        elif operation == "cdr_composition_shuffle":
            sequence = _changed_shuffle(candidate["antibody"]["cdr_h3"], rng)
            if sequence is None:
                continue
            _replace_cdr(candidate, "H3", sequence)
            weight = 0.15
        elif operation == "antigen_composition_shuffle":
            sequence = _changed_shuffle(candidate["antigen"]["sequence"], rng)
            if sequence is None:
                continue
            candidate["antigen"]["sequence"] = sequence
            candidate["antigen"]["state"] = "synthetic_composition_matched_decoy"
            weight = 0.20
        else:
            raise ValueError(f"Unsupported counterfactual operation: {operation}")
        generated.append(_finalize_counterfactual(
            candidate, record, operation, weight, detail))
    return generated


class DeterministicShardWriter:
    def __init__(self, output_dir, shard_size):
        self.output_dir = Path(output_dir)
        self.shard_size = int(shard_size)
        self.shards = []
        self.record_count = 0
        self.fingerprint = hashlib.sha256()
        self._raw = None
        self._gzip = None
        self._count = 0

    def _open(self):
        index = len(self.shards)
        path = self.output_dir / f"part-{index:05d}.jsonl.gz"
        self._raw = path.open("wb")
        self._gzip = gzip.GzipFile(filename="", mode="wb", fileobj=self._raw, mtime=0)
        self._count = 0

    def _close(self):
        if self._gzip is None:
            return
        path = Path(self._raw.name)
        self._gzip.close()
        self._raw.close()
        self.shards.append({
            "path": path.name,
            "records": self._count,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
        self._gzip = None
        self._raw = None

    def write_group(self, records):
        if self._gzip is None or (self._count and self._count + len(records) > self.shard_size):
            self._close()
            self._open()
        for record in records:
            line = (canonical_json(record) + "\n").encode("ascii")
            self._gzip.write(line)
            self.fingerprint.update(line)
            self._count += 1
            self.record_count += 1

    def close(self):
        self._close()


def _source_paths(root, source):
    lmdb_path = root / source["path"]
    ids_path = root / source.get("ids_path", source["path"] + "-ids")
    return lmdb_path, ids_path


def iter_phase3_source(root, source):
    lmdb_path, ids_path = _source_paths(root, source)
    with ids_path.open("rb") as handle:
        identifiers = pickle.load(handle)
    limit = source.get("max_records")
    if limit is not None:
        identifiers = identifiers[:int(limit)]
    env = lmdb.open(
        str(lmdb_path), readonly=True, lock=False, readahead=False, meminit=False, subdir=False)
    try:
        with env.begin() as txn:
            for identifier in identifiers:
                key = identifier if isinstance(identifier, bytes) else str(identifier).encode()
                payload = txn.get(key)
                if payload is not None:
                    yield pickle.loads(payload)
    finally:
        env.close()


def _clean_archive_sequence(value):
    return normalize_sequence("".join(
        char for char in str(value or "").upper() if char in AA_ALPHABET))


def _derived_cluster_split(cluster_id, validation_fraction, seed):
    value = int(stable_id(seed, cluster_id)[:15], 16) / float(16 ** 15)
    return "validation" if value < validation_fraction else "train"


def iter_sabdab2_abag_source(root, source):
    archive = root / source["path"]
    member_name = source.get("member", "splits_final/abag_split.csv")
    allowed_antigens = set(source.get("allowed_antigen_types", ["PROTEIN", "PEPTIDE"]))
    validation_fraction = float(source.get("validation_fraction", 0.1))
    limit = source.get("max_records")
    accepted = 0
    with tarfile.open(archive, "r:gz") as tar:
        stream = tar.extractfile(member_name)
        if stream is None:
            raise FileNotFoundError(f"SAbDab2 split member is missing: {member_name}")
        rows = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8"))
        for row in rows:
            if row.get("ab_ag_split") != "train" or row.get("holo") != "True":
                continue
            antigen_types = {
                item.strip().upper() for item in row.get("agtypes", "").split("/") if item.strip()
            }
            if not antigen_types or not antigen_types.issubset(allowed_antigens):
                continue
            cdrs = {
                name: _clean_archive_sequence(row.get(f"CDR{name}", ""))
                for name in CDR_NAMES
            }
            antigen = _clean_archive_sequence(row.get("agresolvedseqs", ""))
            vh = _clean_archive_sequence(row.get("VH_numerable_seq") or row.get("Hseq"))
            vl = _clean_archive_sequence(row.get("VL_numerable_seq") or row.get("Lseq"))
            if not antigen or not vh or not cdrs.get("H3"):
                continue
            cluster_id = row.get("ab_ag_cluster") or row.get("INSTANCE")
            split = _derived_cluster_split(
                cluster_id, validation_fraction, int(source.get("split_seed", 20260721)))
            yield {
                "pdb_id": row["INSTANCE"],
                "vh_sequence": vh,
                "vl_sequence": vl,
                "cdr_h3_sequence": cdrs["H3"],
                "cdrs": cdrs,
                "antigen_sequence": antigen,
                "official_split": "train",
                "source_cluster_id": cluster_id,
                "factory_split": split,
                "cdr_annotation": {
                    "scheme": "IMGT",
                    "source": "SAbDab2 official abag_split.csv CDR fields",
                    "numbering_backend": "official_precomputed",
                },
            }
            accepted += 1
            if limit is not None and accepted >= int(limit):
                return


def iter_factory_source(root, source):
    if source["type"] == "phase3_lmdb":
        for raw in iter_phase3_source(root, source):
            yield raw, source["split"]
    elif source["type"] == "sabdab2_abag_tar":
        for raw in iter_sabdab2_abag_source(root, source):
            yield raw, raw["factory_split"]
    else:
        raise ValueError(f"Unsupported source type: {source['type']}")


def source_manifest_entry(root, source):
    path = root / source["path"]
    entry = {
        "name": source["name"],
        "release": source["release"],
        "license": source["license"],
        "path": str(Path(source["path"]).as_posix()),
    }
    if source["type"] == "phase3_lmdb":
        lmdb_path, ids_path = _source_paths(root, source)
        entry.update({
            "split": source["split"],
            "lmdb_sha256": sha256_file(lmdb_path),
            "ids_sha256": sha256_file(ids_path),
        })
    else:
        entry.update({
            "archive_sha256": sha256_file(path),
            "member": source.get("member", "splits_final/abag_split.csv"),
            "accepted_official_split": "train",
            "rejected_official_split": "test",
            "derived_validation_by": "stable hash of official ab_ag_cluster",
        })
    return entry


def _validate_source(source):
    searchable = " ".join(str(source.get(key, "")) for key in ("name", "usage", "path")).lower()
    if any(token in searchable for token in EXTERNAL_EVALUATION_TOKENS):
        raise ValueError(f"External evaluation source is forbidden in training factory: {source['name']}")
    required = ("name", "type", "path", "release", "license")
    missing = [key for key in required if not source.get(key)]
    if missing:
        raise ValueError(f"Source {source.get('name', '<unnamed>')} lacks fields: {missing}")
    if source["type"] not in ("phase3_lmdb", "sabdab2_abag_tar"):
        raise ValueError(f"Unsupported source type: {source['type']}")
    if source["type"] == "phase3_lmdb" and not source.get("split"):
        raise ValueError(f"Phase3 source {source['name']} lacks split")


def build_dataset(config, root, output_dir):
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output_dir}")
    staging = output_dir.with_name(output_dir.name + ".building")
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists: {staging}")
    staging.mkdir(parents=True)

    operations = list(config["counterfactuals"]["operations"])
    schema_version = config.get("schema_version", SCHEMA_VERSION)
    writer = DeterministicShardWriter(staging, config.get("shard_size", 50000))
    counts = Counter()
    split_groups = {}
    skipped = 0
    duplicate_source_records_skipped = 0
    seen_source_records = set()
    source_manifest = []
    try:
        for source in config["sources"]:
            if not source.get("enabled", True):
                continue
            _validate_source(source)
            source_manifest.append(source_manifest_entry(root, source))
            for raw, split in iter_factory_source(root, source):
                observed = make_observed_record(
                    raw, source, split, schema_version=schema_version)
                if observed is None:
                    skipped += 1
                    continue
                source_key = (
                    observed["provenance"]["source"],
                    observed["provenance"]["source_record_id"],
                )
                if source_key in seen_source_records:
                    duplicate_source_records_skipped += 1
                    continue
                seen_source_records.add(source_key)
                previous_split = split_groups.setdefault(observed["group_id"], split)
                if previous_split != split:
                    raise ValueError(
                        f"Group leakage across splits for {observed['provenance']['source_record_id']}")
                group = [observed, *generate_counterfactuals(
                    observed, operations, int(config.get("seed", 2026)))]
                writer.write_group(group)
                for record in group:
                    counts[f"split:{record['split']}"] += 1
                    counts[f"tier:{record['evidence']['tier']}"] += 1
                    counts[f"operation:{record['lineage']['operation']}"] += 1
        writer.close()
        manifest = {
            "schema_version": schema_version,
            "dataset_name": config["dataset_name"],
            "records": writer.record_count,
            "groups": len(split_groups),
            "skipped_invalid_records": skipped,
            "duplicate_source_records_skipped": duplicate_source_records_skipped,
            "canonical_fingerprint": writer.fingerprint.hexdigest().upper(),
            "shard_size": int(config.get("shard_size", 50000)),
            "shards": writer.shards,
            "counts": dict(sorted(counts.items())),
            "sources": source_manifest,
            "counterfactual_policy": {
                "operations": operations,
                "synthetic_binding_labels_are_missing": True,
                "synthetic_records_are_not_experimental_nonbinders": True,
            },
            "external_evaluation_exclusions": ["CAID2"],
            "config_sha256": sha256_bytes(canonical_json(config).encode("ascii")),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        audit = verify_dataset(staging)
        (staging / "audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(staging, output_dir)
        return manifest
    except Exception:
        writer.close()
        raise


def verify_dataset(output_dir):
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    counts = Counter()
    observed_ids = set()
    record_ids = set()
    observed_per_group = Counter()
    cluster_splits = {}
    lineage_errors = []
    for shard in manifest["shards"]:
        path = output_dir / shard["path"]
        if sha256_file(path) != shard["sha256"]:
            raise ValueError(f"Shard checksum mismatch: {path}")
        shard_count = 0
        with gzip.open(path, "rt", encoding="ascii") as handle:
            for line in handle:
                digest.update(line.encode("ascii"))
                record = json.loads(line)
                if record["record_id"] in record_ids:
                    raise ValueError(f"Duplicate record_id: {record['record_id']}")
                record_ids.add(record["record_id"])
                if record.get("schema_version") != manifest["schema_version"]:
                    raise ValueError("Record schema mismatch")
                provenance = record["provenance"]
                searchable = f"{provenance['source']} {provenance.get('usage', '')}".lower()
                if any(token in searchable for token in EXTERNAL_EVALUATION_TOKENS):
                    raise ValueError("External evaluation record found in training shards")
                official_split = provenance.get("official_split")
                if official_split is not None and official_split != "train":
                    raise ValueError("Non-training official split found in training shards")
                cluster_id = provenance.get("source_cluster_id")
                if cluster_id is not None:
                    cluster_key = (provenance["source"], str(cluster_id))
                    previous = cluster_splits.setdefault(cluster_key, record["split"])
                    if previous != record["split"]:
                        raise ValueError("Official source cluster crosses derived splits")
                if manifest["schema_version"].startswith("statecontrast_factory_v2"):
                    antibody = record["antibody"]
                    if set(antibody.get("cdrs", {})) != set(CDR_NAMES):
                        raise ValueError("v2 record does not contain all six CDRs")
                    if antibody.get("cdr_annotation", {}).get("scheme") != "IMGT":
                        raise ValueError("v2 record lacks official IMGT annotation")
                    for cdr_name, cdr_sequence in antibody["cdrs"].items():
                        span = antibody.get("cdr_spans", {}).get(cdr_name)
                        if span is None:
                            raise ValueError("v2 CDR lacks an authoritative sequence span")
                        chain = antibody[span["chain"]]
                        if chain[span["start"]:span["end"]] != cdr_sequence:
                            raise ValueError("v2 CDR is inconsistent with its antibody chain")
                operation = record["lineage"]["operation"]
                if operation == "observed_bound":
                    observed_ids.add(record["record_id"])
                    observed_per_group[record["group_id"]] += 1
                    if record["targets"]["binding_label"] != 1:
                        lineage_errors.append(record["record_id"])
                else:
                    parent = record["lineage"]["parent_record_id"]
                    if parent not in observed_ids or record["targets"]["binding_label"] is not None:
                        lineage_errors.append(record["record_id"])
                counts[f"split:{record['split']}"] += 1
                counts[f"tier:{record['evidence']['tier']}"] += 1
                counts[f"operation:{operation}"] += 1
                shard_count += 1
        if shard_count != shard["records"]:
            raise ValueError(f"Shard record count mismatch: {path}")
    if lineage_errors:
        raise ValueError(f"Invalid lineage or labels in {len(lineage_errors)} records")
    invalid_groups = [group for group, count in observed_per_group.items() if count != 1]
    if invalid_groups or len(observed_per_group) != manifest["groups"]:
        raise ValueError("Each group must contain exactly one observed parent")
    if digest.hexdigest().upper() != manifest["canonical_fingerprint"]:
        raise ValueError("Canonical dataset fingerprint mismatch")
    if sum(value for key, value in counts.items() if key.startswith("split:")) != manifest["records"]:
        raise ValueError("Manifest record count mismatch")
    return {
        "status": "pass",
        "records": manifest["records"],
        "groups": manifest["groups"],
        "shards": len(manifest["shards"]),
        "canonical_fingerprint": manifest["canonical_fingerprint"],
        "counts": dict(sorted(counts.items())),
        "lineage_errors": 0,
        "external_evaluation_records": 0,
        "non_training_official_split_records": 0,
        "cluster_split_errors": 0,
    }
