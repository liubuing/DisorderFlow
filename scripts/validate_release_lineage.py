#!/usr/bin/env python
"""Recursively validate frozen DisorderFlow release lineage and local artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENTS = (
    "publication/ECLS_SCOPE_FREEZE.yml",
    "publication/successor_v3_contact_v2_registry.json",
    "configs/benchmarks/successor_v3_contact_v2_frozen_code_rescan.json",
    "configs/benchmarks/successor_v3_contact_v2_future_confirmation.yml",
    "configs/benchmarks/successor_v3_contact_v2_development.yml",
    "results/successor_v3_contact_v2/baselines.json",
    "results/successor_v3_contact_v2/development_evaluation.json",
    "release/ecls_v1/artifact_bundle_manifest.json",
    "release/ecls_v1/source_manifest.json",
)
SOURCE_ONLY_DOCUMENTS = (
    "publication/ECLS_SCOPE_FREEZE.yml",
    "release/ecls_v1/source_manifest.json",
)


def sha256(path: Path, hash_mode="raw") -> str:
    if hash_mode == "raw":
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    if hash_mode == "canonical_lf":
        content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        return hashlib.sha256(content).hexdigest()
    raise ValueError(f"Unsupported hash mode: {hash_mode}")


def load_document(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def local_path(root: Path, value: str) -> Path | None:
    if "://" in value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def collect_references(node: Any, location: str = "$"):
    references = []
    if isinstance(node, dict):
        frozen_code = node.get("frozen_code")
        if isinstance(frozen_code, dict):
            for path_value, digest in frozen_code.items():
                if isinstance(path_value, str) and isinstance(digest, str):
                    # hash_mode "frozen_code" marks a historical pin: a mismatch
                    # is rescued when a sibling frozen_code_rescan map re-hashes
                    # the same path at its current state (see
                    # successor_v3_contact_v2_frozen_code_rescan.json)
                    references.append((
                        f"{location}.frozen_code", path_value, digest, "frozen_code"))
        frozen_code_rescan = node.get("frozen_code_rescan")
        if isinstance(frozen_code_rescan, dict):
            for path_value, digest in frozen_code_rescan.items():
                if isinstance(path_value, str) and isinstance(digest, str):
                    references.append((
                        f"{location}.frozen_code_rescan", path_value, digest, "raw"))
        path_value, digest = node.get("path"), node.get("sha256")
        if isinstance(path_value, str) and isinstance(digest, str):
            references.append((location, path_value, digest, node.get("hash_mode", "raw")))
        for key, value in node.items():
            suffix = "_sha256"
            if key.endswith(suffix) and isinstance(value, str):
                stem = key[:-len(suffix)]
                paired = node.get(stem)
                if isinstance(paired, str):
                    references.append((f"{location}.{stem}", paired, value, "raw"))
            if key != "frozen_code":
                references.extend(collect_references(value, f"{location}.{key}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            references.extend(collect_references(value, f"{location}[{index}]"))
    return references


def collect_rescan_maps(node: Any, into: dict[str, str]):
    if isinstance(node, dict):
        rescan = node.get("frozen_code_rescan")
        if isinstance(rescan, dict):
            for path_value, digest in rescan.items():
                if isinstance(path_value, str) and isinstance(digest, str):
                    into[path_value] = digest
        for value in node.values():
            collect_rescan_maps(value, into)
    elif isinstance(node, list):
        for value in node:
            collect_rescan_maps(value, into)


def validate_document(
        root: Path, relative: str, errors: list[str], checked: dict[str, str],
        rescan_maps: dict[str, str] | None = None):
    document_path = local_path(root, relative)
    if document_path is None or not document_path.is_file():
        errors.append(f"document missing: {relative}")
        return None
    payload = load_document(document_path)
    local_rescan: dict[str, str] = {}
    collect_rescan_maps(payload, local_rescan)
    local_rescan.update(rescan_maps or {})
    for location, reference, expected, hash_mode in collect_references(payload):
        path = local_path(root, reference)
        if path is None:
            errors.append(f"unverifiable remote reference at {relative}:{location}: {reference}")
        elif not path.is_file():
            errors.append(f"missing reference at {relative}:{location}: {reference}")
        else:
            observed = sha256(path, "raw" if hash_mode == "frozen_code" else hash_mode)
            checked[f"{path.resolve().as_posix()}:{hash_mode}"] = observed
            if observed != expected:
                if (hash_mode == "frozen_code"
                        and local_rescan.get(reference) == observed):
                    continue
                errors.append(
                    f"hash mismatch at {relative}:{location}: {reference} "
                    f"expected={expected} observed={observed}")
    return payload


def validate_checkpoint(root: Path, policy: dict, development: dict, errors: list[str]):
    candidate = root / policy["candidate_model"]["path"]
    if not candidate.is_file():
        return
    checkpoint = torch.load(candidate, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    lineage = config.get("lineage", {})
    expected_initializer = development["initializer"]["sha256"]
    if lineage.get("allowed_initializer_sha256") != expected_initializer:
        errors.append("checkpoint initializer lineage does not match development protocol")
    eps = config.get("model", {}).get("diffusion", {}).get("eps_net_opt", {})
    if eps.get("pair_contact") is not True:
        errors.append("candidate checkpoint is not configured with pair_contact=true")


def validate_cross_contracts(root: Path, documents: dict[str, Any], errors: list[str]):
    policy = documents.get("configs/benchmarks/successor_v3_contact_v2_future_confirmation.yml")
    development = documents.get("configs/benchmarks/successor_v3_contact_v2_development.yml")
    baseline = documents.get("results/successor_v3_contact_v2/baselines.json")
    registry = documents.get("publication/successor_v3_contact_v2_registry.json")
    if not all((policy, development, baseline, registry)):
        return
    reference = policy["admission"]["reference_union"]
    if not reference["path"].endswith("reference_union_manifest_v3.json"):
        errors.append("future admission must use reference_union_manifest_v3.json")
    admission = policy["admission"]
    if admission.get("minimum_independent_homology_components", 0) < 12:
        errors.append("future admission must require at least 12 independent homology components")
    baseline_protocol = baseline["inputs"]["protocol"]
    actual_protocol_hash = sha256(root / baseline_protocol["path"])
    if baseline_protocol["sha256"] != actual_protocol_hash:
        errors.append("baseline artifact references a stale development protocol")
    for name, artifact in registry.get("artifacts", {}).items():
        path = root / artifact["path"]
        if path.is_file() and path.stat().st_size != artifact.get("bytes"):
            errors.append(f"registry byte-size mismatch: {name}")
    validate_checkpoint(root, policy, development, errors)


def run_validation(root: Path, document_paths=DEFAULT_DOCUMENTS):
    errors: list[str] = []
    checked: dict[str, str] = {}
    # first pass: collect frozen_code_rescan maps from every whitelisted
    # document so a dedicated rescan document can rescue historical pins in
    # the contracts that were frozen before the sources legitimately evolved
    global_rescan: dict[str, str] = {}
    for relative in document_paths:
        document_path = local_path(root, relative)
        if document_path is not None and document_path.is_file():
            collect_rescan_maps(load_document(document_path), global_rescan)
    documents = {
        relative: validate_document(root, relative, errors, checked, global_rescan)
        for relative in document_paths
    }
    validate_cross_contracts(root, documents, errors)
    return {
        "status": "valid" if not errors else "invalid",
        "documents": len(documents),
        "artifacts_checked": len(checked),
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--document", action="append", dest="documents")
    parser.add_argument(
        "--source-only", action="store_true",
        help="Validate only Git-distributed source and lightweight ECLS evidence")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    documents = args.documents or (
        SOURCE_ONLY_DOCUMENTS if args.source_only else DEFAULT_DOCUMENTS)
    result = run_validation(args.root.resolve(), documents)
    if args.json or result["errors"]:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"lineage-valid documents={result['documents']} "
            f"artifacts={result['artifacts_checked']}")
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
