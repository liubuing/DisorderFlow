"""Validation and merging for residue-level disorder supervision."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SOURCE_CONFIDENCE = {
    "disprot_experimental": 1.0,
    "mobidb_experimental": 1.0,
    "missing_density_proxy": 0.65,
    "nmr_ensemble_proxy": 0.65,
    "rmsf_proxy": 0.55,
    "external_predictor": 0.25,
    "afdb_plddt_ordered_proxy": 0.25,
    "charge_hydropathy_heuristic_v1": 0.10,
}


@dataclass(frozen=True)
class DisorderEvidence:
    values: np.ndarray
    mask: np.ndarray
    confidence: np.ndarray
    sources: tuple[str, ...]
    cluster_id: str


def normalize_evidence(record, expected_length=None):
    """Validate one serialized evidence record and return numeric arrays."""
    values = np.asarray(record["values"], dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("Disorder values must be one-dimensional")
    if expected_length is not None and len(values) != expected_length:
        raise ValueError(f"Disorder length mismatch: {len(values)} != {expected_length}")
    mask = np.asarray(record.get("mask", np.isfinite(values)), dtype=bool)
    if mask.shape != values.shape:
        raise ValueError("Disorder mask shape must match values")
    if np.any(mask & ~np.isfinite(values)):
        raise ValueError("Supervised disorder values must be finite")
    if np.any(mask & ((values < 0) | (values > 1))):
        raise ValueError("Supervised disorder values must be in [0, 1]")
    source = str(record["source"])
    raw_confidence = record.get("confidence", SOURCE_CONFIDENCE.get(source))
    if raw_confidence is None:
        raise ValueError(f"Unknown source requires explicit confidence: {source}")
    confidence = np.asarray(raw_confidence, dtype=np.float32)
    if confidence.ndim == 0:
        confidence = np.full(values.shape, float(confidence), dtype=np.float32)
    if confidence.shape != values.shape:
        raise ValueError("Disorder confidence shape must match values")
    if np.any(mask & ((confidence <= 0) | (confidence > 1))):
        raise ValueError("Supervised confidence must be in (0, 1]")
    cluster_id = str(record.get("cluster_id", "")).strip()
    if not cluster_id:
        raise ValueError("cluster_id is required for leakage-safe supervision")
    return DisorderEvidence(
        values=np.where(mask, values, 0).astype(np.float32),
        mask=mask,
        confidence=np.where(mask, confidence, 0).astype(np.float32),
        sources=(source,),
        cluster_id=cluster_id,
    )


def merge_evidence(records, expected_length=None):
    """Merge evidence by confidence-weighted mean, preserving unknown residues."""
    normalized = [normalize_evidence(record, expected_length) for record in records]
    if not normalized:
        raise ValueError("At least one evidence record is required")
    clusters = {item.cluster_id for item in normalized}
    if len(clusters) != 1:
        raise ValueError("Evidence for one sequence must share a cluster_id")
    numerator = np.zeros_like(normalized[0].values)
    denominator = np.zeros_like(normalized[0].values)
    sources = []
    for item in normalized:
        numerator += item.values * item.confidence
        denominator += item.confidence
        sources.extend(item.sources)
    mask = denominator > 0
    values = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=mask)
    confidence = np.clip(denominator, 0, 1)
    return DisorderEvidence(
        values=values.astype(np.float32),
        mask=mask,
        confidence=confidence.astype(np.float32),
        sources=tuple(sorted(set(sources))),
        cluster_id=normalized[0].cluster_id,
    )


def audit_cluster_splits(split_records):
    """Fail when a homology cluster or exact sequence crosses data splits."""
    cluster_splits = {}
    sequence_splits = {}
    errors = []
    for split, records in split_records.items():
        for record in records:
            cluster = str(record["cluster_id"])
            sequence = str(record["sequence"]).upper()
            previous = cluster_splits.setdefault(cluster, split)
            if previous != split:
                errors.append(f"cluster {cluster} crosses {previous}/{split}")
            previous = sequence_splits.setdefault(sequence, split)
            if previous != split:
                errors.append(f"exact sequence crosses {previous}/{split}")
    return {
        "valid": not errors,
        "n_clusters": len(cluster_splits),
        "n_sequences": len(sequence_splits),
        "errors": sorted(set(errors)),
    }
