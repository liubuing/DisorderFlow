import json
import pickle

import numpy as np
import pytest
import torch

from disorderflow.datasets.disorder_augmented import load_disorder_lookup
from disorderflow.disorder_supervision import (
    audit_cluster_splits,
    merge_evidence,
    normalize_evidence,
)
from disorderflow.utils.transforms.merge import MergeChains
from scripts.build.build_caid_cluster_manifest import build as build_caid_cluster_manifest
from scripts.build.build_disorder_supervision import build_artifact
from scripts.build.prepare_disprot_supervision import experimental_profile


def evidence(source, values, confidence=None, cluster="c1"):
    record = {
        "id": "sample",
        "sequence": "AAA",
        "values": values,
        "source": source,
        "cluster_id": cluster,
    }
    if confidence is not None:
        record["confidence"] = confidence
    return record


def test_normalize_evidence_masks_unknown_residues():
    record = evidence("disprot_experimental", [1.0, 0.0, 0.0])
    record["mask"] = [True, True, False]
    result = normalize_evidence(record)
    assert result.mask.tolist() == [True, True, False]
    assert result.confidence.tolist() == [1.0, 1.0, 0.0]


def test_low_confidence_teacher_cannot_override_experiment():
    experimental = evidence("disprot_experimental", [1.0, 0.0, 1.0])
    teacher = evidence("external_predictor", [0.0, 1.0, 0.0])
    merged = merge_evidence([experimental, teacher])
    assert np.all(merged.values[[0, 2]] > 0.5)
    assert merged.values[1] < 0.5


def test_cluster_audit_rejects_cross_split_cluster():
    row = {"cluster_id": "cluster-a", "sequence": "AAAA"}
    audit = audit_cluster_splits({"train": [row], "external_test": [row]})
    assert not audit["valid"]
    assert any("cluster" in error for error in audit["errors"])


def test_builder_rejects_held_out_cluster(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text(json.dumps(evidence("disprot_experimental", [1, 0, 1])) + "\n")
    with pytest.raises(ValueError, match="Held-out clusters"):
        build_artifact([path], "train", tmp_path / "artifact.pkl", ["c1"])


def test_v3_artifact_preserves_mask_and_confidence(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text(json.dumps(evidence("rmsf_proxy", [0.2, 0.8, 0.4])) + "\n")
    output = tmp_path / "artifact.pkl"
    build_artifact([path], "train", output)
    loaded = load_disorder_lookup(output, expected_split="train", require_envelope=True)
    assert loaded["sample"]["mask"].all()
    assert np.allclose(loaded["sample"]["confidence"], 0.55)


def test_v3_artifact_can_filter_a_larger_structure_store(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text(json.dumps(evidence("disprot_experimental", [1, 0, 1])) + "\n")
    output = tmp_path / "artifact.pkl"
    build_artifact([path], "train", output)
    loaded = load_disorder_lookup(
        output, expected_split="train", expected_ids=["sample", "unlabelled"],
        require_envelope=True)
    assert set(loaded) == {"sample"}


def test_disprot_profile_does_not_treat_unannotated_as_ordered():
    entry = {
        "sequence": "AAAAAA",
        "regions": [
            {"term_namespace": "Structural state", "term_name": "disorder",
             "ec_go": "EXP", "ec_id": "ECO:1", "start": 2, "end": 3},
            {"term_namespace": "Structural state", "term_name": "order",
             "ec_go": "IDA", "ec_id": "ECO:2", "start": 5, "end": 5},
        ],
    }
    values, mask, _ = experimental_profile(entry)
    assert mask == [False, True, True, False, True, False]
    assert values == [0, 1, 1, 0, 0, 0]


def test_caid_manifest_marks_missing_snapshot_entries_unresolved(tmp_path):
    entries = tmp_path / "entries.json"
    entries.write_text(json.dumps({"data": [
        {"disprot_id": "DP1", "uniref50": "UniRef50_A", "acc": "P1"},
    ]}))
    sequences = tmp_path / "sequences.fasta"
    sequences.write_text(">DP1\nAAA\n>DP2\nAAA\n")
    output = tmp_path / "clusters.json"
    audit_path = tmp_path / "clusters.audit.json"

    audit = build_caid_cluster_manifest(entries, sequences, output, audit_path)

    manifest = json.loads(output.read_text())
    assert manifest == {"DP1": "UniRef50_A", "DP2": "unresolved:DP2"}
    assert audit["resolved_count"] == 1
    assert audit["unresolved_ids"] == ["DP2"]


def test_merge_chains_fills_unsupervised_disorder_fields():
    transform = MergeChains()
    data = {"aa": torch.zeros(3, dtype=torch.long)}
    assert not transform._data_attr(data, "disorder_supervision_mask").any()
    assert transform._data_attr(data, "disorder_confidence").sum() == 0
