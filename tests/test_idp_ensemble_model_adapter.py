import json

import numpy as np
import torch

from scripts.audit_idp_ensemble_model_adapter import audit
from scripts.generate_idp_ensemble_v3_candidates import (
    find_observed_subsequence_positions,
)
from scripts.protein_mpnn_prefix_adapter import ProteinMPNNPrefixAdapter


class DummyModel:
    def eval(self):
        return self


class MaskCapturingModel(DummyModel):
    def __init__(self):
        self.chain_mask = None

    def prefix_next_log_probs(
            self, _x, _sequence, _mask, chain_mask, _residue_idx,
            _chain_encoding, _prefix, _target):
        self.chain_mask = chain_mask.clone()
        return torch.zeros((1, 21))


def test_without_frozen_checkpoint_adapter_is_not_ready(tmp_path):
    result = audit(tmp_path / "adapter.json")
    assert result["status"] == "implemented_not_runtime_validated"
    assert result["checks"]["prefix_conditioned_next_residue_logits"] is False
    saved = json.loads((tmp_path / "adapter.json").read_text(encoding="ascii"))
    assert saved["status"] == "implemented_not_runtime_validated"


def test_checkpoint_without_runtime_trace_is_not_eligible(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    result = audit(tmp_path / "adapter.json", [checkpoint])
    assert result["status"] == "implemented_not_runtime_validated"
    assert result["observed_interface"]["runtime_trace"]["reason"] == (
        "runtime_trace_missing"
    )


def test_two_pose_runtime_trace_is_required_for_eligibility(tmp_path):
    checkpoint_a = tmp_path / "a.pt"
    checkpoint_b = tmp_path / "b.pt"
    checkpoint_a.write_bytes(b"a")
    checkpoint_b.write_bytes(b"b")
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps({
            "events": [
                {"pose_id": 0, "position": 0, "prefix": "", "log_probs_finite": True, "log_probs_length": 21},
                {"pose_id": 1, "position": 0, "prefix": "", "log_probs_finite": True, "log_probs_length": 21},
            ]
        }),
        encoding="ascii",
    )
    result = audit(tmp_path / "adapter.json", [checkpoint_a, checkpoint_b], trace)
    assert result["status"] == "adapter_eligible"
    assert result["checks"]["uses_multiple_pose_models"] is True


def test_adapter_rejects_mismatched_prefix_length():
    adapter = ProteinMPNNPrefixAdapter(
        model=DummyModel(),
        tensors={"S": torch.zeros((1, 3), dtype=torch.long)},
        h3_positions=[0, 1],
        native_indices=np.array([0, 0]),
    )
    try:
        adapter.next_log_probs("A", 0)
    except ValueError as error:
        assert "Prefix length" in str(error)
    else:
        raise AssertionError("Expected prefix length validation")


def test_h3_mapping_skips_unresolved_coordinate_gap():
    positions = find_observed_subsequence_positions("XXAKNWAPF-DSYY", "AKNWAPFDS")
    assert positions == [2, 3, 4, 5, 6, 7, 8, 10, 11]


def test_prefix_adapter_exposes_framework_as_fixed_context():
    model = MaskCapturingModel()
    adapter = ProteinMPNNPrefixAdapter(
        model=model,
        tensors={
            "S": torch.zeros((1, 6), dtype=torch.long),
            "X": torch.zeros((1, 6, 4, 3)),
            "mask": torch.ones((1, 6)),
            "chain_M": torch.ones((1, 6)),
            "residue_idx": torch.arange(6).unsqueeze(0),
            "chain_encoding_all": torch.zeros((1, 6), dtype=torch.long),
        },
        h3_positions=[2, 3],
        native_indices=np.array([0, 0]),
    )
    adapter.next_log_probs("", 0)
    assert model.chain_mask.tolist() == [[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]]
