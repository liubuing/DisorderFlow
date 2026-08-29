#!/usr/bin/env python
"""Model-native ProteinMPNN prefix-conditioned adapter."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PROTEIN_MPNN = ROOT / "ProteinMPNN"
if str(PROTEIN_MPNN) not in sys.path:
    sys.path.insert(0, str(PROTEIN_MPNN))

from protein_mpnn_utils import ProteinMPNN  # noqa: E402


class ProteinMPNNPrefixAdapter:
    """Expose one frozen ProteinMPNN pose as a sampler-compatible model."""

    def __init__(self, model, tensors, h3_positions, native_indices, pose_id=None):
        self.model = model.eval()
        self.tensors = tensors
        self.h3_positions = tuple(int(value) for value in h3_positions)
        self.native_indices = np.asarray(native_indices, dtype=np.int64)
        self.pose_id = pose_id
        self.runtime_trace = []
        if len(self.h3_positions) != len(self.native_indices):
            raise ValueError("H3 positions and native indices must have equal length")

    @torch.no_grad()
    def next_log_probs(self, prefix, position):
        if position < 0 or position >= len(self.h3_positions):
            raise ValueError("H3 position is out of range")
        if len(prefix) != position:
            raise ValueError("Prefix length must equal the requested H3 position")
        sequence = self.tensors["S"].clone()
        sequence[:, list(self.h3_positions[:position])] = torch.as_tensor(
            self.native_indices[:position], device=sequence.device
        )
        if prefix:
            sequence[:, list(self.h3_positions[:position])] = torch.as_tensor(
                [self._aa_to_index(aa) for aa in prefix], device=sequence.device
            )
        # ProteinMPNN's chain_M marks an entire design chain. For H3-only
        # autoregression that would incorrectly hide fixed heavy-framework
        # sequence from the target. Restrict the design mask to H3 positions;
        # prefix_next_log_probs still orders future H3 residues after target.
        design_mask = torch.zeros_like(self.tensors["chain_M"])
        design_mask[:, list(self.h3_positions)] = 1.0
        log_probs = self.model.prefix_next_log_probs(
            self.tensors["X"], sequence, self.tensors["mask"],
            design_mask, self.tensors["residue_idx"],
            self.tensors["chain_encoding_all"], self.h3_positions[:position],
            self.h3_positions[position],
        )
        values = log_probs[0].detach().cpu().numpy()
        self.runtime_trace.append({
            "pose_id": self.pose_id,
            "position": position,
            "prefix": prefix,
            "log_probs_finite": bool(np.isfinite(values).all()),
            "log_probs_length": int(len(values)),
        })
        return values

    @staticmethod
    def _aa_to_index(aa):
        alphabet = "ACDEFGHIKLMNPQRSTVWYX"
        try:
            return alphabet.index(aa)
        except ValueError as exc:
            raise ValueError(f"Unsupported amino acid: {aa}") from exc


def build_model(checkpoint, device="cpu", ca_only=False):
    """Load a frozen ProteinMPNN checkpoint; tensor featurization is caller-owned."""
    device = torch.device(device)
    checkpoint = torch.load(checkpoint, map_location=device)
    model = ProteinMPNN(
        ca_only=ca_only,
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        augment_eps=0.0,
        k_neighbors=checkpoint["num_edges"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()
