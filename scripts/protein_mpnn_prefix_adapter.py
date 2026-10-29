#!/usr/bin/env python
"""Model-native ProteinMPNN prefix-conditioned adapter."""

# 本适配器把一个冻结的 ProteinMPNN pose 暴露为"采样器兼容"的模型：
# 逐位把前缀写入 H3 区（天然序列占位 + 已生成前缀覆盖），把设计掩码
# 限制在 H3 位置（避免 chain_M 整链设计把固定框架序列从目标里隐藏），
# 再调用模型原生的 prefix_next_log_probs 得到下一位置的条件对数概率。
# 每次调用都会记录运行时轨迹（pose/位置/前缀/有限性），供诊断复现。

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

    # tensors 为调用方持有的特征（X 坐标/mask/chain_M 等），
    # h3_positions 与 native_indices 必须等长（H3 区与天然残基一一对应）
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
    # 结构参数（128 维节点/边特征、3+3 层）与 ProteinMPNN vanilla 权重一致；
    # k_neighbors 取自检查点记录，保证图连通度与训练时相同
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
