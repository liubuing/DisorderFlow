"""IdP 系综模型适配器的前向兼容审计：守护"无运行时轨迹不得判定适配器可用"的分级、
前缀长度与 H3 映射（跳过未解析坐标缺口）的校验，以及框架区以固定上下文暴露给 ProteinMPNN。"""
import json

import numpy as np
import torch

from scripts.audit_idp_ensemble_model_adapter import audit
from scripts.generate_idp_ensemble_v3_candidates import (
    find_observed_subsequence_positions,
)
from scripts.protein_mpnn_prefix_adapter import ProteinMPNNPrefixAdapter


# 测试替身：仅提供 eval 接口的哑模型。
class DummyModel:
    def eval(self):
        return self


# 测试替身：捕获 chain_mask 参数并返回零 logits 的前向桩，用于断言框架区被冻结。
class MaskCapturingModel(DummyModel):
    def __init__(self):
        self.chain_mask = None

    def prefix_next_log_probs(
            self, _x, _sequence, _mask, chain_mask, _residue_idx,
            _chain_encoding, _prefix, _target):
        self.chain_mask = chain_mask.clone()
        return torch.zeros((1, 21))


# 分级诚实：无冻结 checkpoint 时状态必须为 implemented_not_runtime_validated（而非 eligible），
# 且状态落盘一致——防止未验证的适配器被宣称可用。
def test_without_frozen_checkpoint_adapter_is_not_ready(tmp_path):
    result = audit(tmp_path / "adapter.json")
    assert result["status"] == "implemented_not_runtime_validated"
    assert result["checks"]["prefix_conditioned_next_residue_logits"] is False
    saved = json.loads((tmp_path / "adapter.json").read_text(encoding="ascii"))
    assert saved["status"] == "implemented_not_runtime_validated"


# 轨迹门槛：仅有 checkpoint 而无运行时轨迹时仍不得 eligible（reason=runtime_trace_missing）——
# 防止"有模型文件"冒充"接口已验证"。
def test_checkpoint_without_runtime_trace_is_not_eligible(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    result = audit(tmp_path / "adapter.json", [checkpoint])
    assert result["status"] == "implemented_not_runtime_validated"
    assert result["observed_interface"]["runtime_trace"]["reason"] == (
        "runtime_trace_missing"
    )


# eligible 门槛：多姿态 checkpoint 加上每个 pose 各自的有限 logits 轨迹才可判定 eligible，
# 保证"多姿态可用"的结论有逐 pose 证据支撑。
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


# 前缀长度校验：前缀长度超过 H3 槽位必须报错，防止越界写入打乱残基编号。
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


# 坐标缺口映射：子序列映射必须跳过未解析的 '-' 缺口位（10、11 对应缺口后位置），
# 防止把突变序列错配到缺失坐标上。
def test_h3_mapping_skips_unresolved_coordinate_gap():
    positions = find_observed_subsequence_positions("XXAKNWAPF-DSYY", "AKNWAPFDS")
    assert positions == [2, 3, 4, 5, 6, 7, 8, 10, 11]


# 框架冻结：chain_mask 必须只在 H3 位置为 1（框架区全 0）——
# 防止适配器把框架区也开放为可设计位点。
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
