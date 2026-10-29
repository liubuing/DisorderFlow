"""IDP 集成配对打分测试：验证打分契约、候选池去重、稳健聚合与留一隔离。

守护 scripts.score_idp_ensemble_matched_candidates 的配对评估：
协议必须为 leave_one_conformer_out 且推断单元为独立抗体参考组件、
候选池在 (组件, 方法) 内去重、稳健聚合惩罚离群构象、
留一打分绝不使用被留出构象选集成候选（防泄漏）、
生成产物必须完整后才允许打分。
"""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.score_idp_ensemble_matched_candidates import (
    candidate_pools,
    leave_one_out,
    robust_value,
)

ROOT = Path(__file__).resolve().parents[1]


# 打分契约冻结：留一协议、组件级推断单元、组件数门槛、最小侧链接触数与
# claim_boundary（不得声称优于 ensemble-aware 生成器）必须与预注册一致——
# 防止评估口径漂移造成过度声明。
def test_scoring_contract_uses_component_level_inference():
    config = yaml.safe_load((
        ROOT / "configs/benchmarks/idp_ensemble_matched_scoring_dev_v1.yml"
    ).read_text(encoding="utf-8"))
    assert config["scoring"]["selection_protocol"] == "leave_one_conformer_out"
    assert config["statistics"]["inference_unit"] == "independent_antibody_reference_component"
    assert config["development_gates"]["minimum_valid_components_per_generator"] == 6
    assert config["scoring"]["minimum_sidechain_contacts_per_conformer"] == 1
    assert "not an ensemble-aware BFN generator" in config["claim_boundary"]


# 池内去重：同组件同方法的重复序列必须折叠为唯一集合——防止重复候选放大统计权重。
def test_candidate_pools_deduplicate_within_component_method():
    raw = {"slots": [
        {"status": "success", "component_id": "c", "method": "m", "candidates": [
            {"sequence": "AAAA"}, {"sequence": "AAAA"}, {"sequence": "CCCC"}
        ]}
    ]}
    assert candidate_pools(raw, ["m"])[("c", "m")] == ["AAAA", "CCCC"]


# 稳健聚合方向：存在低离群值的构象组必须低于全优组——防止均值掩盖单构象失败。
def test_robust_value_penalizes_a_low_outlier():
    aggregation = {
        "p25_weight": 0.45,
        "minimum_weight": 0.25,
        "mean_weight": 0.20,
        "standard_deviation_penalty": 0.10,
    }
    assert robust_value([1, 1, 1, 1], aggregation) > robust_value([0, 1, 1, 1], aggregation)


# 留一隔离：第三构象被留出时，集成候选必须仅由前两个构象选出且分数恰为其值（1.0），
# 被留出构象上的 100 分优势不得泄漏——防止用测试构象选择候选的乐观偏差。
def test_leave_one_out_never_uses_heldout_for_ensemble_selection():
    aggregation = {
        "p25_weight": 0.45,
        "minimum_weight": 0.25,
        "mean_weight": 0.20,
        "standard_deviation_penalty": 0.10,
    }
    matrix = np.asarray([
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 100.0],
    ])
    rows = leave_one_out("c", "m", ["AAAA", "CCCC"], matrix, np.zeros(3), aggregation)
    assert rows[2]["ensemble_candidate"] == "AAAA"
    assert rows[2]["ensemble_score"] == pytest.approx(1.0)


# 产物完整性门槛：生成产物必须 status=complete、54 个 slot 全部完成且零失败，
# 否则不允许进入打分——防止在不完整结果上得出结论。
def test_generation_artifact_is_complete_before_scoring():
    raw = json.loads((
        ROOT / "reviewer_outputs/idp_ensemble_matched_generation_dev_v1/raw_generation.json"
    ).read_text(encoding="utf-8"))
    assert raw["status"] == "complete"
    assert raw["completed_slots"] == 54
    assert raw["failures"] == []
