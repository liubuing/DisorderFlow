"""IDP 集成多方法开发基准测试：验证开发契约、组件清单与配对推断的消融隔离。

守护 scripts.analyze_idp_ensemble_multimethod_dev：
配置必须声明为回顾性开发（非确认）且 claim_boundary 不得声称 BFN 优势、
组件清单 6 组件 3 来源且全部有先验结果暴露、
配对推断前必须先按组件×方法聚合（折叠均值）、
缺失基线组件不得插补、符号翻转检验为组件级双侧。
"""
import json
from pathlib import Path

import pytest
import yaml

from scripts.analyze_idp_ensemble_multimethod_dev import (
    component_method_means,
    exact_sign_flip_p,
    paired_summary,
)

ROOT = Path(__file__).resolve().parents[1]


# 开发契约：分类必须是回顾性开发而非确认、组件门槛/推断单元/完整配对臂与
# claim_boundary 冻结——防止开发基准被误读为确认性证据。
def test_development_contract_does_not_claim_confirmation():
    config = yaml.safe_load((
        ROOT / "configs/benchmarks/idp_ensemble_multimethod_dev_v1.yml"
    ).read_text(encoding="utf-8"))
    assert config["classification"] == "retrospective_development_benchmark_not_confirmation"
    assert config["gates"]["minimum_valid_components"] == 12
    assert config["primary"]["inference_unit"] == "independent_antibody_reference_component"
    assert config["gates"]["require_complete_matched_arms"] == [
        "single_conformer_bfn", "proteinmpnn", "esm_if"
    ]
    assert "no BFN superiority" in config["claim_boundary"]


# 组件清单溯源：6 个唯一组件、3 个集成来源、全部暴露于先验结果且 pose 来源声明一致——
# 防止组件重复或来源混入破坏消融可解释性。
def test_manifest_has_six_unique_exposed_components_and_three_sources():
    manifest = json.loads((
        ROOT / "configs/benchmarks/idp_ensemble_multimethod_dev_v1_components.json"
    ).read_text(encoding="utf-8"))
    components = manifest["components"]
    assert len(components) == 6
    assert len({row["component_id"] for row in components}) == 6
    assert len({row["ensemble_source"] for row in components}) == 3
    assert all(row["prior_outcome_exposure"] for row in components)
    assert all(row["pose_provenance"] == "template_transferred_computational_pose" for row in components)


# 聚合先行：配对推断必须用组件×方法的折叠均值（fold 级样本不得直接当独立样本）——
# 防止伪重复（pseudoreplication）夸大显著性。
def test_component_aggregation_occurs_before_paired_inference():
    folds = [
        {"component_id": "a", "fold": "0", "method": "ensemble", "score": 3.0},
        {"component_id": "a", "fold": "1", "method": "ensemble", "score": 1.0},
        {"component_id": "a", "fold": "0", "method": "single", "score": 1.0},
        {"component_id": "a", "fold": "1", "method": "single", "score": 1.0},
        {"component_id": "b", "fold": "0", "method": "ensemble", "score": 2.0},
        {"component_id": "b", "fold": "0", "method": "single", "score": 3.0},
    ]
    scores = component_method_means(folds)
    summary = paired_summary(scores, "ensemble", "single", ["a", "b"], 1000, 17)
    assert summary["valid_components"] == 2
    assert summary["mean_delta"] == pytest.approx(0.0)
    assert summary["positive_component_fraction"] == 0.5


# 缺失不插补：缺基线的组件必须标记 invalid 且 delta 为 None——
# 防止单侧插补制造虚假配对样本。
def test_missing_baseline_component_is_not_imputed():
    scores = {("a", "ensemble"): 2.0, ("a", "single"): 1.0, ("b", "ensemble"): 4.0}
    summary = paired_summary(scores, "ensemble", "single", ["a", "b"], 100, 3)
    assert summary["valid_components"] == 1
    assert summary["rows"][1]["valid"] is False
    assert summary["rows"][1]["delta"] is None


# 符号翻转检验语义：3 个组件全同向时双侧精确 p 值应为 2*(1/2)^3=0.25——
# 防止单侧化或非组件级检验导致 p 值口径漂移。
def test_exact_sign_flip_is_component_level_and_two_sided():
    assert exact_sign_flip_p([1.0, 1.0, 1.0]) == 0.25
