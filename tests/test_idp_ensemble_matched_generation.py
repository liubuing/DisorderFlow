"""IdP 系综匹配生成基线的契约测试：守护生成预算与方法的冻结配置、各参考成分规范化后链组成
与 native H3 一致（含 VHH 双链情形），以及候选槽位的精确数量与 H3 长度校验。"""
import json
from pathlib import Path

import pytest
import yaml

from scripts.generate_idp_ensemble_matched_baselines import (
    canonicalize_component,
    validate_rows,
)

ROOT = Path(__file__).resolve().parents[1]


# 配置冻结：3 个种子、每种子 8 候选、可变区仅重链 H3、三种基线方法全部启用——
# 防止生成预算被悄悄削减造成不公平比较。
def test_generation_contract_freezes_matched_budget_and_methods():
    config = yaml.safe_load((
        ROOT / "configs/benchmarks/idp_ensemble_matched_generation_dev_v1.yml"
    ).read_text(encoding="utf-8"))
    assert config["generation"]["seeds"] == [4101, 4111, 4121]
    assert config["generation"]["candidates_per_seed"] == 8
    assert config["generation"]["mutable_region"] == "heavy_chain_h3_only"
    assert all(config["generation"][method]["enabled"] for method in (
        "proteinmpnn", "single_conformer_bfn", "esm_if"
    ))


# 规范化一致性：全部多方法成分规范化后必须恰含 H/L/P 三链且 native H3 逐字保留——
# 防止链重命名或 H3 突变破坏基线可比性。
def test_all_components_canonicalize_and_match_native_h3(tmp_path):
    manifest = json.loads((
        ROOT / "configs/benchmarks/idp_ensemble_multimethod_dev_v1_components.json"
    ).read_text(encoding="utf-8"))
    for component in manifest["components"]:
        output = tmp_path / component["component_id"] / "canonical.pdb"
        result = canonicalize_component(component, output)
        assert set(result["chains"]) == {"H", "L", "P"}
        assert result["native_h3"] == component["native_h3"]
        assert output.is_file()


# VHH 特例：单域抗体成分只含 H/P 两链，防止用普通抗体的 L 链假设处理 VHH 导致结构化失败。
def test_vhh_component_canonicalizes_as_heavy_and_antigen_only(tmp_path):
    expansion = yaml.safe_load((
        ROOT / "configs/benchmarks/idp_ensemble_expansion_6ref_v1.yml"
    ).read_text(encoding="utf-8"))
    component = next(
        row for row in expansion["components"] if row["component_id"] == "ASYN_2X6M"
    )
    component = {**component, "reference_path": component["reference_pdb"]}
    output = tmp_path / "vhh.pdb"
    result = canonicalize_component(component, output)
    assert set(result["chains"]) == {"H", "P"}
    assert result["native_h3"] == component["native_h3"]


# 槽位校验：候选数必须精确等于预算、H3 长度必须与 native 一致，数量不足或长度异常都要报错——
# 防止方法以更少/更短的候选"偷跑"预算。
def test_candidate_slot_requires_exact_count_and_valid_length():
    component = {"native_h3": "ABCDE"}
    valid = [{"sequence": "ACDEF"} for _ in range(8)]
    validate_rows(valid, component, 8)
    with pytest.raises(ValueError, match="Expected 8"):
        validate_rows(valid[:-1], component, 8)
    with pytest.raises(ValueError, match="Invalid generated H3"):
        validate_rows([{"sequence": "ACDXF"} for _ in range(8)], component, 8)
