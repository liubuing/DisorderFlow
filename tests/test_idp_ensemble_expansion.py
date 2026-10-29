"""IDP 集成扩展（6 参考组件）配置冻结测试：验证谱系集合与合成集成的隔离声明。

守护 idp_ensemble_expansion_6ref_v1 基准配置的冻结溯源：
6 个组件 ID 必须与冻结集合逐一相等且互不相同、
状态标记必须保留"修复后、生成前"的冻结时机、
仅 TAU_6PXR 允许使用合成局部 OpenMM 集成（其余为实验来源）、
VHH 组件必须显式声明无轻链。
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


# 谱系冻结：组件数与 ID 集合必须与预注册完全一致（防静默增删组件），
# 冻结状态与合并后统计（12/11 敏感性）必须原样保留——保证扩展实验可审计。
def test_expansion_freezes_six_distinct_antibody_lineages_before_scoring():
    config = yaml.safe_load((
        ROOT / "configs/benchmarks/idp_ensemble_expansion_6ref_v1.yml"
    ).read_text(encoding="utf-8"))
    components = config["components"]
    assert len(components) == 6
    assert len({row["component_id"] for row in components}) == 6
    assert {row["component_id"] for row in components} == {
        "ABETA_3U0T", "ABETA_7BXV", "TAU_6LRA",
        "ASYN_2X6M", "ASYN_6CT7", "TAU_6PXR",
    }
    assert config["status"] == "frozen_after_6pxr_input_topology_fix_before_replica_generation"
    assert config["statistics"]["primary_total_components_after_merge"] == 12
    assert config["statistics"]["sensitivity_excluding_synthetic_component"] == 11


# 合成集成隔离：仅 TAU_6PXR 可用 synthetic_local_openmm（8 seeds、固定抗体模板转移模式）——
# 防止合成构象悄悄替换实验构象而破坏消融解释。
def test_only_one_expansion_component_uses_synthetic_local_ensemble():
    config = yaml.safe_load((
        ROOT / "configs/benchmarks/idp_ensemble_expansion_6ref_v1.yml"
    ).read_text(encoding="utf-8"))
    synthetic = [
        row for row in config["components"]
        if row["ensemble"]["type"] == "synthetic_local_openmm"
    ]
    assert [row["component_id"] for row in synthetic] == ["TAU_6PXR"]
    assert len(synthetic[0]["ensemble"]["seeds"]) == 8
    assert synthetic[0]["ensemble"]["sampling_mode"] == (
        "peptide_only_then_fixed_antibody_template_transfer"
    )


# VHH 契约：纳米抗体组件必须显式 light_chain=None 且表位字段与冻结值一致——
# 防止把 VHH 当作常规 Fab 处理（误配轻链）。
def test_vhh_component_explicitly_has_no_light_chain():
    config = yaml.safe_load((
        ROOT / "configs/benchmarks/idp_ensemble_expansion_6ref_v1.yml"
    ).read_text(encoding="utf-8"))
    vhh = next(row for row in config["components"] if row["component_id"] == "ASYN_2X6M")
    assert vhh["light_chain"] is None
    assert vhh["epitope"] == "DYEPEA"
    assert vhh["nominal_epitope"] == "GYQDYEPEA"
