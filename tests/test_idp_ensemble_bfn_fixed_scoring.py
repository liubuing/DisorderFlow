"""IdP 系综 BFN 固定打分契约的测试：守护主终点的冻结定义（固定 t、抗原链、推断单元与声明边界），
以及局部指标只统计生成区域的 pLDDT/PAE——防止打分口径漂移或计入非生成区。"""
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.score_idp_ensemble_candidates_bfn import score_value

ROOT = Path(__file__).resolve().parents[1]


# 基准配置冻结：主终点为 bfn_fixed_candidate_iptm、固定 t=0.5、抗原链 P、推断单元为独立抗体参考成分，
# 声明边界必须写明"不是系综感知的 BFN 生成器"——防止夸大声明。
def test_bfn_fixed_scoring_contract_freezes_primary_endpoint():
    config = yaml.safe_load((
        ROOT / "configs/benchmarks/idp_ensemble_bfn_fixed_scoring_dev_v1.yml"
    ).read_text(encoding="utf-8"))
    assert config["primary"]["endpoint"] == "bfn_fixed_candidate_iptm"
    assert config["bfn"]["fixed_t"] == 0.5
    assert config["bfn"]["antigen_chain"] == "P"
    assert config["statistics"]["inference_unit"] == "independent_antibody_reference_component"
    assert "not an ensemble-aware BFN generator" in config["claim_boundary"]


# 局部化打分：pLDDT/PAE 只在 generate_flag 为真的区域求均值（30.0 / 界面 PAE 均值），
# ipTM 为全局量原样透传——防止未生成区域稀释局部指标。
def test_score_value_uses_generated_region_for_local_metrics():
    import torch

    batch = {"generate_flag": torch.tensor([[False, True, True]])}
    output = {
        "iptm": torch.tensor([0.4]),
        "plddt": torch.tensor([[10.0, 20.0, 40.0]]),
        "pae": torch.tensor([[[9.0, 9.0, 9.0], [9.0, 1.0, 3.0], [9.0, 5.0, 7.0]]]),
        "state_compatibility": torch.tensor([0.2]),
    }
    row = score_value(output, batch)
    assert row["iptm"] == pytest.approx(0.4)
    assert row["generated_region_plddt"] == pytest.approx(30.0)
    assert row["generated_region_pae"] == pytest.approx(np.mean([1.0, 3.0, 5.0, 7.0]))
