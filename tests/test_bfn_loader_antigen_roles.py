"""BFN 加载器抗原角色测试：验证区域批次构建、H3 掩码解析与候选界面置信度权重校验。

守护候选界面生成管线：
显式指定的 antigen_chains 必须被标记为 Antigen 且不参与生成、
区域规格按 1-based 闭区间解析为 0-based 索引、
checkpoint 必须携带全部新增置信度权重、
精确序列打分只对生成区掩码计分且不污染输入 batch。
"""
import types

import pytest
import torch

from disorderflow.utils.protein.constants import Fragment
from modules.bfn_loader import (
    _score_exact_candidate_confidence,
    _validate_candidate_interface_checkpoint,
    build_region_batch,
    parse_region_spec,
)


# 通用复合物：抗原链残基必须打上 Antigen 类型且与生成区互斥，
# 防止角色混用导致对抗原残基进行生成或对生成区漏算。
def test_generic_complex_assigns_explicit_antigen_chain():
    batch = build_region_batch(
        "data/anti_abeta_refs/4HIX.pdb",
        "H:96-107",
        context_chains=["L", "A"],
        antigen_chains=["A"],
        device="cpu",
    )
    antigen = batch["fragment_type"] == int(Fragment.Antigen)
    generated = batch["generate_flag"].bool()
    assert int(antigen.sum()) == 6
    assert int(generated.sum()) == 12
    assert not (antigen & generated).any()


# 冻结的 4HIX H3 区域规格 "H:96-107" 必须解析为 12 个 1-based 位置对应的 0-based 索引，
# 防止区间端点 off-by-one 使生成掩码漂移。
def test_frozen_4hix_h3_mask_is_twelve_one_based_positions():
    assert parse_region_spec("H:96-107") == {"H": list(range(95, 107))}


# checkpoint 校验：candidate_interface 置信度头缺少任一新增权重必须报错，
# 权重齐全时通过——防止用旧 checkpoint 静默生成（随机初始化的置信度头）。
def test_candidate_interface_checkpoint_requires_all_new_weights():
    model = types.SimpleNamespace(
        bfn=types.SimpleNamespace(
            receiver=types.SimpleNamespace(
                confidence_head_kind="candidate_interface_v1")),
        state_dict=lambda: {
            "bfn.receiver.candidate_interface_pair.weight": torch.zeros(1),
        },
    )
    with pytest.raises(RuntimeError, match="missing confidence weights"):
        _validate_candidate_interface_checkpoint(model, {})
    _validate_candidate_interface_checkpoint(
        model, {"bfn.receiver.candidate_interface_pair.weight": torch.zeros(1)})


# 精确候选序列的后验打分：只对生成区掩码取 plddt 均值/方差，
# 传入打分器的氨基酸须替换为候选序列且原 batch 不被就地修改——防止打分泄漏或副作用。
def test_generated_candidate_confidence_post_scores_exact_sequence():
    class Core:
        @staticmethod
        def _mask_antigen(batch, candidate_mask):
            return torch.tensor([[False, False, True]])

    class Model:
        bfn = Core()

        @staticmethod
        def score(batch, fixed_t):
            assert fixed_t == 0.5
            assert torch.equal(batch["aa"], torch.tensor([[0, 1, 2]]))
            return {
                "plddt": torch.tensor([[0.1, 0.8, 0.3]]),
                "iptm": torch.tensor([0.7]),
                "pae": torch.tensor([[[0.0, 0.0, 0.4],
                                       [0.0, 0.0, 0.2],
                                       [0.0, 0.0, 0.0]]]),
            }

    batch = {
        "aa": torch.tensor([[9, 9, 2]]),
        "mask": torch.ones(1, 3, dtype=torch.bool),
        "generate_flag": torch.tensor([[True, True, False]]),
    }
    result = _score_exact_candidate_confidence(Model(), batch, "AC")
    assert result == {
        "plddt": pytest.approx(0.45),
        "plddt_std": pytest.approx(0.35),
        "iptm": pytest.approx(0.7),
        "pae": pytest.approx(0.3),
        "pae_scope": "candidate_to_antigen",
    }
    assert torch.equal(batch["aa"], torch.tensor([[9, 9, 2]]))
