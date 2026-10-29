"""状态对比结构管线的契约测试：守护官方 IMGT CDR 的精确映射（含恒定区上下文）、
严格比对不映射错配残基、完整组批采样不拆组且容量不足拒绝、抗原掩码按 fragment_type 判定
（不依赖同伴抗体链），以及 receiver 聚合对多样本 batch 的广播。"""
import torch
from pathlib import Path

from disorderflow.datasets.statecontrast_structural import (
    assign_official_imgt_cdrs,
    exact_alignment_map,
)
from disorderflow.utils.data import CompleteGroupBatchSampler
from disorderflow.utils.protein.constants import CDR, ressymb_to_resindex


# 辅助函数：把氨基酸序列编码为残基索引张量。
def encoded(sequence):
    return torch.tensor([ressymb_to_resindex[aa] for aa in sequence], dtype=torch.long)


# CDR 映射精确性：官方序列在含恒定区更长链中的偏移必须精确对齐（H3 还原一致、三组 CDR flag 计数正确），
# 防止 CDR 注释在带恒定区链上错位。
def test_official_imgt_cdrs_map_exactly_through_constant_domain_context():
    official = "QVQLGFTFSSYTWVRQISSGGAYTKGRFCARDRGFDYWGQGT"
    parsed = official + "ASTKGPSVFPLAP"
    chain = {"aa": encoded(parsed)}
    cdrs = {"H1": "GFTFSSYT", "H2": "ISSGGAYT", "H3": "CARDRGFDYW"}
    spans = {
        name: {
            "start": official.index(sequence),
            "end": official.index(sequence) + len(sequence),
        }
        for name, sequence in cdrs.items()
    }

    indices = assign_official_imgt_cdrs(chain, official, cdrs, spans)

    assert set(indices) == set(cdrs)
    assert "".join(parsed[index] for index in indices["H3"]) == cdrs["H3"]
    assert (chain["cdr_flag"] == int(CDR.H1)).sum().item() == len(cdrs["H1"])
    assert (chain["cdr_flag"] == int(CDR.H2)).sum().item() == len(cdrs["H2"])
    assert (chain["cdr_flag"] == int(CDR.H3)).sum().item() == len(cdrs["H3"])


# 严格比对：单残基错配（D→N）处不得建立映射，其余位置保持一致——防止含错位的比对进入结构对齐。
def test_alignment_map_does_not_map_mismatched_residues():
    mapping = exact_alignment_map("ACDE", "ACNE")
    assert mapping[0] == 0
    assert mapping[1] == 1
    assert 2 not in mapping
    assert mapping[3] == 3


# 测试替身：定义三个组的分组数据集。
class GroupedDataset:
    group_indices = ((0, 1, 2), (3, 4), (5, 6, 7))


# 组完整性：同组样本必须全部落在同一 batch（组不可拆），防止组内样本跨 batch 破坏配对比较。
def test_complete_group_sampler_never_splits_groups():
    sampler = CompleteGroupBatchSampler(
        GroupedDataset(), max_batch_records=5, shuffle=False)
    batches = list(sampler)

    assert batches == [[0, 1, 2, 3, 4], [5, 6, 7]]
    for group in GroupedDataset.group_indices:
        containing = [batch for batch in batches if set(group) & set(batch)]
        assert len(containing) == 1
        assert set(group).issubset(containing[0])


# 容量守护：最大组大小超过 batch 容量必须报错，防止静默拆组掩盖配置错误。
def test_complete_group_sampler_rejects_too_small_capacity():
    try:
        CompleteGroupBatchSampler(GroupedDataset(), max_batch_records=2)
    except ValueError as error:
        assert "exceeds batch capacity" in str(error)
    else:
        raise AssertionError("Expected an undersized batch capacity failure")


# 抗原掩码：必须按 fragment_type 识别抗原（而非推断"非抗体链"），生成位只开放生成区——
# 防止把抗体自身残基误开放为生成目标。
def test_core_antigen_mask_uses_fragment_type_not_partner_antibody_chain():
    from disorderflow.modules.bfn.core import AntibodyBFN_Core

    core = object.__new__(AntibodyBFN_Core)
    batch = {
        "mask": torch.ones((1, 6), dtype=torch.bool),
        "fragment_type": torch.tensor([[1, 1, 2, 2, 3, 3]]),
        "chain_nb": torch.tensor([[0, 0, 1, 1, 2, 2]]),
    }
    generate = torch.tensor([[True, False, True, False, False, False]])

    assert core._mask_antigen(batch, generate).tolist() == [
        [False, False, False, False, True, True]
    ]


# 聚合广播源码契约：抗原计数 clamp(min=1) 与 sum/count 的广播除法必须存在——
# 防止多样本 batch 聚合时的除零或形状错误回归。
def test_receiver_pair_aggregation_broadcasts_for_multi_sample_batches():
    source = Path("disorderflow/modules/bfn/receiver.py").read_text(encoding="utf-8")
    assert "clamp(min=1).unsqueeze(-1)" in source
    assert "sum(dim=2) / antigen_count" in source
