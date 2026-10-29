"""Aβ 4HIX 前瞻实验的契约测试：守护保守突变库的可复现性与 Hamming 上界、组成对照的严格置换性、
候选序列批量展开的精确性，以及多样性短名单的两两距离约束。"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# 辅助函数：动态加载脚本模块（脚本不在包内，需 importlib 按路径导入）。
def load_script():
    path = ROOT / "scripts" / "run_abeta_4hix_prospective.py"
    spec = importlib.util.spec_from_file_location("run_abeta_4hix_prospective", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 保守库契约：同参数两次生成必须逐条相同（固定种子可复现）、无重复、
# 每条与亲本 Hamming 距离在 [1,4] 内且不含 Cys——防止"保守"库混入剧烈突变或不可合成序列。
def test_conservative_library_is_reproducible_and_bounded():
    mod = load_script()
    first = mod.conservative_library("VRYDHYSGSSDY", 40, 17)
    second = mod.conservative_library("VRYDHYSGSSDY", 40, 17)
    assert first == second
    assert len(set(first)) == 40
    assert all(1 <= mod.hamming(row, "VRYDHYSGSSDY") <= 4 for row in first)
    assert all("C" not in row for row in first)


# 组成对照：composition shuffle 必须是氨基酸多重集合的严格置换（sorted 后与亲本一致），
# 保证对照只检验序列顺序效应而非组成差异。
def test_composition_controls_preserve_exact_composition():
    mod = load_script()
    rows = mod.composition_shuffles("VRYDHYSGSSDY", 20, 23)
    assert len(set(rows)) == 20
    assert all(sorted(row) == sorted("VRYDHYSGSSDY") for row in rows)


# 批量展开契约：generate_flag 为真的位置被候选序列精确替换（残基编号正确），mask 与 chain_id 随 batch 维同步扩展，
# 防止展开时错位写入或形状失配。
def test_candidate_batch_expansion_is_exact_and_independent():
    import torch

    from modules.bfn_prospective import inject_candidate_sequences

    batch = {
        "aa": torch.tensor([[9, 9, 9, 9]]),
        "generate_flag": torch.tensor([[False, True, False, True]]),
        "mask": torch.ones(1, 4, dtype=torch.bool),
        "chain_id": [("H", "H", "H", "H")],
    }
    expanded = inject_candidate_sequences(batch, ["AC", "WY"])
    assert torch.equal(expanded["aa"], torch.tensor([[9, 0, 9, 1], [9, 18, 9, 19]]))
    assert expanded["mask"].shape == (2, 4)
    assert len(expanded["chain_id"]) == 2


# 多样性约束：得分排序后近重复序列必须被跳过（b 与 a 距离不足被淘汰，保留距离达标的 c），
# 防止短名单被一族相似序列垄断而丧失探索价值。
def test_diverse_shortlist_enforces_pairwise_distance():
    mod = load_script()
    config = {"selection": {"minimum_pairwise_hamming": 2, "shortlist_maximum": 3}}
    rows = [
        {"sequence": "AAAA", "all_gates_pass": True,
         "scoring": {"mean_complex_minus_stripped": 3.0, "std_complex_minus_stripped": 0.1}},
        {"sequence": "AAAT", "all_gates_pass": True,
         "scoring": {"mean_complex_minus_stripped": 2.0, "std_complex_minus_stripped": 0.1}},
        {"sequence": "AATT", "all_gates_pass": True,
         "scoring": {"mean_complex_minus_stripped": 1.0, "std_complex_minus_stripped": 0.1}},
    ]
    selected = mod.diverse_shortlist(config, rows)
    assert [row["sequence"] for row in selected] == ["AAAA", "AATT"]
