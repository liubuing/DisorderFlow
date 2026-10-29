"""AAYL 可学习性基准测试：验证打分指标在平局、顺序扰动与划分预算下的不变量。

守护 AAYL 抗体结合可学习性评估管线的三条不变量：
平局分数下的指标应等于随机期望（方向性指标不因平局虚高）、
覆盖集合按"精确替换"而非仅位点判定、
冻结划分（splits.json）无候选泄漏且严格满足预算。
"""
import json
import numpy as np

from scripts.benchmark_aayl_learnability import OUT, load, metrics, mutations, top_weights


# 平局分数下 top20_recall/enrichment 应为随机期望、pair_accuracy 恰为 0.5；
# 同时指标必须对样本顺序反转保持不变——防止排序平局处理被改成有偏实现。
def test_tied_screen_is_order_invariant_and_random_expectation():
    y = np.arange(10, dtype=float)
    tied = np.ones(10)
    assert np.isclose(metrics(y, tied)["top20_recall"], .2)
    assert np.isclose(metrics(y, tied)["enrichment"], 1)
    assert metrics(y, tied)["pair_accuracy"] == .5
    p = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    assert np.isclose(top_weights(p, 2).sum(), 2)
    assert metrics(y, p) == metrics(y[::-1], p[::-1])
    assert metrics(y, y)["top20_recall"] == 1


# 覆盖判定必须基于"精确替换组合"而非仅突变位点：
# 已见 (C1A,D2D) 组合不能让未见替换 C→E 也算覆盖——防止覆盖统计虚高。
def test_coverage_requires_exact_substitution_not_just_position():
    seen = mutations("CAAA", "AAAA") | mutations("ADAA", "AAAA")
    assert mutations("CDAA", "AAAA") <= seen
    assert not mutations("CEAA", "AAAA") <= seen


# 冻结划分溯源审计：train/test 无交集、重链序列无泄漏、
# train 全为单突变而 test 为 2-3 突变、family 一致且预算精确满足，
# covered 字段须由 train 覆盖集精确推出——防止候选泄漏进训练集破坏评估可信度。
def test_frozen_splits_have_no_candidate_leakage_and_exact_budgets():
    _, rows, changes = load()
    manifest = json.loads((OUT / "splits.json").read_text())
    assert manifest["rows"] == [r["poi"] for r in rows]
    assert len(manifest["splits"]) == 22
    for s in manifest["splits"]:
        train, test = s["train"], s["test"]
        assert not set(train) & set(test)
        assert not {rows[i]["heavy_sequence"] for i in train} & {rows[i]["heavy_sequence"] for i in test}
        assert all(len(changes[i]) == 1 for i in train)
        assert all(len(changes[i]) in (2, 3) for i in test)
        assert all(rows[i]["family"] == s["family"] for i in train + test)
        if isinstance(s["budget"], int):
            assert len(train) == s["budget"]
        else:
            assert len(train) == (89 if s["family"] == "AAYL49" else 55)
        seen = frozenset().union(*(changes[i] for i in train))
        assert s["covered"] == [i for i in test if changes[i] <= seen]
