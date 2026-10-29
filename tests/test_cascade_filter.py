"""Tests for cascade filtering in modules/cascade_filter.py.

Covers: hard-threshold stage and composite ranking with mocked design dicts.
No model/GPU required.
"""
from cascade_filter import apply_cascade, DEFAULT_THRESHOLDS, DEFAULT_WEIGHTS


# 辅助函数：构造带常用指标的 mock 设计结果字典，便于各测试快速组装输入。
def _mk(seq, ppl=20.0, plddt=0.9, iptm=0.8, entropy=1.0, score=None, recovery=None):
    r = {"sequence": seq, "ppl": ppl, "plddt": plddt, "iptm": iptm,
         "entropy": entropy}
    if score is not None:
        r["score"] = score
    if recovery is not None:
        r["recovery"] = recovery
    return r


# 硬阈值第一阶段：ppl 超限的设计必须被剔除——防止低质量序列进入排序。
def test_hard_threshold_rejects_high_ppl():
    """Designs above ppl_max are filtered out in stage 1."""
    results = [_mk("GOOD1", ppl=20.0), _mk("BADHH", ppl=500.0)]
    filtered, _ = apply_cascade(results, thresholds={"ppl_max": 100.0})
    seqs = [r["sequence"] for r in filtered]
    assert "GOOD1" in seqs and "BADHH" not in seqs


# 硬阈值第一阶段同样适用于熵上限：高熵（序列无序度过高）设计被剔除——
# 防止不可折叠的候选占用下游评估资源。
def test_hard_threshold_rejects_high_entropy():
    results = [_mk("GOOD1", entropy=1.0), _mk("BADHH", entropy=2.9)]
    filtered, _ = apply_cascade(results, thresholds={"entropy_max": 2.5})
    assert len(filtered) == 1
    assert filtered[0]["sequence"] == "GOOD1"


# 复合排序方向性：低 ppl、高置信度的设计必须排在前面且 composite_score 单调更高——
# 防止指标权重或方向写反导致劣质设计占据榜首。
def test_composite_rank_orders_by_quality():
    """Better designs (low ppl, high confidence) rank first."""
    results = [
        _mk("LOWQ", ppl=150.0, plddt=0.4, iptm=0.3, entropy=2.5),
        _mk("HIGH", ppl=15.0, plddt=0.95, iptm=0.9, entropy=0.5),
    ]
    filtered, _ = apply_cascade(results, thresholds={
        "ppl_max": 200.0, "plddt_min": -0.01, "iptm_min": -0.01, "entropy_max": 3.0})
    assert len(filtered) == 2
    assert filtered[0]["sequence"] == "HIGH"
    assert filtered[0]["composite_score"] >= filtered[1]["composite_score"]


# 去重语义：相同序列只保留综合得分最高的一份——防止重复候选稀释候选池多样性。
def test_dedup_keeps_best_duplicate():
    """Identical sequences are deduplicated, keeping the highest-scoring copy."""
    results = [_mk("DUPES", ppl=30.0), _mk("DUPES", ppl=15.0), _mk("UNIQ", ppl=40.0)]
    filtered, _ = apply_cascade(results, thresholds={"ppl_max": 100.0})
    seqs = [r["sequence"] for r in filtered]
    assert seqs.count("DUPES") == 1
    assert "UNIQ" in seqs


# 空输入边界：返回空列表且报告仍为字符串——防止对空批次的崩溃或报告类型漂移。
def test_empty_input_returns_empty():
    filtered, report = apply_cascade([])
    assert filtered == []
    assert isinstance(report, str)
