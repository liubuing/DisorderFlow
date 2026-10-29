"""Tests for negative-design + MD + immunogenicity modules + scorer integration.

All proxy backends (no GPU / no GROMACS / no NetMHCIIpan). Validates that:
  - off_target_docking degrades gracefully when PDB is missing.
  - immunogenicity proxy is in [0,1] and ranks hydrophobic CDRs higher.
  - md_stability 'skip' backend returns None.
  - closed_loop_scorer applies the new off_target/md_rmsd/immunogenicity thresholds.
"""
import os

import pytest

# ── immunogenicity proxy ──
from immunogenicity import predict_immunogenicity


# 值域守护：任何输入（含空串、单残基、极端同聚物）的免疫原性代理分必须落在 [0,1]——
# 防止后端切换或归一化回归导致越界分数。
def test_immunogenicity_proxy_bounded():
    for seq in ["", "A", "ACDEFGHIKLMNPQRSTVWY", "W" * 50, "D" * 50]:
        s = predict_immunogenicity(seq, backend="proxy")
        assert 0.0 <= s <= 1.0


# 方向性：疏水/芳香 stretch 必须比带电 stretch 得分更高（更差）——
# 防止代理打分方向写反把疏水序列当低风险。
def test_immunogenicity_hydrophobic_higher_than_soluble():
    """A hydrophobic/aromatic stretch scores higher (worse) than a charged one."""
    hydrophobic = predict_immunogenicity("WFWFWFWFWFWFWFWFWFWF", backend="proxy")
    charged = predict_immunogenicity("DEDEDEDEDEDEDEDEDEDE", backend="proxy")
    assert hydrophobic > charged


# 优雅降级：netMHCIIpan 二进制缺失时必须回退到 proxy 且返回合法分数——
# 防止外部工具缺失导致整个流程崩溃或返回 None。
def test_immunogenicity_netmhciipan_falls_back_to_proxy():
    """When netMHCIIpan binary is absent, falls back to proxy (never None)."""
    s = predict_immunogenicity("ACDEFGHIKL", backend="netmhciipan",
                               netmhciipan_exe="/nonexistent/netMHCIIpan")
    assert 0.0 <= s <= 1.0  # proxy fallback, not None


# ── off_target_docking graceful degradation ──
# 脱靶 PDB 缺失时必须返回空字典而非抛异常——防止评估管线因缺文件中断。
def test_off_target_scores_missing_pdb_returns_empty():
    from off_target_docking import compute_off_target_scores
    out = compute_off_target_scores(
        [{"full_ab_seq": "ACDE"}], "/nonexistent/off.pdb", "A", verbose=False)
    assert out == {}


# 批量脱靶过滤的降级语义：PDB 缺失 → 分数为空 → 全部设计通过（iptm 默认 0）——
# 防止缺数据时误杀全部候选。
def test_off_target_batch_filter_keeps_passing():
    from off_target_docking import batch_off_target_filter
    # With a missing PDB, scores are empty → all designs pass (iptm defaults 0).
    designs = [{"full_ab_seq": "ACDE", "sequence": "ACDE"}]
    kept = batch_off_target_filter(designs, off_target_iptm_max=0.3,
                                   off_target_pdb="/nonexistent/off.pdb",
                                   off_target_chain="A")
    assert len(kept) == 1
    assert kept[0]["off_target_iptm"] == 0.0


# ── md_stability ──
# skip 后端必须显式返回 None——防止占位实现误报稳定性结果。
def test_md_skip_returns_none():
    from md_stability import run_md_stability
    assert run_md_stability("any.pdb", backend="skip") is None


# gromacs 二进制缺失时 rmsd 必须为 None（优雅降级）——防止伪造数值或崩溃。
def test_md_gromacs_no_binary_returns_none_rmsd():
    """gromacs backend with no gmx installed → rmsd None (graceful)."""
    from md_stability import run_md_stability
    res = run_md_stability("any.pdb", backend="gromacs",
                           gromacs_exe="/nonexistent/gmx", verbose=False)
    assert res["rmsd"] is None


# ── closed_loop_scorer integration (new thresholds) ──
# 硬阈值集成：off_target_iptm 超限的设计必须被拒且计数正确——
# 防止新增阈值未接入 ranker（脱靶结合物混入正选）。
def test_scorer_rejects_off_target_binder():
    from closed_loop_scorer import MultiObjectiveRanker, MULTI_OBJECTIVE_THRESHOLDS
    th = dict(MULTI_OBJECTIVE_THRESHOLDS)
    th["off_target_iptm_max"] = 0.3
    ranker = MultiObjectiveRanker(thresholds=th)
    designs = [
        {"plddt": 80, "iptm": 0.7, "off_target_iptm": 0.1},   # passes
        {"plldt": 80, "iptm": 0.7, "off_target_iptm": 0.6},   # rejected (off-target)
    ]
    # Fix typo'd key on second design:
    designs[1]["plddt"] = 80
    passed, counts = ranker.apply_hard_thresholds(designs)
    assert len(passed) == 1
    assert counts["off_target"] == 1


# 免疫原性阈值集成：immunogenicity 超限的设计必须被拒——防止高免疫原性候选进入短名单。
def test_scorer_rejects_immunogenic_design():
    from closed_loop_scorer import MultiObjectiveRanker, MULTI_OBJECTIVE_THRESHOLDS
    th = dict(MULTI_OBJECTIVE_THRESHOLDS)
    th["immunogenicity_max"] = 0.6
    ranker = MultiObjectiveRanker(thresholds=th)
    designs = [
        {"plddt": 80, "iptm": 0.7, "immunogenicity": 0.2},    # passes
        {"plddt": 80, "iptm": 0.7, "immunogenicity": 0.9},    # rejected
    ]
    passed, counts = ranker.apply_hard_thresholds(designs)
    assert len(passed) == 1
    assert counts["immuno"] == 1


# ── cascade_filter integration ──
# 级联过滤必须尊重显式启用的 off_target/immunogenicity 阈值——
# 防止新指标加入 thresholds 字典后过滤行为失效。
def test_cascade_rejects_off_target_with_enabled_threshold():
    from cascade_filter import apply_cascade
    designs = [
        {"sequence": "GOOD", "plddt": 0.9, "iptm": 0.8, "ppl": 20,
         "off_target_iptm": 0.1, "immunogenicity": 0.2},
        {"sequence": "BBBBOFFARGET", "plddt": 0.9, "iptm": 0.8, "ppl": 20,
         "off_target_iptm": 0.6, "immunogenicity": 0.2},
    ]
    filtered, _ = apply_cascade(designs, thresholds={
        "off_target_iptm_max": 0.3, "immunogenicity_max": 0.9,
        "ppl_max": 100, "entropy_max": 3.0,
    })
    seqs = [r["sequence"] for r in filtered]
    assert "GOOD" in seqs and "BBBBOFFARGET" not in seqs
