"""Tests for state-specific IDP antibody design MVP."""

# 状态特异性打分器测试：验证互补性方向、状态特异性排序与生成确定性。
# 守护 state-specific IDP 抗体设计 MVP：
# 互补性必须奖励针对 aβ 芳香/电荷表位的设计（方向性）、
# 排序偏向正状态优于负状态（specificity_gap 为核心指标）、
# 生成器同种子可复现且产物可排序、
# 序列复杂度必须惩罚同聚物——防止生成器输出退化序列。

from state_specificity_scorer import (
    DesignSpec,
    complementarity_score,
    generate_paratope_candidates,
    load_preset,
    rank_candidates,
    sequence_complexity,
)


# 互补性方向：芳香/电荷互补设计必须优于同聚甘氨酸——防止打分方向写反。
def test_complementarity_rewards_amyloid_aromatic_charge_pattern():
    epitope = "KLVFFAED"
    designed = "DYWYSRNYQHG"
    weak = "GGGGGGGGGGG"
    assert complementarity_score(designed, epitope) > complementarity_score(weak, epitope)


# 状态特异性排序：正状态表位得分高的候选必须排在前面且 gap 为正——
# 守护"区分目标状态与脱靶状态"这一核心设计目标。
def test_rank_candidates_prefers_positive_over_negative_specificity():
    spec = DesignSpec(
        target="abeta42",
        epitope_id="core_16_24",
        epitope_seq="KLVFFAED",
        positive_state="oligomer_core",
        negative_epitopes=["DAEFRHDSGY", "GGGGGGGG"],
    )
    ranked = rank_candidates([
        {"candidate_id": "good", "cdr_h3": "DYWYSRNYQHG"},
        {"candidate_id": "bad", "cdr_h3": "GGGGGGGGGGG"},
    ], spec)
    assert ranked[0]["candidate_id"] == "good"
    assert ranked[0]["specificity_gap"] > ranked[1]["specificity_gap"]


# 生成确定性：同种子两次生成必须完全一致，且产物可排序（rank 从 1 起、含状态分）——
# 防止生成器不可复现或排序字段缺失。
def test_generator_is_deterministic_and_outputs_rankable_candidates():
    spec = load_preset("abeta_core")
    a = generate_paratope_candidates(spec, n=12, length=11, seed=7)
    b = generate_paratope_candidates(spec, n=12, length=11, seed=7)
    assert [x["cdr_h3"] for x in a] == [x["cdr_h3"] for x in b]
    ranked = rank_candidates(a, spec)
    assert len(ranked) == 12
    assert ranked[0]["rank"] == 1
    assert "state_specific_score" in ranked[0]


# 复杂度惩罚：多样化序列必须高于同聚物——防止生成器偏好退化重复序列。
def test_sequence_complexity_penalizes_homopolymers():
    assert sequence_complexity("DYWYSRNYQHG") > sequence_complexity("TTTTTTTTTTT")
