"""Tests for contact-aware state specificity scorer."""

from state_contact_scorer import (
    Contact,
    abeta_fragment_info,
    benchmark_native_vs_decoys,
    conservative_mutant,
    generate_contact_guided_variants,
    benchmark_native_vs_scrambled,
    hotspot_breaking_mutant,
    score_contact_guided_variants,
    score_sequence_on_contact_map,
    select_abeta_like_chain,
)


# 辅助函数：构造小型接触图（paratope DYW × 肽 KFF，三条 3.8-4.2 Å 接触）。
def _toy_contact_map():
    return {
        "paratope_sequence": "DYW",
        "peptide_sequence": "KFF",
        "contacts": [
            Contact(0, "H", 31, "D", 0, "P", 1, "K", 3.8),
            Contact(1, "H", 32, "Y", 1, "P", 2, "F", 4.0),
            Contact(2, "H", 33, "W", 2, "P", 3, "F", 4.2),
        ],
    }


# 接触敏感性：native 序列得分必须高于乱序序列（接触分与热点分皆然）——
# 防止打分器对序列排列不敏感而退化为常数。
def test_native_scores_above_scrambled_hotspot_layout():
    cmap = _toy_contact_map()
    native = score_sequence_on_contact_map("DYW", cmap)
    scrambled = score_sequence_on_contact_map("WYD", cmap)
    assert native["state_contact_score"] > scrambled["state_contact_score"]
    assert native["hotspot_score"] > scrambled["hotspot_score"]


# 基准口径：native 分数不低于乱序均值，native 百分位必须在 [0,1]——防止基准汇总越界或方向颠倒。
def test_benchmark_reports_native_percentile():
    bench = benchmark_native_vs_scrambled(_toy_contact_map(), n_scramble=4, seed=1)
    assert bench["summary"]["native_score"] >= bench["summary"]["scramble_mean_score"]
    assert 0.0 <= bench["summary"]["native_percentile"] <= 1.0


# 片段识别：Aβ 片段必须检出为连续区段 [1,10)，防止片段定位偏移导致链选择错判。
def test_abeta_fragment_info_detects_contiguous_fragment():
    info = abeta_fragment_info("DAEFRHDSGY")
    assert info["is_abeta_like"] is True
    assert info["abeta_start"] == 1
    assert info["abeta_end"] == 10


# 链选择：优先选择 Aβ 样肽链而非短的类 Aβ 片段或长抗体链，防止打分对象选错链。
def test_select_abeta_like_chain_prefers_abeta_over_short_non_abeta():
    chains = {
        "X": [{"aa": a} for a in "EDQIDW"],
        "P": [{"aa": a} for a in "DAEFRHDSGY"],
        "H": [{"aa": "A"} for _ in range(120)],
    }
    selected = select_abeta_like_chain(chains)
    assert selected["chosen_chain"] == "P"


# 热点破坏方向性：破坏热点的突变必须降低接触分——防止打分器把有害突变当改进。
def test_hotspot_breaking_mutant_lowers_toy_score():
    cmap = _toy_contact_map()
    broken = hotspot_breaking_mutant(cmap)
    assert score_sequence_on_contact_map("DYW", cmap)["state_contact_score"] > score_sequence_on_contact_map(broken, cmap)["state_contact_score"]


# 诱饵多样性：基准必须覆盖 scrambled/alanine/glycine/hotspot_breaking/conservative 全部诱饵类型，
# 且保守突变按预期（D→E 等）替换——防止单一诱饵类型虚高特异性结论。
def test_benchmark_native_vs_decoys_includes_multiple_decoy_types():
    bench = benchmark_native_vs_decoys(_toy_contact_map(), n_scramble=4, seed=2)
    types = {d["decoy_type"] for d in bench["decoys"]}
    assert {"scrambled", "alanine", "glycine", "hotspot_breaking", "conservative"}.issubset(types)
    assert conservative_mutant("DYW") == "EFY"


# 变体可排名：接触引导变体必须可全部打分并按 rank=1 起排序——防止变体管道产出不可比对象。
def test_contact_guided_variants_are_rankable():
    cmap = _toy_contact_map()
    variants = generate_contact_guided_variants(cmap, n=5, max_mutations=2, seed=3)
    ranked = score_contact_guided_variants(cmap, variants)
    assert len(ranked) == 5
    assert ranked[0]["rank"] == 1
    assert "candidate_score" in ranked[0]


# 约束遵守：allowed_positions 限定时所有突变只落在允许位（首尾残基不变），防止变异越出授权位点。
def test_contact_guided_variants_respect_allowed_positions():
    cmap = _toy_contact_map()
    variants = generate_contact_guided_variants(
        cmap, n=5, max_mutations=8, seed=3, allowed_positions=[1]
    )
    assert len(variants) == 5
    assert all(v["n_mutations"] == 1 for v in variants)
    assert all(v["sequence"][0] == "D" and v["sequence"][2] == "W" for v in variants)


# 约束校验：空允许集、越界位点、max_mutations=0 都必须显式报错——防止非法约束静默产出空变体。
def test_contact_guided_variants_reject_invalid_constraints():
    cmap = _toy_contact_map()
    import pytest

    with pytest.raises(ValueError, match="at least one position"):
        generate_contact_guided_variants(cmap, allowed_positions=[])
    with pytest.raises(ValueError, match="out-of-range"):
        generate_contact_guided_variants(cmap, allowed_positions=[3])
    with pytest.raises(ValueError, match="max_mutations"):
        generate_contact_guided_variants(cmap, max_mutations=0)
