"""状态对比（state contrast）A/B 工作流的契约测试：守护负状态可打分、正负态特异性差距、
候选排名、接触变量归因分类与引导变体修复、工作流阶段/声明等级的保守性、
已验证参考计数审计，以及真实系综模式对缺失 PDB 的阻塞。"""
from idp_state_ensemble import build_negative_states, positive_state
from state_contact_scorer import Contact, score_sequence_on_contact_map
from state_contrast_scorer import rank_state_contrast_candidates, score_state_contrast
from statecontrast_workflow import integration_claim_level, readiness_requirements, validate_reference_count, workflow_stages
from statecontrast_real_ensembles import real_ensemble_mode_status
from contact_variable_attribution import (
    chemistry_class,
    compile_guidance_rules,
    generate_attribution_guided_variants,
    residue_matches_classes,
    volume_class,
    compare_high_low_gap,
)


# 辅助函数：构造小型接触图（paratope YWK × 肽 FD，三条 4.0-4.5 Å 接触）。
def toy_contact_map():
    return {
        "paratope_sequence": "YWK",
        "peptide_sequence": "FD",
        "paratope_residues": [
            {"chain": "H", "resid": 31, "aa": "Y"},
            {"chain": "H", "resid": 32, "aa": "W"},
            {"chain": "H", "resid": 33, "aa": "K"},
        ],
        "contacts": [
            Contact(0, "H", 31, "Y", 0, "E", 1, "F", 4.0),
            Contact(1, "H", 32, "W", 0, "E", 1, "F", 4.5),
            Contact(2, "H", 33, "K", 1, "E", 2, "D", 4.0),
        ],
    }


# 负状态可打分：构造的 3 个负状态接触图都能对 native 序列正常打分——防止负例构造损坏结构。
def test_negative_states_are_scoreable():
    cmap = toy_contact_map()
    neg = build_negative_states(cmap, "toy")
    assert len(neg) == 3
    for state in neg:
        score = score_sequence_on_contact_map("YWK", state.contact_map)
        assert "state_contact_score" in score


# 特异性差距：正态分数必须为正且结果含 specificity_gap 字段——防止正负对比退化为单一绝对分。
def test_state_contrast_scores_gap():
    cmap = toy_contact_map()
    pos = [positive_state("toy_pos", cmap)]
    neg = build_negative_states(cmap, "toy")
    score = score_state_contrast("YWK", pos, neg)
    assert score["positive_score"] > 0
    assert "specificity_gap" in score


# 排名方向：native 的状态对比排名为 1、弱序列靠后——防止对比分数排序方向写反。
def test_rank_state_contrast_candidates():
    cmap = toy_contact_map()
    pos = [positive_state("toy_pos", cmap)]
    neg = build_negative_states(cmap, "toy")
    variants = [
        {"candidate_id": "native", "sequence": "YWK"},
        {"candidate_id": "weak", "sequence": "AAA"},
    ]
    ranked = rank_state_contrast_candidates(variants, pos, neg)
    assert ranked[0]["statecontrast_rank"] == 1
    assert len(ranked) == 2


# 归因分类锚点：体积/化学类映射符合直觉（G 小、W 大、Y 芳香、K 正电），防止规则引擎分类漂移。
def test_contact_variable_attribution_classes():
    assert volume_class("G") == "small"
    assert volume_class("W") == "large"
    assert chemistry_class("Y") == "aromatic"
    assert chemistry_class("K") == "positive"


# 高低分组：按 specificity_gap 中位切分后高/低组各 1 条并产出特征行——防止归因分析丢样本。
def test_compare_high_low_gap_outputs_features():
    cmap = toy_contact_map()
    rows = [
        {"candidate_id": "a", "sequence": "YWK", "specificity_gap": 1.0},
        {"candidate_id": "b", "sequence": "AAA", "specificity_gap": -1.0},
    ]
    out = compare_high_low_gap(rows, cmap, top_fraction=0.5)
    assert out["n_high"] == 1
    assert out["n_low"] == 1
    assert out["feature_rows"]


# 规则门槛：只有 |fraction_delta| ≥0.20 的信号才进规则（0.25 入选、-0.19 淘汰），
# 且规则按 paratope 位置索引——防止弱信号制造噪声引导。
def test_compile_guidance_rules_uses_only_threshold_signals():
    rows = [
        {
            "reference_pdb": "toy",
            "paratope_index": 1,
            "chain": "H",
            "resid": 31,
            "volume_class": "large",
            "chemistry_class": "aromatic",
            "fraction_delta": 0.25,
        },
        {
            "reference_pdb": "toy",
            "paratope_index": 2,
            "chain": "H",
            "resid": 32,
            "volume_class": "small",
            "chemistry_class": "flexible",
            "fraction_delta": -0.19,
        },
    ]
    rules = compile_guidance_rules(rows, min_abs_delta=0.20)
    assert list(rules["toy"].keys()) == [0]
    assert rules["toy"][0]["preferred"][0]["chemistry_class"] == "aromatic"


# 修复方向：引导变体的目标位置必须落在偏好类（large+aromatic）内——防止"归因引导"随机突变。
def test_attribution_guided_variants_repair_preferred_class():
    cmap = toy_contact_map()
    rules = {
        0: {
            "preferred": [{"volume_class": "large", "chemistry_class": "aromatic", "fraction_delta": 0.3}],
            "avoid": [],
        }
    }
    parents = [{"candidate_id": "bad_parent", "sequence": "AWK"}]
    variants = generate_attribution_guided_variants(cmap, parents, rules, n=4, max_mutations=2, seed=1)
    assert variants
    assert all(residue_matches_classes(v["sequence"][0], "large", "aromatic") for v in variants)


# 工作流契约：首阶段为 domain_intake、GPU 阶段显式标记，CPU-only 支持基准对照且必须含 parent_only 对照——
# 防止控制组缺失或阶段声明漂移。
def test_statecontrast_workflow_contract_marks_cpu_and_gpu_stages():
    stages = workflow_stages()
    assert stages[0]["stage"] == "domain_intake"
    assert any(s["stage"] == "fold_sidecheck" and s["gpu_required"] for s in stages)
    req = readiness_requirements()
    assert req["cpu_only_supported_for_statecontrast_benchmark"] is True
    assert "parent_only" in req["required_controls"]


# 声明保守：最小支持度下只允许 proof_of_concept，充分支持+双验证才允许 validated_domain_workflow——
# 防止声明等级越阶。
def test_integration_claim_level_is_conservative():
    assert integration_claim_level(2, False, False) == "proof_of_concept_position_discovery"
    assert integration_claim_level(8, True, True) == "validated_domain_workflow"


# 参考审计：未审核或可靠性存疑的参考不计入有效数（3 申明→1 有效→fail）——防止引用未验证结构。
def test_verified_reference_count_excludes_unreviewed_entries():
    refs = [
        {"pdb": "GOOD", "reliability": "reliable", "provenance_verified": True},
        {"pdb": "UNREVIEWED", "reliability": "reliable"},
        {"pdb": "BAD", "reliability": "questionable", "provenance_verified": True},
    ]
    result = validate_reference_count(refs, minimum=2, require_verified=True)
    assert result["n_declared_references"] == 3
    assert result["n_references"] == 1
    assert result["status"] == "fail"


# 审计放行：全部可靠且溯源已验证的参考达到最小值时通过——保证审计门槛双向有效。
def test_verified_reference_count_accepts_reviewed_entries():
    refs = [
        {"pdb": str(index), "reliability": "reliable", "provenance_verified": True}
        for index in range(4)
    ]
    result = validate_reference_count(refs, minimum=4, require_verified=True)
    assert result["n_references"] == 4
    assert result["status"] == "pass"


# 系综阻塞：任一系综 PDB 缺失时整体 blocked，且就绪条目在无姿态时接触打分也标记 blocked——
# 防止对缺失结构静默跳过或伪造状态证据。
def test_real_ensemble_status_reports_missing_and_ready_entries(tmp_path):
    ready = tmp_path / "ready.pdb"
    ready.write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 20.00           C  \n",
        encoding="utf-8",
    )
    cfg = {
        "negative_states": {
            "mode": "structure_ensemble",
            "ensembles": [
                {"name": "ready", "pdb": "ready.pdb", "chain": "A", "state_type": "fibril"},
                {"name": "missing", "pdb": "missing.pdb", "chain": "A", "state_type": "oligomer"},
            ],
        }
    }
    status = real_ensemble_mode_status(cfg, tmp_path)
    assert status["ready"] == 1
    assert status["missing"] == 1
    assert status["status"] == "blocked_missing_or_invalid_ensemble_pdbs"
    assert status["entries"][0]["contact_scoring_status"] == "blocked_without_pose"
