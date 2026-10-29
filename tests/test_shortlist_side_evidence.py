"""短名单旁证流水线测试：验证可变区边界、N-糖基化旁证、状态对比消融与 GPU 就绪诊断。

守护 scripts/pipeline 下 shortlist/statecontrast 系列脚本：
可变区切割必须以 J 基序为界、N-糖基化标记须定位到可变区、
v2 状态机阻断需复核候选、导出优先选择真实接触的抗体链、
过拟合对照（随机/打乱规则）与复现聚合、稳定归因的重复方向要求、
突变效应的 present/absent 对比与预算上限、GPU 驱动/运行时状态区分、
以及 aβ 声明的保守性（参考不足时 claim_level 必须是 exploratory_only）。
"""
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


# 辅助函数：以独立模块方式加载 scripts/pipeline 下的脚本（并把目录加入 sys.path）。
def load_script(name):
    path = ROOT / "scripts" / "pipeline" / f"{name}.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 可变区边界：必须由 J 基 motif（WGQGTLVTVSS）收口，CH1 起始段进入 trailing——
# 防止按固定长度切割错位 CDR 区。
def test_variable_region_boundary_uses_j_motif():
    mod = load_script("analyze_shortlist_variable_regions")
    seq = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSNYAMSWVRQAPGKGLEWVSA" \
          "INASGTRTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARGKGYVRYFDV" \
          "WGQGTLVTVSSASTKGPSVF"
    cut = mod.find_variable_region(seq, "heavy")
    assert cut["boundary_status"] == "j_motif_found"
    assert cut["variable_seq"].endswith("WGQGTLVTVSS")
    assert cut["trailing_seq"].startswith("ASTK")


# N-糖基化定位：motif 落在可变区内必须归为 variable（而非恒定区）——
# 防止旁证标记错位误导可开发性审查。
def test_nglyco_flag_localizes_to_variable_region():
    mod = load_script("analyze_shortlist_variable_regions")
    variable = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAINASGTRTYYADSV"
    trailing = "ASTKGPSVF"
    region = mod.flag_region("n_glycosylation_motif", variable + trailing, variable, trailing)
    assert region == "variable"


# 状态机阻断：可变区需复核的候选不得直接通过旁证——防止未审候选进入折叠队列。
def test_v2_status_blocks_variable_region_review():
    mod = load_script("build_shortlist_v2_side_evidence")
    dev = {"developability_status": "developability_pass", "introduced_or_candidate_specific_flags": ""}
    var = {"variable_region_status": "variable_region_review"}
    assert mod.evidence_status(dev, var) == "variable_region_developability_review"


# 干净候选放行：可开发性通过且旗标都在可变区之外时可进入 Fv 折叠旁检——
# 防止状态机过严或过宽（回归守卫）。
def test_v2_status_allows_clean_candidate_for_fold_queue():
    mod = load_script("build_shortlist_v2_side_evidence")
    dev = {"developability_status": "developability_pass", "introduced_or_candidate_specific_flags": ""}
    var = {"variable_region_status": "inherited_flags_outside_variable_region"}
    assert mod.evidence_status(dev, var) == "ready_for_fv_fold_sidecheck"


# 接触链归属：导出构建体时必须优先选择真正参与接触的重/轻链副本——
# 防止导出错误链对导致旁证建立在无关链上。
def test_export_prefers_contacting_duplicate_antibody_chains():
    mod = load_script("export_full_chain_constructs")
    cmap = {"paratope_residues": [{"chain": "A", "resid": 52}, {"chain": "B", "resid": 93}]}
    roles = {"H": "H", "L": "L", "A": "H", "B": "L"}
    assert mod.contact_chain_roles(cmap, roles) == {"H": "A", "L": "B"}


# 5CSZ 挽救：A54 位置需在 N-糖基化 motif 上下文中判定（N-X-S/T 序量）——
# 防止仅凭残基编号误报/漏报糖基化风险。
def test_5csz_rescue_detects_a54_ngt_context():
    mod = load_script("rescue_5csz_nglyco_candidates")
    seq = "SYAAHANRYGKGYVRTIYNMPI"
    idx_by_residue = {("A", 54): 6}
    assert mod.has_nglyco_motif(seq, idx_by_residue)


# 状态对比 delta：high/low gap 组同位置分数差及按体积类聚合必须一致可复算——
# 防止方向（high_gap_enriched）或均值计算写反。
def test_statecontrast_variable_deltas_compare_high_and_low_gap():
    mod = load_script("summarize_statecontrast_variables")
    rows = [
        {
            "reference_pdb": "toy",
            "paratope_index": "1",
            "chain": "H",
            "resid": "31",
            "volume_class": "large",
            "chemistry_class": "aromatic",
            "group": "high_gap",
            "fraction": "0.8",
        },
        {
            "reference_pdb": "toy",
            "paratope_index": "1",
            "chain": "H",
            "resid": "31",
            "volume_class": "large",
            "chemistry_class": "aromatic",
            "group": "low_gap",
            "fraction": "0.3",
        },
    ]
    deltas = mod.compute_deltas(rows)
    assert deltas == [
        {
            "reference_pdb": "toy",
            "paratope_index": "1",
            "chain": "H",
            "resid": "31",
            "volume_class": "large",
            "chemistry_class": "aromatic",
            "high_fraction": 0.8,
            "low_fraction": 0.3,
            "fraction_delta": 0.5,
            "direction": "high_gap_enriched",
        }
    ]
    assert mod.aggregate_by_volume(deltas) == [
        {
            "reference_pdb": "toy",
            "volume_class": "large",
            "mean_delta": 0.5,
            "max_abs_delta": 0.5,
            "n_positions": 1,
        }
    ]


# 过拟合对照：打乱规则必须把规则搬到新位置且保留规则内容——保证阴性对照有效。
def test_overfit_control_shuffles_single_rule_to_new_position():
    mod = load_script("run_statecontrast_ab_overfit_controls")
    rng = __import__("random").Random(1)
    rules = {2: {"preferred": [{"volume_class": "large", "chemistry_class": "aromatic", "fraction_delta": 0.2}], "avoid": []}}
    shuffled = mod.shuffled_position_rules(rules, sequence_len=5, rng=rng)
    assert list(shuffled.keys()) != [2]
    assert list(shuffled.values())[0]["preferred"][0]["chemistry_class"] == "aromatic"


# 复现聚合：guided 必须同时胜过 parent/random/shuffled 三个对照才算赢，
# 仅 1/3 种子全胜时 verdict 必须是 mixed_or_overfit_risk——防止过拟合信号被当有效。
def test_replicate_aggregation_requires_beating_both_controls():
    mod = load_script("run_statecontrast_ab_overfit_replicates")
    rows = []
    for seed, guided, parent, random_rule, shuffled in [
        (1, 0.5, 0.2, 0.4, 0.3),
        (2, 0.5, 0.2, 0.6, 0.3),
        (3, 0.5, 0.2, 0.4, 0.6),
    ]:
        rows.extend([
            {"seed": seed, "reference_pdb": "toy", "arm": "guided_v3_heldout_eval", "top10_mean_gap_delta": guided},
            {"seed": seed, "reference_pdb": "toy", "arm": "top_parent_only_heldout_eval", "top10_mean_gap_delta": parent},
            {"seed": seed, "reference_pdb": "toy", "arm": "random_rule_v3_heldout_eval", "top10_mean_gap_delta": random_rule},
            {"seed": seed, "reference_pdb": "toy", "arm": "shuffled_rule_v3_heldout_eval", "top10_mean_gap_delta": shuffled},
            {"seed": seed, "reference_pdb": "toy", "arm": "v2_discovery_pool_heldout_eval", "top10_mean_gap_delta": 0.1},
        ])
    summary = mod.summarize_metric("toy", rows, "top10_mean_gap_delta")
    assert summary["guided_beats_parent_random_and_shuffled_rate"] == 0.3333
    assert summary["verdict"] == "mixed_or_overfit_risk"


# 稳定归因：只有同方向复现 >= min_recurrence 次的信号才算稳定（反向样本破坏复现）——
# 防止把噪声位点的单次大 delta 当作稳定规则。
def test_stable_attribution_requires_recurrent_same_direction_signal():
    mod = load_script("run_statecontrast_ab_stable_attribution")
    rows = [
        {
            "seed": 1,
            "reference_pdb": "toy",
            "paratope_index": 3,
            "chain": "H",
            "resid": 52,
            "volume_class": "large",
            "chemistry_class": "aromatic",
            "fraction_delta": 0.25,
        },
        {
            "seed": 2,
            "reference_pdb": "toy",
            "paratope_index": 3,
            "chain": "H",
            "resid": 52,
            "volume_class": "large",
            "chemistry_class": "aromatic",
            "fraction_delta": 0.21,
        },
        {
            "seed": 3,
            "reference_pdb": "toy",
            "paratope_index": 3,
            "chain": "H",
            "resid": 52,
            "volume_class": "large",
            "chemistry_class": "aromatic",
            "fraction_delta": -0.30,
        },
    ]
    stable = mod.stable_signal_rows(rows, "toy", min_recurrence=2)
    assert len(stable) == 1
    assert stable[0]["direction"] == "high_gap_enriched"
    assert stable[0]["recurrence"] == 2


# 突变效应：按 effect_key 聚合 present/absent 的 gap delta（支持数门槛），
# 并映射到对位（paratope）位置——防止把单候选波动当突变效应。
def test_mutation_effects_compare_present_vs_absent_candidates():
    mod = load_script("analyze_statecontrast_mutation_effects")
    rows = [
        {"reference_pdb": "toy", "candidate_id": "a", "mutations": "A1Y", "specificity_gap_delta": "0.4"},
        {"reference_pdb": "toy", "candidate_id": "b", "mutations": "A1Y", "specificity_gap_delta": "0.2"},
        {"reference_pdb": "toy", "candidate_id": "c", "mutations": "", "specificity_gap_delta": "0.0"},
        {"reference_pdb": "toy", "candidate_id": "d", "mutations": "G2A", "specificity_gap_delta": "0.0"},
    ]
    mutation_rows, position_rows = mod.mutation_effects(rows, min_support=2)
    top = mutation_rows[0]
    assert top["effect_key"] == "A:1:Y"
    assert top["present_mean"] == 0.3
    assert top["absent_mean"] == 0.0
    assert position_rows[0]["paratope_index"] == 1


# 效应引导生成：稳定突变规则须跨种子复现且生成变体确实应用规则（首残基变 Y）——
# 防止规则提取或应用断链。
def test_effect_guided_v4_recurrent_mutation_rules_and_generation():
    mod = load_script("run_statecontrast_ab_effect_guided_v4")
    rows = []
    for seed in (1, 2):
        rows.extend([
            {"seed": seed, "candidate_id": f"hit_{seed}_1", "mutations": "A1Y", "specificity_gap_delta": "0.4"},
            {"seed": seed, "candidate_id": f"hit_{seed}_2", "mutations": "A1Y", "specificity_gap_delta": "0.3"},
            {"seed": seed, "candidate_id": f"miss_{seed}_1", "mutations": "", "specificity_gap_delta": "0.0"},
            {"seed": seed, "candidate_id": f"miss_{seed}_2", "mutations": "G2A", "specificity_gap_delta": "0.0"},
        ])
    rules = mod.stable_effect_rules(rows, "toy", min_support=2, min_effect=0.1, min_recurrence=2)
    assert rules[0]["rule_type"] == "mutation"
    assert rules[0]["paratope_index"] == 1
    assert rules[0]["new_aa"] == "Y"
    cmap = {"paratope_sequence": "AG", "contacts": [], "paratope_residues": []}
    variants = mod.generate_effect_guided_variants(cmap, [{"candidate_id": "p", "sequence": "AG"}], rules, n=3, max_mutations=2, seed=1, reference_pdb="toy")
    assert variants
    assert all(v["sequence"][0] == "Y" for v in variants)


# 突变预算：激活规则数受 max_mutations 上限约束——防止规则叠加导致过度突变。
def test_effect_guided_v4_caps_active_rules_to_mutation_budget():
    mod = load_script("run_statecontrast_ab_effect_guided_v4")
    cmap = {"paratope_sequence": "AGST", "contacts": [], "paratope_residues": []}
    rules = [
        {"rule_type": "mutation", "paratope_index": 1, "new_aa": "Y"},
        {"rule_type": "mutation", "paratope_index": 2, "new_aa": "Y"},
        {"rule_type": "mutation", "paratope_index": 3, "new_aa": "Y"},
    ]
    variants = mod.generate_effect_guided_variants(cmap, [{"candidate_id": "p", "sequence": "AGST"}], rules, n=3, max_mutations=2, seed=1, reference_pdb="toy")
    assert variants
    assert all(int(v["n_mutations"]) <= 2 for v in variants)


# 报告排序：规则汇总必须按 reference 分组且位内数字排序（"9" 在 "12" 前）——
# 防止字符串排序导致顺序错乱。
def test_final_report_summarizes_position_rules_sorted_by_reference():
    mod = load_script("build_statecontrast_ab_final_report")
    rows = [
        {"reference_pdb": "5CSZ", "paratope_index": "12"},
        {"reference_pdb": "5CSZ", "paratope_index": "9"},
        {"reference_pdb": "4HIX", "paratope_index": "19"},
        {"reference_pdb": "4HIX", "paratope_index": "3"},
    ]
    assert mod.summarize_rules(rows) == {"5CSZ": ["9", "12"], "4HIX": ["3", "19"]}


# GPU 就绪区分：驱动可用但 Python 侧 CUDA 运行时不可用时，
# overall_status 必须精确表达这种半就绪状态——防止误判环境可跑 GPU 任务。
def test_gpu_readiness_distinguishes_driver_from_python_runtime():
    mod = load_script("check_gpu_readiness")
    checks = [
        {"name": "nvidia_smi", "required": False, "status": "pass"},
        {"name": "torch_cuda", "required": False, "status": "pass", "stdout_tail": "2.12.0+cpu\nFalse\n0"},
        {"name": "colabfold_executable", "required": False, "status": "pass"},
        {"name": "project_file:x", "required": True, "status": "pass"},
    ]
    summary = mod.summarize_checks(checks)
    assert summary["overall_status"] == "gpu_driver_ready_python_runtime_not_ready"
    assert summary["gpu_driver_detected"] is True
    assert summary["python_gpu_runtime_detected"] is False


# 声明保守性：声明参考未通过校验（0/2 实际可用）时 status=fail，
# claim_level 必须降级为 exploratory_only 且保留 parent_only 对照——
# 防止参考证据不足时越权声称 aβ 定向结果。
def test_domain_workflow_metadata_keeps_abeta_claim_conservative():
    mod = load_script("run_statecontrast_domain_workflow")
    cfg = mod.load_config(str(ROOT / "configs" / "statecontrast" / "abeta_position_effect_v1.yml"))
    metadata = mod.workflow_metadata(cfg)
    assert metadata["domain"] == "idp_antibody"
    assert metadata["reference_check"]["n_declared_references"] == 2
    assert metadata["reference_check"]["n_references"] == 0
    assert metadata["reference_check"]["status"] == "fail"
    assert metadata["claim_level"] == "exploratory_only"
    assert "parent_only" in metadata["controls"]
