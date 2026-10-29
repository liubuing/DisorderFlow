"""后续实验科学契约的冻结产物审计：守护证据清单的哈希一致性、T2.1 机制的终局阴性判定、
ESM-IF 局部序列掩码与固定位置校验、删失值不得升级为 Kd，以及多支架 v2 留出集冻结、
训练谱系隔离与结构评估"完整但主门槛失败"的记录不被重写。"""
import hashlib
import json
import pickle
from pathlib import Path

import pytest

from scripts.benchmark_h3_candidate_reranking import (
    build_esmif_partial_sequence,
    validate_esmif_sample,
)

ROOT = Path(__file__).resolve().parents[1]


# 哈希校验：证据清单中每个文件的 sha256 必须与磁盘逐字节一致，防止证据文件被静默替换。
def test_followup_evidence_manifest_hashes_match_files():
    manifest = json.loads(
        (ROOT / "publication/followup_evidence_manifest.json").read_text())
    for item in manifest["files"]:
        path = ROOT / item["path"]
        assert path.is_file(), item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]


# 终局性契约：机制不被支持、对当前机制为终局阴性（主对比均值与 CI 上界均 <0），
# 且禁止用同一终局设计重跑——防止"再试一次"式的事后翻案。
def test_t21_current_mechanism_is_terminal_negative():
    decision = json.loads(
        (ROOT / "publication/t21_mechanism_terminal_decision.json").read_text())
    contrast = decision["primary_contrast"]
    assert decision["mechanism_supported"] is False
    assert decision["terminal_for_current_mechanism"] is True
    assert decision["rerun_existing_final_permitted"] is False
    assert contrast["mean"] < 0
    assert contrast["ci95"][1] < 0


# 局部序列构造：仅设计位置替换为 <mask>，固定位置原样保留，并按总长补 <pad>，
# 防止送入 ESM-IF 的条件序列意外覆盖支架。
def test_esmif_partial_sequence_masks_only_design_positions():
    partial = build_esmif_partial_sequence("ACDEFG", [1, 4], 9)
    assert partial == ["A", "<mask>", "D", "E", "<mask>", "G",
                       "<pad>", "<pad>", "<pad>"]


# 固定位置校验：ESM-IF 采样在非设计位点的任何改变必须报错，防止重设计越出授权区间。
def test_esmif_rejects_changes_to_fixed_heavy_positions():
    validate_esmif_sample("AYDEFQ", "ACDEFG", [1, 5])
    with pytest.raises(ValueError, match="fixed heavy-chain positions"):
        validate_esmif_sample("AYXEFQ", "ACDEFG", [1, 5])


# 删失纪律：实验突变面板只允许 exact_* 提取层级；带删失标记的记录不得携带数值、
# 不得标注为 Kd 终点——防止区间/上限估计被当作点估计混入定量分析。
def test_mutation_panel_does_not_promote_censored_values_to_kd():
    panel = json.loads(
        (ROOT / "publication/experimental_mutations_v1.json").read_text())
    assert panel["records"]
    for record in panel["records"]:
        assert record["extraction_tier"].startswith("exact_")
        if "censoring" in record:
            assert "value" not in record
            assert record["endpoint"] != "KD_M"


# 留出集冻结：留出分类声明为"前瞻性内部留出而非外部确认"，20 个独立成分各取一个代表，
# 代表 ID 哈希与清单一致且全部来自可行性审计提出的独立成分集合。
def test_multiscaffold_v2_holdout_is_frozen_from_independent_components():
    holdout = json.loads((
        ROOT / "data/multiscaffold_confirmatory_v2/holdout_manifest.json"
    ).read_text())
    feasibility = json.loads((
        ROOT / "data/multiscaffold_confirmatory_v2/feasibility_audit.json"
    ).read_text())
    representative_hash = hashlib.sha256(
        "\n".join(sorted(holdout["representative_ids"])).encode()).hexdigest()
    proposed = feasibility["schemes"][feasibility["proposed_scheme"]]
    assert holdout["classification"] == (
        "prospective_internal_holdout; not external confirmation")
    assert holdout["n_all_axis_components"] == 20
    assert holdout["n_representatives"] == 20
    assert representative_hash == holdout["representative_ids_sha256"]
    assert set(holdout["representative_ids"]) <= set(proposed["independent_ids"])


# 谱系隔离：两阶段训练契约逐项核对（A 阶段随机初始化不冻结主干、B 阶段唯一初始化来源是 A checkpoint），
# 无序度查找表 827 条 profile 与审计排除记录严格互斥。
def test_multiscaffold_v2_training_lineage_is_isolated():
    contract = json.loads((
        ROOT / "results/ablation/multiscaffold_v2_training_contract.json"
    ).read_text())
    lookup_audit = json.loads((
        ROOT / "data/multiscaffold_confirmatory_v2/disorder_lookup_audit.json"
    ).read_text())
    lookup = pickle.loads((
        ROOT / "data/disorder_supervision/train_afdb_multiscaffold_v2.pkl"
    ).read_bytes())
    assert contract["status"] == "two_stage_training_contract_verified"
    assert contract["stage_a"]["initialization"] == "random"
    assert contract["stage_a"]["freeze_backbone"] is False
    assert contract["stage_a"]["torch_load_guard_smoke_passed"] is True
    assert contract["stage_a"]["one_step_forward_backward_smoke_passed"] is True
    assert contract["stage_b"]["only_permitted_initializer"] == "stage_a_checkpoint"
    assert len(lookup["profiles"]) == 827
    assert lookup_audit["n_excluded_records"] == 1
    assert not set(lookup_audit["excluded_ids"]) & set(lookup["profiles"])


# 结果终局性：评估槽位完整（AF2 960/960、ProDIGY 959 成功 1 失败）但主门槛必须保持失败
# （7 个有效成分、平均 delta 与 bootstrap CI 上界均 <0、门槛全未通过）——防止负面结果被事后改写为通过。
def test_multiscaffold_v2_structural_evaluation_is_complete_but_primary_gate_failed():
    af2 = json.loads((
        ROOT / "results/multiscaffold_confirmatory_v2/af2/results.json"
    ).read_text())
    prodigy = json.loads((
        ROOT / "results/multiscaffold_confirmatory_v2/prodigy/results.json"
    ).read_text())
    analysis = json.loads((
        ROOT / "results/multiscaffold_confirmatory_v2/analysis.json"
    ).read_text())
    assert af2["summary"] == {
        "expected_prediction_slots": 960,
        "recorded_prediction_slots": 960,
        "successes": 960,
        "failures": 0,
    }
    assert prodigy["summary"]["recorded_slots"] == 960
    assert prodigy["summary"]["successes"] == 959
    assert prodigy["summary"]["failures"] == 1
    assert analysis["status"] == "primary_gate_failed"
    assert analysis["primary"]["passed"] is False
    assert analysis["primary"]["valid_components"] == 7
    assert analysis["primary"]["mean_positive_delta"] < 0
    assert analysis["primary"]["component_bootstrap_ci95"][1] < 0
    assert not any(analysis["primary"]["gate_results"].values())
