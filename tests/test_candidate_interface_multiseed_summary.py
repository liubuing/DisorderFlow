"""候选界面多种子汇总的契约测试：守护 std/方差两种离散度口径、按"支架再按实体"的配对自助重采样、
未定义相关性在模型选择键中必须视为失败，以及配对证据的最低支架数与集中度上限。"""
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


# 辅助函数：按路径动态加载多种子汇总脚本模块。
def load_script():
    path = ROOT / "scripts/summarize_candidate_interface_multiseed.py"
    spec = importlib.util.spec_from_file_location(
        "summarize_candidate_interface_multiseed", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 口径区分：std 比值与方差比值必须各按自身口径计算（0.5 vs 0.25），防止二者混用导致不确定性被错误折算。
def test_dispersion_ratios_distinguish_std_and_variance():
    module = load_script()
    std_ratio, variance_ratio = module.dispersion_ratios(1.0, 2.0)
    assert std_ratio == pytest.approx(0.5)
    assert variance_ratio == pytest.approx(0.25)


# 自助重采样单元必须是 scaffold_then_entity（先重采支架再重采实体），避免同一支架的实体被当作独立样本，
# 并要求贡献支架数、单支架配对占比等统计量随重采样落入合理区间。
def test_pair_bootstrap_resamples_scaffolds_and_entities():
    module = load_script()
    records = []
    for scaffold, predictions in (("A", [0.1, 0.2, 0.3]),
                                  ("B", [0.3, 0.2, 0.1])):
        for index, prediction in enumerate(predictions):
            records.append({
                "scaffold_family": scaffold,
                "target_plddt": float(index),
                "pred_plddt": prediction,
                "noise_plddt": 0.0,
            })
    result = module.pair_statistics(records, "plddt", bootstrap_seed=17)
    assert result["pairs"] == 6
    assert result["contributing_scaffolds"] == 2
    assert result["max_scaffold_pair_fraction"] == pytest.approx(0.5)
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["bootstrap_unit"] == "scaffold_then_entity"
    assert 0.0 <= result["bootstrap_95_lower"] <= 1.0


# 选择安全：组内 Spearman 为 None（未定义）时，选择键首元素必须为 -inf，
# 防止相关缺失的指标被当作最优模型静默选中。
def test_selection_key_treats_undefined_correlation_as_failure():
    module = load_script()
    evaluation = {"metrics": {}}
    for name in module.OUTPUTS:
        evaluation["metrics"][name] = {
            "entity_aggregated": {
                "within_scaffold": {"mean_group_spearman": None},
                "mae": 0.0,
            },
        }
    key = module.selection_key(evaluation)
    assert key[0] == float("-inf")


# 证据强度常量：至少 30 对、3 个支架、单支架占比 ≤0.5；并用三支架均衡样本验证统计量满足这些约束，
# 防止阈值放松后单支架数据冒充可靠配对证据。
def test_pair_evidence_requires_multiple_scaffolds_without_concentration():
    module = load_script()
    assert module.MIN_RELIABLE_PAIRS == 30
    assert module.MIN_PAIR_SCAFFOLDS == 3
    assert module.MAX_SCAFFOLD_PAIR_FRACTION == pytest.approx(0.5)

    records = []
    for scaffold, size in (("A", 9), ("B", 7), ("C", 6)):
        for index in range(size):
            records.append({
                "scaffold_family": scaffold,
                "target_iptm": float(index),
                "pred_iptm": float(index),
                "noise_iptm": 0.0,
            })
    result = module.pair_statistics(records, "iptm", bootstrap_seed=23)
    assert result["pairs"] >= module.MIN_RELIABLE_PAIRS
    assert result["contributing_scaffolds"] == 3
    assert result["max_scaffold_pair_fraction"] <= 0.5
