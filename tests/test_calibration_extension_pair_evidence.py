"""候选界面校准扩展的配对证据统计测试：验证可靠配对筛选与重复测量聚合。

守护校准扩展证据管线：
pair_outcomes 只计入目标-噪声差值超过噪声底的可信配对（防止平局灌水）、
pair_statistics 暴露支架家族集中度门槛字段、
aggregate_conditions 按构建体对重复测量取均值后再参与评估。
"""
import importlib.util
from pathlib import Path

import pytest

# 模块级加载被测脚本（evaluate_candidate_interface_calibration_extension.py）。
SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_candidate_interface_calibration_extension.py"
SPEC = importlib.util.spec_from_file_location("ext_pair_evidence", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


# 差值小于可靠性阈值的配对必须被剔除（第三行差值过小不计入）——
# 防止噪声级差异被当作有效证据放大目标优于随机的结论。
def test_pair_outcomes_counts_only_reliable_deltas():
    rows = [
        {"target_iptm": 0.80, "noise_iptm": 0.01},
        {"target_iptm": 0.70, "noise_iptm": 0.01},
        {"target_iptm": 0.805, "noise_iptm": 0.01},
    ]
    outcomes = MODULE.pair_outcomes(rows, "iptm")
    assert len(outcomes) == 2


# 配对统计必须暴露可靠配对数、贡献支架数与最大支架配对占比，
# 支撑"证据不集中于单一支架家族"的集中度门槛判断。
def test_pair_statistics_gate_fields():
    rows = [
        {"scaffold_family": "S1", "target_iptm": 0.8, "noise_iptm": 0.01},
        {"scaffold_family": "S1", "target_iptm": 0.7, "noise_iptm": 0.01},
        {"scaffold_family": "S1", "target_iptm": 0.6, "noise_iptm": 0.01},
        {"scaffold_family": "S2", "target_iptm": 0.5, "noise_iptm": 0.01},
        {"scaffold_family": "S2", "target_iptm": 0.3, "noise_iptm": 0.01},
    ]
    stats = MODULE.pair_statistics(rows, "iptm")
    assert stats["reliable_pairs"] == 4
    assert stats["contributing_scaffolds"] == 2
    assert stats["max_scaffold_pair_fraction"] == pytest.approx(3 / 4)


# 同一构建体的多次测量须先按条件聚合（均值）再进入统计，
# 防止重复测量被当作独立样本导致 p 值虚低。
def test_aggregate_conditions_means_replicates():
    rows = [
        {"construct_id": "C1", "scaffold_family": "S1",
         "target_iptm": 0.8, "noise_iptm": 0.02, "target_plddt": 0.9, "noise_plddt": 0.03,
         "target_pae": 0.1, "noise_pae": 0.01},
        {"construct_id": "C1", "scaffold_family": "S1",
         "target_iptm": 0.6, "noise_iptm": 0.02, "target_plddt": 0.7, "noise_plddt": 0.03,
         "target_pae": 0.3, "noise_pae": 0.01},
    ]
    aggregated = MODULE.aggregate_conditions(rows)
    assert len(aggregated) == 1
    assert aggregated[0]["target_iptm"] == pytest.approx(0.7)
    assert aggregated[0]["noise_iptm"] == pytest.approx(0.02)
