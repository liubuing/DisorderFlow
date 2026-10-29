"""候选界面 PAE 部署冻结脚本的契约测试：守护 std 匹配校准的斜率/截距恢复、
Spearman 相关性的单调方向、以及按支架分组的组内相关性只保留含多设计的支架。"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

# 模块级加载：冻结脚本不在包内，通过 importlib 按路径导入一次供所有测试复用。
SCRIPT = Path(__file__).parents[1] / "scripts" / "freeze_candidate_interface_pae_deployment.py"
SPEC = importlib.util.spec_from_file_location("freeze_pae_deploy", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


# 尺度校准：std 匹配必须恢复线性关系的斜率（2.5）且校正后方差与目标一致，防止部署分数尺度失真。
def test_fit_std_matching_reaches_unit_variance_ratio():
    pred = np.asarray([0.1, 0.2, 0.3, 0.4])
    target = pred * 2.5 + 0.3
    slope, intercept = MODULE.fit_std_matching(pred, target)
    corrected = slope * pred + intercept
    assert np.std(corrected) == pytest.approx(np.std(target), rel=1e-6)
    assert slope == pytest.approx(2.5)


# 偏移校正：纯平移目标下 intercept 必须完全移除均值偏移，防止校准后分数系统性偏高/偏低。
def test_fit_std_matching_removes_mean_offset():
    pred = np.asarray([0.1, 0.2, 0.3])
    target = pred + 5.0
    slope, intercept = MODULE.fit_std_matching(pred, target)
    assert np.mean(slope * pred + intercept) == pytest.approx(np.mean(target), rel=1e-6)


# 相关性符号：同向单调为 +1、反向为 -1，防止实现返回绝对值或错置符号。
def test_spearman_monotone():
    left = [1.0, 2.0, 3.0, 4.0]
    right = [10.0, 20.0, 30.0, 40.0]
    assert MODULE.spearman(left, right) == pytest.approx(1.0)
    assert MODULE.spearman(left, list(reversed(right))) == pytest.approx(-1.0)


# 分组口径：单设计支架（C）不产生组内相关性，结果只含 A/B——防止单样本"相关性"混入汇总。
def test_scaffold_spearman_groups_by_scaffold():
    pred = [0.1, 0.3, 0.5, 0.7, 0.9]
    target = [0.1, 0.2, 0.4, 0.6, 0.8]
    scaffolds = ["A", "A", "B", "B", "C"]
    result = MODULE.scaffold_spearman(pred, target, scaffolds)
    assert set(result) == {"A", "B"}
