"""IBEC E1 生成器行为分析测试：验证多样性度量与解释决策的语义。

守护 scripts.analyze_ibec_e1_generator_behavior：
position_entropy 对完全保守序列为 0、对多样序列为正、
Hamming 距离归一化到 [0,1]、
解释器在多样性占优但可开发性显著回退时必须给出带权衡的决策而非无条件扩探索。
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# 辅助函数：以独立模块方式加载被测分析脚本。
def load_script():
    path = ROOT / "scripts" / "analyze_ibec_e1_generator_behavior.py"
    spec = importlib.util.spec_from_file_location("analyze_ibec_e1_generator_behavior", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 位置熵方向：完全保守（两序列相同）熵为 0、逐位不同熵为正——
# 防止熵计算实现反转或常数化导致多样性结论失真。
def test_position_entropy_detects_diversity():
    mod = load_script()
    assert mod.position_entropy(["AAAA", "AAAA"]) == 0.0
    assert mod.position_entropy(["AAAA", "CCCC"]) > 0.0


# 归一化：Hamming 差异必须按长度归一（2/4=0.5），完全不同序列均值为 1——
# 防止未归一化的计数被当作比例参与跨长度比较。
def test_pairwise_hamming_is_normalized():
    mod = load_script()
    assert mod.hamming_fraction("AAAA", "AACC") == 0.5
    assert mod.mean_pairwise_hamming(["AAAA", "CCCC"]) == 1.0


# 决策语义：多样性指标显著占优但可开发性通过率显著回退（CI 上界 < 0）时，
# 决策必须是 expanded_exploration_with_developability_tradeoff——
# 防止解释器忽略可开发性回退而给出激进结论。
def test_interpretation_reports_quality_tradeoff():
    mod = load_script()
    config = {"interpretation": {
        "diversity_advantage_requires_metrics": 2,
        "developability_noninferiority_margin": -0.05,
    }}
    differences = {
        "unique_fraction": {"mean_difference": 0.1, "ci95": [0.01, 0.2]},
        "mean_position_entropy_nats": {"mean_difference": 0.1, "ci95": [0.01, 0.2]},
        "mean_pairwise_hamming_fraction": {"mean_difference": 0.1, "ci95": [-0.01, 0.2]},
        "developability_pass_fraction_at_0_55": {
            "mean_difference": -0.2, "ci95": [-0.3, -0.1]
        },
    }
    result = mod.interpretation(config, differences)
    assert result["decision"] == "expanded_exploration_with_developability_tradeoff"
