"""发布完整性修复测试：验证氨基酸字母表、序列恢复防护与 T2.1 统计口径分离。

守护发布统计的完整性：
消融脚本必须用标准氨基酸顺序（字典序）、
sequence_recovery 对长度不匹配必须报错（防错位计恢复率）、
T2.1 聚合必须分开报告总体有效率与层级后 QC 有效率、
配对对比的推断单元为官方抗原簇——防止口径混淆导致指标虚高。
"""
import pytest

from scripts.ablation_disorder_conditioning import AA_LETTERS
from scripts.benchmark_vs_mpnn import sequence_recovery
from scripts.t21_statistics import aggregate


# 字母表冻结：消融编码必须用标准 20 氨基酸字典序——防止索引映射与模型权重错位。
def test_ablation_uses_canonical_amino_acid_order():
    assert AA_LETTERS == "ACDEFGHIKLMNPQRSTVWY"


# 长度防护：预测与参考长度不一致必须报错——防止错位比较出虚假恢复率。
def test_sequence_recovery_rejects_length_mismatch():
    with pytest.raises(ValueError, match="Sequence length mismatch"):
        sequence_recovery("ACD", "ACDE")


# 口径分离：总体有效率（1/3）与扭角命中后的 QC 有效率（1/2）必须分开报告；
# 配对对比（supplied - random restraints）均值为 -0.2 且推断单元为官方抗原簇——
# 防止两种分母混淆或推断单元退化为结构级。
def test_t21_reports_overall_and_post_tier_validity_separately():
    results = [
        {
            "id": "a",
            "torsion_hit_tier": True,
            "valid_t2_1": True,
            "metrics": {
                "mean_held_out_contact_recovery": 0.5,
                "mean_rmsd_recovery_angstrom": 0.1,
                "random_restraints_held_out_contact": 0.7,
            },
        },
        {"id": "b", "torsion_hit_tier": True, "valid_t2_1": False},
        {"id": "c", "torsion_hit_tier": False, "valid_t2_1": False},
    ]
    config = {
        "statistics": {"bootstrap_trials": 100, "bootstrap_seed": 1},
        "development_gates": {
            "minimum_valid_structure_fraction": 0.3,
            "minimum_fraction_structures_positive_held_out_recovery": 0.5,
        },
    }
    observed = aggregate(results, config, {"a": "cluster-1"})
    assert observed["overall_valid_structure_fraction"] == pytest.approx(1 / 3)
    assert observed["post_tier_qc_fraction"] == pytest.approx(1 / 2)
    contrast = observed["paired_arm_contrasts"]["supplied_minus_random_restraints"]
    assert contrast["mean"] == pytest.approx(-0.2)
    assert contrast["inference_unit"] == "official_antigen_cluster"
