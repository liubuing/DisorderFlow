"""IdP 系综 v3 非病毒靶点汇总辅助函数的契约测试：守护 Hamming 距离计数与
修饰警告计数（脱酰胺/异构化/氧化各自累加）的正确性，防止汇总口径偏差。"""
from scripts.summarize_idp_ensemble_v3_nonviral import hamming, warning_count


# 汇总锚点：Hamming 距离按位置计数（2/4 不同），警告数等于各类修饰列表长度之和（NG + M 共 2）。
def test_nonviral_summary_helpers():
    assert hamming("AAAA", "ACCA") == 2
    row = {
        "n_linked_glycosylation_motifs": [],
        "deamidation_motifs": ["NG"],
        "isomerization_motifs": [],
        "oxidation_residues": ["M"],
    }
    assert warning_count(row) == 2
