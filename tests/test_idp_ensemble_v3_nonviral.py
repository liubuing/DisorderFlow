"""IdP 系综 v3 非病毒靶点序列 QC 的契约测试：守护 N-糖基化基序检测、超长疏水连续段的检查
（序列 QC 不通过即判 fail），防止不可开发序列进入生成候选池。"""
from scripts.analyze_idp_ensemble_v3_nonviral import sequence_metrics

# 序列 QC 契约：允许氨基酸表、疏水连续段上限、电荷代理上限与三类降解/修饰基序标记。
CONTRACT = {
    "allowed_amino_acids": "ACDEFGHIKLMNPQRSTVWY",
    "maximum_hydrophobic_run": 4,
    "maximum_absolute_charge_proxy": 8,
    "flag_deamidation_motifs": ["NG", "NS", "NN", "NQ"],
    "flag_isomerization_motifs": ["DG", "DS", "DT", "DD"],
    "flag_oxidation_residues": ["M", "W"],
}


# QC 触发：NVT 糖基化基序必须被检出并导致 qc_pass=False；5 连疏水超过上限 4 也必须失败——
# 防止合成可行性问题静默通过。
def test_sequence_qc_flags_glycosylation_and_long_hydrophobic_run():
    glyco = sequence_metrics("ARNVTGY", CONTRACT)
    assert glyco["n_linked_glycosylation_motifs"] == ["NVT"]
    assert glyco["qc_pass"] is False
    hydrophobic = sequence_metrics("AFFFFFG", CONTRACT)
    assert hydrophobic["checks"]["hydrophobic_run_ok"] is False
