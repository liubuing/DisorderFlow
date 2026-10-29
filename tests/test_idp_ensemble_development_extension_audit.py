"""IDP 集成开发扩展审计测试：验证 H3-抗原接触指标统计口径。

守护 scripts.audit_idp_ensemble_development_extension.contact_metrics：
接触 H3 位置数与接触残基对数必须按给定 H3 索引与距离阈值正确计数——
防止接触统计虚高/漏计而误导组件入选判断。
"""
from scripts.audit_idp_ensemble_development_extension import contact_metrics


# 接触计数：一个 H3 位置与一个抗原残基在阈值内应计 1 位置、1 对——
# 防止计数器把"对数"与"位置数"混淆。
def test_contact_metrics_counts_contacting_h3_positions():
    class Atom:
        element = "C"

        def __init__(self, coord):
            self.coord = coord

    class Residue:
        def __init__(self, coord):
            self.atoms = [Atom(coord)]

        def __iter__(self):
            return iter(self.atoms)

    heavy = [{"residue": Residue([0.0, 0.0, 0.0])}]
    antigen = [{"residue": Residue([0.0, 0.0, 4.0])}]
    result = contact_metrics(heavy, antigen, [0], 4.5)
    assert result["contacting_h3_positions"] == 1
    assert result["h3_peptide_residue_contact_pairs"] == 1
