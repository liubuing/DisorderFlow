"""IdP 系综扩展开发集 v2 重审计的契约测试：守护抗原链按组件类型过滤（排除离子/半抗原），
以及类型元数据缺失时回退到坐标中实际观测到氨基酸的链——防止非蛋白链冒充抗原。"""
from scripts.reaudit_idp_ensemble_expanded_development_v2 import (
    resolve_antigen_chains,
    typed_antigen_chains,
)


# 类型过滤：ION/HAPTEN 链必须被剔除，只留 PROTEIN 链——防止小分子配体被当作抗原界面。
def test_typed_antigen_chains_excludes_ions_and_haptens():
    component = {
        "antigen_chains": ["A", "B", "C"],
        "antigen_type": ["PROTEIN", "ION", "HAPTEN"],
    }
    assert typed_antigen_chains(component) == ["A"]


# 坐标回退：类型元数据不可靠时，回退选择坐标中真正含氨基酸残基的链并记录回退方法，
# 防止半抗原链（A）或已被重链占用的标注（AAA）误导链归属。
def test_antigen_chain_fallback_uses_observed_amino_acid_chain():
    class Chain(list):
        pass

    class Residue:
        resname = "ALA"

        def __contains__(self, key):
            return key == "CA"

        def get_resname(self):
            return self.resname

    component = {
        "antigen_chains": ["A", "B", "BBB"],
        "antigen_type": ["HAPTEN", "PEPTIDE"],
        "heavy_chain": "AAA",
        "light_chain": None,
    }
    chains, method = resolve_antigen_chains(
        component, {"AAA": Chain([Residue()]), "BBB": Chain([Residue()])}
    )
    assert chains == ["BBB"]
    assert method == "coordinate_observed_amino_acid_fallback"
