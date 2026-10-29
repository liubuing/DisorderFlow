"""IdP 系综扩展开发集 v2 构建的契约测试：守护靶点本体分类的词形要求
（通用词 "Tau peptide" 不得匹配、全称可匹配）与 PDB ID 的规范化规则。"""
from scripts.build_idp_ensemble_expanded_development_v2 import (
    classify_target,
    normalize_pdb,
)


# 分类精确性：仅完整模式匹配才归类（防止 "Tau" 这类过泛匹配误纳无关肽），
# PDB ID 规范化剥离 pdb_ 前缀并转小写，防止跨来源 ID 不一致。
def test_expanded_target_classification_and_pdb_normalization():
    ontology = {
        "tau": {"patterns": ["microtubule-associated protein tau"]},
        "huntingtin": {"patterns": ["huntingtin"]},
    }
    assert classify_target("Tau peptide", ontology) is None
    assert classify_target("microtubule-associated protein tau", ontology) == "tau"
    assert classify_target("huntingtin peptide", ontology) == "huntingtin"
    assert normalize_pdb("pdb_00004TQE") == "4tqe"
