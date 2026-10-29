"""IdP 系综扩展开发集 v2 物化的契约测试：守护残基计数只统计实际观测到 CA 原子的残基，
防止未解析残基被计入序列长度而扭曲覆盖度指标。"""
from scripts.materialize_idp_ensemble_expanded_development_v2 import residue_count


# 测试替身：模拟带/不带 CA 原子的残基对象。
class Residue:
    def __init__(self, has_ca=True):
        self.resname = "ALA"
        self.has_ca = has_ca

    def __contains__(self, key):
        return key == "CA" and self.has_ca

    def get_resname(self):
        return self.resname


# 计数契约：缺 CA 的残基不得计入（2 个残基只有 1 个计数）——防止坐标缺口虚增残基数。
def test_residue_count_requires_observed_ca():
    assert residue_count([Residue(), Residue(False)]) == 1
