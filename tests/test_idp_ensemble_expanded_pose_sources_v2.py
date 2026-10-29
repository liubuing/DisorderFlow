"""IdP 系综扩展姿态来源 v2 审计的契约测试：守护链序列提取只读含标准 CA 原子的残基，
防止异构/缺失残基混入序列比对。"""
from scripts.audit_idp_ensemble_expanded_pose_sources_v2 import chain_sequence


# 序列提取契约：两个含 CA 的丙氨酸残基应读出 "AA"——防止残基筛选口径漂移导致序列错位。
def test_chain_sequence_reads_standard_ca_residues():
    class Residue:
        resname = "ALA"

        def __contains__(self, key):
            return key == "CA"

        def get_resname(self):
            return self.resname

    assert chain_sequence([Residue(), Residue()]) == "AA"
