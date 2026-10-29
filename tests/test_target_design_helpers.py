"""靶点设计辅助函数的契约测试：守护朝向残基查找返回的是序列内位置（1 起）
而非 PDB resseq 编号——防止两套编号混用导致设计位点错位。"""
from modules.target_design_helpers import find_residues_facing_region


# 编号语义：在极大距离阈值（1000 Å）下应返回全链序列位置 1..219——
# 防止实现误返回原始 PDB resseq 造成下游切片错位。
def test_facing_residues_return_sequence_positions_not_pdb_resseq():
    positions = find_residues_facing_region(
        "data/anti_abeta_refs/4HIX.pdb",
        target_chain="A",
        epitope_resseqs=[1],
        query_chain="H",
        distance_cutoff=1000.0,
    )
    assert positions == list(range(1, 220))
