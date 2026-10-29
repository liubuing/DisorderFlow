"""状态对比 v2 显式状态准备的契约测试：守护 apo 姿态只保留抗体链（去除抗原）并返回内容摘要，
以及姿态选择按来源均衡（两个来源都要保留）——防止状态构造丢失链或被单一来源垄断。"""
from Bio.PDB import PDBParser

from scripts.prepare_statecontrast_v2_explicit_states import (
    select_source_balanced_poses,
    write_apo_pose,
)


# apo 构造：写出的 apo 文件只含抗体链 H、摘要为 64 位哈希——防止抗原残基残留污染负状态。
def test_apo_pose_removes_antigen_and_preserves_antibody(tmp_path):
    source = tmp_path / "complex.pdb"
    source.write_text(
        "ATOM      1  CA  ALA H   1       0.000   0.000   0.000  1.00 10.00           C\n"
        "ATOM      2  CA  GLY A   1       5.000   0.000   0.000  1.00 10.00           C\n"
        "TER\nEND\n",
        encoding="ascii",
    )
    output = tmp_path / "apo.pdb"
    digest = write_apo_pose(source, ["H"], output)
    assert len(digest) == 64
    model = next(iter(PDBParser(QUIET=True).get_structure("apo", output)))
    assert [chain.id for chain in model] == ["H"]


# 来源均衡：own_entry 与受限局部界面采样两个来源都必须出现在 5 条选择中——
# 防止姿态池被单一来源垄断而引入来源偏差。
def test_pose_selection_preserves_both_broad_sources():
    rows = [
        {"pose_id": f"e{i}", "source": "own_entry"} for i in range(6)
    ] + [
        {"pose_id": f"s{i}", "source": "restrained_local_interface_sampling"}
        for i in range(6)
    ]
    selected = select_source_balanced_poses(rows, maximum=5)
    assert len(selected) == 5
    assert {row["source"] for row in selected} == {
        "own_entry", "restrained_local_interface_sampling"}
