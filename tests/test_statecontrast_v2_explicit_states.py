from Bio.PDB import PDBParser

from scripts.prepare_statecontrast_v2_explicit_states import (
    select_source_balanced_poses,
    write_apo_pose,
)


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
