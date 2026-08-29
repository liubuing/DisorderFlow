from modules.target_design_helpers import find_residues_facing_region


def test_facing_residues_return_sequence_positions_not_pdb_resseq():
    positions = find_residues_facing_region(
        "data/anti_abeta_refs/4HIX.pdb",
        target_chain="A",
        epitope_resseqs=[1],
        query_chain="H",
        distance_cutoff=1000.0,
    )
    assert positions == list(range(1, 220))
