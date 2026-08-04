from scripts.build_binder_validation_panel import CDR_SLICES, PANEL, composition_scramble
from scripts.parse_binder_validation_panel import interface_pae


def test_composition_scramble_only_changes_cdr_positions():
    native = PANEL["3STB_native"]
    scrambled = composition_scramble(native, 4201)
    positions = {index for start, end in CDR_SLICES for index in range(start, end)}
    assert sorted(native[index] for index in positions) == sorted(scrambled[index] for index in positions)
    assert any(native[index] != scrambled[index] for index in positions)
    assert all(native[index] == scrambled[index] for index in range(len(native)) if index not in positions)


def test_interface_pae_uses_both_cross_chain_directions():
    score = {"pae": [[0, 2, 4], [6, 0, 8], [10, 12, 0]]}
    assert interface_pae(score, [2, 1]) == 8.5
