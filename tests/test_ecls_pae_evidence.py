from collections import Counter

import numpy as np
import pytest

from scripts.analyze_ecls_pae_evidence import concordance, membership, teacher_value
from scripts.prepare_ecls_hard_controls import near_native_controls


def test_top_membership_ties_preserve_budget():
    weights = membership([0, 1, 1, 1, 2], 3)
    np.testing.assert_allclose(weights, [1, 2/3, 2/3, 2/3, 0])
    assert weights.sum() == pytest.approx(3)


def test_concordance_tied_predictions_get_chance_credit():
    assert concordance([1, 1, 1], [0, 1, 2]) == {"pairs": 3, "accuracy": .5}
    assert concordance([2, 1, 0], [0, 1, 2])["accuracy"] == 0


def test_teacher_is_directional_local_h3_not_reverse_or_global():
    entity = dict(heavy_sequence="ACD", light_sequence="EF", antigen_sequence="GH", h3_sequence="CD")
    matrix = np.zeros((7, 7))
    matrix[1:3, 5:] = 15.5
    matrix[5:, 1:3] = 31
    assert teacher_value(matrix, entity) == .5


def test_teacher_rejects_ambiguous_h3():
    with pytest.raises(ValueError):
        teacher_value(np.zeros((6, 6)), dict(heavy_sequence="AAA", light_sequence="C", antigen_sequence="DE", h3_sequence="AA"))


def test_near_native_controls_preserve_composition_and_only_two_positions():
    seq = "ACDEFGHIKLMN"
    controls = near_native_controls(seq, 20, 7)
    assert controls == near_native_controls(seq, 20, 7)
    assert len({r['h3_sequence'] for r in controls}) == 20
    for row in controls:
        assert Counter(row['h3_sequence']) == Counter(seq)
        assert sum(a != b for a, b in zip(seq, row['h3_sequence'])) == 2


def test_homopolymer_has_no_distinct_swap_control():
    assert near_native_controls("AAAA", 20, 1) == []
