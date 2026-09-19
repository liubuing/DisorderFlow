from scripts.audit_abbibench_sequence_mapping import classify, window_mapping
from scripts.crosswalk_abbibench_aayl_endpoints import finite_value


def test_h3_only_requires_unchanged_light_and_framework():
    assert classify('ACDA', 'LLLL', 'ACDE', 'LLLL', [3]) == 'H3_only_coordinate_compatible'
    assert classify('ACDA', 'LLLA', 'ACDE', 'LLLL', [3]) == 'light_chain_differs'
    assert classify('CCDA', 'LLLL', 'ACDE', 'LLLL', [3]) == 'heavy_changes_outside_H3'
    assert classify('ACDE', 'LLLL', 'ACDE', 'LLLL', [3]) == 'identical_to_coordinate_sequence'


def test_mapping_rejects_ambiguous_and_unobserved_sequence():
    assert window_mapping('ACDEACDE', 'ACDE')['status'] == 'ambiguous_window'
    assert window_mapping('ACD', 'ACDE')['status'] == 'query_longer_than_observed_coordinates'
    assert window_mapping('YYACDEGG', 'ACDE')['offset'] == 2


def test_invalid_or_length_changed_sequences_do_not_enter_h3_set():
    assert classify('ACDX', 'LLLL', 'ACDE', 'LLLL', [3]) == 'noncanonical'
    assert classify('ACD', 'LLLL', 'ACDE', 'LLLL', [3]) == 'length_mismatch'


def test_original_log_affinity_keeps_negative_values_but_excludes_missing():
    assert finite_value('-0.4') == -.4
    assert finite_value('0') == 0
    assert finite_value('') is None
    assert finite_value('nan') is None
    assert finite_value('inf') is None
