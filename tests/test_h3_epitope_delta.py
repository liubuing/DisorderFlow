import numpy as np

from scripts.benchmark_h3_epitope_delta import (
    MPNN_ALPHABET,
    aggregate_inference_units,
    bootstrap_mean_ci,
    percentile_lower_is_better,
    select_inference_units,
    sequence_nll,
)


def test_sequence_nll_uses_only_requested_h3_positions():
    log_probs = np.full((5, len(MPNN_ALPHABET)), -10.0)
    log_probs[1, MPNN_ALPHABET.index("A")] = -1.0
    log_probs[3, MPNN_ALPHABET.index("C")] = -3.0
    assert sequence_nll(log_probs, "AC", [1, 3]) == 2.0


def test_native_percentile_treats_lower_nll_as_better():
    assert percentile_lower_is_better(1.0, [0.5, 1.5, 2.0, 3.0]) == 0.75


def test_inference_unit_selection_is_deterministic():
    records = [
        {"id": "b", "axis_values": {"ag": ["x"], "pdb_id": ["p2"]}},
        {"id": "a", "axis_values": {"ag": ["x"], "pdb_id": ["p1"]}},
        {"id": "c", "axis_values": {"ag": ["y"], "pdb_id": ["p3"]}},
    ]
    assert [row["id"] for row in select_inference_units(records, "ag")] == ["a", "c"]


def test_bootstrap_mean_ci_is_reproducible_and_positive():
    first = bootstrap_mean_ci([0.1, 0.2, 0.3], seed=7, trials=1000)
    second = bootstrap_mean_ci([0.1, 0.2, 0.3], seed=7, trials=1000)
    assert first == second
    assert first[0] > 0


def test_inference_unit_aggregation_averages_duplicate_records():
    audit = [
        {"id": "a", "axis_values": {"ag": ["x"], "pdb_id": ["p1"]}},
        {"id": "b", "axis_values": {"ag": ["x"], "pdb_id": ["p2"]}},
    ]
    results = [
        {"id": "a", "summary": {"ecls_advantage": 0.1,
                                   "native_complex_h3_nll_percentile": 0.8,
                                   "native_epitope_delta_nll_percentile": 0.6}},
        {"id": "b", "summary": {"ecls_advantage": 0.3,
                                   "native_complex_h3_nll_percentile": 1.0,
                                   "native_epitope_delta_nll_percentile": 0.8}},
    ]
    units = aggregate_inference_units(results, audit, "ag")
    assert len(units) == 1
    assert units[0]["n_records"] == 2
    assert units[0]["mean_ecls_advantage"] == 0.2
