import numpy as np

from scripts.benchmark_h3_t1_ensemble import summarize_sequence


def test_ensemble_sequence_summary_uses_shared_reference():
    complex_one = np.zeros((2, 21), dtype=float)
    complex_two = np.zeros((2, 21), dtype=float)
    reference = np.zeros((2, 21), dtype=float)
    complex_one[0, 0] = -1.0
    complex_two[0, 0] = -3.0
    reference[0, 0] = -2.0
    observed = summarize_sequence(
        "A", [complex_one, complex_two], [reference, reference], [0])
    assert observed["per_conformer_ecls"] == [-1.0, 1.0]
    assert observed["mean_ecls"] == 0.0
    assert observed["worst_ecls"] == 1.0
