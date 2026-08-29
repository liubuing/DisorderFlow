from scripts.evaluate_statecontrast_v2_checkpoint import (
    hard_developability_failure,
)


def test_developability_hard_failure_detects_motif_and_hydrophobic_run():
    assert hard_developability_failure("ANVT") is True
    assert hard_developability_failure("AFFFFFA") is True
    assert hard_developability_failure("ARDGSGY") is False
