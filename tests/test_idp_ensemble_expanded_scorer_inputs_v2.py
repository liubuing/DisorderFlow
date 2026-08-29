from scripts.audit_idp_ensemble_expanded_scorer_inputs_v2 import (
    sequence_identity,
)


def test_sequence_identity_is_exact_for_equal_sequences():
    assert sequence_identity("ARDY", "ARDY") == 1.0
    assert sequence_identity("ARDY", "ARAY") < 1.0
