from scripts.select_idp_ensemble_expanded_development_v2 import resolution_value


def test_resolution_value_sorts_missing_last():
    assert resolution_value("1.5") == 1.5
    assert resolution_value("NA") == float("inf")
    assert resolution_value(None) == float("inf")
