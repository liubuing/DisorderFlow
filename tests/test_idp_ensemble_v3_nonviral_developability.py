from modules.state_specificity_scorer import immunogenicity_proxy, sequence_complexity


def test_developability_proxies_are_bounded():
    assert 0.0 <= sequence_complexity("ARDYGGFDY") <= 1.0
    assert 0.0 <= immunogenicity_proxy("ARDYGGFDY") <= 1.0
