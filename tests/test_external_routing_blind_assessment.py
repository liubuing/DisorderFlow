from scripts.assess_external_routing_blind import main


def test_assessment_module_imports():
    assert callable(main)
