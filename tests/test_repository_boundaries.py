from scripts.audit_repository_boundaries import classify


def test_repository_zone_classification_prefers_declared_prefix():
    zones = {"core": ["disorderflow/"], "scripts": ["scripts/"]}
    assert classify(__import__("pathlib").Path("disorderflow/models/x.py"), zones) == "core"
    assert classify(__import__("pathlib").Path("scripts/x.py"), zones) == "scripts"
