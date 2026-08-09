import pytest

from scripts.evaluate_successor_v3_contact_v2_confirmatory import validate_admission


def policy(ready=True):
    return {
        "admission": {
            "current_state": {"ready": ready},
            "reference_union": {"sha256": "a" * 64},
            "minimum_independent_homology_components": 12,
        }
    }


def manifest(count=12):
    return {
        "classification": "future_confirmatory_untouched",
        "checkpoint_accessed_before_manifest_freeze": False,
        "reference_union_sha256": "a" * 64,
        "all_pairwise_axes_isolated": True,
        "records": [{"component_id": f"C{index:03d}"} for index in range(count)],
    }


def test_confirmation_admission_requires_ready_independent_components():
    validate_admission(policy(), manifest(), "manifest.json")
    with pytest.raises(RuntimeError, match="at least 12"):
        validate_admission(policy(), manifest(11), "manifest.json")
    duplicated = manifest()
    duplicated["records"][-1]["component_id"] = "C000"
    with pytest.raises(RuntimeError, match="one representative"):
        validate_admission(policy(), duplicated, "manifest.json")


def test_confirmation_admission_requires_untouched_frozen_manifest():
    exposed = manifest()
    exposed["checkpoint_accessed_before_manifest_freeze"] = True
    with pytest.raises(RuntimeError, match="checkpoint-access"):
        validate_admission(policy(), exposed, "manifest.json")
    with pytest.raises(RuntimeError, match="not marked ready"):
        validate_admission(policy(False), manifest(), "manifest.json")
