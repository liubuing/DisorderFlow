from disorderflow.statecontrast_v2_contract import validate_statecontrast_v2_records


def _record(group, state, source="experimental", teacher=0.5):
    return {
        "group_id": group,
        "state": {"type": state, "source": source, "prior_weight": 1.0},
        "targets": {
            "state_training_weight": 1.0,
            "independent_teacher_score": teacher,
        },
    }


def test_contract_requires_target_and_negative_state():
    invalid = validate_statecontrast_v2_records([_record("g", "target")])
    assert invalid["valid"] is False
    valid = validate_statecontrast_v2_records([
        _record("g", "target"), _record("g", "apo")])
    assert valid["valid"] is True


def test_contract_checks_teacher_coverage():
    rows = [_record("g", "target", teacher=1.0), _record("g", "off_target", teacher=None)]
    result = validate_statecontrast_v2_records(rows, minimum_teacher_coverage=0.8)
    assert result["teacher_coverage"] == 0.5
    assert result["valid"] is False
