import json
from pathlib import Path

import pytest

from scripts.check_idp_ensemble_correction_v3 import check

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/benchmarks/idp_ensemble_correction_v3.yml"


def test_correction_contract_is_fail_closed(tmp_path):
    result = check(CONFIG, tmp_path / "status.json")
    assert result["status"] == "correction_blocked"
    assert result["passed_count"] == 4
    assert result["checks"]["autoregressive_generator"] is True
    assert result["checks"]["untouched_cohort"] is True
    assert result["checks"]["minimum_independent_components"] is True
    assert result["checks"]["broader_mutation_space"] is True
    assert result["checks"]["experimental_measurements"] is False
    assert json.loads((tmp_path / "status.json").read_text(encoding="ascii"))["status"] == (
        "correction_blocked"
    )


def test_correction_status_is_write_once(tmp_path):
    output = tmp_path / "status.json"
    check(CONFIG, output)
    with pytest.raises(FileExistsError):
        check(CONFIG, output)
