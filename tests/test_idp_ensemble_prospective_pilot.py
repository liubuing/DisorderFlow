import json
from pathlib import Path

import pytest

from scripts.prepare_idp_ensemble_prospective_pilot import prepare

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/benchmarks/idp_ensemble_prospective_pilot_v2.yml"


def test_pilot_contract_materializes_six_clean_components(tmp_path, monkeypatch):
    output = tmp_path / "manifest.json"
    config = CONFIG.read_text(encoding="ascii").replace(
        "reviewer_outputs/idp_ensemble_prospective_pilot_v2/candidate_manifest.json",
        str(output).replace("\\", "/"),
    )
    config_path = tmp_path / "config.yml"
    config_path.write_text(config, encoding="ascii")
    monkeypatch.setattr(
        "scripts.prepare_idp_ensemble_prospective_pilot.ROOT", ROOT
    )
    result = prepare(config_path)
    assert result["status"] == "contract_validated_before_candidate_access"
    assert len(result["components"]) == 6
    assert result["budget"]["total_generated_candidates_per_arm"] == 144
    assert json.loads(output.read_text(encoding="ascii"))["components"]


def test_pilot_manifest_is_write_once(tmp_path):
    output = tmp_path / "manifest.json"
    config = CONFIG.read_text(encoding="ascii").replace(
        "reviewer_outputs/idp_ensemble_prospective_pilot_v2/candidate_manifest.json",
        str(output).replace("\\", "/"),
    )
    config_path = tmp_path / "config.yml"
    config_path.write_text(config, encoding="ascii")
    prepare(config_path)
    with pytest.raises(FileExistsError):
        prepare(config_path)
