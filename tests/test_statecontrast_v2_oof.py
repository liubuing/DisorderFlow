import json

import pytest

from scripts.aggregate_statecontrast_v2_oof import aggregate


def test_oof_aggregation_requires_every_frozen_fold(tmp_path):
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"folds": [{"fold": 0}, {"fold": 1}]}))
    gate = tmp_path / "gate.yml"
    gate.write_text("classification: test\n")
    with pytest.raises(ValueError, match="Every frozen fold"):
        aggregate(contract, gate, {}, tmp_path / "out.json")


def test_oof_aggregation_rejects_component_overlap(tmp_path):
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"folds": [{"fold": 0}, {"fold": 1}]}))
    gate = tmp_path / "gate.yml"
    gate.write_text("classification: test\n")
    evaluations = {}
    for fold in (0, 1):
        path = tmp_path / f"fold{fold}.json"
        path.write_text(json.dumps({
            "split": "val", "component_effects": {"same": 1.0},
            "matched_pairs": [], "checkpoint_sha256": str(fold),
        }))
        evaluations[fold] = path
    with pytest.raises(ValueError, match="multiple held-out folds"):
        aggregate(contract, gate, evaluations, tmp_path / "out.json")
