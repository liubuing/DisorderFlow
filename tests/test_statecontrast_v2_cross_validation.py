import json
from pathlib import Path

import yaml

from scripts.prepare_statecontrast_v2_cross_validation import prepare


def test_cv_configs_hold_out_whole_fold(tmp_path, monkeypatch):
    root = tmp_path
    base = root / "base.yml"
    base.write_text(yaml.safe_dump({
        "lineage": {"allowed_initializer": "x.pt", "allowed_initializer_sha256": "a"},
        "dataset": {"train": {}, "val": {}},
    }), encoding="utf-8")
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"cross_validation_folds": 5}), encoding="ascii")
    monkeypatch.setattr("scripts.prepare_statecontrast_v2_cross_validation.ROOT", root)
    contract = prepare(base, manifest, root / "cv", folds=5)
    assert len(contract["folds"]) == 5
    fold = yaml.safe_load((root / "cv/fold_3.yml").read_text())
    assert fold["dataset"]["train"]["heldout_fold"] == 3
    assert fold["dataset"]["train"]["fold_role"] == "train"
    assert fold["dataset"]["val"]["fold_role"] == "validation"
