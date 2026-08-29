import json

from scripts.run_statecontrast_v2_cross_validation import build_plan


def test_cv_runner_builds_one_train_and_evaluation_per_fold(tmp_path):
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({
        "initializer": "init.pt",
        "folds": [
            {"fold": 0, "config": "cv/fold_0.yml"},
            {"fold": 1, "config": "cv/fold_1.yml"},
        ],
    }))
    plan = build_plan(contract, tmp_path / "runs", "cpu", max_iters=2)
    assert len(plan) == 2
    assert all("train_statecontrast_v2.py" in row["train"] for row in plan)
    assert all("--split" in row["evaluate"] for row in plan)
    assert all("--max-iters" in row["train"] for row in plan)
