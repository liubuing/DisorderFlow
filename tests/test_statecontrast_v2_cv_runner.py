"""状态对比 v2 交叉验证运行计划的契约测试：守护每折恰好生成一条训练命令与一条评估命令，
且训练/评估命令携带折配置与 max-iters 参数——防止漏跑或错配折。"""
import json

from scripts.run_statecontrast_v2_cross_validation import build_plan


# 计划完整性：2 折契约产出 2 条计划，每条都调用训练脚本、带 --split 评估与 --max-iters——
# 防止 CV 循环静默少折或参数丢失。
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
