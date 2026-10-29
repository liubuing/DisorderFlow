#!/usr/bin/env python
"""Plan or execute the frozen five-fold StateContrast-v2 workflow."""

# 本脚本服务于 StateContrast-v2 的五折交叉验证流程：
# 读取预先冻结的 contract.json（记录各折的配置与初始化权重），
# 生成可审计的 execution_plan.json；默认仅输出计划（dry-run），
# 传入 --execute 后逐折串行训练并评估，保证复现实验口径一致。
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_plan(contract_path, output_dir, device, max_iters=None):
    # 根据冻结契约构建每折的训练/评估命令行。
    # contract 必须为 ASCII 编码的 JSON（防止意外字符进入冻结产物）。
    contract = json.loads(contract_path.read_text(encoding="ascii"))
    plan = []
    for item in contract["folds"]:
        # 目录布局：output_dir/fold_{k}/<run>/checkpoints/best.pt
        fold = int(item["fold"])
        config = item["config"]
        fold_logdir = output_dir / f"fold_{fold}"
        checkpoint_glob = str(fold_logdir / "*" / "checkpoints" / "best.pt")
        evaluation = output_dir / f"fold_{fold}_evaluation.json"
        train = [
            sys.executable, "train_statecontrast_v2.py", config,
            "--device", device, "--logdir", str(fold_logdir),
            "--init", contract["initializer"], "--tag", f"fold_{fold}",
        ]
        if max_iters is not None:
            # 可选覆盖训练迭代数，仅用于调试/快速冒烟，不影响冻结契约本身。
            train.extend(["--max-iters", str(max_iters)])
        evaluate = [
            sys.executable, "scripts/evaluate_statecontrast_v2_checkpoint.py",
            "--config", config, "--checkpoint", "<BEST_CHECKPOINT>",
            "--split", "val", "--output", str(evaluation), "--device", device,
        ]
        plan.append({
            "fold": fold, "train": train, "evaluate": evaluate,
            "checkpoint_glob": checkpoint_glob, "evaluation": str(evaluation),
        })
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # 默认路径与冻结的交叉验证产物目录保持一致（reviewer_outputs 下）。
    parser.add_argument("--contract", type=Path, default=Path(
        "reviewer_outputs/statecontrast_v2_cross_validation_v2/contract.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "reviewer_outputs/statecontrast_v2_cross_validation_v2/runs"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-iters", type=int)
    # 不加 --execute 时只生成计划；加上后真正跑训练+评估。
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(ROOT / args.contract, output_dir, args.device, args.max_iters)
    plan_path = output_dir / "execution_plan.json"
    # 先落盘执行计划再决定是否执行，便于事后审计每一步命令。
    plan_path.write_text(json.dumps({
        "status": "planned" if not args.execute else "executing",
        "contract": str(args.contract), "folds": plan,
    }, indent=2) + "\n", encoding="ascii")
    if args.execute:
        for row in plan:
            # 串行执行：训练当前折必须成功(check=True)才继续。
            subprocess.run(row["train"], cwd=ROOT, check=True)
            checkpoints = [Path(path) for path in sorted(glob.glob(
                row["checkpoint_glob"]))]
            if not checkpoints:
                raise FileNotFoundError(
                    f"Fold {row['fold']} did not produce best.pt")
            # 用实际产出的 best.pt 替换计划中的占位符；排序后取最后一个
            # （即字典序最新的运行目录）。
            evaluate = [
                str(checkpoints[-1]) if value == "<BEST_CHECKPOINT>" else value
                for value in row["evaluate"]
            ]
            subprocess.run(evaluate, cwd=ROOT, check=True)
    print(json.dumps({
        "status": "complete" if args.execute else "planned",
        "folds": len(plan), "plan": str(plan_path.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
