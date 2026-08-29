#!/usr/bin/env python
"""Plan or execute the frozen five-fold StateContrast-v2 workflow."""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_plan(contract_path, output_dir, device, max_iters=None):
    contract = json.loads(contract_path.read_text(encoding="ascii"))
    plan = []
    for item in contract["folds"]:
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
    parser.add_argument("--contract", type=Path, default=Path(
        "reviewer_outputs/statecontrast_v2_cross_validation_v2/contract.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "reviewer_outputs/statecontrast_v2_cross_validation_v2/runs"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-iters", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(ROOT / args.contract, output_dir, args.device, args.max_iters)
    plan_path = output_dir / "execution_plan.json"
    plan_path.write_text(json.dumps({
        "status": "planned" if not args.execute else "executing",
        "contract": str(args.contract), "folds": plan,
    }, indent=2) + "\n", encoding="ascii")
    if args.execute:
        for row in plan:
            subprocess.run(row["train"], cwd=ROOT, check=True)
            checkpoints = [Path(path) for path in sorted(glob.glob(
                row["checkpoint_glob"]))]
            if not checkpoints:
                raise FileNotFoundError(
                    f"Fold {row['fold']} did not produce best.pt")
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
