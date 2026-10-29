#!/usr/bin/env python
"""Run the non-A-beta initial-guess panel with cached ColabFold MSAs."""
from __future__ import annotations

import csv
import os
import subprocess
import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANEL = PROJECT_ROOT / "outputs/non_abeta_idp_initial_guess_panel_v1/initial_guess_panel.csv"
MSA_DIR = PROJECT_ROOT / "outputs/non_abeta_idp_complex_fold_panel_v1/colabfold_complex_gpu"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/non_abeta_idp_initial_guess_panel_v1/colabfold_initial_guess_gpu"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--constructs", nargs="*")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    with open(PANEL, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.constructs:
        rows = [row for row in rows if row["construct_id"] in set(args.constructs)]
    output_root = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output_root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    environment["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".75"
    completed = 0
    for index, row in enumerate(rows, 1):
        prefix = f"{row['construct_id']}_target_{row['target']}_family_{row['antibody_family']}_VH_VL_IDP"
        # 复用上一阶段（complex fold panel）缓存的单序列 MSA——每个前缀必须
        # 恰好命中一个 A3M，多义或缺失都视为面板不一致而失败
        matches = list(MSA_DIR.glob(f"{prefix}.a3m"))
        if len(matches) != 1:
            raise ValueError(f"Expected one cached MSA for {prefix}, found {len(matches)}")
        job_dir = output_root / row["construct_id"] / f"conformer{row['conformer']}"
        # 断点续跑：按已存在的 seed scores 文件计数，够了就跳过该构造体
        completed_seeds = {
            int(path.stem.rsplit("_seed_", 1)[1])
            for path in job_dir.glob("*_scores_rank_001_*_seed_*.json")
        }
        if len(completed_seeds) >= args.num_seeds:
            completed += 1
            continue
        # initial-guess 用面板预录的初始猜测定界面；multimer_v3、单模型、
        # 3 recycle、MSA 上限 64:128——与 alpha-syn 面板同参数，便于横向比较
        command = [
            "colabfold_batch", str(matches[0]), str(job_dir),
            "--initial-guess", str(PROJECT_ROOT / row["initial_guess_pdb"]),
            "--model-type", "alphafold2_multimer_v3",
            "--num-models", "1", "--num-recycle", "3", "--num-seeds", str(args.num_seeds),
            "--max-msa", "64:128", "--disable-unified-memory", "--rank", "multimer",
        ]
        print(
            f"[{index}/{len(rows)}] {row['construct_id']} conformer {row['conformer']}",
            flush=True,
        )
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)
        completed += 1
    print(f"Initial-guess folds completed: {completed}/{len(rows)}")


if __name__ == "__main__":
    main()
