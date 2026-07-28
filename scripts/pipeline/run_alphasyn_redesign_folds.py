#!/usr/bin/env python
"""Batch the alpha-synuclein redesign panel by conformer on ColabFold GPU."""
from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/alpha_synuclein_redesign_validation_v2"


def main():
    rows = list(csv.DictReader(open(BASE / "initial_guess_panel.csv", newline="", encoding="utf-8")))
    env = dict(os.environ)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".75"
    for conformer in sorted({int(row["conformer"]) for row in rows}):
        conformer_rows = [row for row in rows if int(row["conformer"]) == conformer]
        input_dir = BASE / "batch_inputs" / f"conformer{conformer}"
        input_dir.mkdir(parents=True, exist_ok=True)
        for row in conformer_rows:
            source = BASE / "a3m" / f"{row['construct_id']}.a3m"
            target = input_dir / source.name
            if not target.exists():
                target.write_bytes(source.read_bytes())
        result_dir = BASE / "colabfold_multiseed_gpu" / f"conformer{conformer}"
        existing = list(result_dir.glob("*_scores_rank_*_*_seed_*.json"))
        if len(existing) >= len(conformer_rows) * 3:
            continue
        initial_guess = ROOT / conformer_rows[0]["initial_guess_pdb"]
        print(f"Conformer {conformer}: {len(conformer_rows)} constructs x 3 seeds", flush=True)
        command = [
            "colabfold_batch", str(input_dir), str(result_dir),
            "--initial-guess", str(initial_guess), "--model-type", "alphafold2_multimer_v3",
            "--num-models", "1", "--num-recycle", "3", "--num-seeds", "3",
            "--max-msa", "64:128", "--disable-unified-memory", "--rank", "multimer",
        ]
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    print("Alpha-syn redesign folds complete")


if __name__ == "__main__":
    main()
