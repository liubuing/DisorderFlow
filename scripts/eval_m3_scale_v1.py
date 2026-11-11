#!/usr/bin/env python3
"""Evaluate ECLS at selected M3 training epochs (scale curve)."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
FILTERED = ROOT / "results" / "ecls_cross_arch_v1" / "m3_homology_filtered" / "model_weights"
SCALE = ROOT / "results" / "ecls_cross_arch_v1" / "m3_homology_scale" / "model_weights"
OUT = ROOT / "results" / "ecls_cross_arch_v1" / "m3_scale_eval"
BASE_CONFIG = ROOT / "configs" / "benchmarks" / "peptide_h3_ecls_temporal_final_v1.yml"
BENCH = ROOT / "scripts" / "benchmark_h3_epitope_delta.py"


def checkpoint_for(epoch: int) -> Path | None:
    candidates = [
        FILTERED / f"epoch_{epoch}.pt",
        SCALE / f"epoch_{epoch}.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", default="1,3,5,10,20")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    out = args.out if args.out.is_absolute() else ROOT / args.out
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    curve = {}
    for epoch in [int(value) for value in args.epochs.split(",") if value.strip()]:
        checkpoint = checkpoint_for(epoch)
        if checkpoint is None:
            print(f"epoch {epoch}: checkpoint missing", flush=True)
            continue
        weights_dir = out / f"epoch_{epoch}" / "model_weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, weights_dir / "m3.pt")
        config = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
        config["status"] = f"m3_scale_epoch{epoch}_development"
        config["proteinmpnn"]["weights"] = str(
            weights_dir.relative_to(ROOT)).replace("\\", "/")
        config["proteinmpnn"]["model_name"] = "m3"
        config_path = ROOT / "configs" / "benchmarks" / f"peptide_h3_ecls_m3_scale_epoch{epoch}_v1.yml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        out_dir = out / f"ecls_epoch_{epoch}"
        completed = subprocess.run(
            [sys.executable, str(BENCH), "--config", str(config_path),
             "--out-dir", str(out_dir)],
            capture_output=True, text=True, timeout=1200)
        if completed.returncode:
            print(f"epoch {epoch}: FAILED {completed.stderr[-1000:]}", flush=True)
            continue
        aggregate = json.loads(
            (out_dir / "results.json").read_text(encoding="utf-8"))["aggregate"]
        curve[epoch] = {
            "mean_ecls_advantage": aggregate["mean_ecls_advantage"],
            "median_ecls_advantage": aggregate["median_ecls_advantage"],
            "ci95": aggregate["ecls_advantage_mean_ci95"],
            "fraction_positive": aggregate["fraction_positive_ecls_advantage"],
            "gate_pass": aggregate["gate_pass"],
        }
        print(epoch, curve[epoch], flush=True)
    (out / "scale_curve.json").write_text(
        json.dumps(curve, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(curve, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
