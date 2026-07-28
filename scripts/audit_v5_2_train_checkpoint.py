#!/usr/bin/env python3
"""Audit the bounded v5.2 train.py checkpoint gate."""

import argparse
import json
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from disorderflow.data_factory import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--best", required=True)
    parser.add_argument("--log")
    parser.add_argument("--expected-iteration", type=int, default=3)
    parser.add_argument("--expected-optimizer-steps", type=int, default=2)
    parser.add_argument("--expected-amp-dtype")
    parser.add_argument(
        "--output", default="data/statecontrast_structural_v2_1/formal_training_gate.json")
    args = parser.parse_args()

    source_path = ROOT / args.source
    raw_path = ROOT / args.raw
    best_path = ROOT / args.best
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    best = torch.load(best_path, map_location="cpu", weights_only=False)

    allowed_modules = ("contrastive_head", "contrastive_cdr_conv")
    changed = [
        key for key, value in raw["model"].items()
        if not torch.equal(value, source["model"][key])
    ]
    unexpected_changed = [
        key for key in changed if not any(module in key for module in allowed_modules)
    ]
    parameter_delta_l1 = sum(
        float((raw["model"][key] - source["model"][key]).float().abs().sum())
        for key in changed
    )
    optimizer_steps = sorted({
        int(state["step"].item())
        for state in raw["optimizer"]["state"].values()
        if "step" in state
    })
    completed_optimizer_steps = min(optimizer_steps) if optimizer_steps else 0
    avg_val_loss = float(raw["avg_val_loss"])
    checks = {
        "iteration_reached": int(raw["iteration"]) == args.expected_iteration,
        "optimizer_updates_reached": (
            completed_optimizer_steps >= args.expected_optimizer_steps),
        "finite_validation_loss": math.isfinite(avg_val_loss),
        "raw_checkpoint_marked_raw": raw.get("weights_kind") == "raw",
        "best_checkpoint_marked_ema": best.get("weights_kind") == "ema",
        "trainable_parameters_changed": bool(changed) and parameter_delta_l1 > 0,
        "frozen_parameters_unchanged": not unexpected_changed,
        "ema_differs_from_raw": any(
            not torch.equal(best["model"][key], raw["model"][key]) for key in raw["model"]),
    }
    if args.expected_amp_dtype:
        checks["amp_dtype"] = (
            str(raw["config"].train.get("amp_dtype", "float16"))
            == args.expected_amp_dtype)
    if args.log:
        log_text = (ROOT / args.log).read_text(encoding="utf-8")
        checks["no_nonfinite_training_events"] = (
            "NaN/Inf gradients" not in log_text
            and "NaN or Inf detected in loss" not in log_text)
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "source_checkpoint": str(source_path.relative_to(ROOT)),
        "raw_checkpoint": str(raw_path.relative_to(ROOT)),
        "best_checkpoint": str(best_path.relative_to(ROOT)),
        "raw_checkpoint_sha256": sha256_file(raw_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "iteration": int(raw["iteration"]),
        "successful_optimizer_steps": completed_optimizer_steps,
        "scaler_scale": float(raw["scaler"].get("scale", 0.0)),
        "avg_validation_loss": avg_val_loss,
        "changed_tensors": len(changed),
        "unexpected_changed_tensors": unexpected_changed,
        "parameter_delta_l1": parameter_delta_l1,
    }
    output = ROOT / args.output
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise RuntimeError("v5.2 formal training checkpoint audit failed")


if __name__ == "__main__":
    main()
