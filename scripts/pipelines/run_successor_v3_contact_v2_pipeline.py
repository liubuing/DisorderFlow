#!/usr/bin/env python
"""Single entry point for the successor-v3 contact-v2 development lifecycle."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = (
    "logs/bfn_successor_v3_contact_v2_exposed_2026_08_09__09_06_40_"
    "successor_v3_contact_v2_cb/checkpoints/best.pt")

COMMANDS = {
    "build": [
        "scripts/build/build_successor_v3_contact_dev_dataset.py",
        "--manifest", "data/successor_v3_exploratory/structures_final/structural_manifest.json",
        "--output-dir", "data/successor_v3_contact_dev_v1",
        "--folds", "5", "--eval-fold", "4", "--threads", "8"],
    "baselines": [
        "scripts/benchmark_successor_v3_contact_baselines.py",
        "--dataset-manifest", "data/successor_v3_contact_dev_v1/manifest.json",
        "--structural-manifest", "data/successor_v3_exploratory/structures_final/structural_manifest.json",
        "--isolation-audit", "data/successor_v3_exploratory/isolation_audit.json",
        "--protocol", "configs/benchmarks/successor_v3_contact_v2_development.yml",
        "--output", "results/successor_v3_contact_v2/baselines.json"],
    "train": [
        "train.py", "configs/train/bfn_successor_v3_contact_v2_exposed.yml",
        "--init", "logs/bfn_successor_v3_h3_specificity_2026_08_07__11_35_01_successor_v3_h3_specificity_s3107/checkpoints/best.pt",
        "--device", "cuda", "--num_workers", "0", "--tag", "successor_v3_contact_v2_cb"],
    "evaluate": [
        "scripts/evaluate_successor_v3_contact_v2.py", "--checkpoint", "__CHECKPOINT__",
        "--dataset-manifest", "data/successor_v3_contact_dev_v1/manifest.json",
        "--output", "results/successor_v3_contact_v2/development_evaluation.json"],
    "experiments": [
        "scripts/prepare_successor_v3_contact_v2_experiments.py", "--checkpoint", "__CHECKPOINT__",
        "--dataset-manifest", "data/successor_v3_contact_dev_v1/manifest.json",
        "--structural-manifest", "data/successor_v3_exploratory/structures_final/structural_manifest.json",
        "--output-dir", "experiments/successor_v3_contact_v2"],
    "registry": ["scripts/build/build_successor_v3_contact_v2_registry.py"],
}


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_future_policy():
    path = ROOT / "configs/benchmarks/successor_v3_contact_v2_future_confirmation.yml"
    policy = yaml.safe_load(path.read_text(encoding="ascii"))
    items = [
        policy["candidate_model"], policy["development_evaluation"], policy["baselines"],
        policy["admission"]["reference_union"], *policy["frozen_inputs"].values()]
    failures = [item["path"] for item in items
                if not (ROOT / item["path"]).is_file()
                or sha256(ROOT / item["path"]) != item["sha256"]]
    if failures:
        raise RuntimeError(f"Frozen future policy verification failed: {failures}")
    return policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["status", *COMMANDS, "all"], default="status", nargs="?")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    args = parser.parse_args()
    if args.stage == "status":
        policy = verify_future_policy()
        print(json.dumps({
            "status": policy["status"],
            "candidate_sha256": policy["candidate_model"]["sha256"],
            "future_readiness": policy["admission"]["current_state"],
        }, indent=2))
        return
    stages = list(COMMANDS) if args.stage == "all" else [args.stage]
    checkpoint = args.checkpoint
    for stage in stages:
        command = [sys.executable, *[
            checkpoint if value == "__CHECKPOINT__" else value
            for value in COMMANDS[stage]]]
        if stage in {"baselines", "evaluate", "experiments"}:
            command.extend(["--device", args.device])
        print(f"Running contact-v2 stage: {stage}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        if stage == "train" and args.stage == "all":
            logs = list((ROOT / "logs").glob(
                "bfn_successor_v3_contact_v2_exposed_*_successor_v3_contact_v2_cb"))
            checkpoint = str(max(logs, key=lambda path: path.stat().st_mtime)
                             .joinpath("checkpoints/best.pt").relative_to(ROOT))


if __name__ == "__main__":
    main()
