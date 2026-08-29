#!/usr/bin/env python
"""Run the guarded metadata-only successor-v3 future snapshot workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build.acquire_rcsb_successor_v3_snapshot import acquire  # noqa: E402
from scripts.build.discover_rcsb_successor_v3_snapshot import discover  # noqa: E402
from scripts.check_successor_v3_future_readiness import check  # noqa: E402

DEFAULT_POLICY = ROOT / "configs/benchmarks/successor_v3_future_snapshot_policy.yml"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_utc(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def resolve_from_root(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def cadence_days(policy):
    match = re.fullmatch(
        r"no_more_than_once_per_(\d+)_days", policy["acquisition"]["cadence"]
    )
    if not match:
        raise ValueError("Unsupported future snapshot cadence")
    return int(match.group(1))


def build_plan(policy_path, sabdab_summary, reference, now=None):
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    policy_path = Path(policy_path)
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    sabdab_summary = Path(sabdab_summary)
    reference = Path(reference)
    if not sabdab_summary.is_file():
        raise FileNotFoundError(f"SAbDab summary not found: {sabdab_summary}")
    if not reference.is_file():
        raise FileNotFoundError(f"Reference union not found: {reference}")
    expected_reference = policy["reference_union"]
    if reference.name != "reference_union_manifest_v3.json":
        raise ValueError("Explicit reference must be reference_union_manifest_v3.json")
    if sha256(reference) != expected_reference["sha256"]:
        raise ValueError("Explicit reference union hash does not match frozen policy")

    current_discovery = resolve_from_root(policy["current_state"]["discovery"])
    discovery = json.loads(current_discovery.read_text(encoding="utf-8"))
    current_acquisition = Path(discovery["inputs"]["acquisition"])
    if not current_acquisition.is_absolute():
        current_acquisition = ROOT / current_acquisition
    acquisition = json.loads(current_acquisition.read_text(encoding="utf-8"))
    last_retrieved = parse_utc(acquisition["retrieved_at_utc"])
    next_permitted = last_retrieved + timedelta(days=cadence_days(policy))
    snapshot_date = now.strftime("%Y_%m_%d")
    snapshot_dir = ROOT / f"data/successor_v3_rcsb_snapshot_{snapshot_date}"
    discovery_path = snapshot_dir / "discovery_after_exposure_v3.json"
    readiness_path = (
        ROOT / "publication"
        / f"successor_v3_future_readiness_after_exposure_{snapshot_date}.json"
    )
    return {
        "schema_version": 1,
        "classification": "metadata_only_future_confirmatory_readiness_plan",
        "status": "eligible_to_execute" if now >= next_permitted else "cadence_gate_closed",
        "now_utc": now.isoformat().replace("+00:00", "Z"),
        "last_snapshot_retrieved_at_utc": last_retrieved.isoformat().replace(
            "+00:00", "Z"
        ),
        "next_permitted_at_utc": next_permitted.isoformat().replace("+00:00", "Z"),
        "eligible_to_execute": now >= next_permitted,
        "inputs": {
            "policy": str(policy_path),
            "policy_sha256": sha256(policy_path),
            "sabdab_summary": str(sabdab_summary),
            "sabdab_summary_sha256": sha256(sabdab_summary),
            "reference_union": str(reference),
            "reference_union_sha256": sha256(reference),
        },
        "outputs": {
            "snapshot_directory": str(snapshot_dir),
            "acquisition": str(snapshot_dir / "acquisition.json"),
            "discovery": str(discovery_path),
            "readiness": str(readiness_path),
        },
        "access_boundary": (
            "RCSB entry IDs and SAbDab summary metadata only; no coordinate, "
            "checkpoint, materialization, or model access"
        ),
    }


def execute_plan(plan):
    if not plan["eligible_to_execute"]:
        raise RuntimeError(
            f"Cadence gate closed until {plan['next_permitted_at_utc']}"
        )
    outputs = {key: Path(value) for key, value in plan["outputs"].items()}
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite future snapshot output: {path}")
    inputs = plan["inputs"]
    acquisition = acquire(outputs["snapshot_directory"])
    discovery = discover(
        outputs["acquisition"],
        Path(inputs["sabdab_summary"]),
        Path(inputs["reference_union"]),
        outputs["discovery"],
        expected_reference_sha256=inputs["reference_union_sha256"],
    )
    readiness = check(outputs["discovery"], outputs["readiness"])
    return {
        "plan": plan,
        "acquisition": {
            "total_count": acquisition["total_count"],
            "entry_ids_sha256": acquisition["entry_ids_sha256"],
        },
        "discovery": discovery["counts"],
        "readiness": readiness,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--sabdab-summary", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = build_plan(
        resolve_from_root(args.policy),
        resolve_from_root(args.sabdab_summary),
        resolve_from_root(args.reference),
    )
    result = execute_plan(plan) if args.execute else plan
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
