#!/usr/bin/env python
"""Fail-closed readiness check for final untouched IDP confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(config_path, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite final readiness: {output}")
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    checkpoint = ROOT / config["generator"]["checkpoint"]
    checkpoint_ready = (
        checkpoint.is_file()
        and sha256(checkpoint) == config["generator"]["checkpoint_sha256"]
    )
    discovery_path = ROOT / config["outputs"]["discovery"]
    discovery = (
        json.loads(discovery_path.read_text(encoding="ascii"))
        if discovery_path.is_file() else {}
    )
    cohort_path = ROOT / config["outputs"]["cohort"]
    cohort = (
        json.loads(cohort_path.read_text(encoding="ascii"))
        if cohort_path.is_file() else {}
    )
    checks = {
        "pipeline_contract_frozen": config["status"] == (
            "final_pipeline_frozen_before_new_idp_cohort_access"
        ),
        "checkpoint_frozen": checkpoint_ready,
        "metadata_idp_pool_ready": (
            discovery.get("eligible_exact_new_idp_entries", 0)
            >= config["scope"]["minimum_exact_new_structures"]
        ),
        "untouched_idp_cohort_ready": (
            cohort.get("status") == "untouched_idp_cohort_admitted"
            and cohort.get("independent_lineage_count", 0)
            >= config["scope"]["minimum_independent_antibody_lineages"]
            and cohort.get("target_count", 0) >= config["scope"]["minimum_targets"]
        ),
        "blinded_measurements_complete": False,
    }
    payload = {
        "schema_version": 1,
        "status": "final_confirmation_ready" if all(checks.values()) else (
            "final_confirmation_blocked"
        ),
        "classification": config["classification"],
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "checks": checks,
        "passed_count": sum(checks.values()),
        "required_count": len(checks),
        "observed_exact_new_idp_entries": discovery.get(
            "eligible_exact_new_idp_entries", 0
        ),
        "decision": (
            "execute_frozen_confirmation"
            if all(checks.values())
            else "acquire_exact_new_idp_structures_and_blinded_measurements"
        ),
        "claim_boundary": config["claim_boundary"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/benchmarks/idp_ensemble_final_confirmation_v1.yml"),
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    result = check(config_path, ROOT / config["outputs"]["readiness"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
