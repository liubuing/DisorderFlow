#!/usr/bin/env python
"""Freeze the v3 candidate execution plan after cohort admission."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def prepare(config_path, admission_path, output_path):
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite v3 panel plan: {output_path}")
    config = yaml.safe_load(Path(config_path).read_text(encoding="ascii"))
    admission = json.loads(Path(admission_path).read_text(encoding="ascii"))
    if admission.get("status") != "untouched_cohort_admitted":
        raise RuntimeError("Untouched cohort has not passed admission")
    observed = admission.get("observed", {})
    cohort = config["cohort"]
    if observed.get("exact_new_structures", 0) < cohort["minimum_exact_new_structures"]:
        raise RuntimeError("Untouched cohort has fewer than 12 exact-new structures")
    if observed.get("independent_lineage_components", 0) < cohort["minimum_independent_antibody_lineage_components"]:
        raise RuntimeError("Untouched cohort has fewer than 12 independent lineages")
    if observed.get("targets", 0) < cohort["minimum_targets"]:
        raise RuntimeError("Untouched cohort has fewer than 3 targets")

    mutation = config["mutation_space"]
    components = admission["components"]
    if len({row["component_id"] for row in components}) != len(components):
        raise RuntimeError("Cohort contains duplicate component IDs")
    lineages = {
        row.get("antibody_lineage_cluster", row.get("lineage"))
        for row in components
    }
    if None in lineages or len(lineages) != len(components):
        raise RuntimeError("Cohort contains duplicate antibody lineages")
    plan = {
        "schema_version": 1,
        "status": "v3_panel_plan_frozen_before_candidate_access",
        "classification": "untouched_confirmation_candidate_execution_plan",
        "config": str(Path(config_path)),
        "config_sha256": sha256(Path(config_path)),
        "admission": {
            "path": str(admission_path),
            "sha256": sha256(Path(admission_path)),
        },
        "components": [
            {
                "component_id": row["component_id"],
                "target": row["target"],
                "antibody_lineage": row.get(
                    "antibody_lineage_cluster", row.get("lineage")
                ),
                "epitope_cluster": row.get("epitope_cluster"),
            }
            for row in components
        ],
        "arms": config["comparison"]["paired_arms"],
        "seeds": config["comparison"]["independent_generation_seeds"],
        "candidates_per_component_arm": config["comparison"]["candidates_per_component_arm"],
        "native_controls_per_component": config["comparison"]["native_controls_per_component"],
        "mutation_space": {
            "mutable_regions": mutation["mutable_regions"],
            "substitution_buckets": mutation["substitution_buckets"],
            "maximum_h3_substitutions": mutation["maximum_h3_substitutions"],
            "candidate_count_per_bucket": mutation["candidate_count_per_bucket"],
        },
        "decision": "candidate_generation_may_begin_only_after_this_plan_is_frozen",
        "claim_boundary": (
            "general antibody multi-conformer execution plan on non-IDP-specific "
            "targets; no generated sequence, IDP-specific confirmation, or "
            "experimental result"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="ascii")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/benchmarks/idp_ensemble_correction_v3.yml"))
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(resolve(args.config), resolve(args.admission), resolve(args.output)), indent=2))


if __name__ == "__main__":
    main()
