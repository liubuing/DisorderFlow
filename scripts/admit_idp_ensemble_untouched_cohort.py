#!/usr/bin/env python
"""Create a fail-closed untouched-cohort admission artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def admit(discovery_path, registry_path, output, curation_path=None):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite cohort admission: {output}")
    discovery = json.loads(discovery_path.read_text(encoding="ascii"))
    exact_new = discovery.get("candidate_entry_ids", [])
    curation = (
        json.loads(Path(curation_path).read_text(encoding="ascii"))
        if curation_path else None
    )
    admitted = bool(
        curation
        and curation.get("status") == "cohort_candidate_curated"
        and curation.get("component_count", 0) >= 12
        and curation.get("independent_lineage_count", 0) >= 12
        and curation.get("target_count", 0) >= 3
        and max(curation.get("cluster_counts", {"empty": 99}).values()) <= 2
    )
    payload = {
        "schema_version": 1,
        "status": "untouched_cohort_admitted" if admitted else "untouched_cohort_blocked",
        "classification": "metadata_only_cohort_admission",
        "inputs": {
            "discovery": str(discovery_path),
            "discovery_sha256": sha256(discovery_path),
            "historical_registry": str(registry_path),
            "historical_registry_sha256": sha256(registry_path),
        },
        "admission_rule": {
            "minimum_exact_new_structures": 12,
            "minimum_independent_lineage_components": 12,
            "minimum_targets": 3,
            "historical_registry_is_ineligible": True,
        },
        "observed": {
            "exact_new_structures": curation.get("component_count", len(exact_new)) if curation else len(exact_new),
            "independent_lineage_components": curation.get("independent_lineage_count", 0) if curation else 0,
            "targets": curation.get("target_count", 0) if curation else 0,
            "candidate_entry_ids": exact_new,
        },
        "decision": "cohort_admitted_before_candidate_access" if admitted else "do_not_materialize_coordinates_or_generate_candidates",
        "claim_boundary": "metadata admission only; no confirmation cohort or performance claim",
    }
    if curation_path:
        payload["inputs"]["curation"] = str(curation_path)
        payload["inputs"]["curation_sha256"] = sha256(curation_path)
        payload["components"] = curation.get("components", [])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("reviewer_outputs/idp_ensemble_development_extension_registry_v1/components.csv"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--curation", type=Path)
    args = parser.parse_args()
    print(json.dumps(admit(args.discovery, args.registry, args.output, args.curation), indent=2))


if __name__ == "__main__":
    main()
