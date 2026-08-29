#!/usr/bin/env python
"""Classify exact-new antibody metadata for final IDP cohort eligibility."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TARGET_PATTERNS = {
    "amyloid_beta": [r"amyloid[- ]?beta", r"\babeta\b", r"aβ"],
    "tau": [r"microtubule[- ]associated protein tau", r"\btau\b"],
    "alpha_synuclein": [r"alpha[- ]synuclein", r"\bsnca\b"],
    "prion_protein": [r"prion protein", r"\bprp\b"],
}
ANTIBODY_PATTERN = re.compile(
    r"antibody|\bfab\b|\bvhh\b|\bscfv\b|heavy chain|light chain",
    re.IGNORECASE,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_text(record):
    fields = [record.get("title") or ""]
    fields.extend(
        entity.get("description") or ""
        for entity in record.get("polymer_entities", [])
    )
    return " | ".join(fields)


def discover(config_path, metadata_path, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite IDP metadata discovery: {output}")
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    metadata = json.loads(metadata_path.read_text(encoding="ascii"))
    eligible, excluded = [], []
    allowed_targets = set(config["scope"]["required_targets"])
    for record in metadata["records"]:
        text = record_text(record)
        targets = [
            target for target, patterns in TARGET_PATTERNS.items()
            if target in allowed_targets
            and any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
        ]
        antibody_present = bool(ANTIBODY_PATTERN.search(text))
        if len(targets) == 1 and antibody_present:
            eligible.append({
                "entry_id": record["entry_id"],
                "target": targets[0],
                "title": record["title"],
                "classification": "metadata_candidate_requires_coordinate_admission",
            })
        else:
            reasons = []
            if not targets:
                reasons.append("no_configured_idp_target_metadata")
            elif len(targets) > 1:
                reasons.append("ambiguous_idp_target_metadata")
            if not antibody_present:
                reasons.append("no_antibody_entity_metadata")
            excluded.append({
                "entry_id": record["entry_id"],
                "title": record["title"],
                "reasons": reasons,
            })
    target_count = len({row["target"] for row in eligible})
    payload = {
        "schema_version": 1,
        "status": (
            "metadata_idp_pool_ready"
            if len(eligible) >= config["scope"]["minimum_exact_new_structures"]
            and target_count >= config["scope"]["minimum_targets"]
            else "metadata_idp_pool_blocked"
        ),
        "classification": "metadata_only_exact_new_idp_discovery",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "source_metadata": str(metadata_path),
        "source_metadata_sha256": sha256(metadata_path),
        "queried_exact_new_entries": len(metadata["records"]),
        "eligible_exact_new_idp_entries": len(eligible),
        "eligible_target_count": target_count,
        "minimum_required_entries": config["scope"]["minimum_exact_new_structures"],
        "minimum_required_targets": config["scope"]["minimum_targets"],
        "eligible_entries": eligible,
        "excluded_entries": excluded,
        "decision": (
            "freeze_coordinate_admission"
            if len(eligible) >= config["scope"]["minimum_exact_new_structures"]
            and target_count >= config["scope"]["minimum_targets"]
            else "wait_for_later_cadence_snapshot_without_changing_protocol"
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
    parser.add_argument(
        "--metadata", type=Path,
        default=Path(
            "reviewer_outputs/idp_ensemble_correction_v3/"
            "exact_new_metadata_2026_08_15.json"
        ),
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    result = discover(
        config_path, ROOT / args.metadata, ROOT / config["outputs"]["discovery"]
    )
    print(json.dumps({
        "status": result["status"],
        "queried": result["queried_exact_new_entries"],
        "eligible_idp_entries": result["eligible_exact_new_idp_entries"],
        "eligible_targets": result["eligible_target_count"],
        "decision": result["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
