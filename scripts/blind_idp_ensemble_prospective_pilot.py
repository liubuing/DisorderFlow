#!/usr/bin/env python
"""Select fixed-budget pilot candidates and emit an opaque assay manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def blind(config_path, candidates_path, manifest_path, key_path):
    config_path = Path(config_path)
    candidates_path = Path(candidates_path)
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    if sha256(config_path) != "2cb4ca3b5a6d0f134c9e61615f2356d3b0e9990aa62d2795ecc55f43f0c90003":
        raise ValueError("Pilot v2 config hash mismatch")
    candidates = json.loads(candidates_path.read_text(encoding="ascii"))
    if candidates["config_sha256"] != sha256(config_path):
        raise ValueError("Candidate/config provenance mismatch")
    if manifest_path.exists() or key_path.exists():
        raise FileExistsError("Refusing to overwrite blinded pilot outputs")
    rng = random.Random(config["blinding"]["randomization_seed"])
    selected, key = [], {}
    counts = config["selection"]["experimental_candidate_substitution_counts"]
    for component_id, arms in sorted(candidates["components"].items()):
        for arm in ("ensemble_product_of_experts", "single_state"):
            rows = [row for row in arms[arm] if row["substitutions"] in counts]
            chosen = []
            for count in counts:
                pool = [row for row in rows if row["substitutions"] == count]
                chosen.append(max(pool, key=lambda row: (row["generation_score"], row["sequence"])))
            for row in chosen:
                code = f"P{len(selected) + 1:03d}"
                key[code] = {
                    "component_id": component_id,
                    "arm": arm,
                    "sequence": row["sequence"],
                    "substitutions": row["substitutions"],
                }
                selected.append({
                    "construct_code": code,
                    "component_id": component_id,
                    "substitutions": row["substitutions"],
                    "native_control": False,
                })
        native_code = f"P{len(selected) + 1:03d}"
        key[native_code] = {
            "component_id": component_id,
            "arm": "native_control",
            "sequence": arms["ensemble_product_of_experts_native_control"]["sequence"],
            "substitutions": 0,
        }
        selected.append({
            "construct_code": native_code,
            "component_id": component_id,
            "substitutions": 0,
            "native_control": True,
        })
    rng.shuffle(selected)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "status": "blinded_construct_manifest_ready_for_experimental_qc",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "candidate_input": str(candidates_path.relative_to(ROOT)),
        "candidate_input_sha256": sha256(candidates_path),
        "constructs": selected,
        "claim_boundary": config["claim_boundary"],
    }, indent=2) + "\n", encoding="ascii")
    key_path.write_text(json.dumps({
        "schema_version": 1,
        "status": "sealed_blind_key_do_not_share_with_assay_analyst",
        "manifest_sha256": sha256(manifest_path),
        "constructs": key,
    }, indent=2) + "\n", encoding="ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmarks/idp_ensemble_prospective_pilot_v2.yml")
    parser.add_argument("--candidates", default="reviewer_outputs/idp_ensemble_prospective_pilot_v2/candidates.json")
    parser.add_argument("--manifest", default="reviewer_outputs/idp_ensemble_prospective_pilot_v2/blinded_construct_manifest.json")
    parser.add_argument("--key", default="reviewer_outputs/idp_ensemble_prospective_pilot_v2/blind_key.json")
    args = parser.parse_args()
    blind(ROOT / args.config, ROOT / args.candidates, ROOT / args.manifest, ROOT / args.key)


if __name__ == "__main__":
    main()
