#!/usr/bin/env python
"""Generate matched positionwise PoE and single-state H3 pilot candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_npz(record):
    path = ROOT / record["npz"]
    if sha256(path) != record["npz_sha256"]:
        raise ValueError(f"Conditional NPZ hash mismatch: {path}")
    payload = np.load(path)
    indices = [
        int(index) for index in np.flatnonzero(np.asarray(payload["design_mask"]) > 0)
    ]
    log_p = np.asarray(payload["log_p"][0], dtype=np.float64)
    if not np.isfinite(log_p[indices]).all():
        raise ValueError(f"Non-finite conditional probabilities: {path}")
    return log_p, indices


def conditional_distribution(component, arm):
    rows = component["conformers"]
    states = [row["complex"] for row in rows]
    log_probs = []
    indices = None
    for row in states:
        log_p, observed = load_npz(row)
        if indices is None:
            indices = observed
        elif indices != observed:
            raise ValueError("H3 design masks differ across conformers")
        log_probs.append(log_p[indices])
    if arm == "single_state":
        combined = log_probs[0]
    elif arm == "ensemble_product_of_experts":
        combined = np.mean(np.stack(log_probs, axis=0), axis=0)
    else:
        raise ValueError(f"Unknown arm: {arm}")
    combined = combined - np.max(combined, axis=-1, keepdims=True)
    probabilities = np.exp(combined)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return probabilities, indices


def sample_h3(native, probabilities, rng, substitutions):
    if substitutions < 1 or substitutions > len(native):
        raise ValueError("Invalid substitution count")
    sequence = list(native)
    positions = rng.choice(len(native), size=substitutions, replace=False)
    for position in positions:
        row = probabilities[position].copy()
        row[ALPHABET.index(native[position])] = 0.0
        row /= row.sum()
        sequence[position] = ALPHABET[int(rng.choice(len(ALPHABET), p=row))]
    return "".join(sequence)


def candidate_score(sequence, native, probabilities):
    return float(sum(
        np.log(probabilities[index, ALPHABET.index(aa)])
        for index, aa in enumerate(sequence)
    ))


def generate_component(component, seeds, substitution_schedule):
    output = {}
    for arm in ("ensemble_product_of_experts", "single_state"):
        probabilities, indices = conditional_distribution(component, arm)
        if len(indices) != len(component["native_h3"]):
            raise ValueError("H3 design mask length mismatch")
        arm_rows = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            accepted = []
            for sample_index, substitutions in enumerate(substitution_schedule):
                attempts = 0
                while attempts < 10000:
                    attempts += 1
                    candidate = sample_h3(
                        component["native_h3"], probabilities, rng, substitutions
                    )
                    if candidate not in accepted:
                        accepted.append(candidate)
                        arm_rows.append({
                            "seed": seed,
                            "sample_index": sample_index,
                            "substitutions": substitutions,
                            "sequence": candidate,
                            "generation_score": candidate_score(
                                candidate, component["native_h3"], probabilities
                            ),
                        })
                        break
                else:
                    raise RuntimeError(
                        f"Unable to fill {component['component_id']} {arm} seed {seed}"
                    )
        output[arm] = arm_rows
        output[f"{arm}_native_control"] = {
            "sequence": component["native_h3"],
            "substitutions": 0,
            "generation_score": candidate_score(
                component["native_h3"], component["native_h3"], probabilities
            ),
        }
    return output


def generate(config_path, score_path=None, out_path=None):
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    if str(config.get("status", "")).startswith("superseded"):
        raise ValueError("Refusing to generate from a superseded pilot contract")
    score_item = config["conditional_score_artifact"]
    score_path = Path(score_path or ROOT / score_item["path"])
    if sha256(score_path) != score_item["sha256"]:
        raise ValueError("Conditional score artifact hash mismatch")
    score = json.loads(score_path.read_text(encoding="ascii"))
    clean_ids = {
        row["component_id"]
        for row in yaml.safe_load(
            (ROOT / config["component_registry"]["path"]).read_text(encoding="ascii")
        )["components"]
        if not row.get("sensitivity_only", False)
    }
    components = [row for row in score["component_scores"] if row["component_id"] in clean_ids]
    expected = config["matched_budget"]["components"]
    if len(components) != expected:
        raise ValueError(f"Expected {expected} clean components, found {len(components)}")
    budget = config["matched_budget"]
    generated = {
        component["component_id"]: generate_component(
            component,
            budget["seeds"],
            budget["substitutions_per_seed"],
        )
        for component in components
    }
    output = {
        "schema_version": 1,
        "status": "complete",
        "classification": config["classification"],
        "generator": "positionwise_conditional_product_of_experts_vs_pose_0_single_state",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "score_artifact": {"path": str(score_path.relative_to(ROOT)), "sha256": sha256(score_path)},
        "budget": budget,
        "components": generated,
        "claim_boundary": config["claim_boundary"],
    }
    output_path = Path(out_path or ROOT / config["outputs"]["raw_candidates"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite candidates: {output_path}")
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmarks/idp_ensemble_prospective_pilot_v2.yml")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    print(json.dumps(generate(ROOT / args.config, out_path=args.out), indent=2))


if __name__ == "__main__":
    main()
