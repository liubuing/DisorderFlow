#!/usr/bin/env python
"""Prepare and sequence-audit the frozen nonviral exploratory v3 subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
HYDROPHOBIC = set("AILMFWVY")
POSITIVE = set("KRH")
NEGATIVE = set("DE")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def longest_hydrophobic_run(sequence):
    longest = current = 0
    for amino_acid in sequence:
        current = current + 1 if amino_acid in HYDROPHOBIC else 0
        longest = max(longest, current)
    return longest


def sequence_metrics(sequence, contract):
    allowed = set(contract["allowed_amino_acids"])
    charge = sum(aa in POSITIVE for aa in sequence) - sum(
        aa in NEGATIVE for aa in sequence
    )
    glycosylation = [
        match.group(0)
        for match in re.finditer(r"N[^P][ST]", sequence)
    ]
    deamidation = [motif for motif in contract["flag_deamidation_motifs"] if motif in sequence]
    isomerization = [motif for motif in contract["flag_isomerization_motifs"] if motif in sequence]
    oxidation = sorted({aa for aa in contract["flag_oxidation_residues"] if aa in sequence})
    run = longest_hydrophobic_run(sequence)
    checks = {
        "valid_amino_acids": set(sequence) <= allowed,
        "hydrophobic_run_ok": run <= contract["maximum_hydrophobic_run"],
        "charge_proxy_ok": abs(charge) <= contract["maximum_absolute_charge_proxy"],
    }
    return {
        "sequence": sequence,
        "length": len(sequence),
        "charge_proxy": charge,
        "hydrophobic_fraction": round(
            sum(aa in HYDROPHOBIC for aa in sequence) / len(sequence), 6
        ),
        "longest_hydrophobic_run": run,
        "n_linked_glycosylation_motifs": glycosylation,
        "deamidation_motifs": deamidation,
        "isomerization_motifs": isomerization,
        "oxidation_residues": oxidation,
        "checks": checks,
        "qc_pass": all(checks.values()) and not glycosylation,
    }


def analyze(config_path, candidates_path, subset_path, analysis_path, status_path):
    for path in (subset_path, analysis_path, status_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite nonviral artifact: {path}")
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    candidates = json.loads(candidates_path.read_text(encoding="ascii"))
    included_targets = set(config["source"]["included_targets"])
    included, excluded = {}, []
    for component_id, component in candidates["components"].items():
        if component["target"] in included_targets:
            included[component_id] = component
        else:
            excluded.append({
                "component_id": component_id,
                "target": component["target"],
                "reason": "viral_or_outside_frozen_nonviral_targets",
            })
    expected = config["selection"]
    candidate_count = sum(
        len(arm["candidates"])
        for component in included.values()
        for arm in component["arms"].values()
    )
    targets = {component["target"] for component in included.values()}
    if (
        len(included) != expected["expected_components"]
        or len(targets) != expected["expected_targets"]
        or candidate_count != expected["expected_candidates"]
    ):
        raise RuntimeError("Nonviral subset does not match the frozen counts")
    subset = {
        "schema_version": 1,
        "status": "nonviral_subset_frozen",
        "classification": config["classification"],
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "source_candidates": str(candidates_path),
        "source_candidates_sha256": sha256(candidates_path),
        "component_count": len(included),
        "target_count": len(targets),
        "candidate_count": candidate_count,
        "native_control_count": len(included),
        "components": included,
        "excluded_components": excluded,
        "claim_boundary": config["claim_boundary"],
    }
    records = []
    for component_id, component in included.items():
        for arm_name, arm in component["arms"].items():
            for candidate in arm["candidates"]:
                records.append({
                    "component_id": component_id,
                    "target": component["target"],
                    "arm": arm_name,
                    "substitution_bucket": candidate["substitution_bucket"],
                    **sequence_metrics(candidate["sequence"], config["sequence_qc"]),
                })
    analysis = {
        "schema_version": 1,
        "status": "nonviral_sequence_qc_complete",
        "classification": config["classification"],
        "subset": str(subset_path),
        "record_count": len(records),
        "qc_pass_count": sum(row["qc_pass"] for row in records),
        "flagged_count": sum(not row["qc_pass"] for row in records),
        "records": records,
        "claim_boundary": config["claim_boundary"],
    }
    experiment = {
        "schema_version": 1,
        "status": "experimental_measurements_intentionally_skipped",
        "measurement_gate_passed": False,
        "reason": "user_requested_computational_nonviral_branch",
        "effect_on_correction_gate": "original_v3_remains_4_of_5_and_blocked",
        "claim_boundary": config["claim_boundary"],
    }
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    subset_path.write_text(json.dumps(subset, indent=2) + "\n", encoding="ascii")
    analysis_path.write_text(json.dumps(analysis, indent=2) + "\n", encoding="ascii")
    status_path.write_text(json.dumps(experiment, indent=2) + "\n", encoding="ascii")
    return subset, analysis, experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/benchmarks/idp_ensemble_v3_nonviral_exploratory.yml"),
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    outputs = config["outputs"]
    subset, analysis, experiment = analyze(
        config_path,
        ROOT / config["source"]["candidates"],
        ROOT / outputs["subset"],
        ROOT / outputs["analysis"],
        ROOT / outputs["experiment_status"],
    )
    print(json.dumps({
        "subset_status": subset["status"],
        "components": subset["component_count"],
        "targets": subset["target_count"],
        "candidates": subset["candidate_count"],
        "qc_pass": analysis["qc_pass_count"],
        "flagged": analysis["flagged_count"],
        "experiment_status": experiment["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
