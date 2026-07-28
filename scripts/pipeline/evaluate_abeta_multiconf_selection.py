#!/usr/bin/env python
"""Evaluate ensemble selection on held-out A-beta conformers."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "modules"))

from ensemble_pose_transfer import fixed_paratope_contact_map  # noqa: E402
from state_contact_scorer import extract_contact_map, score_sequence_on_contact_map  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/statecontrast/abeta_position_effect_v3_4ref_loro.yml")
    parser.add_argument("--pose-audit", default="outputs/abeta_multiconf_pose_panel_v1/pose_panel_audit.json")
    parser.add_argument("--candidates", default="outputs/statecontrast_domain_abeta_position_4ref_loro_v1/v4_position_effect/statecontrast_effect_guided_v4_scores.csv")
    parser.add_argument("--out", default="outputs/abeta_multiconf_selection_eval_v1")
    parser.add_argument("--random-replicates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2031)
    return parser.parse_args()


def load_unique_candidates(path, reference_id):
    rows = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["reference_pdb"] != reference_id:
                continue
            current = rows.get(row["sequence"])
            if current is None or float(row.get("statecontrast_score", 0)) > float(current.get("statecontrast_score", 0)):
                rows[row["sequence"]] = row
    return list(rows.values())


def robust_value(values):
    values = np.asarray(values, dtype=float)
    return float(
        0.45 * np.percentile(values, 25)
        + 0.25 * values.min()
        + 0.20 * values.mean()
        - 0.10 * values.std()
    )


def summarize(rows):
    return {
        "heldout_conformers": len(rows),
        "mean_minus_native": float(np.mean([row["ensemble_minus_native"] for row in rows])),
        "mean_minus_single_state": float(np.mean([row["ensemble_minus_single_state"] for row in rows])),
        "mean_minus_random": float(np.mean([row["ensemble_minus_random"] for row in rows])),
        "beat_native_rate": sum(row["ensemble_beats_native"] for row in rows) / len(rows),
        "beat_single_state_rate": sum(row["ensemble_beats_single_state"] for row in rows) / len(rows),
        "beat_random_rate": sum(row["ensemble_beats_random"] for row in rows) / len(rows),
    }


def main():
    args = parse_args()
    with open(PROJECT_ROOT / args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with open(PROJECT_ROOT / args.pose_audit, encoding="utf-8") as handle:
        audit = json.load(handle)
    if audit["status"] != "pass":
        raise SystemExit("Pose-panel audit has not passed")

    pose_rows = defaultdict(list)
    for row in audit["entries"]:
        if row["status"] == "pass":
            pose_rows[row["reference_pdb"]].append(row)

    results = []
    rng = random.Random(args.seed)
    for reference in config["references"]:
        reference_id = reference["pdb"]
        template_map = extract_contact_map(
            str(PROJECT_ROOT / reference["complex_pdb"]), peptide_chain=reference["peptide_chain"]
        )
        maps = [
            fixed_paratope_contact_map(PROJECT_ROOT / pose["pose_pdb"], template_map)
            for pose in pose_rows[reference_id]
        ]
        candidates = load_unique_candidates(PROJECT_ROOT / args.candidates, reference_id)
        sequences = [row["sequence"] for row in candidates]
        score_matrix = np.array([
            [score_sequence_on_contact_map(sequence, contact_map)["state_contact_score"] for contact_map in maps]
            for sequence in sequences
        ])
        native_scores = np.array([
            score_sequence_on_contact_map(template_map["paratope_sequence"], contact_map)["state_contact_score"]
            for contact_map in maps
        ])

        for holdout in range(len(maps)):
            training = [index for index in range(len(maps)) if index != holdout]
            ensemble_index = max(
                range(len(sequences)), key=lambda index: robust_value(score_matrix[index, training])
            )
            single_indices = [int(np.argmax(score_matrix[:, index])) for index in training]
            single_holdout = [float(score_matrix[index, holdout]) for index in single_indices]
            random_holdout = [
                float(score_matrix[rng.randrange(len(sequences)), holdout])
                for _ in range(args.random_replicates)
            ]
            ensemble_score = float(score_matrix[ensemble_index, holdout])
            native_score = float(native_scores[holdout])
            single_median = float(np.median(single_holdout))
            random_median = float(np.median(random_holdout))
            results.append({
                "reference_pdb": reference_id,
                "heldout_conformer": pose_rows[reference_id][holdout]["conformer"],
                "n_training_conformers": len(training),
                "n_candidates": len(sequences),
                "ensemble_candidate": candidates[ensemble_index]["candidate_id"],
                "ensemble_candidate_uid": (
                    f"{reference_id}_"
                    f"{hashlib.sha256(sequences[ensemble_index].encode('ascii')).hexdigest()[:12]}"
                ),
                "ensemble_holdout_score": round(ensemble_score, 4),
                "native_holdout_score": round(native_score, 4),
                "ensemble_minus_native": round(ensemble_score - native_score, 4),
                "single_state_selection_median": round(single_median, 4),
                "ensemble_minus_single_state": round(ensemble_score - single_median, 4),
                "random_selection_median": round(random_median, 4),
                "ensemble_minus_random": round(ensemble_score - random_median, 4),
                "ensemble_beats_native": ensemble_score > native_score,
                "ensemble_beats_single_state": ensemble_score > single_median,
                "ensemble_beats_random": ensemble_score > random_median,
            })

    by_reference = {}
    for reference_id in sorted({row["reference_pdb"] for row in results}):
        by_reference[reference_id] = summarize([
            row for row in results if row["reference_pdb"] == reference_id
        ])
    overall = summarize(results)
    status = (
        "pass"
        if overall["mean_minus_single_state"] >= 0 and overall["beat_random_rate"] >= 0.75
        else "partial"
    )
    summary = {
        "status": status,
        "selection_protocol": "leave-one-conformer-out with no heldout score used for selection",
        "overall": overall,
        "by_reference": by_reference,
        "claim_boundary": "contact-topology evaluation on computational template-transferred poses",
    }

    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "leave_one_conformer_out.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with open(out_dir / "selection_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(out_dir / "selection_report.md", "w", encoding="utf-8") as handle:
        handle.write("# Multi-Conformation Selection Evaluation\n\n")
        handle.write(f"Status: `{status}`; held-out evaluations: {len(results)}.\n\n")
        handle.write("Each candidate is selected without using the held-out conformer.\n\n")
        handle.write("| Reference | Mean vs native | Mean vs single | Mean vs random | Beat native | Beat single | Beat random |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for reference_id, row in by_reference.items():
            handle.write(
                f"| {reference_id} | {row['mean_minus_native']:.4f} | "
                f"{row['mean_minus_single_state']:.4f} | {row['mean_minus_random']:.4f} | "
                f"{row['beat_native_rate']:.2f} | {row['beat_single_state_rate']:.2f} | "
                f"{row['beat_random_rate']:.2f} |\n"
            )
        handle.write("\nThis evaluates a scoring protocol on predicted poses, not experimental binding.\n")
    print(f"Leave-one-conformer-out: {len(results)} evaluations; status={status}")


if __name__ == "__main__":
    main()
