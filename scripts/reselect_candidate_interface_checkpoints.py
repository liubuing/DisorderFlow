#!/usr/bin/env python
"""Re-select v2 raw checkpoints by lowest validation loss (train-only criterion)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_FILES = {
    "2041": "results/confidence_candidate_interface_v2/s2041_all_evaluation.json",
    "2053": "results/confidence_candidate_interface_v2/s2053_all_evaluation.json",
    "2069": "results/confidence_candidate_interface_v2/s2069_all_evaluation.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reselect(output: Path) -> dict:
    selections = []
    for seed, relative in EVAL_FILES.items():
        path = ROOT / relative
        document = json.loads(path.read_text(encoding="utf-8"))
        raw = [
            row for row in document["evaluations"]
            if row["weights_kind"] == "raw" and row.get("stored_avg_val_loss") is not None
        ]
        if not raw:
            raise ValueError(f"No raw checkpoints with validation loss for seed {seed}")
        selected = min(raw, key=lambda row: row["stored_avg_val_loss"])
        selections.append({
            "seed": seed,
            "checkpoint": selected["checkpoint"],
            "checkpoint_sha256": sha256(Path(selected["checkpoint"])),
            "iteration": selected["iteration"],
            "weights_kind": selected["weights_kind"],
            "val_loss": selected["stored_avg_val_loss"],
            "n_raw_candidates": len(raw),
        })

    report = {
        "schema_version": "candidate_interface_v2_checkpoint_reselection_v1",
        "classification": "development_only",
        "rationale": (
            "The prior selection maximized calibration Spearman, but the original "
            "calibration exhibited variance collapse (Spearman ~0), so the criterion "
            "degenerated and selected undertrained checkpoints (iter 100/300/400). "
            "Re-selection uses lowest validation loss, a train-only signal independent "
            "of calibration, test, and the extension scaffolds."),
        "selection_criterion": "lowest raw-checkpoint validation loss",
        "selections": selections,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="results/candidate_interface_multiscaffold_calibration_ext/checkpoint_reselection.json")
    args = parser.parse_args()
    print(json.dumps(reselect(ROOT / args.output), indent=2))


if __name__ == "__main__":
    main()
