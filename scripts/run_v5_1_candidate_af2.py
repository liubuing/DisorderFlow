#!/usr/bin/env python3
"""Run GPU AF2-Multimer screening for corrected v5.1 A-beta candidates."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from build_design_variant_dataset import _batch_af2_wsl  # noqa: E402

ABETA42 = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default="results/v5_1_candidates/abeta42/af2_candidates.csv")
    parser.add_argument("--output", default="results/v5_1_candidates/abeta42/af2_screen.json")
    parser.add_argument("--structures", default="results/v5_1_candidates/abeta42/af2_structures")
    parser.add_argument("--recycles", type=int, default=3)
    parser.add_argument("--max-candidates", type=int)
    args = parser.parse_args()

    with open(args.candidates, newline="", encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle))
    if args.max_candidates:
        candidates = candidates[:args.max_candidates]
    if not candidates:
        raise RuntimeError("No quality-gated candidates are available for AF2 screening")
    local_fv = "heavy_sequence" in candidates[0] and "light_sequence" in candidates[0]
    sequences = [
        f"{candidate['heavy_sequence']}:{candidate['light_sequence']}"
        if local_fv else candidate["full_antibody_sequence"]
        for candidate in candidates
    ]
    results = _batch_af2_wsl(
        sequences, ABETA42, args.recycles, warmup_seq=sequences[0],
        output_dir=args.structures)

    records = []
    for candidate, result in zip(candidates, results):
        record = dict(candidate)
        if not result or not result.get("success"):
            record.update({"af2_success": False, "af2_error": (result or {}).get("error", "missing result")})
            records.append(record)
            continue
        plddt = np.asarray(result.get("plddt_seq", []), dtype=np.float64)
        antibody_length = (
            len(candidate["heavy_sequence"]) + len(candidate["light_sequence"])
            if local_fv else len(candidate["full_antibody_sequence"])
        )
        record.update({
            "af2_success": True,
            "af2_iptm": float(result.get("iptm", 0.0)),
            "af2_ptm": float(result.get("ptm", 0.0)),
            "af2_antibody_plddt": float(plddt[:antibody_length].mean()) if len(plddt) >= antibody_length else None,
            "af2_epitope_plddt": float(plddt[antibody_length:].mean()) if len(plddt) > antibody_length else None,
            "af2_interface_pae": float(result.get("interface_pae", math.inf)),
            "af2_max_pae": float(result.get("max_pae", math.inf)),
            "af2_elapsed": float(result.get("elapsed", 0.0)),
            "af2_structure": str(Path(args.structures) / f"{len(records):03d}.pdb"),
        })
        record["af2_pass"] = (
            record["af2_iptm"] >= 0.25
            and record["af2_antibody_plddt"] is not None
            and record["af2_antibody_plddt"] >= 0.60
            and record["af2_interface_pae"] <= 20.0
        )
        record["af2_rank_score"] = (
            0.50 * record["af2_iptm"]
            + 0.25 * record["af2_antibody_plddt"]
            + 0.15 * max(0.0, 1.0 - record["af2_interface_pae"] / 30.0)
            + 0.10 * float(candidate.get("contact_score", candidate.get("factual_contact", 0.0)))
        )
        records.append(record)

    records.sort(
        key=lambda record: (record.get("af2_pass", False), record.get("af2_rank_score", -1.0)),
        reverse=True)
    for rank, record in enumerate(records, 1):
        record["af2_rank"] = rank
    valid = [record for record in records if record.get("af2_success")]
    report = {
        "target": "A-beta42",
        "target_sequence": ABETA42,
        "backend": "AlphaFold2-Multimer v3, WSL CUDA, single-sequence features",
        "recycles": args.recycles,
        "n_candidates": len(candidates),
        "n_success": len(valid),
        "n_pass": sum(record.get("af2_pass", False) for record in records),
        "summary": {
            "mean_iptm": float(np.mean([record["af2_iptm"] for record in valid])) if valid else None,
            "max_iptm": max((record["af2_iptm"] for record in valid), default=None),
            "mean_antibody_plddt": float(np.mean([record["af2_antibody_plddt"] for record in valid])) if valid else None,
            "min_interface_pae": min((record["af2_interface_pae"] for record in valid), default=None),
        },
        "gates": {
            "af2_iptm_min": 0.25,
            "antibody_plddt_min": 0.60,
            "interface_pae_max": 20.0,
        },
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
