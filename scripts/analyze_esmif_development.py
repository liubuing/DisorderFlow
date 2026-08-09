#!/usr/bin/env python
"""Summarize the matched-position ESM-IF development baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="results/publication/h3_candidate_reranking_dev_v1/results.json")
    parser.add_argument("--out", default="results/ablation/esmif_h3_development_v1.json")
    args = parser.parse_args()
    input_path = ROOT / args.input
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = [row for row in payload["results"] if row["generator"] == "esm_if"]
    candidates = [candidate for row in rows for candidate in row["candidates"]]
    failures = [row for row in payload["failures"] if row["generator"] == "esm_if"]
    expected = len({row["unit"] for row in rows}) * len({row["seed"] for row in rows}) * 8
    aggregate = payload["aggregate"]["generators"]["esm_if"]
    output = {
        "schema_version": 1,
        "status": "development_only; no sealed final accessed",
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "model": "esm_if1_gvp4_t16_142M_UR50",
        "design_contract": "official H3 positions masked; all other heavy positions fixed; context coordinates retained without context sequences",
        "n_antigen_units": len({row["unit"] for row in rows}),
        "n_seed_rows": len(rows),
        "n_raw_candidates": sum(row["raw_candidates"] for row in rows),
        "expected_raw_candidates": expected,
        "n_unique_non_native_candidates": len(candidates),
        "n_failures": len(failures),
        "mean_candidate_native_recovery": float(np.mean([
            candidate["recovery"] for candidate in candidates])),
        "mean_ecls_native_normalized_rank": aggregate["mean_ecls_nnr"],
        "ecls_over_random_ci95": aggregate["ecls_over_random_ci95"],
        "mean_ecls_gain_over_complex_nll": aggregate["mean_gain"],
        "gain_ci95": aggregate["gain_ci95"],
        "decision": "ESM-IF is operational as a matched-position development baseline; these data are not homology-independent confirmatory evidence.",
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
