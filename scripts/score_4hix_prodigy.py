#!/usr/bin/env python
"""Score frozen 4HIX AF2 structures with independent PRODIGY affinity."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--af2", default=str(ROOT / "results/ablation/4hix_matched_af2_v1/results.json")
    )
    parser.add_argument("--out", default=str(ROOT / "results/ablation/4hix_prodigy_v1.json"))
    args = parser.parse_args()
    af2_path = Path(args.af2)
    af2 = json.loads(af2_path.read_text())
    pdb_dir = af2_path.parent / "pdb"
    executable = Path(sys.executable).parent / ("prodigy.exe" if os.name == "nt" else "prodigy")
    command = [str(executable), str(pdb_dir), "--selection", "A,B", "C", "-q", "-np", "4"]
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(command, capture_output=True, text=True, env=env, timeout=300)
    if completed.returncode:
        raise RuntimeError(completed.stderr[-4000:])
    energies = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) == 2:
            energies[fields[0].removesuffix("_model0")] = float(fields[1])

    rows = []
    for row in af2["results"]:
        if not row.get("pdb"):
            continue
        stem = Path(row["pdb"]).stem
        if stem not in energies:
            raise ValueError(f"Missing PRODIGY output for {stem}")
        rows.append(
            {
                "id": row["id"],
                "arm": row["arm"],
                "h3": row["h3"],
                "seed": row["seed"],
                "predicted_delta_g_kcal_mol": energies[stem],
                "source_pdb": row["pdb"],
                "source_pdb_sha256": row["pdb_sha256"],
            }
        )

    by_candidate = defaultdict(list)
    candidate_arm = {}
    for row in rows:
        by_candidate[row["id"]].append(row["predicted_delta_g_kcal_mol"])
        candidate_arm[row["id"]] = row["arm"]
    candidates = {
        candidate_id: {
            "arm": candidate_arm[candidate_id],
            "n_seeds": len(values),
            "median_delta_g_kcal_mol": float(np.median(values)),
            "min_delta_g_kcal_mol": float(np.min(values)),
            "max_delta_g_kcal_mol": float(np.max(values)),
        }
        for candidate_id, values in by_candidate.items()
    }
    by_arm = defaultdict(list)
    for candidate in candidates.values():
        by_arm[candidate["arm"]].append(candidate["median_delta_g_kcal_mol"])
    arms = {
        arm: {
            "n_candidates": len(values),
            "mean_candidate_median_delta_g_kcal_mol": float(np.mean(values)),
            "min_candidate_median_delta_g_kcal_mol": float(np.min(values)),
            "max_candidate_median_delta_g_kcal_mol": float(np.max(values)),
        }
        for arm, values in by_arm.items()
    }
    output = {
        "schema_version": 1,
        "status": "independent_computational_score_not_experimental_affinity",
        "prodigy_version": importlib.metadata.version("prodigy-prot"),
        "selection": "A,B C",
        "temperature_celsius": 25.0,
        "caveat": (
            "PRODIGY is applied to low-confidence single-sequence AF2 models; "
            "large seed spread prohibits an affinity claim."
        ),
        "rows": rows,
        "candidates": candidates,
        "arms": arms,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(json.dumps(arms, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
