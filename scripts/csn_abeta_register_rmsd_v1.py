#!/usr/bin/env python3
"""Register-aware CSN Abeta RMSD sensitivity analysis.

Post hoc sensitivity analysis, not a preregistered confirmatory endpoint.
The frozen primary analysis used a 5-residue AEFRH core that mixes two
peptide registers (A2-start and D1-start). This script splits the cohort by
register and compares 6-residue cores within each register:

  R0 (A2-start): AEFRHD
  R1 (D1-start): DAEFRH

It also compares the two L3 entries (3IFN R0 vs 3IFL R1) on the shared
AEFRHD core to estimate a within-lineage register effect.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from csn_abeta_rmsd_v1 import (  # noqa: E402
    build_records,
    collect_pairs,
    decide,
    entry_pair_values,
    rmsd,
    sha256,
)

ENTRIES = ROOT / "results" / "csn_abeta_v1" / "abeta_entries.csv"
OUT = ROOT / "results" / "csn_abeta_v1"
TIERS = {"same_bound_conformer_max": 0.5, "related_max": 1.0, "ncs_floor_max": 0.5}
COHORTS = {
    "R0_A2_start": {
        "cores": ["AEFRH", "AEFRHD"],
        "cohort": {
            "L1": ["2IPU", "2R0W"],
            "L2": ["3BAE", "3BKJ"],
            "L3": ["3IFN"],
            "L4": ["3IFO"],
            "L5": ["3IFP"],
            "L9": ["6CO3"],
        },
    },
    "R1_D1_start": {
        "cores": ["AEFRH", "DAEFRH"],
        "cohort": {
            "L3": ["3IFL"],
            "L6": ["4HIX", "4OJF", "4ONF"],
            "L7": ["5CSZ"],
            "L8": ["5W3P"],
        },
    },
}


def summarize_group(entries, cohort, core):
    records, missing = build_records(entries, cohort, core)
    pairs = collect_pairs(records)
    cross = [pair for pair in pairs if pair["pair_type"] == "cross_lineage"]
    cross_entry = entry_pair_values(cross)
    ncs = [pair["rmsd"] for pair in pairs if pair["pair_type"] == "ncs"]
    within = [pair["rmsd"] for pair in pairs if pair["pair_type"] == "within_lineage"]
    summary = {
        "core": core,
        "n_entries": len({record["pdb"] for record in records}),
        "n_lineages": len({record["lineage"] for record in records}),
        "n_chains": len(records),
        "n_cross_pairs": len(cross),
        "ncs": {
            "n": len(ncs),
            "mean": float(np.mean(ncs)) if ncs else None,
            "median": float(np.median(ncs)) if ncs else None,
        },
        "within_lineage": {
            "n": len(within),
            "mean": float(np.mean(within)) if within else None,
        },
        "cross_lineage": {
            "n": len(cross),
            "mean": float(np.mean([pair["rmsd"] for pair in cross])) if cross else None,
            "median": float(np.median([pair["rmsd"] for pair in cross])) if cross else None,
            "entry_pair_median": float(np.median(list(cross_entry.values()))) if cross_entry else None,
            "entry_pair_mean": float(np.mean(list(cross_entry.values()))) if cross_entry else None,
        },
        "tiers": {
            "same_bound_conformer": int(sum(
                1 for pair in cross if pair["rmsd"] <= TIERS["same_bound_conformer_max"])),
            "related": int(sum(
                1 for pair in cross
                if TIERS["same_bound_conformer_max"] < pair["rmsd"] <= TIERS["related_max"])),
            "different": int(sum(
                1 for pair in cross if pair["rmsd"] > TIERS["related_max"])),
        },
        "missing": missing,
    }
    if summary["ncs"]["n"] == 0:
        summary["verdict"] = "ncs_unavailable"
    else:
        summary["verdict"] = decide(summary, TIERS)
    return summary, records


def main() -> int:
    entries = pd.read_csv(ENTRIES)
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = {}
    records_by_group = {}
    for label, spec in COHORTS.items():
        outputs[label] = {}
        for core in spec["cores"]:
            summary, records = summarize_group(entries, spec["cohort"], core)
            summary["classification"] = (
                "post_hoc_register_aware_sensitivity_not_confirmatory")
            outputs[label][core] = summary
            if core == spec["cores"][0]:
                records_by_group[label] = records

    # Within-lineage register effect: 3IFN (R0) vs 3IFL (R1), shared AEFRHD.
    register_effect = None
    r0_records = [record for record in records_by_group["R0_A2_start"]
                  if record["pdb"] == "3IFN"]
    r1_records = [record for record in records_by_group["R1_D1_start"]
                  if record["pdb"] == "3IFL"]
    if r0_records and r1_records:
        # Reload both on the shared 6-mer AEFRHD.
        shared_records, _missing = build_records(
            entries, {"L3": ["3IFN", "3IFL"]}, "AEFRHD")
        values = [rmsd(first["coords"], second["coords"])
                  for first in shared_records for second in shared_records
                  if first["pdb"] == "3IFN" and second["pdb"] == "3IFL"]
        register_effect = {
            "entries": ["3IFN", "3IFL"],
            "core": "AEFRHD",
            "rmsd": float(np.mean(values)) if values else None,
            "note": "Same lineage, different peptide register; estimates register sensitivity.",
        }

    payload = {
        "schema_version": 1,
        "created": date.today().isoformat(),
        "analysis": "register_aware_sensitivity",
        "classification": "post_hoc_not_confirmatory",
        "source_entries": str(ENTRIES.relative_to(ROOT)).replace("\\", "/"),
        "source_entries_sha256": sha256(ENTRIES),
        "groups": outputs,
        "within_lineage_register_effect": register_effect,
    }
    path = OUT / "register_summary.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
