#!/usr/bin/env python3
"""Parse H3 structure predictions using absolute gates without ipTM ranking."""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_scores(directory, construct_id, antibody_length, interface=False):
    records = []
    for path in sorted(directory.glob(f"{construct_id}_scores_*.json")):
        score = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "score_json": str(path),
            "mean_plddt": float(np.mean(score["plddt"][:antibody_length])),
            "iptm": float(score.get("iptm", 0.0)),
        }
        if interface:
            pae = np.asarray(score["pae"], dtype=np.float64)
            cross = np.concatenate((pae[:antibody_length, antibody_length:].ravel(),
                                    pae[antibody_length:, :antibody_length].ravel()))
            row["interface_pae"] = float(cross.mean())
        records.append(row)
    return records


def mean(records, key):
    values = [row[key] for row in records if key in row]
    return float(np.mean(values)) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--fab-results", required=True, type=Path)
    parser.add_argument("--abeta11-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    output = []
    for source in panel["records"]:
        construct_id = source["construct_id"]
        fab = parse_scores(args.fab_results, construct_id, 443)
        complex_records = parse_scores(args.abeta11_results, construct_id, 443, interface=True)
        fab_plddt = mean(fab, "mean_plddt")
        complex_iptm = mean(complex_records, "iptm")
        interface_pae = mean(complex_records, "interface_pae")
        output.append({
            "construct_id": construct_id,
            "h3_mutations": source["h3_mutations"],
            "developability_status": source["developability_status"],
            "fab_predictions": len(fab),
            "fab_mean_plddt": fab_plddt,
            "fab_fold_gate": fab_plddt is not None and fab_plddt >= 70.0,
            "abeta11_predictions": len(complex_records),
            "abeta11_mean_iptm": complex_iptm,
            "abeta11_mean_interface_pae": interface_pae,
            "multimer_absolute_gate": (
                complex_iptm is not None and complex_iptm >= 0.25
                and interface_pae is not None and interface_pae <= 20.0
            ),
            "multimer_status": (
                "passed_absolute_gate" if complex_records and complex_iptm >= 0.25 and interface_pae <= 20.0
                else "failed_absolute_gate" if complex_records
                else "not_evaluated_resource_blocked"
            ),
        })
    report = {
        "gates": {"fab_mean_plddt_min": 70.0, "multimer_mean_iptm_min": 0.25,
                  "multimer_mean_interface_pae_max": 20.0},
        "selection_policy": "ipTM is an absolute rejection gate only; records retain pre-structure panel order",
        "multimer_execution_note": (
            "A-beta1-11 prediction requested approximately 34 GB pinned host memory and did not "
            "complete on the available 8 GB GPU. Missing multimer results are not treated as failures."
        ),
        "records": output,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "complete_fab": sum(row["fab_predictions"] >= 5 for row in output),
        "fab_pass": sum(row["fab_fold_gate"] for row in output),
        "complete_abeta11": sum(row["abeta11_predictions"] >= 5 for row in output),
        "multimer_absolute_pass": sum(row["multimer_absolute_gate"] for row in output),
    }, indent=2))


if __name__ == "__main__":
    main()
