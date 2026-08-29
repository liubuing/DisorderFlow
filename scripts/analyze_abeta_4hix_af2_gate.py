#!/usr/bin/env python
"""Apply frozen AF2, PRODIGY, stability, and diversity gates to 3D6 H3 candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import sysconfig
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hamming(first, second):
    if len(first) != len(second):
        raise ValueError("Cannot compare sequences with different lengths")
    return sum(left != right for left, right in zip(first, second, strict=True))


def validate_inputs(config, config_path, af2_path):
    if sha256(ROOT / config["input"]["prospective_results"]) != config["input"][
        "prospective_results_sha256"
    ]:
        raise ValueError("Prospective result hash mismatch")
    af2 = json.loads(af2_path.read_text(encoding="utf-8"))
    if af2["config_sha256"] != sha256(config_path):
        raise ValueError("AF2 result was not generated from the current frozen config")
    if af2["status"] != "af2_complete" or af2["summary"]["failures"] != 0:
        raise ValueError("AF2 panel is incomplete or contains failed slots")
    if af2["summary"]["successes"] != int(config["af2"]["prediction_slots"]):
        raise ValueError("AF2 successful slot count differs from frozen contract")
    return af2


def run_prodigy(af2):
    pdb_paths = [ROOT / row["pdb"] for row in af2["results"]]
    pdb_dirs = {path.parent for path in pdb_paths}
    if len(pdb_dirs) != 1:
        raise ValueError("AF2 structures do not share one frozen directory")
    for path, row in zip(pdb_paths, af2["results"], strict=True):
        if not path.exists() or sha256(path) != row["pdb_sha256"]:
            raise ValueError(f"AF2 structure hash mismatch: {path}")
    executable_name = "prodigy.exe" if os.name == "nt" else "prodigy"
    candidates = [
        shutil.which("prodigy"),
        str(Path(sys.executable).parent / executable_name),
        str(Path(sysconfig.get_path("scripts", scheme="nt_user")) / executable_name)
        if os.name == "nt" else None,
    ]
    executable = next(
        (Path(value) for value in candidates if value and Path(value).exists()), None
    )
    if executable is None:
        raise FileNotFoundError("prodigy executable was not found in PATH or Python Scripts")
    command = [
        str(executable), str(next(iter(pdb_dirs))),
        "--selection", "A,B", "C", "-q", "-np", "4",
    ]
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        command, capture_output=True, text=True, env=env, timeout=900
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr[-4000:])
    energies = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) == 2:
            try:
                energies[fields[0].removesuffix("_model0")] = float(fields[1])
            except ValueError:
                continue
    output = {}
    for row in af2["results"]:
        stem = Path(row["pdb"]).stem
        if stem not in energies:
            raise ValueError(f"Missing PRODIGY score for {stem}")
        output[row["prediction_id"]] = energies[stem]
    return output, command


def summarize_entities(af2, energies):
    entities = {row["entity_id"]: row for row in af2["entities"]}
    grouped = defaultdict(list)
    for row in af2["results"]:
        grouped[row["entity_id"]].append(row)
    summaries = []
    for entity_id in sorted(grouped):
        rows = grouped[entity_id]
        iptm = np.asarray([row["iptm"] for row in rows], dtype=float)
        interface_pae = np.asarray([row["interface_pae"] for row in rows], dtype=float)
        delta_g = np.asarray([energies[row["prediction_id"]] for row in rows], dtype=float)
        entity = entities[entity_id]
        summaries.append({
            "entity_id": entity_id,
            "entity_type": entity["entity_type"],
            "pre_af2_rank": entity["pre_af2_rank"],
            "h3_sequence": entity["h3_sequence"],
            "heavy_sequence": entity["heavy_sequence"],
            "light_sequence": entity["light_sequence"],
            "n_successful_seeds": len(rows),
            "median_iptm": float(np.median(iptm)),
            "worst_iptm": float(np.min(iptm)),
            "best_iptm": float(np.max(iptm)),
            "iptm_seed_range": float(np.max(iptm) - np.min(iptm)),
            "median_interface_pae": float(np.median(interface_pae)),
            "worst_interface_pae": float(np.max(interface_pae)),
            "median_prodigy_delta_g_kcal_mol": float(np.median(delta_g)),
            "worst_prodigy_delta_g_kcal_mol": float(np.max(delta_g)),
            "seed_rows": [
                {
                    "seed": row["af2_seed"],
                    "iptm": row["iptm"],
                    "interface_pae": row["interface_pae"],
                    "prodigy_delta_g_kcal_mol": energies[row["prediction_id"]],
                    "pdb": row["pdb"],
                    "pdb_sha256": row["pdb_sha256"],
                }
                for row in sorted(rows, key=lambda item: item["af2_seed"])
            ],
        })
    return summaries


def fixed_rank_score(row, weights):
    iptm = min(1.0, max(0.0, row["median_iptm"]))
    worst = min(1.0, max(0.0, row["worst_iptm"]))
    pae_inverse = min(1.0, max(0.0, 1.0 - row["median_interface_pae"] / 32.0))
    dg_inverse = min(1.0, max(0.0, -row["median_prodigy_delta_g_kcal_mol"] / 15.0))
    return (
        weights["median_iptm"] * iptm
        + weights["worst_iptm"] * worst
        + weights["median_interface_pae_inverse"] * pae_inverse
        + weights["median_prodigy_dg_inverse"] * dg_inverse
    )


def apply_gates(config, summaries):
    selection = config["selection"]
    by_id = {row["entity_id"]: row for row in summaries}
    native = by_id["native"]
    mpnn = by_id["proteinmpnn_best"]
    baselines = {
        "native": {key: native[key] for key in (
            "median_iptm", "worst_iptm", "median_interface_pae",
            "median_prodigy_delta_g_kcal_mol")},
        "proteinmpnn_best": {key: mpnn[key] for key in (
            "median_iptm", "worst_iptm", "median_interface_pae",
            "median_prodigy_delta_g_kcal_mol")},
    }
    for row in summaries:
        if row["entity_type"] != "candidate":
            continue
        checks = {
            "all_seed_success": (
                not selection["require_all_seed_success"]
                or row["n_successful_seeds"] == len(config["af2"]["seeds"])
            ),
            "median_iptm_noninferior_native": row["median_iptm"] >= (
                native["median_iptm"]
                + selection["minimum_median_iptm_relative_to_native"]
            ),
            "worst_iptm_noninferior_native": row["worst_iptm"] >= (
                native["worst_iptm"]
                + selection["minimum_worst_iptm_relative_to_native"]
            ),
            "iptm_seed_stability": row["iptm_seed_range"] <= selection[
                "maximum_iptm_seed_range"
            ],
            "interface_pae_noninferior_native": row["median_interface_pae"] <= (
                native["median_interface_pae"]
                + selection["maximum_median_interface_pae_relative_to_native"]
            ),
            "prodigy_noninferior_native": row[
                "median_prodigy_delta_g_kcal_mol"
            ] <= (
                native["median_prodigy_delta_g_kcal_mol"]
                + selection["maximum_median_prodigy_dg_relative_to_native_kcal_mol"]
            ),
            "not_below_both_iptm_controls": (
                not selection["require_not_below_both_native_and_proteinmpnn_median_iptm"]
                or row["median_iptm"] >= min(native["median_iptm"], mpnn["median_iptm"])
            ),
        }
        row["gate_checks"] = checks
        row["all_gates_pass"] = all(checks.values())
        row["fixed_rank_score"] = fixed_rank_score(row, selection["ranking"])
    return baselines


def diverse_shortlist(config, summaries):
    candidates = [
        row for row in summaries
        if row["entity_type"] == "candidate" and row["all_gates_pass"]
    ]
    candidates.sort(key=lambda row: (
        -row["fixed_rank_score"], -row["median_iptm"],
        row["median_interface_pae"], row["entity_id"],
    ))
    selected = []
    minimum = int(config["selection"]["minimum_pairwise_hamming"])
    for row in candidates:
        if all(hamming(row["h3_sequence"], prior["h3_sequence"]) >= minimum
               for prior in selected):
            selected.append(row)
        if len(selected) == int(config["selection"]["shortlist_target"]):
            break
    return selected


def export_shortlist(config, shortlist):
    csv_path = ROOT / config["output"]["shortlist_csv"]
    fasta_path = ROOT / config["output"]["shortlist_fasta"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "final_rank", "entity_id", "h3_sequence", "pre_af2_rank", "fixed_rank_score",
        "median_iptm", "worst_iptm", "iptm_seed_range", "median_interface_pae",
        "median_prodigy_delta_g_kcal_mol", "status",
    ]
    rows = []
    fasta = []
    for rank, candidate in enumerate(shortlist, 1):
        rows.append({
            "final_rank": rank,
            "entity_id": candidate["entity_id"],
            "h3_sequence": candidate["h3_sequence"],
            "pre_af2_rank": candidate["pre_af2_rank"],
            "fixed_rank_score": candidate["fixed_rank_score"],
            "median_iptm": candidate["median_iptm"],
            "worst_iptm": candidate["worst_iptm"],
            "iptm_seed_range": candidate["iptm_seed_range"],
            "median_interface_pae": candidate["median_interface_pae"],
            "median_prodigy_delta_g_kcal_mol": candidate[
                "median_prodigy_delta_g_kcal_mol"],
            "status": "computationally_prioritized_manual_construct_review_required",
        })
        fasta.extend([
            f">{candidate['entity_id']}|heavy|manual_construct_review_required",
            candidate["heavy_sequence"],
            f">{candidate['entity_id']}|light|manual_construct_review_required",
            candidate["light_sequence"],
        ])
    with csv_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    fasta_path.write_text("\n".join(fasta) + "\n", encoding="ascii")
    return csv_path, fasta_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmarks/abeta_4hix_af2_gate_v1.yml")
    parser.add_argument("--af2")
    parser.add_argument("--out")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    af2_path = ROOT / (args.af2 or config["output"]["af2_results"])
    af2 = validate_inputs(config, config_path, af2_path)
    energies, prodigy_command = run_prodigy(af2)
    summaries = summarize_entities(af2, energies)
    baselines = apply_gates(config, summaries)
    shortlist = diverse_shortlist(config, summaries)
    csv_path, fasta_path = export_shortlist(config, shortlist)
    target = int(config["selection"]["shortlist_target"])
    output = {
        "schema_version": 1,
        "status": (
            "af2_gate_complete_target_reached" if len(shortlist) == target
            else "af2_gate_complete_insufficient_eligible_candidates"
        ),
        "config": args.config,
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__)),
        "af2_results": str(af2_path.relative_to(ROOT).as_posix()),
        "af2_results_sha256": sha256(af2_path),
        "prodigy_version": importlib.metadata.version("prodigy-prot"),
        "prodigy_command": prodigy_command,
        "baselines": baselines,
        "summary": {
            "candidate_entities": sum(row["entity_type"] == "candidate" for row in summaries),
            "all_gates_pass": sum(
                row.get("all_gates_pass", False) for row in summaries
                if row["entity_type"] == "candidate"
            ),
            "diverse_shortlist": len(shortlist),
            "shortlist_target": target,
        },
        "shortlist": [
            {"final_rank": rank, **row} for rank, row in enumerate(shortlist, 1)
        ],
        "entities": summaries,
        "claim_boundary": config["claim_boundary"],
    }
    out_path = ROOT / (args.out or config["output"]["analysis"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    temporary.replace(out_path)
    print(json.dumps({"status": output["status"], **output["summary"]}, indent=2))
    print(f"Wrote {out_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {fasta_path}")


if __name__ == "__main__":
    main()
