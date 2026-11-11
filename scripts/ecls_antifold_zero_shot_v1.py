#!/usr/bin/env python3
"""Score the frozen ECLS temporal panel with AntiFold zero-shot.

M2 of the cross-architecture preregistration. The complex/apo PDB
construction, H3 positions and shuffle protocol are identical to the frozen
ProteinMPNN ECLS analysis; only the per-residue log-probability source changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_h3_epitope_delta import (  # noqa: E402
    aggregate_inference_units,
    bootstrap_mean_ci,
    read_record,
    select_inference_units,
    sha256_text,
    write_record_backbone,
)
from benchmark_h3_scorer_sanity import composition_shuffles  # noqa: E402

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
ANTIFOLD_PYTHON = ROOT / ".venv_antifold314" / "Scripts" / "python.exe"
ANTIFOLD_REPO = ROOT / ".tools" / "AntiFold"
PREREG = ROOT / "results" / "ecls_cross_arch_v1" / "preregistration.json"


def run_antifold(pdb_path: Path, out_dir: Path, include_antigen: bool, device: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    chains = {
        line[21] for line in pdb_path.read_text(encoding="ascii").splitlines()
        if line.startswith("ATOM")
    }
    args = [
        str(ANTIFOLD_PYTHON), "-m", "antifold.main",
        "--pdb_file", str(pdb_path),
        "--out_dir", str(out_dir),
        "--device", device,
        "--num_seq_per_target", "0",
        "--verbose", "0",
    ]
    if "L" in chains:
        args.extend(["--heavy_chain", "H", "--light_chain", "L"])
    else:
        # Single-domain VHH: ANARCII requires nanobody mode.
        args.extend(["--nanobody_chain", "H"])
    if include_antigen and "P" in chains:
        args.extend(["--antigen_chain", "P"])
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ANTIFOLD_REPO)
    completed = subprocess.run(
        args, capture_output=True, text=True, timeout=900, cwd=str(ROOT), env=env)
    if completed.returncode:
        raise RuntimeError(
            f"AntiFold failed for {pdb_path.name}: "
            f"{(completed.stderr or completed.stdout)[-2000:]}")
    csv_files = sorted(out_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"AntiFold produced no CSV in {out_dir}")
    return pd.read_csv(csv_files[0])


def logp_matrix(frame: pd.DataFrame, h3_indices, native_sequence: str):
    # AntiFold's pdb_pos is IMGT numbering, not the original PDB numbering.
    # ANARCII preserves residue order, so the H-row order matches the input
    # heavy-chain order. Use row order first and fall back to pdb_pos.
    heavy = frame[frame["pdb_chain"] == "H"].reset_index(drop=True)
    by_position = {int(row["pdb_pos"]): row for _, row in heavy.iterrows()}
    rows = []
    for offset, index in enumerate(h3_indices):
        row = heavy.iloc[index] if index < len(heavy) else None
        if row is None or str(row["pdb_res"]) != native_sequence[offset]:
            row = by_position.get(index + 1)
        if row is None:
            raise KeyError(f"AntiFold CSV missing H position {index + 1}")
        if str(row["pdb_res"]) != native_sequence[offset]:
            raise ValueError(
                f"AntiFold residue mismatch at H{index + 1}: "
                f"{row['pdb_res']} != {native_sequence[offset]}")
        rows.append([float(row[aa]) for aa in ALPHABET])
    return np.asarray(rows, dtype=float)


def nll(logp: np.ndarray, sequence: str) -> float:
    indices = [ALPHABET.index(aa) for aa in sequence]
    return -sum(logp[i, index] for i, index in enumerate(indices)) / len(sequence)


def score_record(record, h3_indices, native_sequence, config, work_dir, device):
    state_frames = {}
    for state, include_antigen in (("complex", True), ("apo", False)):
        state_dir = work_dir / record["id"] / state
        pdb_path = state_dir / f"{record['id']}_{state}.pdb"
        write_record_backbone(record, pdb_path, include_antigen)
        frame = run_antifold(pdb_path, state_dir / "antifold", include_antigen, device)
        state_frames[state] = logp_matrix(frame, h3_indices, native_sequence)
    shuffles = composition_shuffles(
        native_sequence, int(config["shuffle_trials"]),
        int(config["proteinmpnn"]["seed"]) + sum(ord(char) for char in record["id"]))
    sequences = [native_sequence, *shuffles]
    rows = []
    for sequence in sequences:
        complex_nll = nll(state_frames["complex"], sequence)
        apo_nll = nll(state_frames["apo"], sequence)
        rows.append({
            "sequence": sequence,
            "complex_h3_nll": round(complex_nll, 6),
            "apo_h3_nll": round(apo_nll, 6),
            "epitope_delta_nll": round(complex_nll - apo_nll, 6),
        })
    native = rows[0]
    controls = rows[1:]
    metrics = {}
    for metric in ("complex_h3_nll", "apo_h3_nll", "epitope_delta_nll"):
        values = [row[metric] for row in controls]
        metrics[f"native_{metric}"] = native[metric]
        metrics[f"shuffle_mean_{metric}"] = round(sum(values) / len(values), 6)
        metrics[f"native_{metric}_percentile"] = round(
            sum(value > native[metric] for value in values) / len(values), 6)
    metrics["ecls_advantage"] = round(
        metrics["shuffle_mean_epitope_delta_nll"]
        - metrics["native_epitope_delta_nll"], 6)
    return {
        "id": record["id"],
        "native": native,
        "composition_shuffles": controls,
        "summary": metrics,
        "provenance": {
            "h3_indices": h3_indices,
            "model": "AntiFold",
            "device": device,
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs" / "benchmarks" / "peptide_h3_ecls_temporal_final_v1.yml")
    parser.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "results" / "ecls_cross_arch_v1" / "antifold")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    audit_path = ROOT / config["split_audit"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    records = audit["records"][config["benchmark_split"]]
    if config.get("sampling", {}).get("one_record_per_inference_unit"):
        records = select_inference_units(records, config["inference_axis"])
    if args.max_records is not None:
        records = records[:args.max_records]

    selected_ids_sha256 = sha256_text("\n".join(sorted(
        record["id"] for record in records)))
    final_contract = config.get("final_contract")
    if final_contract and args.max_records is None:
        if len(records) != int(final_contract["n_records"]):
            raise ValueError("Record count does not match frozen final contract")
        if selected_ids_sha256 != final_contract["ids_sha256"]:
            raise ValueError("Selected ID hash does not match frozen final contract")

    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg.get("status") != "frozen_before_scoring":
        raise RuntimeError("Preregistration is not frozen; refusing to score")

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    work_dir = out_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    lmdb_path = ROOT / config["lmdb"]

    results = []
    for index, record in enumerate(records, 1):
        h3_indices = record["h3_coordinate_mapping"]["parsed_h3_indices"]
        native_sequence = record["axis_values"]["cdr_h3_sequence_exact"][0]
        print(f"[{index}/{len(records)}] {record['id']}", flush=True)
        results.append(score_record(
            read_record(lmdb_path, record["id"]), h3_indices, native_sequence,
            config, work_dir, args.device))

    unit_results = aggregate_inference_units(
        results, records, config["inference_axis"])
    advantages = [row["mean_ecls_advantage"] for row in unit_results]
    bootstrap = bootstrap_mean_ci(
        advantages, int(config["proteinmpnn"]["seed"]),
        int(config.get("statistics", {}).get("bootstrap_trials", 10000)))
    summary = {
        "schema_version": 1,
        "status": "cross_architecture_zero_shot_not_confirmatory",
        "model": "AntiFold",
        "n_records": len(results),
        "n_inference_units": len(unit_results),
        "selected_ids_sha256": selected_ids_sha256,
        "mean_ecls_advantage": round(float(np.mean(advantages)), 6),
        "median_ecls_advantage": round(float(np.median(advantages)), 6),
        "ecls_advantage_mean_ci95": [round(value, 6) for value in bootstrap],
        "fraction_positive_ecls_advantage": round(
            sum(value > 0 for value in advantages) / len(advantages), 6),
        "unit_results": unit_results,
        "prereg_sha256": hashlib.file_digest(
            PREREG.open("rb"), "sha256").hexdigest(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "antifold_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "antifold_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "n_records": summary["n_records"],
        "n_inference_units": summary["n_inference_units"],
        "mean_ecls_advantage": summary["mean_ecls_advantage"],
        "median_ecls_advantage": summary["median_ecls_advantage"],
        "ci95": summary["ecls_advantage_mean_ci95"],
        "fraction_positive": summary["fraction_positive_ecls_advantage"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
