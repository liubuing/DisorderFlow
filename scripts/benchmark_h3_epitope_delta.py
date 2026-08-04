#!/usr/bin/env python
"""Evaluate peptide-conditioned ProteinMPNN H3 likelihood on development complexes."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import lmdb
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_h3_scorer_sanity import composition_shuffles, sha256  # noqa: E402


ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA1_TO_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
BACKBONE_NAMES = ("N", "CA", "C", "O")


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_record(lmdb_path, record_id):
    env = lmdb.open(str(lmdb_path), subdir=False, readonly=True, lock=False, readahead=False)
    with env.begin() as transaction:
        payload = transaction.get(str(record_id).encode())
    env.close()
    if payload is None:
        raise KeyError(f"Missing LMDB record: {record_id}")
    return pickle.loads(payload)


def chain_sequence(chain):
    return "".join(ALPHABET[int(value)] for value in chain["aa"])


def write_record_backbone(record, path, include_antigen):
    chain_specs = [("H", record["heavy"]), ("L", record.get("light"))]
    if include_antigen:
        chain_specs.append(("P", record["antigen"]))
    lines = []
    serial = 1
    manifest = {}
    for chain_id, chain in chain_specs:
        if chain is None:
            continue
        sequence = chain_sequence(chain)
        positions = np.asarray(chain["pos_heavyatom"], dtype=float)
        masks = np.asarray(chain["mask_heavyatom"], dtype=bool)
        for index, aa in enumerate(sequence, 1):
            if not masks[index - 1, :3].all():
                raise ValueError(f"Missing N/CA/C backbone at {record['id']} {chain_id}{index}")
            for atom_index, atom_name in enumerate(BACKBONE_NAMES):
                if not masks[index - 1, atom_index]:
                    continue
                x, y, z = positions[index - 1, atom_index]
                lines.append(
                    f"ATOM  {serial:5d} {atom_name:^4s} {AA1_TO_3[aa]:>3s} "
                    f"{chain_id:1s}{index:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                    f"  1.00  0.00          {atom_name[0]:>2s}\n")
                serial += 1
        lines.append("TER\n")
        manifest[chain_id] = {"length": len(sequence), "sequence": sequence}
    lines.append("END\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="ascii")
    return manifest


def run_mpnn_state(config, record, h3_indices, state, work_dir):
    include_antigen = state == "complex"
    state_dir = work_dir / record["id"] / state
    pdb_path = state_dir / f"{record['id']}_{state}.pdb"
    manifest = write_record_backbone(record, pdb_path, include_antigen)
    fixed_positions = [
        position for position in range(1, manifest["H"]["length"] + 1)
        if position - 1 not in set(h3_indices)
    ]
    name = pdb_path.stem
    fixed_path = state_dir / "fixed_positions.jsonl"
    fixed_path.write_text(
        json.dumps({name: {"H": fixed_positions}}) + "\n", encoding="ascii")
    output = state_dir / "mpnn"
    result_path = output / "conditional_probs_only" / f"{name}.npz"
    mpnn = config["proteinmpnn"]
    command = [
        sys.executable,
        (ROOT / mpnn["script"]).resolve().as_posix(),
        "--pdb_path", pdb_path.resolve().as_posix(),
        "--pdb_path_chains", "H",
        "--fixed_positions_jsonl", fixed_path.resolve().as_posix(),
        "--path_to_model_weights", (ROOT / mpnn["weights"]).resolve().as_posix(),
        "--model_name", mpnn["model_name"],
        "--conditional_probs_only", "1",
        "--conditional_probs_only_backbone", "1",
        "--num_seq_per_target", "1",
        "--batch_size", "1",
        "--seed", str(mpnn["seed"]),
        "--suppress_print", "1",
        "--out_folder", output.resolve().as_posix(),
    ]
    if not result_path.exists():
        completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
        if completed.returncode:
            raise RuntimeError(
                f"ProteinMPNN failed for {record['id']} {state}: {completed.stderr[-2000:]}")
    if not result_path.exists():
        raise FileNotFoundError(f"Missing ProteinMPNN output: {result_path}")
    result = np.load(result_path)
    for index in h3_indices:
        if result["design_mask"][index] <= 0 or result["mask"][index] <= 0:
            raise ValueError(f"Invalid H3 MPNN mask at {record['id']} {state} index {index}")
    return result["log_p"][0], manifest, command


def sequence_nll(log_probs, sequence, h3_indices):
    return -sum(
        float(log_probs[index, MPNN_ALPHABET.index(aa)])
        for index, aa in zip(h3_indices, sequence)
    ) / len(sequence)


def percentile_lower_is_better(native, controls):
    return sum(value > native for value in controls) / len(controls)


def benchmark_record(config, audit_record, lmdb_path, work_dir):
    record = read_record(lmdb_path, audit_record["id"])
    h3_indices = audit_record["h3_coordinate_mapping"]["parsed_h3_indices"]
    native_sequence = audit_record["axis_values"]["cdr_h3_sequence_exact"][0]
    if "".join(chain_sequence(record["heavy"])[index] for index in h3_indices) != native_sequence:
        raise ValueError(f"H3 mapping mismatch for {record['id']}")
    complex_logp, complex_manifest, complex_command = run_mpnn_state(
        config, record, h3_indices, "complex", work_dir)
    apo_logp, apo_manifest, apo_command = run_mpnn_state(
        config, record, h3_indices, "apo", work_dir)
    shuffles = composition_shuffles(
        native_sequence, int(config["shuffle_trials"]),
        int(config["proteinmpnn"]["seed"]) + sum(ord(char) for char in record["id"]))
    sequences = [native_sequence, *shuffles]
    rows = []
    for sequence in sequences:
        complex_nll = sequence_nll(complex_logp, sequence, h3_indices)
        apo_nll = sequence_nll(apo_logp, sequence, h3_indices)
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
            percentile_lower_is_better(native[metric], values), 6)
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
            "complex_manifest": complex_manifest,
            "apo_manifest": apo_manifest,
            "complex_command": complex_command,
            "apo_command": apo_command,
        },
    }


def median(values):
    return float(np.median(np.asarray(values, dtype=float)))


def select_inference_units(records, axis):
    selected = {}
    for record in sorted(records, key=lambda item: item["id"]):
        values = record["axis_values"].get(axis, [])
        unit = values[0] if values else record["axis_values"]["pdb_id"][0]
        selected.setdefault(unit, record)
    return list(selected.values())


def bootstrap_mean_ci(values, seed, trials=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(trials, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def aggregate_inference_units(results, audit_records, axis):
    record_by_id = {record["id"]: record for record in audit_records}
    grouped = {}
    for result in results:
        record = record_by_id[result["id"]]
        values = record["axis_values"].get(axis, [])
        unit = values[0] if values else record["axis_values"]["pdb_id"][0]
        grouped.setdefault(unit, []).append(result)
    units = []
    for unit in sorted(grouped):
        rows = grouped[unit]
        units.append({
            "unit": unit,
            "n_records": len(rows),
            "record_ids": sorted(row["id"] for row in rows),
            "mean_ecls_advantage": float(np.mean([
                row["summary"]["ecls_advantage"] for row in rows])),
            "mean_native_complex_percentile": float(np.mean([
                row["summary"]["native_complex_h3_nll_percentile"] for row in rows])),
            "mean_native_delta_percentile": float(np.mean([
                row["summary"]["native_epitope_delta_nll_percentile"] for row in rows])),
        })
    return units


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "benchmarks" / "peptide_h3_epitope_delta_dev_v1.yml"))
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results" / "publication" / "h3_epitope_delta_dev_v1"))
    parser.add_argument("--max-complexes", type=int, default=None)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    audit_path = (ROOT / config["split_audit"]).resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    benchmark_split = config.get("benchmark_split", "development")
    development = audit["records"][benchmark_split]
    inference_axis = config.get("inference_axis", "official_antigen_cluster")
    if config.get("sampling", {}).get("one_record_per_inference_unit"):
        development = select_inference_units(development, inference_axis)
    if args.max_complexes is not None:
        development = development[:args.max_complexes]
    selected_ids_sha256 = sha256_text("\n".join(sorted(
        record["id"] for record in development)))
    final_contract = config.get("final_contract")
    if final_contract:
        if args.max_complexes is not None:
            raise ValueError("A frozen final cannot be truncated with --max-complexes")
        if len(development) != int(final_contract["n_records"]):
            raise ValueError("Final input record count does not match frozen contract")
        if selected_ids_sha256 != final_contract["ids_sha256"]:
            raise ValueError("Final input ID hash does not match frozen contract")
        units = {
            (record["axis_values"].get(inference_axis) or record["axis_values"]["pdb_id"])[0]
            for record in development
        }
        if len(units) != int(final_contract["n_inference_units"]):
            raise ValueError("Final inference-unit count does not match frozen contract")
    out_dir = Path(args.out_dir)
    work_dir = out_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    lmdb_path = (ROOT / config.get("lmdb", config.get("development_lmdb"))).resolve()
    results = [
        benchmark_record(config, record, lmdb_path, work_dir)
        for record in development
    ]
    unit_results = aggregate_inference_units(results, development, inference_axis)
    complex_percentiles = [
        row["mean_native_complex_percentile"] for row in unit_results]
    delta_percentiles = [
        row["mean_native_delta_percentile"] for row in unit_results]
    gates = config["development_gates"]
    aggregate = {
        "n_complexes": len(results),
        "median_native_complex_percentile": round(median(complex_percentiles), 6),
        "median_native_delta_percentile": round(median(delta_percentiles), 6),
        "fraction_delta_percentile_at_least_0_8": round(
            sum(value >= 0.8 for value in delta_percentiles) / len(delta_percentiles), 6),
    }
    advantages = [row["mean_ecls_advantage"] for row in unit_results]
    advantage_ci = bootstrap_mean_ci(
        advantages, int(config["proteinmpnn"]["seed"]),
        int(config.get("statistics", {}).get("bootstrap_trials", 10000)))
    aggregate.update({
        "inference_axis": inference_axis,
        "n_inference_units": len(advantages),
        "mean_ecls_advantage": round(float(np.mean(advantages)), 6),
        "median_ecls_advantage": round(median(advantages), 6),
        "ecls_advantage_mean_ci95": [round(value, 6) for value in advantage_ci],
        "fraction_positive_ecls_advantage": round(
            sum(value > 0 for value in advantages) / len(advantages), 6),
    })
    if "minimum_median_ecls_advantage" in gates:
        aggregate["development_pass"] = bool(
            aggregate["median_ecls_advantage"]
            >= float(gates["minimum_median_ecls_advantage"])
            and aggregate["fraction_positive_ecls_advantage"]
            >= float(gates["minimum_fraction_positive_ecls_advantage"])
            and (
                not gates.get("require_ecls_mean_ci95_lower_above_zero")
                or aggregate["ecls_advantage_mean_ci95"][0] > 0
            )
        )
    else:
        aggregate["development_pass"] = bool(
            aggregate["median_native_delta_percentile"]
            >= float(gates["minimum_median_native_delta_percentile"])
            and aggregate["fraction_delta_percentile_at_least_0_8"]
            >= float(gates["minimum_fraction_delta_percentile_at_least_0_8"])
            and (
                not gates.get("require_delta_median_above_complex_median")
                or aggregate["median_native_delta_percentile"]
                > aggregate["median_native_complex_percentile"]
            )
        )
    evaluation_class = "final" if final_contract else "development"
    aggregate["gate_pass"] = aggregate["development_pass"]
    output = {
        "schema_version": 1,
        "status": (
            "final_evaluated_once; no rerun permitted"
            if final_contract else "development_only; publication final not evaluated"),
        "evaluation_class": evaluation_class,
        "config_sha256": sha256(config_path),
        "split_audit_sha256": sha256(audit_path),
        "benchmark_split": benchmark_split,
        "selected_ids_sha256": selected_ids_sha256,
        "aggregate": aggregate,
        "inference_units": unit_results,
        "results": results,
    }
    (out_dir / "results.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(json.dumps({"aggregate": aggregate, "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
