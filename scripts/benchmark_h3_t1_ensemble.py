#!/usr/bin/env python
"""Development-only T1 peptide-ensemble ECLS benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_h3_epitope_delta import (  # noqa: E402
    bootstrap_mean_ci,
    chain_sequence,
    composition_shuffles,
    read_record,
    sequence_nll,
    write_record_backbone,
)
from generate_h3_peptide_ensemble import generate_ensemble  # noqa: E402
from modules.h3_interface_contacts import write_standardized_backbone  # noqa: E402


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def select_units(records, axis):
    selected = {}
    for record in sorted(records, key=lambda item: item["id"]):
        values = record["axis_values"].get(axis, [])
        unit = values[0] if values else record["axis_values"]["pdb_id"][0]
        selected.setdefault(unit, record)
    return list(selected.values())


def run_conditional_probs(config, pdb_path, h3_indices, output_dir, chain_ids):
    output_dir.mkdir(parents=True, exist_ok=True)
    standardized = output_dir / "standardized.pdb"
    manifest = write_standardized_backbone(
        str(pdb_path), str(standardized), chain_ids)
    heavy_length = manifest["chains"]["H"]["length"]
    fixed = [
        position for position in range(1, heavy_length + 1)
        if position - 1 not in set(h3_indices)]
    name = standardized.stem
    fixed_path = output_dir / "fixed_positions.jsonl"
    fixed_path.write_text(json.dumps({name: {"H": fixed}}) + "\n", encoding="ascii")
    mpnn_output = output_dir / "mpnn"
    result_path = mpnn_output / "conditional_probs_only" / f"{name}.npz"
    command = [
        sys.executable, (ROOT / config["script"]).resolve().as_posix(),
        "--pdb_path", standardized.resolve().as_posix(),
        "--pdb_path_chains", "H",
        "--fixed_positions_jsonl", fixed_path.resolve().as_posix(),
        "--path_to_model_weights", (ROOT / config["weights"]).resolve().as_posix(),
        "--model_name", config["model_name"],
        "--conditional_probs_only", "1",
        "--conditional_probs_only_backbone", "1",
        "--num_seq_per_target", "1", "--batch_size", "1",
        "--seed", str(config["seed"]), "--suppress_print", "1",
        "--out_folder", mpnn_output.resolve().as_posix(),
    ]
    if not result_path.exists():
        completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
        if completed.returncode:
            raise RuntimeError(f"ProteinMPNN failed: {completed.stderr[-2000:]}")
    result = np.load(result_path)
    for index in h3_indices:
        if result["design_mask"][index] <= 0 or result["mask"][index] <= 0:
            raise ValueError(f"Invalid ProteinMPNN H3 mask at index {index}")
    return result["log_p"][0], manifest, command


def summarize_sequence(sequence, conformer_log_probs, reference_log_probs, h3_indices):
    if len(conformer_log_probs) != len(reference_log_probs):
        raise ValueError("Complex and peptide-stripped conformer counts do not match")
    values = [
        sequence_nll(log_probs, sequence, h3_indices)
        - sequence_nll(reference, sequence, h3_indices)
        for log_probs, reference in zip(conformer_log_probs, reference_log_probs)]
    return {
        "sequence": sequence,
        "mean_ecls": float(np.mean(values)),
        "median_ecls": float(np.median(values)),
        "worst_ecls": float(np.max(values)),
        "std_ecls": float(np.std(values)),
        "per_conformer_ecls": values,
    }


def benchmark_record(config, audit_record, lmdb_path, out_dir):
    record = read_record(lmdb_path, audit_record["id"])
    h3_indices = audit_record["h3_coordinate_mapping"]["parsed_h3_indices"]
    native = audit_record["axis_values"]["cdr_h3_sequence_exact"][0]
    if "".join(chain_sequence(record["heavy"])[index] for index in h3_indices) != native:
        raise ValueError(f"Official H3 mapping mismatch for {record['id']}")
    record_dir = out_dir / "work" / record["id"]
    cached_result = record_dir / "record_result.json"
    if cached_result.exists():
        return json.loads(cached_result.read_text(encoding="utf-8"))
    deposited_pdb = record_dir / "deposited_complex.pdb"
    manifest = write_record_backbone(record, deposited_pdb, include_antigen=True)
    antibody_chains = [chain for chain in ("H", "L") if chain in manifest]
    generation_config = {
        "chains": {"antibody": antibody_chains, "peptide": "P"},
        "sampling": config["sampling"],
    }
    ensemble = generate_ensemble(
        deposited_pdb, record_dir / "ensemble", generation_config, record["id"])
    accepted = [
        Path(row["pdb"]) for row in ensemble["conformers"]
        if row["status"] == "accepted"]
    requirements = config["ensemble_requirements"]
    valid = bool(
        len(accepted) >= int(requirements["minimum_accepted_conformers"])
        and ensemble["acceptance_fraction"] >= float(
            requirements["minimum_acceptance_fraction"])
        and ensemble["dispersion"]["mean_pairwise_rmsd"] >= float(
            requirements["minimum_mean_pairwise_peptide_backbone_rmsd"])
    )
    if not valid:
        result = {
            "id": record["id"], "valid_ensemble": False,
            "ensemble_audit": ensemble,
        }
        cached_result.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
        return result
    chain_ids = [*antibody_chains, "P"]
    conformer_log_probs = []
    conformer_reference_log_probs = []
    conformer_provenance = []
    for index, pdb_path in enumerate(accepted):
        log_probs, score_manifest, command = run_conditional_probs(
            config["ecls"], pdb_path, h3_indices,
            record_dir / "scoring" / f"conformer_{index + 1}", chain_ids)
        conformer_log_probs.append(log_probs)
        reference_log_probs, reference_score_manifest, reference_score_command = (
            run_conditional_probs(
                config["ecls"], pdb_path, h3_indices,
                record_dir / "scoring" / f"conformer_{index + 1}_peptide_stripped",
                antibody_chains))
        conformer_reference_log_probs.append(reference_log_probs)
        conformer_provenance.append({
            "pdb": str(pdb_path),
            "complex_manifest": score_manifest,
            "complex_command": command,
            "peptide_stripped_manifest": reference_score_manifest,
            "peptide_stripped_command": reference_score_command,
        })
    shuffles = composition_shuffles(
        native, int(config["ecls"]["shuffle_trials"]),
        int(config["ecls"]["seed"]) + sum(ord(char) for char in record["id"]))
    rows = [
        summarize_sequence(
            sequence, conformer_log_probs, conformer_reference_log_probs, h3_indices)
        for sequence in [native, *shuffles]
    ]
    native_row = rows[0]
    controls = rows[1:]
    mean_advantage = float(np.mean([
        row["mean_ecls"] for row in controls]) - native_row["mean_ecls"])
    median_advantage = float(np.mean([
        row["median_ecls"] for row in controls]) - native_row["median_ecls"])
    worst_advantage = float(np.mean([
        row["worst_ecls"] for row in controls]) - native_row["worst_ecls"])
    deposited_log_probs, _, _ = run_conditional_probs(
        config["ecls"], deposited_pdb, h3_indices,
        record_dir / "scoring" / "deposited", chain_ids)
    deposited_reference_log_probs, deposited_reference_manifest, deposited_reference_command = (
        run_conditional_probs(
            config["ecls"], deposited_pdb, h3_indices,
            record_dir / "scoring" / "deposited_peptide_stripped", antibody_chains))
    deposited_native_ecls = (
        sequence_nll(deposited_log_probs, native, h3_indices)
        - sequence_nll(deposited_reference_log_probs, native, h3_indices))
    deposited_control_ecls = [
        sequence_nll(deposited_log_probs, sequence, h3_indices)
        - sequence_nll(deposited_reference_log_probs, sequence, h3_indices)
        for sequence in shuffles]
    deposited_advantage = float(np.mean(deposited_control_ecls) - deposited_native_ecls)
    unit_values = audit_record["axis_values"].get(config["inference_axis"], [])
    result = {
        "id": record["id"],
        "unit": unit_values[0] if unit_values else record["id"],
        "valid_ensemble": True,
        "ensemble_audit": ensemble,
        "native": native_row,
        "composition_shuffles": controls,
        "metrics": {
            "ensemble_mean_ecls_advantage": mean_advantage,
            "ensemble_median_ecls_advantage": median_advantage,
            "ensemble_worst_ecls_advantage": worst_advantage,
            "deposited_single_ecls_advantage": deposited_advantage,
            "ensemble_minus_single_advantage": mean_advantage - deposited_advantage,
            "native_ecls_conformer_std": native_row["std_ecls"],
        },
        "provenance": {
            "conformers": conformer_provenance,
            "deposited_peptide_stripped_manifest": deposited_reference_manifest,
            "deposited_peptide_stripped_command": deposited_reference_command,
        },
    }
    cached_result.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    return result


def aggregate(results, config):
    valid = [row for row in results if row["valid_ensemble"]]
    advantages = [
        row["metrics"]["ensemble_mean_ecls_advantage"] for row in valid]
    gains = [row["metrics"]["ensemble_minus_single_advantage"] for row in valid]
    dispersion = [
        row["ensemble_audit"]["dispersion"]["mean_pairwise_rmsd"] for row in valid]
    if not advantages:
        return {"development_pass": False, "n_valid_ensembles": 0}
    statistics_config = config["statistics"]
    advantage_ci = bootstrap_mean_ci(
        advantages, int(statistics_config["bootstrap_seed"]),
        int(statistics_config["bootstrap_trials"]))
    gain_ci = bootstrap_mean_ci(
        gains, int(statistics_config["bootstrap_seed"]) + 1,
        int(statistics_config["bootstrap_trials"]))
    valid_fraction = len(valid) / len(results)
    positive_fraction = sum(value > 0 for value in advantages) / len(advantages)
    gates = config["development_gates"]
    passed = bool(
        valid_fraction >= float(gates["minimum_clusters_with_valid_ensemble_fraction"])
        and (not gates["require_ensemble_native_advantage_ci95_lower_above_zero"]
             or advantage_ci[0] > 0)
        and positive_fraction >= float(
            gates["minimum_fraction_positive_ensemble_advantage"])
        and float(np.mean(dispersion)) >= float(gates["minimum_mean_pairwise_rmsd"])
    )
    return {
        "n_records": len(results),
        "n_valid_ensembles": len(valid),
        "valid_ensemble_fraction": valid_fraction,
        "mean_ensemble_ecls_advantage": float(np.mean(advantages)),
        "ensemble_ecls_advantage_ci95": advantage_ci,
        "fraction_positive_ensemble_advantage": positive_fraction,
        "mean_ensemble_minus_single_advantage": float(np.mean(gains)),
        "ensemble_minus_single_advantage_ci95": gain_ci,
        "mean_pairwise_peptide_backbone_rmsd": float(np.mean(dispersion)),
        "mean_native_ecls_conformer_std": float(np.mean([
            row["metrics"]["native_ecls_conformer_std"] for row in valid])),
        "development_pass": passed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/benchmarks/peptide_h3_t1_ensemble_dev_v1.yml"))
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results/publication/h3_t1_ensemble_dev_v1"))
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["policy"]["temporal_final_access"] != "forbidden":
        raise ValueError("T1 development protocol must forbid temporal-final access")
    audit = json.loads((ROOT / config["split_audit"]).read_text(encoding="utf-8"))
    selected = select_units(
        audit["records"][config["benchmark_split"]], config["inference_axis"])
    selected_hash = sha256_text("\n".join(sorted(row["id"] for row in selected)))
    if len(selected) != int(config["selection"]["n_records"]):
        raise ValueError("Frozen selected-record count mismatch")
    if selected_hash != config["selection"]["ids_sha256"]:
        raise ValueError("Frozen selected-record hash mismatch")
    if args.max_records is not None:
        selected = selected[:args.max_records]
    out_dir = Path(args.out_dir)
    lmdb_path = ROOT / config["lmdb"]
    results = []
    for index, record in enumerate(selected, 1):
        results.append(benchmark_record(config, record, lmdb_path, out_dir))
        print(f"Completed {index}/{len(selected)}: {record['id']}", flush=True)
    output = {
        "schema_version": 1,
        "status": "development_only; temporal final not accessed",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "selected_ids_sha256": selected_hash,
        "aggregate": aggregate(results, config),
        "results": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(json.dumps({"aggregate": output["aggregate"], "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
