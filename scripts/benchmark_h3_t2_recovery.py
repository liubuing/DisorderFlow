#!/usr/bin/env python
"""Development-only T2 perturbation recovery and ECLS benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_h3_epitope_delta import (  # noqa: E402
    bootstrap_mean_ci, chain_sequence, composition_shuffles, read_record,
    sequence_nll, write_record_backbone,
)
from benchmark_h3_t1_ensemble import run_conditional_probs  # noqa: E402
from generate_h3_peptide_t2 import generate_t2  # noqa: E402


def select_units(records, axis):
    selected = {}
    for record in sorted(records, key=lambda item: item["id"]):
        values = record["axis_values"].get(axis, [])
        unit = values[0] if values else record["axis_values"]["pdb_id"][0]
        selected.setdefault(unit, record)
    return list(selected.values())


def selected_hash(records):
    value = "\n".join(sorted(record["id"] for record in records))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def state_advantage(config, pdb_paths, reference_chain_ids, native, shuffles, h3_indices,
                    output_dir):
    values = {sequence: [] for sequence in [native, *shuffles]}
    for index, pdb_path in enumerate(pdb_paths):
        complex_logp, _, _ = run_conditional_probs(
            config, pdb_path, h3_indices, output_dir / f"state_{index}_complex",
            [*reference_chain_ids, "P"])
        stripped_logp, _, _ = run_conditional_probs(
            config, pdb_path, h3_indices, output_dir / f"state_{index}_stripped",
            reference_chain_ids)
        for sequence in values:
            values[sequence].append(
                sequence_nll(complex_logp, sequence, h3_indices)
                - sequence_nll(stripped_logp, sequence, h3_indices))
    native_mean = float(np.mean(values[native]))
    control_means = [float(np.mean(values[sequence])) for sequence in shuffles]
    return {
        "native_mean_ecls": native_mean,
        "shuffle_mean_ecls": float(np.mean(control_means)),
        "ecls_advantage": float(np.mean(control_means) - native_mean),
        "native_per_replica_ecls": values[native],
    }


def benchmark_record(config, audit_record, lmdb_path, out_dir):
    record = read_record(lmdb_path, audit_record["id"])
    h3_indices = audit_record["h3_coordinate_mapping"]["parsed_h3_indices"]
    native = audit_record["axis_values"]["cdr_h3_sequence_exact"][0]
    if "".join(chain_sequence(record["heavy"])[index] for index in h3_indices) != native:
        raise ValueError(f"Official H3 mapping mismatch for {record['id']}")
    record_dir = out_dir / "work" / record["id"]
    cache = record_dir / "record_result.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    deposited = record_dir / "deposited_complex.pdb"
    manifest = write_record_backbone(record, deposited, include_antigen=True)
    antibody_chains = [chain for chain in ("H", "L") if chain in manifest]
    generation_config = {
        "chains": {"antibody": antibody_chains, "peptide": "P"},
        "sampling": config["sampling"],
    }
    audit = generate_t2(
        deposited, record_dir / "t2", generation_config, record["id"])
    accepted = [row for row in audit["replicas"] if row["status"] == "accepted"]
    requirements = config["ensemble_requirements"]
    valid = bool(
        len(accepted) >= int(requirements["minimum_accepted_replicas"])
        and audit["acceptance_fraction"] >= float(
            requirements["minimum_acceptance_fraction"]))
    unit_values = audit_record["axis_values"].get(config["inference_axis"], [])
    result = {
        "id": record["id"], "unit": unit_values[0] if unit_values else record["id"],
        "valid_t2": valid, "t2_audit": audit,
    }
    if valid:
        initial_paths = [Path(row["initial_pdb"]) for row in accepted]
        final_paths = [Path(row["pdb"]) for row in accepted]
        shuffles = composition_shuffles(
            native, int(config["ecls"]["shuffle_trials"]),
            int(config["ecls"]["seed"]) + sum(ord(char) for char in record["id"]))
        initial_score = state_advantage(
            config["ecls"], initial_paths, antibody_chains, native, shuffles,
            h3_indices, record_dir / "scoring" / "initial")
        final_score = state_advantage(
            config["ecls"], final_paths, antibody_chains, native, shuffles,
            h3_indices, record_dir / "scoring" / "final")
        held_out = [row["recovery"]["held_out_contact_recovery"] for row in accepted]
        rmsd = [row["recovery"]["rmsd_recovery_angstrom"] for row in accepted]
        result.update({
            "initial_ecls": initial_score, "final_ecls": final_score,
            "metrics": {
                "mean_held_out_contact_recovery": float(np.mean(held_out)),
                "fraction_replicas_positive_held_out_recovery": float(
                    np.mean(np.asarray(held_out) > 0)),
                "mean_rmsd_recovery_angstrom": float(np.mean(rmsd)),
                "fraction_replicas_positive_rmsd_recovery": float(
                    np.mean(np.asarray(rmsd) > 0)),
                "final_ecls_advantage": final_score["ecls_advantage"],
                "ecls_advantage_recovery": (
                    final_score["ecls_advantage"] - initial_score["ecls_advantage"]),
            },
        })
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    return result


def aggregate(results, config):
    valid = [row for row in results if row["valid_t2"]]
    if not valid:
        return {"n_valid_clusters": 0, "development_pass": False}
    contact = [row["metrics"]["mean_held_out_contact_recovery"] for row in valid]
    rmsd = [row["metrics"]["mean_rmsd_recovery_angstrom"] for row in valid]
    ecls = [row["metrics"]["final_ecls_advantage"] for row in valid]
    ecls_recovery = [row["metrics"]["ecls_advantage_recovery"] for row in valid]
    stats = config["statistics"]
    trials = int(stats["bootstrap_trials"])
    seed = int(stats["bootstrap_seed"])
    contact_ci = bootstrap_mean_ci(contact, seed, trials)
    rmsd_ci = bootstrap_mean_ci(rmsd, seed + 1, trials)
    ecls_ci = bootstrap_mean_ci(ecls, seed + 2, trials)
    valid_fraction = len(valid) / len(results)
    positive_contact_fraction = float(np.mean(np.asarray(contact) > 0))
    gates = config["development_gates"]
    passed = bool(
        valid_fraction >= float(gates["minimum_valid_cluster_fraction"])
        and (not gates["require_held_out_recovery_ci95_lower_above_zero"]
             or contact_ci[0] > 0)
        and positive_contact_fraction >= float(
            gates["minimum_fraction_clusters_positive_held_out_recovery"])
        and (not gates["require_positive_mean_final_ecls_advantage"]
             or float(np.mean(ecls)) > 0))
    return {
        "n_records": len(results), "n_valid_clusters": len(valid),
        "valid_cluster_fraction": valid_fraction,
        "mean_held_out_contact_recovery": float(np.mean(contact)),
        "held_out_contact_recovery_ci95": contact_ci,
        "fraction_clusters_positive_held_out_recovery": positive_contact_fraction,
        "mean_rmsd_recovery_angstrom": float(np.mean(rmsd)),
        "rmsd_recovery_ci95": rmsd_ci,
        "mean_final_ecls_advantage": float(np.mean(ecls)),
        "final_ecls_advantage_ci95": ecls_ci,
        "mean_ecls_advantage_recovery": float(np.mean(ecls_recovery)),
        "development_pass": passed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/benchmarks/peptide_h3_t2_recovery_dev_v1.yml"))
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results/publication/h3_t2_recovery_dev_v1"))
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["policy"]["temporal_final_access"] != "forbidden":
        raise ValueError("T2 development must forbid temporal-final access")
    audit = json.loads((ROOT / config["split_audit"]).read_text(encoding="utf-8"))
    selected = select_units(
        audit["records"][config["benchmark_split"]], config["inference_axis"])
    observed_hash = selected_hash(selected)
    if len(selected) != int(config["selection"]["n_records"]):
        raise ValueError("Frozen T2 record count mismatch")
    if observed_hash != config["selection"]["ids_sha256"]:
        raise ValueError("Frozen T2 record hash mismatch")
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
        "selected_ids_sha256": observed_hash,
        "aggregate": aggregate(results, config), "results": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(json.dumps({"aggregate": output["aggregate"], "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
