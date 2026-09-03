#!/usr/bin/env python
"""Build candidate-interface LMDBs from the full-PAE multiscaffold AF2 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import lmdb
import numpy as np
import torch
from Bio.PDB import PDBParser

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_candidate_interface_confidence_v1 import (  # noqa: E402
    region_spec,
    sha256,
    unbatch_and_trim,
)
from disorderflow.utils.data import update_lmdb_record_digest  # noqa: E402


NOISE_SCALES = {"plddt": 0.05, "iptm": 0.05, "pae_normalized": 0.10}


def load_inputs(results_paths, split_path):
    result_sets = [json.loads(path.read_text(encoding="utf-8")) for path in results_paths]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    for results in result_sets:
        if results.get("status") != "af2_complete":
            raise ValueError("AF2 results must be complete before dataset construction")
        if results.get("split_manifest_sha256") != sha256(split_path):
            raise ValueError("AF2 results and split manifest hashes do not match")
    entity_contracts = [[row["entity_id"] for row in results["entities"]]
                        for results in result_sets]
    if any(contract != entity_contracts[0] for contract in entity_contracts[1:]):
        raise ValueError("AF2 protocols have different entity contracts")
    protocol_numbers = [results.get("af2_protocol", {}).get("model_number", 1)
                        for results in result_sets]
    if len(set(protocol_numbers)) != len(protocol_numbers):
        raise ValueError("AF2 protocols must use distinct model numbers")
    return result_sets, split


def residue_plddt_and_sequences(path):
    model = PDBParser(QUIET=True).get_structure(path.stem, str(path))[0]
    sequences = {}
    values = []
    aa3 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    for chain in model:
        chain_sequence = []
        for residue in chain:
            if residue.resname not in aa3 or "CA" not in residue:
                continue
            chain_sequence.append(aa3[residue.resname])
            values.append(float(residue["CA"].bfactor))
        sequences[chain.id] = "".join(chain_sequence)
    return np.asarray(values, dtype=np.float32), sequences


def load_prediction_arrays(result):
    pdb_path = ROOT / result["pdb"]
    pae_path = ROOT / result["pae_npz"]
    if sha256(pdb_path) != result["pdb_sha256"]:
        raise ValueError(f"PDB hash mismatch: {pdb_path}")
    if sha256(pae_path) != result["pae_sha256"]:
        raise ValueError(f"PAE hash mismatch: {pae_path}")
    plddt, observed = residue_plddt_and_sequences(pdb_path)
    pae = np.load(pae_path, allow_pickle=False)["pae"].astype(np.float32)
    return plddt / 100.0, pae, observed


def aggregate_targets(result_sets, entities, components, allowed_components=None):
    rows_by_entity = {}
    for results in result_sets:
        for row in results["results"]:
            if (allowed_components is not None
                    and row["component_id"] not in allowed_components):
                continue
            rows_by_entity.setdefault(row["entity_id"], []).append(row)
    expected_replicates = sum(len(results["requested"]["seeds"])
                              for results in result_sets)
    aggregates = {}
    for entity_id, rows in rows_by_entity.items():
        if len(rows) != expected_replicates or any(row["status"] != "success" for row in rows):
            raise ValueError(f"Incomplete replicate set: {entity_id}")
        entity = entities[entity_id]
        component = components[entity["component_id"]]
        plddt_values, pae_values, iptm_values = [], [], []
        for row in rows:
            plddt, pae, observed = load_prediction_arrays(row)
            expected = {
                "A": entity["heavy_sequence"], "B": entity["light_sequence"],
                "C": entity["antigen_sequence"],
            }
            if observed != expected:
                raise ValueError(f"PDB sequence mismatch: {row['pdb']}")
            plddt_values.append(plddt)
            pae_values.append(pae)
            iptm_values.append(float(row["iptm"]))
        plddt_stack = np.stack(plddt_values)
        pae_stack = np.stack(pae_values)
        iptm_array = np.asarray(iptm_values, dtype=np.float32)
        h3 = np.asarray(component["h3_heavy_indices_zero_based"])
        antigen_start = len(entity["heavy_sequence"]) + len(entity["light_sequence"])
        antigen = np.arange(antigen_start, antigen_start + len(entity["antigen_sequence"]))
        plddt_std = float(plddt_stack[:, h3].mean(axis=1).std())
        iptm_std = float(iptm_array.std())
        pae_summary = pae_stack[:, h3][:, :, antigen].mean(axis=(1, 2)) / 31.0
        pae_std = float(pae_summary.std())
        scaled_variance = np.mean([
            (plddt_std / NOISE_SCALES["plddt"]) ** 2,
            (iptm_std / NOISE_SCALES["iptm"]) ** 2,
            (pae_std / NOISE_SCALES["pae_normalized"]) ** 2,
        ])
        aggregates[entity_id] = {
            "plddt": plddt_stack.mean(axis=0).astype(np.float32),
            "iptm": float(iptm_array.mean()),
            "pae": pae_stack.mean(axis=0).astype(np.float32),
            "plddt_std": plddt_std,
            "iptm_std": iptm_std,
            "pae_std": pae_std,
            "plddt_sem": plddt_std / np.sqrt(expected_replicates),
            "iptm_sem": iptm_std / np.sqrt(expected_replicates),
            "pae_sem": pae_std / np.sqrt(expected_replicates),
            "weight": float(np.clip(1.0 / (1.0 + scaled_variance), 0.1, 1.0)),
            "replicates": expected_replicates,
        }
    return aggregates


def build_entry(
        result, entity, component, scaffold_id, protocol_id, target,
        uncertainty_version="v1"):
    from modules.bfn_loader import build_region_batch

    pdb_path = ROOT / result["pdb"]
    pae_path = ROOT / result["pae_npz"]
    plddt, pae, observed = load_prediction_arrays(result)
    expected = {
        "A": entity["heavy_sequence"],
        "B": entity["light_sequence"],
        "C": entity["antigen_sequence"],
    }
    if observed != expected:
        raise ValueError(f"PDB sequence mismatch: {pdb_path}")
    full_length = sum(map(len, expected.values()))
    if plddt.shape != (full_length,) or pae.shape != (full_length, full_length):
        raise ValueError(f"AF2 array shape mismatch: {result['prediction_id']}")
    if not np.isfinite(plddt).all() or not np.isfinite(pae).all() or np.ptp(pae) <= 0:
        raise ValueError(f"Invalid AF2 arrays: {result['prediction_id']}")
    if abs(float(plddt.mean()) - float(result["plddt"])) > 5e-4:
        raise ValueError(f"PDB pLDDT does not match result: {result['prediction_id']}")

    positions = {
        "A": [index + 1 for index in component["h3_heavy_indices_zero_based"]]
    }
    batch = build_region_batch(
        str(pdb_path), region_spec(positions), context_chains=["B", "C"],
        antigen_chains=["C"],
        antigen_context_cap=(16 if uncertainty_version == "v2" else 0),
        preserve_context_chain_order=(uncertainty_version == "v2"),
        device="cpu")
    patch_idx = batch["patch_idx"][0][batch["mask"][0].bool()].long().numpy()
    model_batch, valid_length = unbatch_and_trim(batch)
    if len(patch_idx) != valid_length:
        raise ValueError("Patch index length mismatch")
    if int(model_batch["generate_flag"].sum()) != len(positions["A"]):
        raise ValueError("H3 candidate mask was truncated")

    entry = {
        "schema_version": "candidate_interface_confidence_v1",
        "pdb_id": result["prediction_id"],
        "sequence": "".join(expected.values()),
        "scaffold_id": scaffold_id,
        "scaffold_family": entity["component_id"],
        "target": component["antigen_name"],
        "construct_id": entity["entity_id"],
        "construct_type": entity["entity_type"],
        "pose_id": "single_sequence_pose",
        "af2_seed": result["af2_seed"],
        "protocol_id": protocol_id,
        "candidate_positions": positions,
        "batch": model_batch,
        "af2_plddt": torch.from_numpy(target["plddt"][patch_idx]),
        "af2_iptm": torch.tensor(target["iptm"], dtype=torch.float32),
        "af2_pae_matrix": torch.from_numpy(target["pae"][np.ix_(patch_idx, patch_idx)]),
        "af2_pae_normalized": False,
        "af2_candidate_plddt_std": target["plddt_std"],
        "af2_iptm_std": target["iptm_std"],
        "af2_interface_pae_normalized_std": target["pae_std"],
        "af2_candidate_plddt_sem": target["plddt_sem"],
        "af2_iptm_sem": target["iptm_sem"],
        "af2_interface_pae_normalized_sem": target["pae_sem"],
        "confidence_sample_weight": target["weight"],
        "af2_replicate_count": target["replicates"],
        "source": "dual_protocol_aggregate_multiscaffold_af2_v1",
        "source_pae_path": result["pae_npz"],
        "source_pae_sha256": result["pae_sha256"],
        "source_pdb_path": result["pdb"],
        "source_pdb_sha256": result["pdb_sha256"],
    }
    if uncertainty_version == "v2":
        h3 = np.asarray(component["h3_heavy_indices_zero_based"])
        antigen_start = len(entity["heavy_sequence"]) + len(entity["light_sequence"])
        patch_antigen = patch_idx[patch_idx >= antigen_start]
        if not len(patch_antigen):
            raise ValueError("Candidate-interface patch has no antigen residues")
        entry["_replicate_candidate_plddt"] = float(plddt[h3].mean())
        entry["_replicate_iptm"] = float(result["iptm"])
        entry["_replicate_candidate_to_antigen_pae"] = {
            int(index): float(pae[np.ix_(h3, [index])].mean() / 31.0)
            for index in patch_antigen
        }
        entry["_patch_global_indices"] = patch_idx.tolist()
    return entry


def hierarchical_sem(rows, key):
    """Conservative crossed model/seed component estimate for the mean."""
    protocols = sorted({row["protocol_id"] for row in rows})
    seeds = sorted({int(row["af2_seed"]) for row in rows})
    grid = {
        (row["protocol_id"], int(row["af2_seed"])): float(row[key])
        for row in rows
    }
    if len(protocols) < 2 or len(seeds) < 2 or len(grid) != len(protocols) * len(seeds):
        values = np.asarray([float(row[key]) for row in rows])
        return float(values.std(ddof=1) / np.sqrt(len(values)))
    matrix = np.asarray([
        [grid[(protocol, seed)] for seed in seeds] for protocol in protocols
    ])
    grand_mean = matrix.mean()
    model_means = matrix.mean(axis=1)
    seed_means = matrix.mean(axis=0)
    residual = matrix - model_means[:, None] - seed_means[None, :] + grand_mean
    model_component = model_means.std(ddof=1) / np.sqrt(len(protocols))
    seed_component = seed_means.std(ddof=1) / np.sqrt(len(seeds))
    residual_component = residual.std(ddof=1) / np.sqrt(matrix.size)
    return float(np.sqrt(
        model_component ** 2 + seed_component ** 2 + residual_component ** 2))


def apply_v2_uncertainty(entries):
    """Attach sample SD and hierarchical SEM on one shared antigen patch."""
    by_construct = {}
    for entry in entries:
        by_construct.setdefault(entry["construct_id"], []).append(entry)
    summary_keys = {
        "plddt": "_replicate_candidate_plddt",
        "iptm": "_replicate_iptm",
    }
    for rows in by_construct.values():
        replicate_count = len(rows)
        if replicate_count < 2:
            raise ValueError("v2 uncertainty requires at least two replicates")
        common_antigen = set.intersection(*(
            set(row["_replicate_candidate_to_antigen_pae"]) for row in rows))
        if not common_antigen:
            raise ValueError("Replicate antigen patches have no shared residues")
        for row in rows:
            pae_by_residue = row["_replicate_candidate_to_antigen_pae"]
            row["_replicate_interface_pae_normalized"] = float(np.mean([
                pae_by_residue[index] for index in common_antigen
            ]))
            patch_indices = row["_patch_global_indices"]
            row["batch"]["pae_supervision_antigen_mask"] = torch.tensor(
                [index in common_antigen for index in patch_indices],
                dtype=torch.bool)
        all_summary_keys = {
            **summary_keys,
            "pae_normalized": "_replicate_interface_pae_normalized",
        }
        uncertainty = {}
        sem = {}
        for name, key in all_summary_keys.items():
            values = np.asarray([row[key] for row in rows], dtype=np.float64)
            uncertainty[name] = float(values.std(ddof=1))
            sem[name] = hierarchical_sem(rows, key)
        scaled_variance = np.mean([
            (uncertainty[name] / NOISE_SCALES[name]) ** 2
            for name in all_summary_keys
        ])
        weight = float(np.clip(1.0 / (1.0 + scaled_variance), 0.1, 1.0))
        for row in rows:
            row["af2_candidate_plddt_std"] = uncertainty["plddt"]
            row["af2_iptm_std"] = uncertainty["iptm"]
            row["af2_interface_pae_normalized_std"] = uncertainty["pae_normalized"]
            row["af2_candidate_plddt_sem"] = sem["plddt"]
            row["af2_iptm_sem"] = sem["iptm"]
            row["af2_interface_pae_normalized_sem"] = sem["pae_normalized"]
            row["confidence_sample_weight"] = weight
            row["af2_replicate_count"] = replicate_count
            for key in all_summary_keys.values():
                del row[key]
            del row["_replicate_candidate_to_antigen_pae"]
            del row["_patch_global_indices"]


def write_lmdb(path, entries):
    env = lmdb.open(str(path), map_size=1 << 34)
    digest = hashlib.sha256()
    with env.begin(write=True) as transaction:
        for index, entry in enumerate(entries):
            key = f"{index:08d}".encode()
            value = pickle.dumps(entry)
            transaction.put(key, value)
            update_lmdb_record_digest(digest, key, value)
        length_key = b"__len__"
        length_value = pickle.dumps(len(entries))
        transaction.put(length_key, length_value)
        update_lmdb_record_digest(digest, length_key, length_value)
    env.sync()
    env.close()
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", nargs="+", default=[
            "results/candidate_interface_multiscaffold_v1/af2/results.json",
            "results/candidate_interface_multiscaffold_v1/af2_model2/results.json",
        ])
    parser.add_argument(
        "--split-manifest", default="data/candidate_interface_multiscaffold_v1/split_manifest.json")
    parser.add_argument(
        "--out", default="data/confidence_candidate_interface_multiscaffold_dual_sem_v1")
    parser.add_argument(
        "--uncertainty-version", choices=("v1", "v2"), default="v1",
        help="v2 uses sample SD and each replicate's evaluated antigen patch")
    parser.add_argument(
        "--include-test", action="store_true",
        help="Explicitly unseal and export the final-test split")
    parser.add_argument(
        "--wait-seconds", type=int, default=0,
        help="Wait up to this many seconds for AF2 results to become complete")
    args = parser.parse_args()

    results_paths = [ROOT / path for path in args.results]
    split_path = ROOT / args.split_manifest
    output = ROOT / args.out
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    deadline = time.monotonic() + args.wait_seconds
    while True:
        try:
            result_sets, split = load_inputs(results_paths, split_path)
            break
        except (FileNotFoundError, ValueError) as error:
            if not args.wait_seconds or time.monotonic() >= deadline:
                raise
            print(f"Waiting for complete AF2 results: {error}", flush=True)
            time.sleep(60)
    output.mkdir(parents=True)

    entities = {row["entity_id"]: row for row in result_sets[0]["entities"]}
    source_manifest = json.loads((
        ROOT / split["source_holdout_manifest"]).read_text(encoding="utf-8"))
    components = {
        row["component_id"]: row["representative"]
        for row in source_manifest["components"]
    }
    selected_splits = ["train", "calibration"]
    if args.include_test:
        selected_splits.append("test")
    allowed_components = set().union(*(
        set(split[split_name]) for split_name in selected_splits))
    aggregates = aggregate_targets(
        result_sets, entities, components, allowed_components)

    summaries = {}
    records = {}
    for split_name in selected_splits:
        allowed = set(split[split_name])
        tagged_rows = [
            (row, f"af2_multimer_model_{results.get('af2_protocol', {}).get('model_number', 1)}")
            for results in result_sets for row in results["results"]
            if row["component_id"] in allowed and row["status"] == "success"
        ]
        expected = sum(
            entity["component_id"] in allowed for entity in result_sets[0]["entities"]
        ) * sum(len(results["requested"]["seeds"]) for results in result_sets)
        if len(tagged_rows) != expected:
            raise ValueError(f"Incomplete {split_name} split: {len(tagged_rows)}/{expected}")
        group_keys = sorted({(row["component_id"], protocol_id, row["af2_seed"])
                             for row, protocol_id in tagged_rows})
        group_ids = {key: index for index, key in enumerate(group_keys)}
        entries = [
            build_entry(
                row, entities[row["entity_id"]], components[row["component_id"]],
                group_ids[(row["component_id"], protocol_id, row["af2_seed"])],
                protocol_id, aggregates[row["entity_id"]], args.uncertainty_version)
            for row, protocol_id in tagged_rows
        ]
        if args.uncertainty_version == "v2":
            apply_v2_uncertainty(entries)
        lmdb_sha256 = write_lmdb(output / f"{split_name}.lmdb", entries)
        counts = Counter(entry["scaffold_family"] for entry in entries)
        summaries[split_name] = {
            "records": len(entries),
            "components": len(counts),
            "groups": len(group_keys),
            "records_by_component": dict(sorted(counts.items())),
        }
        if args.uncertainty_version == "v2":
            summaries[split_name]["lmdb_records_sha256"] = lmdb_sha256
        records[split_name] = [{
            key: entry[key] for key in (
                "construct_id", "scaffold_family", "protocol_id", "af2_seed", "scaffold_id",
                "confidence_sample_weight", "af2_iptm_std",
                "source_pae_path", "source_pae_sha256", "source_pdb_path",
                "source_pdb_sha256",
            )
        } for entry in entries]

    manifest = {
        "schema_version": (
            "candidate_interface_multiscaffold_dual_dataset_v3"
            if args.uncertainty_version == "v2"
            else "candidate_interface_multiscaffold_dual_dataset_v2"),
        "classification": "development_only",
        "source_results": [{
            "path": path_name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        } for path_name, path in zip(args.results, results_paths, strict=True)],
        "split_manifest": args.split_manifest,
        "split_manifest_sha256": sha256(split_path),
        "test_exported": args.include_test,
        "group_unit": "component_id+protocol_id+af2_seed",
        "target_aggregation": "mean over 2 AF2 models x 3 seeds per entity",
        "noise_scales": NOISE_SCALES,
        "stability_weight": "clip(1/(1+mean((std/noise_scale)^2)), 0.1, 1.0)",
        "pairwise_noise_threshold": (
            "sqrt(SEM_i^2+SEM_j^2), with hierarchical model/seed SEM"
            if args.uncertainty_version == "v2"
            else "sqrt(SEM_i^2+SEM_j^2), with SEM=replicate_std/sqrt(6)"),
        "summary": summaries,
        "records": records,
    }
    if args.uncertainty_version == "v2":
        manifest["uncertainty_estimator"] = (
            "sample SD (ddof=1); crossed model/seed component SEM; PAE uses the "
            "antigen-residue intersection shared by all six replicate patches")
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
