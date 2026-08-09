#!/usr/bin/env python
"""Development-only cross-generator ECLS native-spike reranking benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "modules"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_h3_epitope_delta import (  # noqa: E402
    bootstrap_mean_ci,
    chain_sequence,
    read_record,
    run_mpnn_state,
    sequence_nll,
    write_record_backbone,
)
from benchmark_vs_mpnn import sequence_recovery  # noqa: E402

AA = set("ACDEFGHIKLMNPQRSTVWY")


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def select_units(records, axis):
    selected = {}
    for record in sorted(records, key=lambda item: item["id"]):
        values = record["axis_values"].get(axis, [])
        unit = values[0] if values else record["axis_values"]["pdb_id"][0]
        selected.setdefault(unit, record)
    return list(selected.values())


def validate_candidate(sequence, length):
    sequence = str(sequence).strip().upper()
    if len(sequence) != length or any(aa not in AA for aa in sequence):
        raise ValueError(f"Invalid generated H3 sequence: {sequence!r}")
    return sequence


def normalized_native_rank(native_score, candidate_scores):
    """Tie-aware normalized rank; one is best and random expectation is 0.5."""
    scores = [native_score, *candidate_scores]
    if len(scores) == 1:
        return 1.0
    better = sum(score < native_score for score in candidate_scores)
    tied = sum(score == native_score for score in candidate_scores)
    rank = 1.0 + better + 0.5 * tied
    return 1.0 - (rank - 1.0) / (len(scores) - 1.0)


def parse_mpnn_fasta(path, h3_indices, heavy_length):
    rows = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.startswith(">T="):
            continue
        if index + 1 >= len(lines):
            raise ValueError(f"Missing sequence after ProteinMPNN header in {path}")
        full_sequence = lines[index + 1].strip().split("/")[0]
        if len(full_sequence) != heavy_length:
            raise ValueError(
                f"ProteinMPNN heavy length {len(full_sequence)} != {heavy_length}")
        fields = {}
        for field in line[1:].split(","):
            if "=" in field:
                key, value = field.split("=", 1)
                fields[key.strip()] = value.strip()
        rows.append({
            "sequence": "".join(full_sequence[position] for position in h3_indices),
            "generator_score": float(fields.get("score", "nan")),
            "full_heavy_sequence": full_sequence,
        })
    return rows


def generate_mpnn(config, record, h3_indices, pdb_path, manifest, seed, work_dir):
    settings = config["generation"]["proteinmpnn"]
    name = pdb_path.stem
    fixed = [
        position for position in range(1, manifest["H"]["length"] + 1)
        if position - 1 not in set(h3_indices)
    ]
    fixed_path = work_dir / "fixed_positions.jsonl"
    fixed_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_path.write_text(json.dumps({name: {"H": fixed}}) + "\n", encoding="ascii")
    output = work_dir / "mpnn"
    fasta = output / "seqs" / f"{name}.fa"
    command = [
        sys.executable, (ROOT / settings["script"]).resolve().as_posix(),
        "--pdb_path", pdb_path.resolve().as_posix(),
        "--pdb_path_chains", "H",
        "--fixed_positions_jsonl", fixed_path.resolve().as_posix(),
        "--path_to_model_weights", (ROOT / settings["weights"]).resolve().as_posix(),
        "--model_name", settings["model_name"],
        "--num_seq_per_target", str(config["generation"]["candidates_per_seed"]),
        "--batch_size", "1",
        "--sampling_temp", str(settings["temperature"]),
        "--seed", str(seed),
        "--save_score", "1",
        "--suppress_print", "1",
        "--out_folder", output.resolve().as_posix(),
    ]
    if not fasta.exists():
        completed = subprocess.run(command, capture_output=True, text=True, timeout=900)
        if completed.returncode:
            raise RuntimeError(f"ProteinMPNN generation failed: {completed.stderr[-2000:]}")
    rows = parse_mpnn_fasta(fasta, h3_indices, manifest["H"]["length"])
    return rows, command


def generate_bfn(config, h3_indices, pdb_path, context_chains, seed):
    from bfn_loader import run_bfn_design

    checkpoint = (ROOT / config["generation"]["bfn"]["checkpoint"]).resolve()
    os.environ["DISORDERFLOW_CHECKPOINT"] = str(checkpoint)
    if h3_indices != list(range(h3_indices[0], h3_indices[-1] + 1)):
        raise ValueError("BFN benchmark requires a contiguous official H3 mapping")
    region = f"H:{h3_indices[0] + 1}-{h3_indices[-1] + 1}"
    rows = run_bfn_design(
        str(pdb_path), region,
        num_samples=int(config["generation"]["candidates_per_seed"]),
        stochastic=bool(config["generation"]["bfn"]["stochastic"]),
        context_chains=context_chains,
        device="cpu",
        sampling_seed=seed,
    )
    return [
        {"sequence": row["sequence"], "generator_score": row["ppl"], **row}
        for row in rows
    ]


def load_esmif():
    from esmif_compat import install_torch_scatter_fallback

    install_torch_scatter_fallback()
    import esm

    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    return model.cpu().eval(), alphabet


def build_esmif_partial_sequence(heavy_sequence, h3_indices, total_length):
    design_positions = set(h3_indices)
    if not design_positions:
        raise ValueError("ESM-IF design positions are empty")
    if min(design_positions) < 0 or max(design_positions) >= len(heavy_sequence):
        raise ValueError("ESM-IF design positions fall outside the heavy chain")
    partial = ["<pad>"] * total_length
    for index, aa in enumerate(heavy_sequence):
        partial[index] = "<mask>" if index in design_positions else aa
    return partial


def validate_esmif_sample(sampled_heavy, native_heavy, h3_indices):
    if len(sampled_heavy) != len(native_heavy):
        raise ValueError("ESM-IF changed the heavy-chain length")
    design_positions = set(h3_indices)
    changed_fixed = [
        index for index, (sampled, native) in enumerate(
            zip(sampled_heavy, native_heavy, strict=True))
        if index not in design_positions and sampled != native
    ]
    if changed_fixed:
        raise ValueError(
            f"ESM-IF changed fixed heavy-chain positions: {changed_fixed[:10]}")


def generate_esmif(config, model, pdb_path, h3_indices, chain_ids, seed):
    from esm.inverse_folding import multichain_util

    coords, sequences = multichain_util.load_complex_coords(str(pdb_path), chain_ids)
    heavy_sequence = sequences["H"]
    if len(heavy_sequence) <= h3_indices[-1]:
        raise ValueError("ESM-IF heavy sequence is shorter than the official H3 mapping")
    all_coords = multichain_util._concatenate_coords(coords, "H")
    partial = build_esmif_partial_sequence(
        heavy_sequence, h3_indices, len(all_coords))
    rows = []
    for sample_index in range(int(config["generation"]["candidates_per_seed"])):
        torch.manual_seed(seed + sample_index)
        sampled = model.sample(
            all_coords, partial_seq=partial,
            temperature=float(config["generation"]["esm_if"]["temperature"]))
        sampled_heavy = sampled[:len(heavy_sequence)]
        validate_esmif_sample(sampled_heavy, heavy_sequence, h3_indices)
        rows.append({
            "sequence": "".join(sampled_heavy[index] for index in h3_indices),
            "full_heavy_sequence": sampled_heavy,
            "generator_score": None,
        })
    return rows


def unique_candidates(rows, native, expected_length):
    output = []
    seen = set()
    native_generated = False
    failures = 0
    for row in rows:
        try:
            sequence = validate_candidate(row["sequence"], expected_length)
        except ValueError:
            failures += 1
            continue
        native_generated |= sequence == native
        if sequence in seen or sequence == native:
            continue
        seen.add(sequence)
        output.append({**row, "sequence": sequence})
    return output, native_generated, failures


def score_pool(native, candidates, complex_logp, apo_logp, h3_indices):
    def scores(sequence):
        complex_nll = sequence_nll(complex_logp, sequence, h3_indices)
        apo_nll = sequence_nll(apo_logp, sequence, h3_indices)
        return complex_nll, apo_nll, complex_nll - apo_nll

    native_complex, native_apo, native_ecls = scores(native)
    scored = []
    for row in candidates:
        complex_nll, apo_nll, ecls = scores(row["sequence"])
        scored.append({
            **row,
            "complex_nll": round(complex_nll, 6),
            "apo_nll": round(apo_nll, 6),
            "ecls": round(ecls, 6),
            "recovery": round(sequence_recovery(row["sequence"], native), 6),
        })
    return {
        "native_scores": {
            "complex_nll": native_complex, "apo_nll": native_apo, "ecls": native_ecls},
        "candidates": scored,
        "metrics": {
            "n_unique_candidates": len(scored),
            "ecls_nnr": normalized_native_rank(native_ecls, [row["ecls"] for row in scored]),
            "complex_nll_nnr": normalized_native_rank(
                native_complex, [row["complex_nll"] for row in scored]),
            "apo_nll_nnr": normalized_native_rank(
                native_apo, [row["apo_nll"] for row in scored]),
        },
    }


def aggregate(results, config):
    generators = sorted({row["generator"] for row in results})
    generator_rows = {}
    all_differences = []
    for generator in generators:
        rows = [row for row in results if row["generator"] == generator]
        by_unit = {}
        for row in rows:
            by_unit.setdefault(row["unit"], []).append(row)
        units = []
        for unit, unit_rows in sorted(by_unit.items()):
            ecls = float(np.mean([row["metrics"]["ecls_nnr"] for row in unit_rows]))
            complex_nll = float(np.mean([
                row["metrics"]["complex_nll_nnr"] for row in unit_rows]))
            units.append({
                "unit": unit,
                "ecls_nnr": ecls,
                "complex_nll_nnr": complex_nll,
                "gain": ecls - complex_nll,
            })
        differences = [row["gain"] for row in units]
        all_differences.extend(differences)
        ecls_over_random = [row["ecls_nnr"] - 0.5 for row in units]
        generator_rows[generator] = {
            "n_units": len(units),
            "mean_ecls_nnr": float(np.mean([row["ecls_nnr"] for row in units])),
            "mean_complex_nll_nnr": float(np.mean([
                row["complex_nll_nnr"] for row in units])),
            "mean_gain": float(np.mean(differences)),
            "ecls_over_random_ci95": bootstrap_mean_ci(
                ecls_over_random, seed=2465, trials=int(config["statistics"]["bootstrap_trials"])),
            "gain_ci95": bootstrap_mean_ci(
                differences, seed=2466, trials=int(config["statistics"]["bootstrap_trials"])),
            "units": units,
        }
    by_unit = {}
    for generator_result in generator_rows.values():
        for row in generator_result["units"]:
            by_unit.setdefault(row["unit"], []).append(row)
    pooled_units = []
    for unit, rows in sorted(by_unit.items()):
        pooled_units.append({
            "unit": unit,
            "ecls_nnr": float(np.mean([row["ecls_nnr"] for row in rows])),
            "complex_nll_nnr": float(np.mean([
                row["complex_nll_nnr"] for row in rows])),
            "gain": float(np.mean([row["gain"] for row in rows])),
        })
    trials = int(config["statistics"]["bootstrap_trials"])
    pooled_differences = [row["gain"] for row in pooled_units]
    ci = bootstrap_mean_ci(pooled_differences, seed=2467, trials=trials)
    mean_gain = float(np.mean(pooled_differences))
    ecls_values = [row["ecls_nnr"] for row in pooled_units]
    positive_generators = sum(
        row["mean_gain"] > 0 for row in generator_rows.values())
    gates = config["development_gates"]
    passed = bool(
        mean_gain >= float(gates["minimum_mean_nnr_gain_over_complex_nll"])
        and positive_generators >= int(gates["minimum_generators_with_positive_mean_gain"])
        and (
            not gates["require_gain_ci95_lower_above_zero"] or ci[0] > 0
        )
    )
    return {
        "generators": generator_rows,
        "n_inference_units": len(pooled_units),
        "mean_ecls_nnr": float(np.mean(ecls_values)),
        "ecls_over_random_ci95": bootstrap_mean_ci(
            [value - 0.5 for value in ecls_values], seed=2468, trials=trials),
        "overall_mean_gain": mean_gain,
        "overall_gain_ci95": ci,
        "positive_generators": positive_generators,
        "development_pass": passed,
        "units": pooled_units,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "benchmarks" / "peptide_h3_candidate_reranking_dev_v1.yml"))
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results" / "publication" / "h3_candidate_reranking_dev_v1"))
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    audit_path = (ROOT / config["split_audit"]).resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    selected = select_units(
        audit["records"][config["benchmark_split"]], config["inference_axis"])
    selected_hash = sha256_text("\n".join(sorted(row["id"] for row in selected)))
    if len(selected) != int(config["selection"]["n_records"]):
        raise ValueError("Selected record count does not match the frozen contract")
    if selected_hash != config["selection"]["ids_sha256"]:
        raise ValueError("Selected record hash does not match the frozen contract")
    if args.max_records is not None:
        selected = selected[:args.max_records]

    out_dir = Path(args.out_dir)
    work_dir = out_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "results.json"
    if args.aggregate_only:
        output = json.loads(result_path.read_text(encoding="utf-8"))
        output["aggregate"] = aggregate(output["results"], config)
        result_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
        print(json.dumps({"aggregate": output["aggregate"], "out_dir": str(out_dir)}, indent=2))
        return
    lmdb_path = (ROOT / config["lmdb"]).resolve()
    esm_model = None
    if config["generation"]["esm_if"]["enabled"]:
        esm_model, _ = load_esmif()
    scoring_config = {"proteinmpnn": config["ecls"]}
    results = []
    failures = []
    for record_index, audit_record in enumerate(selected, 1):
        record = read_record(lmdb_path, audit_record["id"])
        h3_indices = audit_record["h3_coordinate_mapping"]["parsed_h3_indices"]
        native = audit_record["axis_values"]["cdr_h3_sequence_exact"][0]
        if "".join(chain_sequence(record["heavy"])[index] for index in h3_indices) != native:
            raise ValueError(f"Official H3 mapping mismatch for {record['id']}")
        record_work = work_dir / record["id"]
        complex_pdb = record_work / f"{record['id']}_complex.pdb"
        manifest = write_record_backbone(record, complex_pdb, include_antigen=True)
        context_chains = [chain for chain in ("L", "P") if chain in manifest]
        ecls_work = config["ecls"].get("precomputed_work_dir")
        ecls_work = (ROOT / ecls_work).resolve() if ecls_work else record_work / "ecls"
        complex_logp, _, _ = run_mpnn_state(
            scoring_config, record, h3_indices, "complex", ecls_work)
        apo_logp, _, _ = run_mpnn_state(
            scoring_config, record, h3_indices, "apo", ecls_work)
        unit = (audit_record["axis_values"].get(config["inference_axis"])
                or audit_record["axis_values"]["pdb_id"])[0]
        for seed in config["generation"]["seeds"]:
            generators = {}
            if config["generation"]["bfn"]["enabled"]:
                try:
                    generators["bfn"] = generate_bfn(
                        config, h3_indices, complex_pdb, context_chains, int(seed))
                except Exception as error:  # noqa: BLE001
                    failures.append({
                        "id": record["id"], "generator": "bfn", "seed": seed,
                        "error": str(error)})
            if config["generation"]["proteinmpnn"]["enabled"]:
                try:
                    generators["proteinmpnn"], _ = generate_mpnn(
                        config, record, h3_indices, complex_pdb, manifest, int(seed),
                        record_work / f"proteinmpnn_seed_{seed}")
                except Exception as error:  # noqa: BLE001
                    failures.append({
                        "id": record["id"], "generator": "proteinmpnn", "seed": seed,
                        "error": str(error)})
            if config["generation"]["esm_if"]["enabled"]:
                try:
                    generators["esm_if"] = generate_esmif(
                        config, esm_model, complex_pdb, h3_indices,
                        list(manifest), int(seed))
                except Exception as error:  # noqa: BLE001
                    failures.append({
                        "id": record["id"], "generator": "esm_if", "seed": seed,
                        "error": str(error)})
            for generator, raw_rows in generators.items():
                candidates, native_generated, invalid = unique_candidates(
                    raw_rows, native, len(native))
                pool = score_pool(native, candidates, complex_logp, apo_logp, h3_indices)
                results.append({
                    "id": record["id"], "unit": unit, "generator": generator,
                    "seed": seed, "native_sequence": native,
                    "raw_candidates": len(raw_rows),
                    "native_generated": native_generated,
                    "invalid_candidates": invalid,
                    **pool,
                })
        print(f"Completed {record_index}/{len(selected)}: {record['id']}", flush=True)

    aggregate_result = aggregate(results, config) if results else None
    output = {
        "schema_version": 1,
        "status": "development_only; temporal final was not accessed",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "selected_ids_sha256": selected_hash,
        "aggregate": aggregate_result,
        "failures": failures,
        "results": results,
    }
    result_path.write_text(
        json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(json.dumps({
        "aggregate": aggregate_result,
        "n_failures": len(failures),
        "out_dir": str(out_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
