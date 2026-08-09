#!/usr/bin/env python
"""Generate every frozen v2 candidate attempt without scoring or selection."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from Bio.PDB import PDBIO, Chain, MMCIFParser, Model, Structure
from Bio.SeqUtils import seq1

ROOT = Path(__file__).resolve().parents[1]
AA = set("ACDEFGHIKLMNPQRSTVWY")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "modules"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_h3_candidate_reranking import (  # noqa: E402
    generate_esmif,
    generate_mpnn,
    load_esmif,
    validate_candidate,
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="ascii")
    temporary.replace(path)


def canonicalize_structure(representative, output_path):
    source = ROOT / representative["cif_path"]
    parser = MMCIFParser(QUIET=True, auth_chains=True, auth_residues=True)
    source_model = next(parser.get_structure(representative["instance"], source).get_models())
    roles = [
        (representative["heavy_chain"], "H", representative["heavy_sequence"]),
        (representative["light_chain"], "L", representative["light_sequence"]),
        (representative["antigen_chain"], "P", representative["antigen_sequence"]),
    ]
    structure = Structure.Structure(representative["instance"])
    model = Model.Model(0)
    structure.add(model)
    for source_id, target_id, expected_sequence in roles:
        if source_id not in source_model:
            raise KeyError(f"Missing source chain {source_id}")
        target = Chain.Chain(target_id)
        residue_index = 1
        observed_sequence = []
        for residue in source_model[source_id].get_residues():
            amino_acid = seq1(residue.resname, custom_map={"MSE": "M"})
            if residue.id[0] != " " or "CA" not in residue or amino_acid not in AA:
                continue
            if not all(atom in residue for atom in ("N", "CA", "C")):
                raise ValueError(
                    f"Incomplete backbone in {source_id} residue {residue.id}")
            copied = copy.deepcopy(residue)
            copied.detach_parent()
            copied.id = (" ", residue_index, " ")
            target.add(copied)
            observed_sequence.append(amino_acid)
            residue_index += 1
        observed_sequence = "".join(observed_sequence)
        if observed_sequence != expected_sequence:
            raise ValueError(
                f"Canonical {target_id} sequence differs from frozen manifest")
        model.add(target)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PDBIO()
    writer.set_structure(structure)
    writer.save(str(output_path))
    return source


def reset_bfn(checkpoint):
    import modules.bfn_loader as loader

    os.environ["DISORDERFLOW_CHECKPOINT"] = str(checkpoint)
    loader._bfn_model = None
    loader._bfn_config = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return loader


def full_heavy(native, h3_indices, h3):
    output = list(native)
    if len(h3_indices) != len(h3):
        raise ValueError("Generated H3 length differs from frozen design positions")
    for index, amino_acid in zip(h3_indices, h3, strict=True):
        output[index] = amino_acid
    return "".join(output)


def bfn_rows(loader, settings, representative, pdb_path, seed, samples, device, profile=None):
    h3_indices = representative["h3_heavy_indices_zero_based"]
    if h3_indices != list(range(h3_indices[0], h3_indices[-1] + 1)):
        raise ValueError("BFN generation requires contiguous frozen H3 indices")
    region = f"H:{h3_indices[0] + 1}-{h3_indices[-1] + 1}"
    rows = loader.run_bfn_design(
        str(pdb_path), region, num_samples=samples,
        stochastic=bool(settings["stochastic"]), context_chains=["L", "P"],
        device=device, antigen_chains=["P"],
        epitope_disorder_profile=profile,
        disorder_guided=profile is not None,
        disorder_guided_strength=float(settings.get("disorder_guided_strength", 1.0)),
        sampling_seed=seed,
    )
    return [{**row, "generator_score": row["ppl"]} for row in rows]


def disorder_profile(loader, pdb_path, expected_sequence, device):
    from modules.idp_disorder_analysis import predict_disorder

    model, config = loader.load_bfn(device)
    prediction = predict_disorder(model, config, str(pdb_path), chain_id="P", device=device)
    if prediction["sequence"] != expected_sequence:
        raise ValueError(
            f"Antigen sequence mismatch: {prediction['sequence']} != {expected_sequence}")
    return prediction["disorder_scores"].astype(float).tolist()


def project_profile_to_antigen_patch(loader, pdb_path, h3_indices, profile, device):
    region = f"H:{h3_indices[0] + 1}-{h3_indices[-1] + 1}"
    batch = loader.build_region_batch(
        str(pdb_path), region, context_chains=["L", "P"],
        device=device, antigen_chains=["P"])
    model, _config = loader.load_bfn(device)
    antigen_mask = model.bfn._mask_antigen(batch, batch["generate_flag"].bool())
    # Canonical PDB chains are renumbered 1..N; resseq therefore maps the
    # contact patch back to the full antigen profile across dataset variants.
    residue_numbers = batch["resseq"][antigen_mask].long().tolist()
    if not residue_numbers or min(residue_numbers) < 1 or max(residue_numbers) > len(profile):
        raise ValueError("Antigen patch residue numbers fall outside the full profile")
    return [profile[number - 1] for number in residue_numbers]


def attempt_rows(component, arm, seed, samples, generated, error, provenance, elapsed):
    rows = []
    for sample_index in range(samples):
        attempt_id = f"{component['component_id']}|{arm}|{seed}|{sample_index:02d}"
        if error is not None or sample_index >= len(generated):
            rows.append({
                "attempt_id": attempt_id,
                "component_id": component["component_id"],
                "representative_id": component["representative_id"],
                "arm": arm,
                "seed": seed,
                "sample_index": sample_index,
                "status": "failed",
                "sequence": None,
                "full_heavy_sequence": None,
                "generator_score": None,
                "error": error or f"generator returned {len(generated)}/{samples} candidates",
                "seed_wall_seconds": elapsed,
                **provenance,
            })
            continue
        row = generated[sample_index]
        try:
            sequence = validate_candidate(
                row["sequence"], len(component["representative"]["cdr_h3_sequence"]))
            full_sequence = row.get("full_heavy_sequence") or full_heavy(
                component["representative"]["heavy_sequence"],
                component["representative"]["h3_heavy_indices_zero_based"], sequence)
            rows.append({
                "attempt_id": attempt_id,
                "component_id": component["component_id"],
                "representative_id": component["representative_id"],
                "arm": arm,
                "seed": seed,
                "sample_index": sample_index,
                "status": "success",
                "sequence": sequence,
                "full_heavy_sequence": full_sequence,
                "generator_score": row.get("generator_score"),
                "native_generated": sequence == component["representative"]["cdr_h3_sequence"],
                "error": None,
                "seed_wall_seconds": elapsed,
                **{key: value for key, value in row.items()
                   if key not in {"sequence", "full_heavy_sequence", "generator_score"}},
                **provenance,
            })
        except Exception as candidate_error:  # noqa: BLE001
            rows.extend(attempt_rows(
                component, arm, seed, 1, [], str(candidate_error), provenance, elapsed))
            rows[-1]["sample_index"] = sample_index
            rows[-1]["attempt_id"] = attempt_id
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/benchmarks/multiscaffold_confirmatory_v2_generation.yml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--work-dir", default="data/multiscaffold_confirmatory_v2/generation_work")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--components", nargs="*")
    parser.add_argument("--arms", nargs="*")
    parser.add_argument("--only-arms", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protocol_path = ROOT / config["protocol"]
    holdout_path = ROOT / config["holdout_manifest"]
    lineage_path = ROOT / config["checkpoint_lineage"]
    for path, expected in [
        (protocol_path, config["protocol_sha256"]),
        (holdout_path, config["holdout_manifest_sha256"]),
    ]:
        if sha256(path) != expected:
            raise ValueError(f"Frozen hash mismatch: {path}")
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    if lineage["holdout_manifest_sha256"] != config["holdout_manifest_sha256"]:
        raise ValueError("Checkpoint lineage does not reference the frozen holdout")

    components = holdout["components"]
    if args.components:
        requested = set(args.components)
        components = [row for row in components
                      if row["component_id"] in requested or row["representative_id"] in requested]
        if len(components) != len(requested):
            raise ValueError("One or more requested components were not found")
    arm_settings = config["generation"]["arms"]
    arms = args.arms or list(arm_settings)
    if not set(arms) <= set(arm_settings):
        raise ValueError("Unknown generation arm")
    execution_arms = args.only_arms or arms
    if not set(execution_arms) <= set(arms):
        raise ValueError("Execution arms must be part of the requested contract")
    seeds = args.seeds or config["generation"]["seeds"]
    samples = args.samples or int(config["generation"]["candidates_per_seed"])
    output_path = ROOT / (args.out or config["output"]["path"])
    work_dir = ROOT / args.work_dir

    output = {
        "schema_version": 1,
        "status": "generation_in_progress",
        "config": args.config,
        "config_sha256": sha256(config_path),
        "protocol_sha256": sha256(protocol_path),
        "holdout_manifest_sha256": sha256(holdout_path),
        "checkpoint_lineage_sha256": sha256(lineage_path),
        "requested": {
            "components": [row["component_id"] for row in components],
            "arms": arms,
            "seeds": seeds,
            "samples_per_seed": samples,
            "attempts": len(components) * len(arms) * len(seeds) * samples,
        },
        "attempts": [],
        "claim_boundary": config["claim_boundary"],
    }
    if args.resume and output_path.exists():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        for key in ("config_sha256", "protocol_sha256", "holdout_manifest_sha256",
                    "checkpoint_lineage_sha256", "requested"):
            if previous[key] != output[key]:
                raise ValueError(f"Resume contract mismatch: {key}")
        output = previous
        output["status"] = "generation_in_progress"
    completed_ids = {
        row["attempt_id"] for row in output["attempts"]
        if not args.retry_failures or row["status"] == "success"}

    prepared = {}
    for component in components:
        representative = component["representative"]
        pdb_path = work_dir / "structures" / f"{component['component_id']}.pdb"
        source = canonicalize_structure(representative, pdb_path)
        prepared[component["component_id"]] = (pdb_path, sha256(source), sha256(pdb_path))

    esm_model = None
    active_bfn_checkpoint = None
    for arm in execution_arms:
        settings = arm_settings[arm]
        loader = None
        if settings["type"] == "bfn":
            checkpoint = ROOT / settings["checkpoint"]
            if sha256(checkpoint) != settings["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint hash mismatch for {arm}")
            if active_bfn_checkpoint != checkpoint:
                loader = reset_bfn(checkpoint)
                active_bfn_checkpoint = checkpoint
            else:
                import modules.bfn_loader as loader
        elif settings["type"] == "esm_if":
            esm_model, _alphabet = load_esmif()

        for component in components:
            representative = component["representative"]
            pdb_path, source_sha, pdb_sha = prepared[component["component_id"]]
            profile = None
            if settings["type"] == "bfn" and settings["disorder_profile"] != "none":
                profile = disorder_profile(
                    loader, pdb_path, representative["antigen_sequence"], args.device)
                profile = project_profile_to_antigen_patch(
                    loader, pdb_path,
                    representative["h3_heavy_indices_zero_based"], profile, args.device)
            provenance = {
                "input_cif_sha256": source_sha,
                "canonical_pdb_sha256": pdb_sha,
                "checkpoint_or_model_sha256": settings.get(
                    "checkpoint_sha256",
                    settings.get("model_weights_sha256", settings.get("model"))),
            }
            for seed in seeds:
                expected_ids = {
                    f"{component['component_id']}|{arm}|{seed}|{index:02d}"
                    for index in range(samples)}
                if expected_ids <= completed_ids:
                    continue
                generated = []
                error = None
                started = time.perf_counter()
                try:
                    if settings["type"] == "bfn":
                        generated = bfn_rows(
                            loader, settings, representative, pdb_path,
                            seed, samples, args.device, profile)
                    elif settings["type"] == "proteinmpnn":
                        adapter_config = {"generation": {
                            "candidates_per_seed": samples,
                            "proteinmpnn": settings,
                        }}
                        manifest = {"H": {"length": len(representative["heavy_sequence"])}}
                        generated, _command = generate_mpnn(
                            adapter_config, representative,
                            representative["h3_heavy_indices_zero_based"], pdb_path,
                            manifest, seed,
                            work_dir / component["component_id"] / arm / str(seed))
                    elif settings["type"] == "esm_if":
                        adapter_config = {"generation": {
                            "candidates_per_seed": samples,
                            "esm_if": settings,
                        }}
                        generated = generate_esmif(
                            adapter_config, esm_model, pdb_path,
                            representative["h3_heavy_indices_zero_based"],
                            ["H", "L", "P"], seed)
                    else:
                        raise ValueError(f"Unsupported arm type: {settings['type']}")
                except Exception as generation_error:  # noqa: BLE001
                    error = f"{type(generation_error).__name__}: {generation_error}"
                elapsed = time.perf_counter() - started
                rows = attempt_rows(
                    component, arm, seed, samples, generated, error, provenance, elapsed)
                output["attempts"] = [
                    row for row in output["attempts"] if row["attempt_id"] not in expected_ids]
                output["attempts"].extend(rows)
                completed_ids.update(expected_ids)
                atomic_json(output_path, output)
                print(
                    f"{component['component_id']} {arm} seed={seed}: "
                    f"{sum(row['status'] == 'success' for row in rows)}/{samples}", flush=True)

    expected = output["requested"]["attempts"]
    output["attempts"].sort(key=lambda row: row["attempt_id"])
    output["summary"] = {
        "expected_attempts": expected,
        "recorded_attempts": len(output["attempts"]),
        "successes": sum(row["status"] == "success" for row in output["attempts"]),
        "failures": sum(row["status"] == "failed" for row in output["attempts"]),
    }
    output["status"] = (
        "raw_generation_complete" if len(output["attempts"]) == expected
        else "raw_generation_incomplete")
    atomic_json(output_path, output)
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
