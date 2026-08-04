#!/usr/bin/env python
"""Benchmark ProteinMPNN backbone-conditional H3 likelihood on reviewed complexes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "modules"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from benchmark_h3_scorer_sanity import composition_shuffles, sha256  # noqa: E402
from h3_interface_contacts import (  # noqa: E402
    ResidueID,
    extract_h3_interface,
    write_standardized_backbone,
)


MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def _residue_id(row):
    return ResidueID(row["chain"], int(row["resseq"]), row.get("icode", ""))


def prepare_reference(reference, work_dir, contact_config):
    source = (PROJECT_ROOT / reference["path"]).resolve()
    if sha256(source) != reference["sha256"].upper():
        raise ValueError(f"{reference['pdb']} SHA256 mismatch")
    interface = extract_h3_interface(
        str(source), reference["heavy_chain"], reference["peptide_chain"],
        light_chain=reference["light_chain"],
        expected_h3_sequence=reference["expected_h3_sequence"],
        expected_peptide_sequence=reference["expected_peptide_sequence"],
        contact_cutoff=float(contact_config["contact_cutoff_angstrom"]),
        sidechain_cutoff=float(contact_config["sidechain_cutoff_angstrom"]),
    )
    standardized_dir = work_dir / reference["pdb"]
    standardized = standardized_dir / f"{reference['pdb']}.pdb"
    manifest = write_standardized_backbone(
        str(source), str(standardized),
        [reference["heavy_chain"], reference["light_chain"], reference["peptide_chain"]])
    heavy_manifest = manifest["chains"][reference["heavy_chain"]]
    standard_by_original = {
        _residue_id(row["original"]): int(row["standardized_resseq"])
        for row in heavy_manifest["residue_mapping"]
    }
    h3_positions = [
        standard_by_original[_residue_id(row)] for row in interface["h3_residues"]
    ]
    fixed_positions = [
        position for position in range(1, heavy_manifest["length"] + 1)
        if position not in set(h3_positions)
    ]
    fixed_path = work_dir / f"{reference['pdb']}_fixed_positions.jsonl"
    fixed_payload = {
        reference["pdb"]: {reference["heavy_chain"]: fixed_positions}
    }
    fixed_path.write_text(json.dumps(fixed_payload) + "\n", encoding="ascii")
    return interface, manifest, standardized, fixed_path, h3_positions


def run_mpnn(config, reference, standardized, fixed_path, work_dir):
    mpnn = config["proteinmpnn"]
    output = work_dir / f"{reference['pdb']}_mpnn"
    command = [
        sys.executable,
        str((PROJECT_ROOT / mpnn["script"]).resolve()),
        "--pdb_path", standardized.resolve().as_posix(),
        "--pdb_path_chains", reference["heavy_chain"],
        "--fixed_positions_jsonl", fixed_path.resolve().as_posix(),
        "--path_to_model_weights", (PROJECT_ROOT / mpnn["weights"]).resolve().as_posix(),
        "--model_name", mpnn["model_name"],
        "--conditional_probs_only", "1",
        "--conditional_probs_only_backbone", "1",
        "--num_seq_per_target", "1",
        "--batch_size", "1",
        "--seed", str(mpnn["seed"]),
        "--suppress_print", "1",
        "--out_folder", output.resolve().as_posix(),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
    if completed.returncode:
        raise RuntimeError(
            f"ProteinMPNN failed for {reference['pdb']}: {completed.stderr[-2000:]}")
    result_path = output / "conditional_probs_only" / reference["pdb"]
    if not result_path.with_suffix(".npz").exists():
        raise FileNotFoundError(f"Missing ProteinMPNN output: {result_path}.npz")
    return np.load(result_path.with_suffix(".npz")), command


def score_sequences(log_p, sequences, global_indices):
    probabilities = log_p[0]
    rows = []
    for sequence in sequences:
        nll = -sum(
            float(probabilities[index, MPNN_ALPHABET.index(aa)])
            for index, aa in zip(global_indices, sequence)
        ) / len(sequence)
        rows.append({"sequence": sequence, "h3_nll": round(nll, 6)})
    return rows


def benchmark_reference(config, contact_config, reference, work_dir):
    interface, manifest, standardized, fixed_path, h3_positions = prepare_reference(
        reference, work_dir, contact_config)
    mpnn_result, command = run_mpnn(
        config, reference, standardized, fixed_path, work_dir)
    # ProteinMPNN's tied_featurize concatenates masked/designed chains before
    # visible chains. This runner masks only the heavy chain, so its offset is 0.
    global_indices = [position - 1 for position in h3_positions]
    design_mask = mpnn_result["design_mask"]
    if not all(design_mask[index] > 0 for index in global_indices):
        raise ValueError(f"ProteinMPNN design mask does not cover all H3 positions for {reference['pdb']}")
    native_sequence = interface["h3_sequence"]
    shuffles = composition_shuffles(
        native_sequence, int(config["shuffle_trials"]),
        int(config["proteinmpnn"]["seed"]) + sum(ord(char) for char in reference["pdb"]))
    rows = score_sequences(
        mpnn_result["log_p"], [native_sequence, *shuffles], global_indices)
    native = rows[0]
    controls = rows[1:]
    control_nll = [row["h3_nll"] for row in controls]
    percentile = sum(value > native["h3_nll"] for value in control_nll) / len(control_nll)
    threshold = float(config["diagnostic_gates"]["minimum_native_shuffle_percentile"])
    return {
        "pdb": reference["pdb"],
        "antibody": reference["antibody"],
        "native": native,
        "composition_shuffles": controls,
        "summary": {
            "h3_sequence": native_sequence,
            "n_h3_positions": len(global_indices),
            "native_h3_nll": native["h3_nll"],
            "shuffle_count": len(controls),
            "shuffle_mean_h3_nll": round(sum(control_nll) / len(control_nll), 6),
            "native_shuffle_percentile": round(percentile, 6),
            "diagnostic_pass": percentile >= threshold,
        },
        "provenance": {
            "standardization_manifest": manifest,
            "h3_standardized_positions": h3_positions,
            "h3_global_indices_zero_based": global_indices,
            "proteinmpnn_command": command,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "benchmarks" / "peptide_h3_mpnn_sanity_v1.yml"))
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "results" / "publication" / "h3_mpnn_sanity_v1"))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    reference_config_path = (PROJECT_ROOT / config["reference_config"]).resolve()
    contact_config = yaml.safe_load(reference_config_path.read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    work_dir = out_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    results = [
        benchmark_reference(config, contact_config, reference, work_dir)
        for reference in contact_config["references"]
    ]
    pass_fraction = sum(row["summary"]["diagnostic_pass"] for row in results) / len(results)
    required = float(config["diagnostic_gates"]["required_reference_pass_fraction"])
    output = {
        "schema_version": 1,
        "purpose": config["purpose"],
        "config_sha256": sha256(config_path),
        "reference_config_sha256": sha256(reference_config_path),
        "reference_pass_fraction": pass_fraction,
        "overall_pass": pass_fraction >= required,
        "results": results,
    }
    (out_dir / "results.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(json.dumps({
        "overall_pass": output["overall_pass"],
        "reference_pass_fraction": pass_fraction,
        "results": {row["pdb"]: row["summary"] for row in results},
        "out_dir": str(out_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
