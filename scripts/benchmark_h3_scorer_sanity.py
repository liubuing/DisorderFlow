#!/usr/bin/env python
"""Run the CPU-only H3 interface scorer implementation diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "modules"))

from h3_interface_contacts import (  # noqa: E402
    extract_h3_interface,
    score_h3_sequence,
    serialize_interface,
)


def composition_shuffles(sequence, n, seed):
    rng = random.Random(seed)
    native = str(sequence)
    seen = {native}
    controls = []
    attempts = 0
    max_unique = _multiset_permutations(native) - 1
    target = min(n, max_unique)
    while len(controls) < target and attempts < max(n * 100, 1000):
        attempts += 1
        chars = list(native)
        rng.shuffle(chars)
        shuffled = "".join(chars)
        if shuffled not in seen:
            seen.add(shuffled)
            controls.append(shuffled)
    return controls


def _multiset_permutations(sequence):
    import math

    total = math.factorial(len(sequence))
    for aa in set(sequence):
        total //= math.factorial(sequence.count(aa))
    return total


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def benchmark_reference(reference, config, config_path):
    pdb_path = (PROJECT_ROOT / reference["path"]).resolve()
    observed_sha = sha256(pdb_path)
    if observed_sha != reference["sha256"].upper():
        raise ValueError(
            f"{reference['pdb']} SHA256 mismatch: {observed_sha} != {reference['sha256']}")
    interface = extract_h3_interface(
        str(pdb_path),
        reference["heavy_chain"],
        reference["peptide_chain"],
        light_chain=reference["light_chain"],
        expected_h3_sequence=reference["expected_h3_sequence"],
        expected_peptide_sequence=reference["expected_peptide_sequence"],
        contact_cutoff=float(config["contact_cutoff_angstrom"]),
        sidechain_cutoff=float(config["sidechain_cutoff_angstrom"]),
    )
    native_sequence = interface["h3_sequence"]
    native = score_h3_sequence(native_sequence, interface)
    shuffles = composition_shuffles(
        native_sequence,
        int(config["shuffle_trials"]),
        int(config["seed"]) + sum(ord(char) for char in reference["pdb"]),
    )
    shuffle_rows = [score_h3_sequence(sequence, interface) for sequence in shuffles]
    shuffle_scores = [row["chemistry_score"] for row in shuffle_rows]
    alanine_rows = []
    for index, native_aa in enumerate(native_sequence):
        if native_aa == "A":
            continue
        sequence = native_sequence[:index] + "A" + native_sequence[index + 1:]
        alanine_rows.append({
            "position": index,
            "native_aa": native_aa,
            **score_h3_sequence(sequence, interface),
        })
    n_shuffles = len(shuffle_scores)
    percentile = None
    if n_shuffles:
        percentile = sum(score < native["chemistry_score"] for score in shuffle_scores) / n_shuffles
    contact_positions = sorted({
        contact.h3_index for contact in interface["contacts"]
        if contact.min_sidechain_distance is not None
    })
    gates = config.get("diagnostic_gates", {})
    minimum_positions = int(gates.get("minimum_contacting_h3_positions", 1))
    minimum_contacts = int(gates.get("minimum_sidechain_contacts", 1))
    minimum_percentile = float(gates.get("minimum_native_shuffle_percentile", 0.0))
    minimum_margin = float(gates.get("minimum_native_minus_shuffle_mean", 0.0))
    shuffle_mean = sum(shuffle_scores) / n_shuffles if n_shuffles else None
    native_margin = (
        native["chemistry_score"] - shuffle_mean if shuffle_mean is not None else None)
    diagnostic_pass = bool(
        n_shuffles
        and len(contact_positions) >= minimum_positions
        and native["n_sidechain_contacts"] >= minimum_contacts
        and percentile is not None
        and percentile >= minimum_percentile
        and native_margin is not None
        and native_margin >= minimum_margin
    )
    return {
        "pdb": reference["pdb"],
        "antibody": reference["antibody"],
        "provenance": {
            "path": reference["path"],
            "sha256": observed_sha,
            "config": str(config_path),
        },
        "interface": serialize_interface(interface),
        "native": native,
        "controls": {
            "composition_shuffles": shuffle_rows,
            "single_alanine": alanine_rows,
        },
        "summary": {
            "n_h3_positions": len(native_sequence),
            "n_contacting_h3_positions": len(contact_positions),
            "contacting_h3_positions": contact_positions,
            "n_geometry_contacts": len(interface["contacts"]),
            "n_sidechain_contacts": native["n_sidechain_contacts"],
            "native_chemistry_score": native["chemistry_score"],
            "shuffle_count": n_shuffles,
            "shuffle_mean": (
                round(shuffle_mean, 6) if shuffle_mean is not None else None),
            "shuffle_max": max(shuffle_scores) if n_shuffles else None,
            "native_shuffle_percentile": (
                round(percentile, 6) if percentile is not None else None),
            "native_minus_shuffle_mean": (
                round(native_margin, 6) if native_margin is not None else None),
            "diagnostic_pass": diagnostic_pass,
        },
    }


def write_report(result, path):
    lines = [
        "# H3 Interface Scorer Sanity Benchmark",
        "",
        "This is an implementation diagnostic. It is not evidence of binding affinity or specificity.",
        "",
        "| PDB | H3 | Geometry contacts | Sidechain contacts | Native score | Shuffle mean | Percentile | Diagnostic |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in result["results"]:
        summary = row["summary"]
        lines.append(
            f"| {row['pdb']} | {summary['n_h3_positions']} | "
            f"{summary['n_geometry_contacts']} | {summary['n_sidechain_contacts']} | "
            f"{summary['native_chemistry_score']} | {summary['shuffle_mean']} | "
            f"{summary['native_shuffle_percentile']} | "
            f"{'PASS' if summary['diagnostic_pass'] else 'FAIL'} |")
    lines.extend([
        "",
        f"Overall diagnostic: **{'PASS' if result['overall_pass'] else 'FAIL'}**",
        "",
        "A pass only shows that the fixed contact-chemistry heuristic behaves as expected on the reviewed controls.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "benchmarks" / "peptide_h3_sanity_v1.yml"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "results" / "publication" / "h3_scorer_sanity_v1"),
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    results = [
        benchmark_reference(reference, config, config_path)
        for reference in config["references"]
    ]
    pass_fraction = sum(
        row["summary"]["diagnostic_pass"] for row in results) / max(1, len(results))
    required_pass_fraction = float(
        config.get("diagnostic_gates", {}).get("required_reference_pass_fraction", 1.0))
    overall_pass = pass_fraction >= required_pass_fraction
    output = {
        "schema_version": 1,
        "purpose": config["purpose"],
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "overall_pass": overall_pass,
        "reference_pass_fraction": pass_fraction,
        "diagnostic_gates": config.get("diagnostic_gates", {}),
        "results": results,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="ascii")
    write_report(output, out_dir / "report.md")
    print(json.dumps({
        "overall_pass": overall_pass,
        "out_dir": str(out_dir),
        "references": {
            row["pdb"]: row["summary"] for row in results
        },
    }, indent=2))


if __name__ == "__main__":
    main()
