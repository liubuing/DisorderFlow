#!/usr/bin/env python3
"""Generate RMSF-conditioned A-beta42 antibody candidates with corrected v5.1."""

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "modules"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import bfn_loader  # noqa: E402
from antibody_epitope_complex import position_epitope  # noqa: E402
from evaluate_v5_1_phase3 import load_checkpoint  # noqa: E402
from idp_antibody_design import (  # noqa: E402
    _extract_sequence_from_pdb,
    _parse_cdr_ranges,
    graft_cdrs,
)


ABETA42 = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"
SCAFFOLDS = [
    ("3STB", "data/misfolding_targets/3STB.pdb", "A", "A:26-33,51-58,97-113"),
    ("5IMK", "data/misfolding_targets/5IMK.pdb", "B", "B:26-33,51-58,97-113"),
]
SEEDS = [0, 2, 4]
AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def read_ca(path):
    sequence = []
    coordinates = []
    seen = set()
    with open(path, encoding="ascii") as handle:
        for line in handle:
            if not line.startswith("ATOM") or line[21:22] != "P" or line[12:16].strip() != "CA":
                continue
            residue = (line[22:26], line[26:27])
            if residue in seen:
                continue
            seen.add(residue)
            sequence.append(AA3.get(line[17:20].strip(), "X"))
            coordinates.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return "".join(sequence), np.asarray(coordinates, dtype=np.float64)


def align_to_reference(mobile, reference):
    mobile_center = mobile.mean(axis=0)
    reference_center = reference.mean(axis=0)
    covariance = (mobile - mobile_center).T @ (reference - reference_center)
    left, _, right = np.linalg.svd(covariance)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    return (mobile - mobile_center) @ rotation + reference_center


def rmsf_profile(seed_paths):
    structures = []
    for path in seed_paths:
        sequence, coordinates = read_ca(path)
        if sequence != ABETA42 or coordinates.shape != (42, 3):
            raise ValueError(f"A-beta seed does not match the 42-residue contract: {path}")
        structures.append(coordinates)
    reference = structures[0]
    aligned = np.stack([reference] + [align_to_reference(item, reference) for item in structures[1:]])
    mean = aligned.mean(axis=0)
    rmsf = np.sqrt(np.mean(np.sum((aligned - mean) ** 2, axis=-1), axis=0))
    condition = np.tanh(rmsf / 2.0)
    return rmsf, condition


def max_run(sequence):
    longest = current = 1
    for left, right in zip(sequence, sequence[1:]):
        current = current + 1 if left == right else 1
        longest = max(longest, current)
    return longest


def shannon(sequence):
    counts = Counter(sequence)
    return -sum((count / len(sequence)) * math.log(count / len(sequence)) for count in counts.values())


def nglyco_count(sequence):
    return sum(
        sequence[index] == "N" and sequence[index + 1] != "P" and sequence[index + 2] in "ST"
        for index in range(len(sequence) - 2)
    )


def quality_metrics(cdr, full_sequence, scaffold_sequence, result):
    metrics = {
        "unique_aa": len(set(cdr)),
        "shannon": shannon(cdr),
        "max_run": max_run(cdr),
        "new_cysteines": max(0, cdr.count("C")),
        "new_nglyco_motifs": max(0, nglyco_count(full_sequence) - nglyco_count(scaffold_sequence)),
    }
    failures = []
    if metrics["unique_aa"] < 8:
        failures.append("low_diversity")
    if metrics["shannon"] < 1.8:
        failures.append("low_shannon")
    if metrics["max_run"] > 3:
        failures.append("homopolymer")
    if metrics["new_cysteines"]:
        failures.append("new_cysteine")
    if metrics["new_nglyco_motifs"]:
        failures.append("new_nglyco")
    if result["ppl"] > 200:
        failures.append("high_ppl")
    if not 0.25 <= result["entropy"] <= 2.95:
        failures.append("entropy_out_of_range")
    metrics["quality_failures"] = failures
    metrics["passes_quality"] = not failures
    return metrics


def candidate_score(result):
    contact = result.get("contact_score", 0.0)
    plddt = result.get("plddt", 0.0)
    iptm = result.get("iptm", 0.0)
    inverse_ppl = 1.0 / (1.0 + math.log1p(result["ppl"]))
    entropy_quality = max(0.0, 1.0 - abs(result["entropy"] - 1.8) / 1.8)
    return 0.50 * contact + 0.15 * plddt + 0.10 * iptm + 0.15 * inverse_ppl + 0.10 * entropy_quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="results/v5_1_candidates/abeta42")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--zero-controls", type=int, default=2)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=3100)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    complex_dir = output_dir / "complexes"
    output_dir.mkdir(parents=True, exist_ok=True)
    complex_dir.mkdir(exist_ok=True)
    seed_paths = [PROJECT_ROOT / f"data/abeta_conformations/pdbs/abeta42_seed{seed}_42.pdb" for seed in range(5)]
    rmsf, condition = rmsf_profile(seed_paths)

    model, config = load_checkpoint(args.checkpoint, args.device)
    bfn_loader._bfn_model = model
    bfn_loader._bfn_config = config

    candidates = []
    paired_controls = []
    for scaffold_index, (name, scaffold_path, chain, region_spec) in enumerate(SCAFFOLDS):
        scaffold_sequence = _extract_sequence_from_pdb(scaffold_path, chain)
        cdr_ranges = _parse_cdr_ranges(region_spec)
        for seed_index in SEEDS:
            epitope_path = PROJECT_ROOT / f"data/abeta_conformations/pdbs/abeta42_seed{seed_index}_42.pdb"
            complex_info = position_epitope(
                scaffold_path, epitope_path, scaffold_chain=chain, epitope_chain="P",
                distance=8.0, output_dir=complex_dir)
            pair_seed = args.seed + scaffold_index * 1000 + seed_index * 100
            factual = bfn_loader.run_bfn_design(
                complex_info["pdb_path"], region_spec, num_samples=args.samples,
                stochastic=True, context_chains=["P"], device=args.device,
                disorder_guided=True, disorder_guided_strength=0.35,
                epitope_disorder_profile=condition, sampling_seed=pair_seed)
            zero = bfn_loader.run_bfn_design(
                complex_info["pdb_path"], region_spec, num_samples=args.zero_controls,
                stochastic=True, context_chains=["P"], device=args.device,
                disorder_guided=True, disorder_guided_strength=0.35,
                epitope_disorder_profile=np.zeros_like(condition), sampling_seed=pair_seed)

            for sample_index, result in enumerate(factual):
                full_sequence, mutations = graft_cdrs(scaffold_sequence, result["sequence"], cdr_ranges)
                quality = quality_metrics(result["sequence"], full_sequence, scaffold_sequence, result)
                candidates.append({
                    "candidate_id": f"{name}_seed{seed_index}_factual_{sample_index:02d}",
                    "scaffold": name,
                    "conformation_seed": seed_index,
                    "condition_arm": "rmsf_factual",
                    "cdr_sequence": result["sequence"],
                    "full_antibody_sequence": full_sequence,
                    "mutation_count": len(mutations),
                    "mutations": [f"{position}:{old}>{new}" for position, old, new in mutations],
                    **result,
                    **quality,
                    "candidate_score": candidate_score(result),
                })
            for sample_index, (factual_result, zero_result) in enumerate(zip(factual, zero)):
                paired_controls.append({
                    "scaffold": name,
                    "conformation_seed": seed_index,
                    "sample_index": sample_index,
                    "factual_cdr": factual_result["sequence"],
                    "zero_cdr": zero_result["sequence"],
                    "hamming_fraction": sum(a != b for a, b in zip(factual_result["sequence"], zero_result["sequence"]))
                    / len(factual_result["sequence"]),
                    "contact_delta": factual_result.get("contact_score", 0.0) - zero_result.get("contact_score", 0.0),
                })
            print(f"Generated {name} seed {seed_index}: {len(factual)} factual + {len(zero)} zero", flush=True)

    candidates.sort(key=lambda item: (item["passes_quality"], item["candidate_score"]), reverse=True)
    selected = []
    seen_sequences = set()
    per_context = Counter()
    for candidate in candidates:
        context = (candidate["scaffold"], candidate["conformation_seed"])
        sequence = candidate["full_antibody_sequence"]
        if not candidate["passes_quality"] or sequence in seen_sequences or per_context[context] >= 2:
            continue
        selected.append(candidate)
        seen_sequences.add(sequence)
        per_context[context] += 1
        if len(selected) == args.top:
            break
    for rank, candidate in enumerate(selected, 1):
        candidate["rank"] = rank

    report = {
        "checkpoint": args.checkpoint,
        "weights_kind": "ema",
        "target": "A-beta42",
        "target_sequence": ABETA42,
        "condition_contract": {
            "source": "five locally cached A-beta42 conformations",
            "alignment": "global CA Kabsch alignment",
            "raw_label": "per-residue CA RMSF",
            "model_input": "tanh(RMSF / 2.0)",
            "rmsf": rmsf.tolist(),
            "profile": condition.tolist(),
        },
        "generation": {
            "factual_candidates": len(candidates),
            "paired_zero_controls": len(paired_controls),
            "quality_pass": sum(item["passes_quality"] for item in candidates),
            "selected_for_af2": len(selected),
            "contrastive_used": False,
        },
        "paired_condition_effect": {
            "mean_hamming_fraction": float(np.mean([item["hamming_fraction"] for item in paired_controls])),
            "mean_contact_delta": float(np.mean([item["contact_delta"] for item in paired_controls])),
            "pairs": paired_controls,
        },
        "selected": selected,
        "all_candidates": candidates,
    }
    (output_dir / "generation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    fields = [
        "rank", "candidate_id", "scaffold", "conformation_seed", "cdr_sequence",
        "full_antibody_sequence", "candidate_score", "contact_score", "ppl", "entropy",
        "plddt", "iptm", "pae", "mutation_count", "unique_aa", "shannon",
    ]
    with (output_dir / "af2_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    with (output_dir / "af2_candidates.fasta").open("w", encoding="ascii") as handle:
        for candidate in selected:
            handle.write(f">{candidate['candidate_id']}|rank={candidate['rank']}\n")
            handle.write(f"{candidate['full_antibody_sequence']}:{ABETA42}\n")
    print(json.dumps({key: value for key, value in report["generation"].items()}, indent=2))
    print(json.dumps(report["paired_condition_effect"] | {"pairs": len(paired_controls)}, indent=2))


if __name__ == "__main__":
    main()
