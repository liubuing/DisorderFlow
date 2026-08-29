#!/usr/bin/env python
"""Run the frozen prospective 4HIX/3D6 conservative H3 candidate protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from Bio.PDB import PDBParser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

AA = set("ACDEFGHIKLMNPQRSTVWY")
AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
CONSERVATIVE_GROUPS = (
    set("AVLIM"), set("FYW"), set("STNQ"), set("DE"), set("KRH"), set("GAS")
)
HYDROPHOBIC = set("AILMFWYV")
POSITIVE = set("KRH")
NEGATIVE = set("DE")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hamming(first, second):
    if len(first) != len(second):
        raise ValueError("Cannot compare sequences of different lengths")
    return sum(a != b for a, b in zip(first, second, strict=True))


def chain_sequence(path, chain_id):
    model = PDBParser(QUIET=True).get_structure(Path(path).stem, str(path))[0]
    return "".join(AA3[residue.resname] for residue in model[chain_id] if residue.resname in AA3)


def allowed_substitutions(amino_acid):
    allowed = set()
    for group in CONSERVATIVE_GROUPS:
        if amino_acid in group:
            allowed.update(group)
    allowed.discard(amino_acid)
    return sorted(allowed)


def conservative_library(native, size, seed, minimum=1, maximum=4):
    rng = random.Random(seed)
    mutable = [index for index, aa in enumerate(native) if allowed_substitutions(aa)]
    output = []
    seen = {native}
    attempts = 0
    while len(output) < size and attempts < size * 500:
        attempts += 1
        count = rng.randint(minimum, maximum)
        positions = rng.sample(mutable, min(count, len(mutable)))
        sequence = list(native)
        for position in positions:
            sequence[position] = rng.choice(allowed_substitutions(native[position]))
        sequence = "".join(sequence)
        if sequence not in seen:
            seen.add(sequence)
            output.append(sequence)
    if len(output) != size:
        raise RuntimeError(f"Generated only {len(output)}/{size} conservative variants")
    return output


def composition_shuffles(native, count, seed):
    rng = random.Random(seed)
    output = []
    seen = {native}
    while len(output) < count:
        values = list(native)
        rng.shuffle(values)
        sequence = "".join(values)
        if sequence not in seen:
            seen.add(sequence)
            output.append(sequence)
    return output


def net_charge(sequence):
    return sum(aa in POSITIVE for aa in sequence) - sum(aa in NEGATIVE for aa in sequence)


def hydrophobic_fraction(sequence):
    return sum(aa in HYDROPHOBIC for aa in sequence) / len(sequence)


def nglyco_count(sequence):
    return len(re.findall(r"N[^P][ST]", sequence))


def validate_contract(config):
    design = config["design"]
    scaffold = ROOT / design["scaffold"]
    if sha256(scaffold) != design["scaffold_sha256"]:
        raise ValueError("Scaffold hash mismatch")
    heavy = chain_sequence(scaffold, design["heavy_chain"])
    if heavy[95:107] != design["native_h3"] or design["region_spec"] != "H:96-107":
        raise ValueError("Frozen 4HIX H3 mapping is not the expected 12-residue mask")
    for pose in config["coordinate_panel"]:
        path = ROOT / pose["path"]
        if sha256(path) != pose["sha256"]:
            raise ValueError(f"Coordinate hash mismatch: {pose['id']}")
        if chain_sequence(path, "H")[95:107] != design["native_h3"]:
            raise ValueError(f"Heavy-chain mapping mismatch: {pose['id']}")
        if not chain_sequence(path, pose["antigen_chain"]):
            raise ValueError(f"Missing antigen chain: {pose['id']}")
    checkpoint = ROOT / config["model"]["checkpoint"]
    if sha256(checkpoint) != config["model"]["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    control = ROOT / config["generation"]["proteinmpnn_control"]
    if sha256(control) != config["generation"]["proteinmpnn_control_sha256"]:
        raise ValueError("ProteinMPNN control hash mismatch")


def candidate_metadata(config, smoke=False):
    native = config["design"]["native_h3"]
    generation = config["generation"]
    library_size = 12 if smoke else int(generation["conservative_library_size"])
    shuffle_count = 8 if smoke else int(generation["composition_shuffle_controls"])
    rows = [{"sequence": native, "sources": ["native"]}]
    rows.extend({"sequence": sequence, "sources": ["conservative_library"]} for sequence in
                conservative_library(native, library_size, generation["conservative_library_seed"]))
    rows.extend({"sequence": sequence, "sources": ["composition_shuffle"]} for sequence in
                composition_shuffles(native, shuffle_count, generation["shuffle_seed"]))
    mpnn = json.loads((ROOT / generation["proteinmpnn_control"]).read_text(encoding="utf-8"))
    rows.extend({"sequence": row["sequence"], "sources": ["proteinmpnn"]}
                for row in mpnn["results"])
    return merge_candidates(rows)


def merge_candidates(rows):
    merged = {}
    for row in rows:
        sequence = row["sequence"].strip().upper()
        if not sequence or any(aa not in AA for aa in sequence):
            continue
        entry = merged.setdefault(sequence, {"sequence": sequence, "sources": []})
        entry["sources"] = sorted(set(entry["sources"] + row.get("sources", [])))
    return list(merged.values())


def generate_bfn_rows(config, model, smoke=False):
    from modules.bfn_loader import run_bfn_design

    pose_by_id = {pose["id"]: pose for pose in config["coordinate_panel"]}
    pose_ids = config["generation"]["bfn_poses"][:1 if smoke else None]
    seeds = config["generation"]["seeds"][:1 if smoke else None]
    samples = 2 if smoke else int(config["generation"]["bfn_candidates_per_pose_seed"])
    rows = []
    for pose_id in pose_ids:
        pose = pose_by_id[pose_id]
        for seed in seeds:
            generated = run_bfn_design(
                str(ROOT / pose["path"]), config["design"]["region_spec"],
                num_samples=samples, stochastic=True,
                context_chains=[config["design"]["light_chain"], pose["antigen_chain"]],
                antigen_chains=[pose["antigen_chain"]], device=next(model.parameters()).device,
                sampling_seed=seed,
            )
            rows.extend({"sequence": row["sequence"],
                         "sources": [f"bfn:{pose_id}:{seed}"],
                         "generator_metrics": row} for row in generated)
    return rows


def tensor_rows(result, sequences, pose_id, state):
    rows = []
    for index, sequence in enumerate(sequences):
        contact = result["contact"][index]
        rows.append({
            "sequence": sequence,
            "pose_id": pose_id,
            "state": state,
            "state_compatibility": float(result["state_compatibility"][index].cpu()),
            "iptm": float(result["iptm"][index].cpu()),
            "mean_contact_probability": float(torch.sigmoid(contact).mean().cpu()),
        })
    return rows


def score_candidates(config, model, candidates):
    from modules.bfn_prospective import score_bfn_candidates

    sequences = [row["sequence"] for row in candidates]
    batch_size = int(config["model"]["score_batch_size"])
    device = next(model.parameters()).device
    rows = []
    for pose in config["coordinate_panel"]:
        path = str(ROOT / pose["path"])
        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start:start + batch_size]
            complex_result = score_bfn_candidates(
                path, config["design"]["region_spec"], chunk,
                context_chains=[config["design"]["light_chain"], pose["antigen_chain"]],
                antigen_chains=[pose["antigen_chain"]], device=device, model=model,
                fixed_t=config["model"]["fixed_t"])
            stripped_result = score_bfn_candidates(
                path, config["design"]["region_spec"], chunk,
                context_chains=[config["design"]["light_chain"]],
                device=device, model=model, fixed_t=config["model"]["fixed_t"])
            complex_rows = tensor_rows(complex_result, chunk, pose["id"], "complex")
            stripped_rows = tensor_rows(stripped_result, chunk, pose["id"], "stripped")
            for complex_row, stripped_row in zip(complex_rows, stripped_rows, strict=True):
                complex_row["complex_minus_stripped"] = (
                    complex_row["state_compatibility"] - stripped_row["state_compatibility"])
                rows.append(complex_row)
                rows.append(stripped_row)
    return rows


def aggregate_scores(config, candidates, score_rows):
    native = config["design"]["native_h3"]
    complex_by_sequence = {}
    for row in score_rows:
        if row["state"] == "complex":
            complex_by_sequence.setdefault(row["sequence"], []).append(row)
    native_rows = complex_by_sequence[native]
    native_by_pose = {row["pose_id"]: row for row in native_rows}
    mpnn_sequences = {row["sequence"] for row in candidates if "proteinmpnn" in row["sources"]}
    mpnn_means = [np.mean([row["complex_minus_stripped"] for row in complex_by_sequence[sequence]])
                  for sequence in mpnn_sequences if sequence != native]
    mpnn_median = float(np.median(mpnn_means)) if mpnn_means else float("nan")
    native_mean = float(np.mean([row["complex_minus_stripped"] for row in native_rows]))
    native_worst = float(np.min([row["complex_minus_stripped"] for row in native_rows]))
    for candidate in candidates:
        rows = complex_by_sequence[candidate["sequence"]]
        deltas = [row["complex_minus_stripped"] for row in rows]
        candidate["scoring"] = {
            "mean_complex_minus_stripped": float(np.mean(deltas)),
            "std_complex_minus_stripped": float(np.std(deltas)),
            "worst_complex_minus_stripped": float(np.min(deltas)),
            "poses_better_than_native": sum(
                row["complex_minus_stripped"] > native_by_pose[row["pose_id"]]["complex_minus_stripped"]
                for row in rows),
            "coordinate_sensitive_poses": sum(
                abs(value) >= config["selection"]["minimum_abs_complex_minus_stripped"]
                for value in deltas),
            "mean_contact_probability": float(np.mean(
                [row["mean_contact_probability"] for row in rows])),
        }
    return {"native_mean": native_mean, "native_worst": native_worst,
            "proteinmpnn_median": mpnn_median}


def add_developability(config, candidates):
    from scripts.pipeline.score_shortlist_developability import score_sequence

    design = config["design"]
    heavy = chain_sequence(ROOT / design["scaffold"], design["heavy_chain"])
    native = design["native_h3"]
    native_full = score_sequence(heavy)["sequence_risk"]
    native_h3_charge = net_charge(native)
    native_h3_hydro = hydrophobic_fraction(native)
    native_nglyco = nglyco_count(native)
    for candidate in candidates:
        sequence = candidate["sequence"]
        full_heavy = heavy[:95] + sequence + heavy[107:]
        score = score_sequence(full_heavy)
        candidate["mutation_count"] = hamming(sequence, native)
        candidate["developability"] = {
            **score,
            "risk_increase_over_native": score["sequence_risk"] - native_full,
            "h3_charge_change": net_charge(sequence) - native_h3_charge,
            "h3_hydrophobic_fraction_increase": hydrophobic_fraction(sequence) - native_h3_hydro,
            "new_h3_n_glycosylation_motifs": max(0, nglyco_count(sequence) - native_nglyco),
            "new_h3_cysteines": max(0, sequence.count("C") - native.count("C")),
        }


def apply_gates(config, candidates, baselines):
    selection = config["selection"]
    design = config["design"]
    for candidate in candidates:
        score = candidate["scoring"]
        dev = candidate["developability"]
        checks = {
            "prospective_source": bool(set(candidate["sources"]) & {"conservative_library"}) or
                                  any(source.startswith("bfn:") for source in candidate["sources"]),
            "mutation_count": design["mutation_count"]["minimum"] <= candidate["mutation_count"] <=
                              design["mutation_count"]["maximum"],
            "poses_better_than_native": score["poses_better_than_native"] >=
                                        selection["minimum_poses_better_than_native"],
            "mean_better_than_native": score["mean_complex_minus_stripped"] >=
                                       baselines["native_mean"] +
                                       selection["minimum_mean_state_delta_over_native"],
            "worst_pose": score["worst_complex_minus_stripped"] >=
                          baselines["native_worst"] - selection["maximum_worst_pose_deficit"],
            "proteinmpnn_baseline": (not selection["require_above_proteinmpnn_median"] or
                                     score["mean_complex_minus_stripped"] >=
                                     baselines["proteinmpnn_median"]),
            "coordinate_sensitivity": score["coordinate_sensitive_poses"] >=
                                      selection["minimum_coordinate_sensitive_poses"],
            "developability_risk": dev["sequence_risk"] <=
                                   selection["maximum_full_heavy_developability_risk"],
            "risk_increase": dev["risk_increase_over_native"] <=
                             selection["maximum_risk_increase_over_native"],
            "charge": abs(dev["h3_charge_change"]) <=
                      selection["maximum_absolute_h3_charge_change"],
            "hydrophobicity": dev["h3_hydrophobic_fraction_increase"] <=
                              selection["maximum_h3_hydrophobic_fraction_increase"],
            "n_glycosylation": (not selection["forbid_new_h3_n_glycosylation_motif"] or
                                dev["new_h3_n_glycosylation_motifs"] == 0),
            "cysteine": not selection["forbid_new_h3_cysteine"] or dev["new_h3_cysteines"] == 0,
        }
        candidate["gate_checks"] = checks
        candidate["all_gates_pass"] = all(checks.values())


def diverse_shortlist(config, candidates):
    eligible = [row for row in candidates if row["all_gates_pass"]]
    eligible.sort(key=lambda row: (
        -row["scoring"]["mean_complex_minus_stripped"],
        row["scoring"]["std_complex_minus_stripped"], row["sequence"]))
    selected = []
    minimum = int(config["selection"]["minimum_pairwise_hamming"])
    for row in eligible:
        if all(hamming(row["sequence"], prior["sequence"]) >= minimum for prior in selected):
            selected.append(row)
        if len(selected) == int(config["selection"]["shortlist_maximum"]):
            break
    return selected


def preflight(config, candidates, score_rows):
    native_sequence = config["design"]["native_h3"]
    native = next(row for row in candidates if row["sequence"] == native_sequence)
    complex_rows = [
        row for row in score_rows
        if row["sequence"] == native_sequence and row["state"] == "complex"
    ]
    pose_class = {row["id"]: row["class"] for row in config["coordinate_panel"]}
    matched_pose_values = [
        row["state_compatibility"] for row in complex_rows
        if pose_class[row["pose_id"]] == "template_transferred_abeta42_pose"
    ]
    pose_range = max(matched_pose_values) - min(matched_pose_values)
    max_conditioning = max(abs(row["complex_minus_stripped"]) for row in complex_rows)
    range_pass = pose_range >= config["preflight"]["minimum_pose_output_range"]
    conditioning_pass = (
        max_conditioning >= config["preflight"]["minimum_max_complex_minus_stripped_abs"]
    )
    return {
        "native_mean_complex_minus_stripped": native["scoring"]["mean_complex_minus_stripped"],
        "matched_abeta42_pose_output_range": pose_range,
        "maximum_complex_minus_stripped_abs": max_conditioning,
        "matched_pose_range_pass": range_pass,
        "antigen_conditioning_pass": conditioning_pass,
        "coordinate_signal_present": native["scoring"]["coordinate_sensitive_poses"] >=
                                     config["selection"]["minimum_coordinate_sensitive_poses"] and
                                     range_pass and conditioning_pass,
    }


def export_shortlist(config, output_path, shortlist):
    """Export auditable triage files; these are not synthesis specifications."""
    csv_path = output_path.with_name("shortlist_for_af2_prep.csv")
    fasta_path = output_path.with_name("shortlist_for_af2_prep.fasta")
    fields = [
        "rank", "candidate_id", "h3_sequence", "mutation_count", "sources",
        "mean_complex_minus_stripped", "std_complex_minus_stripped",
        "worst_complex_minus_stripped", "poses_better_than_native",
        "developability_risk", "status",
    ]
    rows = []
    fasta = []
    for index, candidate in enumerate(shortlist, 1):
        candidate_id = f"3D6-H3-PV1-{index:02d}"
        rows.append({
            "rank": index,
            "candidate_id": candidate_id,
            "h3_sequence": candidate["sequence"],
            "mutation_count": candidate["mutation_count"],
            "sources": ";".join(candidate["sources"]),
            "mean_complex_minus_stripped": candidate["scoring"][
                "mean_complex_minus_stripped"],
            "std_complex_minus_stripped": candidate["scoring"][
                "std_complex_minus_stripped"],
            "worst_complex_minus_stripped": candidate["scoring"][
                "worst_complex_minus_stripped"],
            "poses_better_than_native": candidate["scoring"]["poses_better_than_native"],
            "developability_risk": candidate["developability"]["sequence_risk"],
            "status": "computational_triage_only_not_synthesis_ready",
        })
        fasta.append(
            f">{candidate_id} computational_triage_only_not_synthesis_ready\n"
            f"{candidate['sequence']}"
        )
    with csv_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    fasta_path.write_text("\n".join(fasta) + "\n", encoding="ascii")
    return csv_path, fasta_path


def source_summary(candidates, shortlist):
    def counts(rows):
        output = {}
        for row in rows:
            for source in row["sources"]:
                family = source.split(":", 1)[0]
                output[family] = output.get(family, 0) + 1
        return dict(sorted(output.items()))

    return {
        "candidate_source_memberships": counts(candidates),
        "shortlist_source_memberships": counts(shortlist),
        "bfn_direct_generation_reached_shortlist": any(
            source.startswith("bfn:")
            for row in shortlist for source in row["sources"]
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmarks/abeta_4hix_prospective_v1.yml")
    parser.add_argument("--out")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-bfn-generation", action="store_true")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_contract(config)
    os.environ["DISORDERFLOW_CHECKPOINT"] = str(ROOT / config["model"]["checkpoint"])
    from modules import bfn_loader

    bfn_loader._bfn_model = None
    bfn_loader._bfn_config = None
    model, _ = bfn_loader.load_bfn(args.device)
    candidates = candidate_metadata(config, smoke=args.smoke)
    if not args.skip_bfn_generation:
        candidates = merge_candidates(candidates + generate_bfn_rows(config, model, smoke=args.smoke))
    score_rows = score_candidates(config, model, candidates)
    baselines = aggregate_scores(config, candidates, score_rows)
    add_developability(config, candidates)
    apply_gates(config, candidates, baselines)
    shortlist = diverse_shortlist(config, candidates)
    minimum = int(config["selection"]["shortlist_minimum"])
    preflight_result = preflight(config, candidates, score_rows)
    status = "smoke_complete_not_for_selection" if args.smoke else (
        "go_for_af2_and_experimental_prep"
        if preflight_result["coordinate_signal_present"] and len(shortlist) >= minimum
        else "no_go_computational_gates")
    output = {
        "schema_version": 1,
        "status": status,
        "config": args.config,
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__)),
        "smoke": args.smoke,
        "baselines": baselines,
        "summary": {
            "unique_candidates": len(candidates),
            "all_gates_pass": sum(row["all_gates_pass"] for row in candidates),
            "diverse_shortlist": len(shortlist),
            "required_shortlist": minimum,
        },
        "sources": source_summary(candidates, shortlist),
        "preflight": preflight_result,
        "shortlist": [{"rank": index + 1, **row} for index, row in enumerate(shortlist)],
        "candidates": candidates,
        "pose_scores": score_rows,
        "claim_boundary": config["claim_boundary"],
    }
    output_path = ROOT / (args.out or config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    temporary.replace(output_path)
    csv_path, fasta_path = export_shortlist(config, output_path, shortlist)
    print(json.dumps({"status": status, **output["summary"]}, indent=2))
    print(f"Wrote {output_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {fasta_path}")


if __name__ == "__main__":
    main()
