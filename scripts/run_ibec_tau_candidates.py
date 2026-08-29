#!/usr/bin/env python
"""Run the frozen correct-mask Tau H3-only candidate protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from Bio.PDB import PDBParser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "modules"))

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
AA = set(AA3.values())
CONSERVATIVE_GROUPS = (
    set("AVLIM"), set("FYW"), set("STNQ"), set("DE"), set("KRH"), set("GAS")
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sequence_sha256(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def hamming(first, second):
    if len(first) != len(second):
        raise ValueError("Unequal H3 lengths")
    return sum(a != b for a, b in zip(first, second, strict=True))


def normalize_gate_checks(checks):
    return {name: bool(passed) for name, passed in checks.items()}


def chain_sequence_and_resids(path, chain_id):
    model = PDBParser(QUIET=True).get_structure(Path(path).stem, str(path))[0]
    residues = [r for r in model[chain_id] if r.resname in AA3 and "CA" in r]
    return (
        "".join(AA3[r.resname] for r in residues),
        [r.id[1] for r in residues],
    )


def allowed_substitutions(amino_acid):
    allowed = set()
    for group in CONSERVATIVE_GROUPS:
        if amino_acid in group:
            allowed.update(group)
    allowed.discard(amino_acid)
    return sorted(allowed)


def conservative_library(native, size, seed, minimum, maximum):
    rng = random.Random(seed)
    mutable = [i for i, aa in enumerate(native) if allowed_substitutions(aa)]
    output, seen, attempts = [], {native}, 0
    while len(output) < size and attempts < size * 500:
        attempts += 1
        count = rng.randint(minimum, maximum)
        positions = rng.sample(mutable, min(count, len(mutable)))
        sequence = list(native)
        for position in positions:
            sequence[position] = rng.choice(allowed_substitutions(native[position]))
        candidate = "".join(sequence)
        if candidate not in seen:
            seen.add(candidate)
            output.append(candidate)
    if len(output) != size:
        raise RuntimeError(f"Conservative library incomplete: {len(output)}/{size}")
    return output


def composition_shuffles(native, count, seed):
    rng = random.Random(seed)
    output, seen = [], {native}
    while len(output) < count:
        values = list(native)
        rng.shuffle(values)
        candidate = "".join(values)
        if candidate not in seen:
            seen.add(candidate)
            output.append(candidate)
    return output


def validate_inputs(config):
    checks = [
        ("sanity_status", "sanity_status_sha256"),
        ("pose_panel_audit", "pose_panel_audit_sha256"),
        ("reference_complex", "reference_complex_sha256"),
    ]
    for name, key in checks:
        if sha256(ROOT / config["inputs"][name]) != config["inputs"][key]:
            raise ValueError(f"Frozen input hash mismatch: {name}")
    checkpoint = ROOT / config["model"]["checkpoint"]
    if sha256(checkpoint) != config["model"]["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    sanity = json.loads((ROOT / config["inputs"]["sanity_status"]).read_text())
    if sanity["decision"] != "go_tau_second_target_extension":
        raise ValueError("Tau sanity did not pass")
    weights = ROOT / config["generation"]["proteinmpnn"]["weights"]
    if sha256(weights / "v_48_020.pt") != config["generation"]["proteinmpnn"]["weights_sha256"]:
        raise ValueError("ProteinMPNN weight hash mismatch")


def tau_pose_rows(config):
    audit = json.loads((ROOT / config["inputs"]["pose_panel_audit"]).read_text())
    rows = sorted(
        (row for row in audit["entries"]
         if row["target"] == "tau" and row.get("selected_for_panel")),
        key=lambda row: row["conformer"],
    )
    if len(rows) != 5:
        raise ValueError("Expected exactly five selected Tau poses")
    for row in rows:
        expected = config["inputs"]["pose_sha256"][f"conformer{row['conformer']}"]
        if sha256(ROOT / row["pose_pdb"]) != expected:
            raise ValueError(f"Pose hash mismatch: {row['pose_pdb']}")
    return rows


def build_fixed_positions(reference_complex, target):
    heavy, heavy_resids = chain_sequence_and_resids(reference_complex, target["heavy_chain"])
    light, light_resids = chain_sequence_and_resids(reference_complex, target["light_chain"])
    antigen, antigen_resids = chain_sequence_and_resids(
        reference_complex, target["reference_antigen_chain"]
    )
    native = heavy[95:108]
    if native != target["h3_sequence"] or len(native) != int(target["h3_length"]):
        raise ValueError(f"Tau H3 mismatch: {native}")
    # ProteinMPNN skips the N-terminal PCA and indexes each parsed chain
    # 1..N over standard residues. The H3 is ordinals 96-108 on chain A.
    design = list(range(96, 109))
    fixed_heavy = [
        index for index in range(1, len(heavy) + 1)
        if not 96 <= index <= 108
    ]
    return {
        "fixed_positions": {
            target["heavy_chain"]: fixed_heavy,
            target["light_chain"]: list(range(1, len(light) + 1)),
            target["reference_antigen_chain"]: list(range(1, len(antigen) + 1)),
        },
        "heavy": heavy,
        "light": light,
        "antigen": antigen,
        "native_h3": native,
        "mpnn_design_ordinals": design,
        "h3_resid_window": heavy_resids[95:108],
    }


def run_proteinmpnn(config, reference, work_dir):
    settings = config["generation"]["proteinmpnn"]
    name = Path(config["inputs"]["reference_complex"]).stem
    fixed_path = work_dir / "fixed_positions.jsonl"
    fixed_path.write_text(
        json.dumps({name: reference["fixed_positions"]}) + "\n", encoding="ascii"
    )
    rows, commands = [], []
    native_anchor = None
    for seed in settings["seeds"]:
        output = work_dir / f"seed_{seed}"
        fasta = output / "seqs" / f"{name}.fa"
        command = [
            sys.executable,
            (ROOT / "ProteinMPNN/protein_mpnn_run.py").resolve().as_posix(),
            "--pdb_path", (ROOT / config["inputs"]["reference_complex"]).resolve().as_posix(),
            "--pdb_path_chains",
            f"{config['target']['heavy_chain']} {config['target']['light_chain']} "
            f"{config['target']['reference_antigen_chain']}",
            "--fixed_positions_jsonl", fixed_path.resolve().as_posix(),
            "--path_to_model_weights", (ROOT / settings["weights"]).resolve().as_posix(),
            "--model_name", "v_48_020",
            "--num_seq_per_target", str(settings["samples_per_seed"]),
            "--batch_size", "1",
            "--sampling_temp", str(settings["temperature"]),
            "--seed", str(seed),
            "--save_score", "1",
            "--suppress_print", "1",
            "--out_folder", output.resolve().as_posix(),
        ]
        commands.append(command)
        started = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True, timeout=900)
        if completed.returncode:
            raise RuntimeError(completed.stderr[-3000:])
        elapsed = time.perf_counter() - started
        lines = fasta.read_text(encoding="utf-8").splitlines()
        native_full = None
        for index, line in enumerate(lines):
            if line.startswith(">") and not line.startswith(">T="):
                native_full = lines[index + 1].strip().split("/")[0]
                break
        if native_full is None or len(native_full) != len(reference["heavy"]):
            raise ValueError("ProteinMPNN native row missing or length mismatch")
        if native_anchor is None:
            native_anchor = native_full.find(reference["native_h3"])
            if native_anchor < 0:
                raise ValueError("Native H3 anchor not found in ProteinMPNN output")
        seed_rows = []
        for index, line in enumerate(lines):
            if not line.startswith(">T="):
                continue
            full = lines[index + 1].strip().split("/")[0]
            if len(full) != len(reference["heavy"]):
                raise ValueError(
                    f"ProteinMPNN heavy length {len(full)} != {len(reference['heavy'])}"
                )
            h3 = full[native_anchor:native_anchor + 13]
            fields = dict(
                field.split("=", 1) for field in line[1:].split(",") if "=" in field
            )
            seed_rows.append({
                "sequence": h3,
                "generator_score": float(fields.get("score", "nan")),
                "seed": seed,
                "wall_seconds": elapsed,
            })
        rows.extend(seed_rows)
    return rows, commands


def run_bfn_control(config, reference, device):
    from modules import bfn_loader
    from modules.bfn_loader import run_bfn_design

    os.environ["DISORDERFLOW_CHECKPOINT"] = str(ROOT / config["model"]["checkpoint"])
    bfn_loader._bfn_model = None
    bfn_loader._bfn_config = None
    rows = []
    settings = config["generation"]["bfn_control"]
    for seed in settings["seeds"]:
        generated = run_bfn_design(
            str(ROOT / config["inputs"]["reference_complex"]),
            config["target"]["h3_region_spec"],
            num_samples=int(settings["samples_per_seed"]),
            stochastic=True,
            context_chains=[
                config["target"]["light_chain"],
                config["target"]["reference_antigen_chain"],
            ],
            antigen_chains=[config["target"]["reference_antigen_chain"]],
            device=device,
            sampling_seed=seed,
        )
        rows.extend({
            "sequence": row["sequence"],
            "generator_score": row["ppl"],
            "seed": seed,
            **{key: value for key, value in row.items() if key != "sequence"},
        } for row in generated)
    return rows


def merge_candidates(rows):
    merged = {}
    for row in rows:
        sequence = str(row["sequence"]).strip().upper()
        if not sequence or len(sequence) != 13 or any(aa not in AA for aa in sequence):
            continue
        entry = merged.setdefault(sequence, {"sequence": sequence, "sources": []})
        entry["sources"] = sorted(set(entry["sources"] + row.get("sources", [])))
        if "generator_score" in row and row["generator_score"] is not None:
            entry.setdefault("generator_scores", []).append(row["generator_score"])
    return list(merged.values())


def score_bfn_candidates_on_pose(config, model, pose_path, sequences, target, device):
    from modules.bfn_prospective import score_bfn_candidates

    batch_size = int(config["model"]["score_batch_size"])
    complex_rows, stripped_rows = [], []
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start:start + batch_size]
        complex_result = score_bfn_candidates(
            str(pose_path), target["h3_region_spec"], chunk,
            context_chains=[target["light_chain"], target["pose_antigen_chain"]],
            antigen_chains=[target["pose_antigen_chain"]],
            device=device, model=model,
            fixed_t=float(config["model"]["fixed_t"]),
        )
        stripped_result = score_bfn_candidates(
            str(pose_path), target["h3_region_spec"], chunk,
            context_chains=[target["light_chain"]],
            device=device, model=model,
            fixed_t=float(config["model"]["fixed_t"]),
        )
        for index, sequence in enumerate(chunk):
            complex_rows.append({
                "sequence": sequence,
                "state_compatibility": float(complex_result["state_compatibility"][index].cpu()),
                "iptm_diagnostic": float(complex_result["iptm"][index].cpu()),
            })
            stripped_rows.append({
                "sequence": sequence,
                "state_compatibility": float(stripped_result["state_compatibility"][index].cpu()),
            })
    return complex_rows, stripped_rows


def build_pose_contact_maps(config, pose_rows):
    from ensemble_pose_transfer import fixed_paratope_contact_map
    from state_contact_scorer import extract_contact_map

    template = extract_contact_map(
        str(ROOT / config["inputs"]["reference_complex"]),
        peptide_chain=config["target"]["reference_antigen_chain"],
    )
    maps = []
    for row in pose_rows:
        contact_map = fixed_paratope_contact_map(
            ROOT / row["pose_pdb"], template,
            peptide_chain=config["target"]["pose_antigen_chain"],
        )
        maps.append((row["conformer"], contact_map))
    return template, maps


def contact_score_for_h3(native_h3, candidate_h3, template, pose_maps, h3_to_paratope):
    from state_contact_scorer import score_sequence_on_contact_map

    base = list(template["paratope_sequence"])
    for h3_index, paratope_index in h3_to_paratope.items():
        base[paratope_index] = candidate_h3[h3_index]
    sequence = "".join(base)
    scores = []
    for _, contact_map in pose_maps:
        scores.append(
            score_sequence_on_contact_map(sequence, contact_map)["state_contact_score"]
        )
    native_base = list(template["paratope_sequence"])
    native_sequence = "".join(native_base)
    native_scores = [
        score_sequence_on_contact_map(native_sequence, contact_map)["state_contact_score"]
        for _, contact_map in pose_maps
    ]
    scores = np.asarray(scores, dtype=float)
    native_scores = np.asarray(native_scores, dtype=float)
    deltas = scores - native_scores
    return {
        "contact_mean": float(scores.mean()),
        "contact_min": float(scores.min()),
        "contact_std": float(scores.std()),
        "contact_mean_delta_native": float(deltas.mean()),
        "contact_min_delta_native": float(deltas.min()),
        "contact_p25_delta_native": float(np.percentile(deltas, 25)),
        "contact_pose_scores": [float(value) for value in scores],
    }


def developability_row(config, reference, sequence):
    from scripts.pipeline.score_shortlist_developability import score_sequence

    heavy = reference["heavy"]
    full = heavy[:95] + sequence + heavy[108:]
    score = score_sequence(full)
    native_score = score_sequence(heavy)
    native_h3 = reference["native_h3"]
    return {
        "full_heavy_developability_risk": score["sequence_risk"],
        "risk_increase_over_native": score["sequence_risk"] - native_score["sequence_risk"],
        "new_h3_cysteines": max(0, sequence.count("C") - native_h3.count("C")),
        "new_h3_n_glycosylation_motifs": max(
            0, len(re.findall(r"N[^P][ST]", sequence))
            - len(re.findall(r"N[^P][ST]", native_h3))
        ),
        "full_heavy_sequence": full,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmarks/ibec_tau_candidates_v1.yml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out")
    parser.add_argument("--skip-mpnn", action="store_true")
    parser.add_argument("--skip-bfn", action="store_true")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_inputs(config)
    target = config["target"]
    pose_rows = tau_pose_rows(config)
    reference = build_fixed_positions(config["inputs"]["reference_complex"], target)

    generation = config["generation"]
    candidate_rows = [{"sequence": reference["native_h3"], "sources": ["native"]}]
    candidate_rows += [
        {"sequence": value, "sources": ["conservative_library"]}
        for value in conservative_library(
            reference["native_h3"],
            int(generation["conservative_library_size"]),
            int(generation["conservative_library_seed"]),
            int(config["selection"]["mutation_count"]["minimum"]),
            int(config["selection"]["mutation_count"]["maximum"]),
        )
    ]
    candidate_rows += [
        {"sequence": value, "sources": ["composition_shuffle"]}
        for value in composition_shuffles(
            reference["native_h3"],
            int(generation["composition_shuffle_count"]),
            int(generation["composition_shuffle_seed"]),
        )
    ]
    mpnn_commands = []
    if not args.skip_mpnn:
        work_dir = ROOT / "results/ibec/tau_candidates_v1/mpnn_work"
        work_dir.mkdir(parents=True, exist_ok=True)
        mpnn_rows, mpnn_commands = run_proteinmpnn(config, reference, work_dir)
        candidate_rows += [
            {**row, "sources": ["proteinmpnn"]} for row in mpnn_rows
        ]
    if not args.skip_bfn:
        bfn_rows = run_bfn_control(config, reference, args.device)
        candidate_rows += [
            {**row, "sources": [f"bfn:{row['seed']}"]} for row in bfn_rows
        ]
    candidates = merge_candidates(candidate_rows)
    sequences = [row["sequence"] for row in candidates]

    from modules import bfn_loader
    from modules.bfn_prospective import score_bfn_candidates

    os.environ["DISORDERFLOW_CHECKPOINT"] = str(ROOT / config["model"]["checkpoint"])
    bfn_loader._bfn_model = None
    bfn_loader._bfn_config = None
    model, _ = bfn_loader.load_bfn(args.device)

    cms_by_sequence = {}
    for row in pose_rows:
        complex_rows, stripped_rows = score_bfn_candidates_on_pose(
            config, model, ROOT / row["pose_pdb"], sequences, target, args.device
        )
        for complex_row, stripped_row in zip(complex_rows, stripped_rows, strict=True):
            entry = cms_by_sequence.setdefault(
                complex_row["sequence"], {"pose_values": [], "pose_iptm": []}
            )
            entry["pose_values"].append(
                complex_row["state_compatibility"] - stripped_row["state_compatibility"]
            )
            entry["pose_iptm"].append(complex_row["iptm_diagnostic"])

    template, pose_maps = build_pose_contact_maps(config, pose_rows)
    h3_to_paratope = {int(k): int(v) for k, v in target["contact_map_h3_to_paratope"].items()}
    native_entry = cms_by_sequence[reference["native_h3"]]
    native_cms = np.asarray(native_entry["pose_values"], dtype=float)
    native_contact = contact_score_for_h3(
        reference["native_h3"], reference["native_h3"], template, pose_maps, h3_to_paratope
    )

    selection = config["selection"]
    enriched = []
    for candidate in candidates:
        sequence = candidate["sequence"]
        cms = np.asarray(cms_by_sequence[sequence]["pose_values"], dtype=float)
        contact = contact_score_for_h3(
            reference["native_h3"], sequence, template, pose_maps, h3_to_paratope
        )
        developability = developability_row(config, reference, sequence)
        mutation_count = hamming(sequence, reference["native_h3"])
        row = {
            **candidate,
            "mutation_count": mutation_count,
            "bfn_mean_cms": float(cms.mean()),
            "bfn_worst_cms": float(cms.min()),
            "bfn_std_cms": float(cms.std()),
            "bfn_pose_values": [float(value) for value in cms],
            **contact,
            **developability,
        }
        checks = {
            "eligible_source": bool(set(candidate["sources"]) & set(selection["eligible_sources"])),
            "mutation_count": (
                int(selection["mutation_count"]["minimum"]) <= mutation_count
                <= int(selection["mutation_count"]["maximum"])
            ),
            "bfn_mean_noninferior": row["bfn_mean_cms"] >= native_cms.mean() + float(
                selection["bfn_mean_cms_noninferiority_vs_native"]
            ),
            "bfn_worst_noninferior": row["bfn_worst_cms"] >= native_cms.min() + float(
                selection["bfn_worst_cms_noninferiority_vs_native"]
            ),
            "contact_mean_delta": row["contact_mean_delta_native"] >= float(
                selection["contact_mean_delta_minimum"]
            ),
            "contact_min_delta": row["contact_min_delta_native"] >= float(
                selection["contact_min_delta_minimum"]
            ),
            "developability_risk": row["full_heavy_developability_risk"] <= float(
                selection["maximum_full_heavy_developability_risk"]
            ),
            "risk_increase": row["risk_increase_over_native"] <= float(
                selection["maximum_risk_increase_over_native"]
            ),
            "no_new_cysteine": (
                not selection["forbid_new_h3_cysteine"] or row["new_h3_cysteines"] == 0
            ),
            "no_new_n_glycosylation": (
                not selection["forbid_new_h3_n_glycosylation_motif"]
                or row["new_h3_n_glycosylation_motifs"] == 0
            ),
        }
        checks = normalize_gate_checks(checks)
        row["gate_checks"] = checks
        row["all_gates_pass"] = all(checks.values())
        enriched.append(row)

    eligible = sorted(
        (row for row in enriched if row["all_gates_pass"]),
        key=lambda row: (
            -row["contact_mean_delta_native"], row["bfn_std_cms"], row["sequence"]
        ),
    )
    selected = []
    minimum_hamming = int(selection["minimum_pairwise_hamming"])
    for row in eligible:
        if all(
            hamming(row["sequence"], prior["sequence"]) >= minimum_hamming
            for prior in selected
        ):
            selected.append(row)
        if len(selected) == int(selection["shortlist_target"]):
            break

    output_dir = ROOT / (args.out or config["output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "tau_candidates.csv").open("w", newline="", encoding="ascii") as handle:
        fields = list(enriched[0])
        fields.remove("gate_checks")
        fields.remove("bfn_pose_values")
        fields.remove("contact_pose_scores")
        fields.remove("full_heavy_sequence")
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(enriched)
    status = (
        "tau_candidate_shortlist_complete"
        if len(selected) >= int(selection["shortlist_minimum"])
        else "tau_candidate_gates_insufficient"
    )
    source_counts = {}
    for row in enriched:
        for source in row["sources"]:
            family = source.split(":", 1)[0]
            source_counts[family] = source_counts.get(family, 0) + 1
    output = {
        "schema_version": 1,
        "status": status,
        "config": args.config,
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__)),
        "target": target,
        "reference": {
            "native_h3": reference["native_h3"],
            "h3_sequence_sha256": sequence_sha256(reference["native_h3"]),
            "native_bfn_mean_cms": float(native_cms.mean()),
            "native_bfn_worst_cms": float(native_cms.min()),
            "native_contact": native_contact,
        },
        "summary": {
            "unique_candidates": len(candidates),
            "all_gates_pass": sum(row["all_gates_pass"] for row in enriched),
            "diverse_shortlist": len(selected),
            "shortlist_minimum": int(selection["shortlist_minimum"]),
        },
        "candidate_source_memberships": source_counts,
        "native_bfn_pose_values": [float(value) for value in native_cms],
        "shortlist": [{"rank": index + 1, **row} for index, row in enumerate(selected)],
        "mpnn_commands": mpnn_commands,
        "claim_boundary": config["claim_boundary"],
    }
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(json.dumps({"status": status, **output["summary"]}, indent=2))
    print(f"Wrote {result_path}")


if __name__ == "__main__":
    main()
