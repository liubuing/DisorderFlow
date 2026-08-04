#!/usr/bin/env python3
"""Rerank low-mutation anti-A-beta variants with corrected disorder conditioning."""

import argparse
import copy
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "modules"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from disorderflow.datasets.protein import preprocess_protein_structure  # noqa: E402
from disorderflow.utils.data import PaddingCollate  # noqa: E402
from disorderflow.utils.misc import seed_all  # noqa: E402
from disorderflow.utils.train import recursive_to  # noqa: E402
from disorderflow.utils.transforms import get_transform  # noqa: E402
from evaluate_v5_1_phase3 import load_checkpoint  # noqa: E402
from generate_v5_1_abeta_candidates import ABETA42, rmsf_profile  # noqa: E402
from state_contact_scorer import extract_contact_map  # noqa: E402
from disorderflow.utils.protein.constants import ChothiaCDRRange  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"
REFERENCES = {
    "4HIX": {"path": "data/anti_abeta_refs/4HIX.pdb", "peptide_chain": "A"},
    "5CSZ": {"path": "data/anti_abeta_refs/5CSZ.pdb", "peptide_chain": "E"},
}


def load_fasta(path):
    records = {}
    header = None
    sequence = []
    with open(path, encoding="ascii") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    records[header] = "".join(sequence)
                header = line[1:]
                sequence = []
            elif line:
                sequence.append(line)
    if header is not None:
        records[header] = "".join(sequence)
    return records


def construct_chains(records, construct_id):
    output = {}
    for header, sequence in records.items():
        if not header.startswith(f"{construct_id}|"):
            continue
        fields = header.split("|")
        output[fields[1]] = sequence
    return output


def mutation_map(row, contact_map):
    mutations = {}
    for token in row["mutations"].split(";"):
        match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", token.strip())
        if not match:
            raise ValueError(f"Invalid paratope mutation: {token}")
        native, index, replacement = match.groups()
        residue = contact_map["paratope_residues"][int(index) - 1]
        if residue["aa"] != native:
            raise ValueError(f"Mutation native mismatch for {row['candidate_id']}: {token}")
        mutations[(residue["chain"], int(residue["resid"]))] = replacement
    return mutations


def build_batch(reference, mutations, profile, device):
    design_chains = sorted({chain for chain, _ in mutations})
    peptide_chain = reference["peptide_chain"]
    structure = preprocess_protein_structure(
        reference["path"], chain_ids=design_chains + [peptide_chain])
    regions = {}
    for chain, residue in mutations:
        regions.setdefault(chain, []).append(residue)
    transform = get_transform([
        {"type": "mask_region", "regions": regions},
        {"type": "merge_protein"},
        {"type": "patch_protein"},
    ])
    batch = recursive_to(PaddingCollate()([transform(structure)]), device)
    mask_gen = batch["generate_flag"].bool()
    ordered_mutations = [
        mutations[key] for key in sorted(mutations, key=lambda key: (design_chains.index(key[0]), key[1]))
    ]
    if int(mask_gen.sum()) != len(ordered_mutations):
        raise ValueError(f"Mutation mask mismatch: {int(mask_gen.sum())} != {len(ordered_mutations)}")
    batch["aa"][mask_gen] = torch.tensor(
        [AA.index(value) for value in ordered_mutations], device=device)
    chain_nb = batch.get("chain_nb")
    if chain_nb is None:
        raise ValueError("Transformed complex has no chain_nb for peptide identification")
    real_chain_nb = chain_nb[batch["mask"].bool()]
    chain_values, chain_counts = torch.unique(real_chain_nb, return_counts=True)
    matches = chain_values[chain_counts == len(profile)]
    if len(matches) != 1:
        counts = {int(value): int(count) for value, count in zip(chain_values, chain_counts)}
        raise ValueError(f"Could not uniquely identify {len(profile)}-residue peptide chain: {counts}")
    peptide_chain_nb = matches[0]
    mask_antigen = batch["mask"].bool() & (chain_nb == peptide_chain_nb)
    antigen_count = int(mask_antigen.sum())
    if antigen_count != len(profile):
        raise ValueError(f"Antigen profile mismatch: {antigen_count} != {len(profile)}")
    condition = torch.zeros_like(batch["aa"], dtype=torch.float32)
    condition[mask_antigen] = torch.as_tensor(profile, dtype=torch.float32, device=device)
    batch["mask_antigen"] = mask_antigen
    batch["epitope_disorder_profile"] = condition
    batch["fixed_t"] = 0.5
    return batch


def evaluate(model, source_batch, device, factual, seed):
    batch = recursive_to(copy.deepcopy(source_batch), device)
    if not factual:
        batch["epitope_disorder_profile"].zero_()
    captured = {}

    def hook(_module, _inputs, output):
        captured["output"] = output

    handle = model.bfn.receiver.register_forward_hook(hook)
    try:
        seed_all(seed)
        with torch.inference_mode():
            model(batch)
    finally:
        handle.remove()
    output = captured["output"]
    generated = batch["generate_flag"].bool() & batch["mask"].bool()
    logits = output[0][..., :20][generated].float()
    targets = batch["aa"][generated].long()
    return {
        "nll": F.cross_entropy(logits, targets).item(),
        "recovery": (logits.argmax(dim=-1) == targets).float().mean().item(),
        "contact": torch.sigmoid(output[8][generated]).mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--library", default="outputs/abeta_contact_guided_selected_nglyco_rescued_v2/selected_contact_guided_library.csv")
    parser.add_argument("--construct-fasta", default="outputs/abeta_full_chain_constructs_nglyco_rescued_v2/full_chain_constructs.fasta")
    parser.add_argument("--output-dir", default="results/v5_1_candidates/abeta42_local")
    parser.add_argument("--max-mutations", type=int, default=4)
    parser.add_argument("--cdr", choices=["H3"], help="Reject candidates with mutations outside this Chothia CDR")
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=4100)
    args = parser.parse_args()

    with open(args.library, newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["n_mutations"]) <= args.max_mutations]
    fasta = load_fasta(args.construct_fasta)
    seed_paths = [PROJECT_ROOT / f"data/abeta_conformations/pdbs/abeta42_seed{index}_42.pdb" for index in range(5)]
    _, full_profile = rmsf_profile(seed_paths)
    model, _ = load_checkpoint(args.checkpoint, args.device)

    contact_maps = {
        name: extract_contact_map(info["path"], peptide_chain=info["peptide_chain"])
        for name, info in REFERENCES.items()
    }
    records = []
    failures = []
    for index, row in enumerate(rows):
        reference_name = row["reference_pdb"]
        reference = REFERENCES[reference_name]
        contact_map = contact_maps[reference_name]
        peptide_sequence = contact_map["peptide_sequence"]
        start = ABETA42.find(peptide_sequence)
        if start < 0:
            failures.append({"candidate_id": row["candidate_id"], "error": "peptide_not_in_abeta42"})
            continue
        try:
            mutations = mutation_map(row, contact_map)
            if args.cdr == "H3":
                h3_start, h3_end = ChothiaCDRRange.H3
                outside = [
                    f"{chain}:{residue}" for chain, residue in mutations
                    if chain not in ("H", "A") or not (h3_start <= residue <= h3_end)
                ]
                if outside:
                    raise ValueError(f"mutations_outside_H3: {','.join(outside)}")
            batch = build_batch(reference, mutations, full_profile[start:start + len(peptide_sequence)], args.device)
            factual = evaluate(model, batch, args.device, True, args.seed + index)
            zero = evaluate(model, batch, args.device, False, args.seed + index)
        except Exception as error:  # noqa: BLE001
            failures.append({"candidate_id": row["candidate_id"], "error": str(error)})
            continue
        construct_id = row["candidate_id"]
        if reference_name == "5CSZ" and f"{construct_id}_N52H|heavy" in fasta:
            construct_id = f"{construct_id}_N52H"
        chains = construct_chains(fasta, f"{reference_name}_{construct_id}")
        if not chains:
            failures.append({"candidate_id": row["candidate_id"], "error": "construct_sequence_missing"})
            continue
        condition_advantage = zero["nll"] - factual["nll"]
        combined = (
            0.40 * float(row["candidate_score"])
            + 0.25 * min(float(row["contact_retention"]), 1.2) / 1.2
            + 0.20 * (1.0 / (1.0 + math.exp(-5.0 * condition_advantage)))
            + 0.15 * factual["contact"]
        )
        records.append({
            **row,
            "construct_id": f"{reference_name}_{construct_id}",
            "heavy_sequence": chains.get("heavy", ""),
            "light_sequence": chains.get("light", ""),
            "factual_nll": factual["nll"],
            "zero_nll": zero["nll"],
            "condition_advantage": condition_advantage,
            "factual_contact": factual["contact"],
            "corrected_rank_score": combined,
        })
    records.sort(key=lambda row: row["corrected_rank_score"], reverse=True)
    selected = records[:args.top]
    for rank, row in enumerate(selected, 1):
        row["corrected_rank"] = rank

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": args.checkpoint,
        "weights_kind": "ema",
        "source_library": args.library,
        "selection_contract": (
            f"contact/developability-gated local variants, <={args.max_mutations} mutations, "
            f"CDR={args.cdr or 'unrestricted'}, corrected factual-vs-zero reranking"
        ),
        "n_input": len(rows),
        "n_scored": len(records),
        "n_failed": len(failures),
        "n_selected": len(selected),
        "mean_condition_advantage": float(np.mean([row["condition_advantage"] for row in records])) if records else None,
        "selected": selected,
        "all_scored": records,
        "failures": failures,
    }
    (output_dir / "local_rerank_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output_dir / "af2_local_candidates.fasta").open("w", encoding="ascii") as handle:
        for row in selected:
            handle.write(f">{row['construct_id']}|rank={row['corrected_rank']}\n")
            handle.write(f"{row['heavy_sequence']}:{row['light_sequence']}:{ABETA42}\n")
    fields = [
        "corrected_rank", "construct_id", "reference_pdb", "candidate_id", "n_mutations",
        "mutations", "candidate_score", "contact_retention", "condition_advantage",
        "factual_nll", "zero_nll", "factual_contact", "corrected_rank_score",
        "heavy_sequence", "light_sequence",
    ]
    with (output_dir / "af2_local_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    print(json.dumps({key: value for key, value in report.items() if key not in ("selected", "all_scored", "failures")}, indent=2))


if __name__ == "__main__":
    main()
