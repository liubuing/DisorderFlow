#!/usr/bin/env python
"""Build the candidate-interface confidence successor dataset from frozen AF2 bundles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import lmdb
import numpy as np
import torch
from Bio.PDB import PDBParser

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

TRAIN_ROOTS = (
    "outputs/alpha_synuclein_redesign_validation_v2/colabfold_multiseed_gpu",
    "results/binder_validation/3stb_panel_v1/multimer_results",
)
VAL_ROOTS = (
    "outputs/non_abeta_idp_complex_fold_panel_v1/colabfold_complex_gpu",
    "outputs/non_abeta_idp_initial_guess_panel_v1/colabfold_initial_guess_gpu",
    "outputs/non_abeta_idp_initial_guess_panel_v1/colabfold_initial_guess_multiseed_gpu",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def chain_sequences(path):
    model = PDBParser(QUIET=True).get_structure(Path(path).stem, str(path))[0]
    return {
        chain.id: "".join(
            AA3[residue.resname]
            for residue in chain
            if residue.resname in AA3 and "CA" in residue
        )
        for chain in model
    }


def read_construct_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_constructs():
    rows = read_construct_rows(
        ROOT / "outputs/alpha_synuclein_redesign_validation_v2/redesign_constructs.csv")
    rows += read_construct_rows(
        ROOT / "outputs/non_abeta_idp_initial_guess_panel_v1/initial_guess_panel.csv")
    constructs = {}
    for row in rows:
        construct_id = row["construct_id"]
        constructs.setdefault(construct_id, {
            "family": row["antibody_family"],
            "target": row["target"],
            "construct_type": row["construct_type"],
            "chains": {
                "A": row["vh_sequence"],
                "B": row["vl_sequence"],
                "C": row["antigen_sequence"],
            },
        })

    manifest = json.loads((
        ROOT / "results/binder_validation/3stb_panel_v1/manifest.json"
    ).read_text(encoding="utf-8"))
    for row in manifest["records"]:
        constructs[row["name"]] = {
            "family": "3STB",
            "target": "abeta42",
            "construct_type": (
                "native_control" if row["name"] == "3STB_native" else "candidate"),
            "chains": {"A": row["vhh_sequence"], "B": manifest["target_sequence"]},
        }
    return constructs, manifest


def variable_positions(constructs, family, chain_id):
    family_rows = [
        row for row in constructs.values()
        if row["family"] == family and chain_id in row["chains"]
    ]
    native = next(
        row for row in family_rows if row["construct_type"] == "native_control")
    native_sequence = native["chains"][chain_id]
    positions = {
        index + 1
        for row in family_rows
        for index, (reference, value) in enumerate(
            zip(native_sequence, row["chains"][chain_id], strict=True))
        if reference != value
    }
    return sorted(positions)


def family_contracts(constructs, manifest):
    contracts = {}
    for family in ("8B9V", "5MP3"):
        positions = {
            chain_id: variable_positions(constructs, family, chain_id)
            for chain_id in ("A", "B")
        }
        positions = {chain: values for chain, values in positions.items() if values}
        if not positions:
            raise ValueError(f"No variable positions found for {family}")
        contracts[family] = {
            "candidate_positions": positions,
            "antigen_chain": "C",
        }
    contracts["3STB"] = {
        "candidate_positions": {
            "A": [
                index + 1
                for start, end in manifest["cdr_slices_zero_based_half_open"]
                for index in range(start, end)
            ],
        },
        "antigen_chain": "B",
    }
    return contracts


def region_spec(positions):
    return ";".join(
        f"{chain}:{','.join(str(value) for value in values)}"
        for chain, values in positions.items()
    )


def paired_pdb(score_path):
    name = score_path.name.replace("_scores_", "_unrelaxed_")
    return score_path.with_name(Path(name).with_suffix(".pdb").name)


def construct_for_path(path, constructs):
    candidates = [
        construct_id for construct_id in constructs
        if path.name.startswith(construct_id + "_")
    ]
    if not candidates:
        raise ValueError(f"Cannot map AF2 bundle to construct: {path}")
    return max(candidates, key=len)


def pose_and_seed(path):
    pose = next(
        (part for part in path.parts if re.fullmatch(r"conformer\d+", part)),
        "single_pose",
    )
    match = re.search(r"_seed_(\d+)", path.name)
    if not match:
        raise ValueError(f"AF2 seed missing from filename: {path.name}")
    return pose, int(match.group(1))


def discover_bundles(roots, constructs):
    rows = []
    seen = set()
    seen_conditions = set()
    for root_name in roots:
        root = ROOT / root_name
        for score_path in sorted(root.rglob("*_scores_rank_*.json")):
            pdb_path = paired_pdb(score_path)
            if not pdb_path.is_file():
                continue
            construct_id = construct_for_path(score_path, constructs)
            family = constructs[construct_id]["family"]
            if roots == VAL_ROOTS and family != "5MP3":
                continue
            if roots == TRAIN_ROOTS and family not in {"8B9V", "3STB"}:
                continue
            identity = (sha256(score_path), sha256(pdb_path))
            if identity in seen:
                continue
            seen.add(identity)
            pose, seed = pose_and_seed(score_path)
            protocol_id = Path(root_name).name
            condition = (family, protocol_id, pose, seed, construct_id)
            if condition in seen_conditions:
                continue
            seen_conditions.add(condition)
            rows.append({
                "score_path": score_path,
                "pdb_path": pdb_path,
                "construct_id": construct_id,
                "family": family,
                "pose_id": pose,
                "af2_seed": seed,
                "protocol_id": protocol_id,
                "source_identity": identity,
            })
    return rows


def unbatch_and_trim(batch):
    padded_length = batch["mask"].shape[1]
    valid_length = int(batch["mask"][0].sum())
    output = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            value = value[0]
            if value.ndim and value.shape[0] == padded_length:
                value = value[:valid_length]
            output[key] = value.cpu()
        elif isinstance(value, list) and len(value) == padded_length:
            output[key] = [
                item[0] if isinstance(item, (tuple, list)) and len(item) == 1 else item
                for item in value[:valid_length]
            ]
        else:
            output[key] = value
    return output, valid_length


def build_entry(bundle, constructs, contracts, scaffold_id):
    from modules.bfn_loader import build_region_batch

    metadata = constructs[bundle["construct_id"]]
    contract = contracts[bundle["family"]]
    observed = chain_sequences(bundle["pdb_path"])
    if observed != metadata["chains"]:
        raise ValueError(f"PDB sequence mismatch: {bundle['pdb_path']}")

    design_chains = list(contract["candidate_positions"])
    antigen_chain = contract["antigen_chain"]
    context_chains = [
        chain for chain in metadata["chains"] if chain not in design_chains
    ]
    merged_chain_order = design_chains + context_chains
    batch = build_region_batch(
        str(bundle["pdb_path"]),
        region_spec(contract["candidate_positions"]),
        context_chains=context_chains,
        antigen_chains=[antigen_chain],
        device="cpu",
    )
    if set(metadata["chains"]) != set(observed):
        raise ValueError(f"Unexpected chains in {bundle['pdb_path']}: {sorted(observed)}")

    score = json.loads(bundle["score_path"].read_text(encoding="utf-8"))
    full_length = sum(len(observed[chain]) for chain in merged_chain_order)
    plddt = np.asarray(score["plddt"], dtype=np.float32)
    pae = np.asarray(score["pae"], dtype=np.float32)
    if plddt.shape != (full_length,) or pae.shape != (full_length, full_length):
        raise ValueError(f"AF2 array shape mismatch: {bundle['score_path']}")
    if not np.isfinite(plddt).all() or not np.isfinite(pae).all() or np.ptp(pae) <= 0:
        raise ValueError(f"Invalid AF2 confidence arrays: {bundle['score_path']}")

    patch_idx = batch["patch_idx"][0][batch["mask"][0].bool()].long().numpy()
    model_batch, valid_length = unbatch_and_trim(batch)
    if len(patch_idx) != valid_length:
        raise ValueError("Patch index length mismatch")
    if int(model_batch["generate_flag"].sum()) != sum(
            len(values) for values in contract["candidate_positions"].values()):
        raise ValueError("Candidate mask was truncated")

    return {
        "schema_version": "candidate_interface_confidence_v1",
        "pdb_id": bundle["construct_id"],
        "sequence": "".join(observed[chain] for chain in merged_chain_order),
        "scaffold_id": scaffold_id,
        "scaffold_family": bundle["family"],
        "target": metadata["target"],
        "construct_id": bundle["construct_id"],
        "construct_type": metadata["construct_type"],
        "pose_id": bundle["pose_id"],
        "af2_seed": bundle["af2_seed"],
        "protocol_id": bundle["protocol_id"],
        "candidate_positions": contract["candidate_positions"],
        "batch": model_batch,
        "af2_plddt": torch.from_numpy(plddt[patch_idx] / 100.0),
        "af2_iptm": torch.tensor(float(score["iptm"]), dtype=torch.float32),
        "af2_pae_matrix": torch.from_numpy(pae[np.ix_(patch_idx, patch_idx)]),
        "af2_pae_normalized": False,
        "source": "frozen_colabfold_bundle",
        "source_score_path": bundle["score_path"].relative_to(ROOT).as_posix(),
        "source_score_sha256": bundle["source_identity"][0],
        "source_pdb_path": bundle["pdb_path"].relative_to(ROOT).as_posix(),
        "source_pdb_sha256": bundle["source_identity"][1],
    }


def write_lmdb(path, entries):
    env = lmdb.open(str(path), map_size=1 << 32)
    with env.begin(write=True) as transaction:
        for index, entry in enumerate(entries):
            transaction.put(f"{index:08d}".encode(), pickle.dumps(entry))
        transaction.put(b"__len__", pickle.dumps(len(entries)))
    env.sync()
    env.close()


def summarize(entries):
    families = defaultdict(int)
    groups = defaultdict(set)
    for entry in entries:
        families[entry["scaffold_family"]] += 1
        groups[entry["scaffold_id"]].add(entry["construct_id"])
    return {
        "records": len(entries),
        "families": dict(sorted(families.items())),
        "groups": len(groups),
        "minimum_constructs_per_group": min(map(len, groups.values())),
        "maximum_constructs_per_group": max(map(len, groups.values())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="data/confidence_candidate_interface_v1")
    parser.add_argument("--limit-per-split", type=int)
    args = parser.parse_args()
    output = ROOT / args.out
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)

    constructs, source_manifest = load_constructs()
    contracts = family_contracts(constructs, source_manifest)
    split_bundles = {
        "train": discover_bundles(TRAIN_ROOTS, constructs),
        "val_scaffold_disjoint": discover_bundles(VAL_ROOTS, constructs),
    }
    if args.limit_per_split:
        split_bundles = {
            name: rows[:args.limit_per_split]
            for name, rows in split_bundles.items()
        }

    group_keys = sorted({
        (row["family"], row["protocol_id"], row["pose_id"], row["af2_seed"])
        for rows in split_bundles.values() for row in rows
    })
    group_ids = {key: index for index, key in enumerate(group_keys)}
    split_entries = {}
    for split, bundles in split_bundles.items():
        entries = []
        for bundle in bundles:
            group_key = (
                bundle["family"], bundle["protocol_id"],
                bundle["pose_id"], bundle["af2_seed"])
            entries.append(build_entry(
                bundle, constructs, contracts, group_ids[group_key]))
        if not entries:
            raise RuntimeError(f"No records found for {split}")
        split_summary = summarize(entries)
        if args.limit_per_split is None and split_summary[
                "minimum_constructs_per_group"] < 2:
            raise ValueError(
                f"{split} contains a group with fewer than two constructs")
        split_entries[split] = entries
        write_lmdb(output / f"{split}.lmdb", entries)

    manifest = {
        "schema_version": "candidate_interface_confidence_dataset_v1",
        "classification": "development_only",
        "split_contract": {
            "train_families": ["8B9V", "3STB"],
            "validation_families": ["5MP3"],
            "group_unit": "scaffold_family+protocol+antigen_pose+af2_seed",
        },
        "candidate_positions": contracts,
        "summary": {
            split: summarize(entries) for split, entries in split_entries.items()
        },
        "records": {
            split: [{
                key: entry[key] for key in (
                    "construct_id", "scaffold_family", "target", "pose_id",
                    "af2_seed", "protocol_id", "scaffold_id", "source_score_path",
                    "source_score_sha256", "source_pdb_path", "source_pdb_sha256",
                )
            } for entry in entries]
            for split, entries in split_entries.items()
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    print(json.dumps(manifest["summary"], indent=2))
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
