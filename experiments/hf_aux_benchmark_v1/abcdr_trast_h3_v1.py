#!/usr/bin/env python
"""Trastuzumab H3-variant library benchmark on the 1N8Z scaffold.

Data: AbCDR-Binding trast_set (422 CDR-H3 variants; 152 with 11-aa H3, the
length of the 1N8Z trastuzumab H3). Binding endpoint: Binder (higher = stronger;
verified by positive BLOSUM-Binder correlation).

For each 11-aa variant, graft its H3 into the 1N8Z H chain (backbone fixed),
then:
  ECLS(s) = NLL_complex(s) - NLL_antibody_stripped(s)
  advantage = mean ECLS(composition-matched shuffled H3) - ECLS(variant H3)

Correlate advantage with Binder. Development benchmark; not a binding predictor.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices
from Bio.PDB import Atom, Chain, Model, PDBIO, PDBParser, Residue, Structure
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/hf_aux_benchmark_v1/abcdr_binding/curated/trast_set.csv"
PDB = ROOT / "data/hf_aux_benchmark_v1/abcdr_binding/1n8z.pdb"
OUT = ROOT / "results/hf_aux_benchmark_v1/abcdr_binding/trast_h3"
WORK = OUT / "work"
MPNN = ROOT / "ProteinMPNN" / "protein_mpnn_run.py"
WEIGHTS = ROOT / "ProteinMPNN" / "vanilla_model_weights"
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}
AA1_TO_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
BLOSUM = substitution_matrices.load("BLOSUM62")
WT_H3 = "WGGDGFYAMDY"
H3_START = 98  # 0-based in 1N8Z chain B
H3_LEN = 11


def standard_aa(residue):
    return AA3.get(residue.resname.upper())


def chain_residues(structure, chain_id):
    model = next(structure.get_models())
    try:
        chain = model[chain_id]
    except KeyError:
        return []
    return [(residue, aa) for residue in chain
            if residue.id[0] == " " and (aa := standard_aa(residue)) is not None]


def write_structure(structure, variant_h3, path, chain_ids):
    """Write selected chains with the variant H3 in chain B."""
    model = next(structure.get_models())
    new_structure = Structure.Structure("x")
    new_model = Model.Model(0)
    new_structure.add(new_model)
    lengths = {}
    for chain_id in chain_ids:
        source = model[chain_id]
        target = Chain.Chain(chain_id)
        new_model.add(target)
        count = 0
        for residue in source:
            if residue.id[0] != " ":
                continue
            aa = standard_aa(residue)
            if aa is None:
                continue
            count += 1
            resname = residue.resname
            if chain_id == "B" and H3_START <= count - 1 < H3_START + H3_LEN:
                resname = AA1_TO_3[variant_h3[count - 1 - H3_START]]
            new_residue = Residue.Residue((" ", count, " "), resname, residue.segid)
            for atom in residue:
                if atom.element == "H":
                    continue
                new_residue.add(Atom.Atom(
                    atom.name, atom.coord, atom.bfactor, atom.occupancy,
                    atom.altloc, atom.fullname, atom.serial_number, atom.element,
                ))
            target.add(new_residue)
        lengths[chain_id] = count
    path.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(new_structure)
    io.save(str(path))
    return lengths


def run_mpnn(name, pdb_path, scored_chain, fixed):
    out_dir = WORK / name
    result_path = out_dir / "conditional_probs_only" / f"{pdb_path.stem}.npz"
    if not result_path.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        fixed_path = out_dir / "fixed_positions.jsonl"
        fixed_path.write_text(json.dumps({pdb_path.stem: fixed}) + "\n", encoding="ascii")
        command = [
            sys.executable, str(MPNN),
            "--pdb_path", pdb_path.as_posix(),
            "--pdb_path_chains", scored_chain,
            "--fixed_positions_jsonl", fixed_path.as_posix(),
            "--path_to_model_weights", WEIGHTS.as_posix(),
            "--model_name", "v_48_020",
            "--conditional_probs_only", "1",
            "--conditional_probs_only_backbone", "0",
            "--num_seq_per_target", "1",
            "--batch_size", "1",
            "--seed", "2309",
            "--suppress_print", "1",
            "--out_folder", out_dir.as_posix(),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=300)
        if completed.returncode or not result_path.exists():
            raise RuntimeError(f"ProteinMPNN failed for {name}: {completed.stderr[-1000:]}")
    data = np.load(result_path)
    log_p = data["log_p"][0]
    rows = np.where(data["design_mask"] > 0)[0]
    return {index + 1: log_p[row] for index, row in enumerate(rows)}


def nll(logp, sequence, indices):
    return -sum(float(logp[i][ALPHABET.index(aa)]) for i, aa in zip(indices, sequence)) / len(sequence)


def shuffles(sequence, seed, trials):
    rng = random.Random(seed)
    seen = {sequence}
    out = []
    attempts = 0
    while len(out) < trials and attempts < trials * 50:
        attempts += 1
        candidate = list(sequence)
        rng.shuffle(candidate)
        candidate = "".join(candidate)
        if candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA)
    frame = frame[frame.CDRH3.str.len() == H3_LEN].reset_index(drop=True)
    print(f"variants {len(frame)}", flush=True)
    structure = PDBParser(QUIET=True).get_structure("1n8z", str(PDB))
    hseq = "".join(aa for _, aa in chain_residues(structure, "B"))
    if hseq[H3_START:H3_START + H3_LEN] != WT_H3:
        raise RuntimeError(f"unexpected 1N8Z H3: {hseq[H3_START:H3_START+H3_LEN]}")
    h3_indices = list(range(H3_START + 1, H3_START + 1 + H3_LEN))
    wt_blosum = [BLOSUM[WT_H3[i], WT_H3[i]] for i in range(H3_LEN)]

    rows = []
    for index, record in frame.iterrows():
        variant = record.CDRH3
        name = f"trast_{index:04d}"
        complex_pdb = WORK / name / f"{name}_complex.pdb"
        stripped_pdb = WORK / name / f"{name}_stripped.pdb"
        lengths = write_structure(structure, variant, complex_pdb, ("B", "A", "C"))
        write_structure(structure, variant, stripped_pdb, ("B", "A"))
        try:
            complex_logp = run_mpnn(
                f"{name}_complex", complex_pdb, "B",
                {"B": [], "A": list(range(1, lengths["A"] + 1)),
                 "C": list(range(1, lengths["C"] + 1))})
            stripped_logp = run_mpnn(
                f"{name}_stripped", stripped_pdb, "B",
                {"B": [], "A": list(range(1, lengths["A"] + 1))})
        except Exception as exc:  # noqa: BLE001
            print(f"mpnn failed {index}: {exc}", flush=True)
            continue
        native_ecls = nll(complex_logp, variant, h3_indices) - nll(stripped_logp, variant, h3_indices)
        controls = []
        for candidate in shuffles(variant, 20261112 + index, 200):
            controls.append(nll(complex_logp, candidate, h3_indices)
                            - nll(stripped_logp, candidate, h3_indices))
        advantage = float(np.mean(controls) - native_ecls)
        blosum = sum(BLOSUM[WT_H3[i], variant[i]] for i in range(H3_LEN))
        rows.append({
            "h3": variant, "binder": float(record.Binder),
            "advantage": advantage, "native_ecls": float(native_ecls),
            "shuffle_mean_ecls": float(np.mean(controls)),
            "blosum_vs_wt": float(blosum),
        })
        print(f"{index} {variant} Binder={record.Binder:.3f} adv={advantage:.4f}", flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "scores.csv", index=False)
    summary = {"n": int(len(scores))}
    for score in ("advantage", "blosum_vs_wt"):
        rho, p = spearmanr(scores[score], scores["binder"])
        summary[score] = {"rho": float(rho), "p": float(p)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
