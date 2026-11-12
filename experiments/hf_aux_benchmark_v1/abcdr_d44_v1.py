#!/usr/bin/env python
"""D44 single-mutation binding benchmark.

Data: AbCDR-Binding d44_set (2,048 single-point mutants of anti-lysozyme D44.1;
Zenodo 10.5281/zenodo.18762978). Parent structure: 1MLC (D44.1 Fab + lysozyme).

Scores per mutation:
  dlogp_complex = logP(mut|complex) - logP(wt|complex)
  dlogp_stripped = logP(mut|chain alone) - logP(wt|chain alone)
  dils = dlogp_complex - dlogp_stripped
  blosum62 = BLOSUM62(wt, mut)

Endpoint: Binder (higher = stronger; verified by positive BLOSUM-Binder rho).
Expected direction: positive correlation for dlogp/dils.
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices
from Bio.PDB import Atom, Chain, Model, PDBIO, PDBParser, Residue, Structure
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/hf_aux_benchmark_v1/abcdr_binding/curated/d44_set.csv"
PDB = ROOT / "data/abbibench_structural_audit_v1/structures/1mlc_bae.pdb"
OUT = ROOT / "results/hf_aux_benchmark_v1/abcdr_binding/d44"
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
BLOSUM = substitution_matrices.load("BLOSUM62")


def standard_aa(residue):
    return AA3.get(residue.resname.upper())


def chain_residues(structure, chain_id):
    model = next(structure.get_models())
    try:
        chain = model[chain_id]
    except KeyError:
        return []
    out = []
    for residue in chain:
        if residue.id[0] != " ":
            continue
        aa = standard_aa(residue)
        if aa is None:
            continue
        out.append((residue, aa))
    return out


def write_renumbered(structure, chain_ids, path):
    model = next(structure.get_models())
    new_structure = Structure.Structure("renumbered")
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
            resname = "MET" if residue.resname.upper() == "MSE" else residue.resname
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
        completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
        if completed.returncode or not result_path.exists():
            raise RuntimeError(f"ProteinMPNN failed for {name}: {completed.stderr[-1000:]}")
    data = np.load(result_path)
    log_p = data["log_p"][0]
    rows = np.where(data["design_mask"] > 0)[0]
    return {index + 1: log_p[row] for index, row in enumerate(rows)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA)
    heavy = Counter(frame.Heavy_Sequence).most_common(1)[0][0]
    light = Counter(frame.Light_Sequence).most_common(1)[0][0]
    structure = PDBParser(QUIET=True).get_structure("d44", str(PDB))
    hseq = "".join(aa for _, aa in chain_residues(structure, "B"))
    lseq = "".join(aa for _, aa in chain_residues(structure, "A"))
    if not (hseq.startswith(heavy) and lseq.startswith(light)):
        raise RuntimeError("dataset sequences are not prefixes of 1MLC H/L")
    print(f"WT H {len(heavy)} L {len(light)}; PDB H {len(hseq)} L {len(lseq)}", flush=True)

    complex_name = "d44_complex"
    complex_pdb = WORK / complex_name / f"{complex_name}.pdb"
    lengths = write_renumbered(structure, ["B", "A", "E"], complex_pdb)
    h_complex = run_mpnn("d44_H_complex", complex_pdb, "B",
                         {"B": [], "A": list(range(1, lengths["A"] + 1)),
                          "E": list(range(1, lengths["E"] + 1))})
    l_complex = run_mpnn("d44_L_complex", complex_pdb, "A",
                         {"A": [], "B": list(range(1, lengths["B"] + 1)),
                          "E": list(range(1, lengths["E"] + 1))})
    h_pdb = WORK / "d44_H_stripped" / "d44_H_stripped.pdb"
    write_renumbered(structure, ["B"], h_pdb)
    h_stripped = run_mpnn("d44_H_stripped", h_pdb, "B", {"B": []})
    l_pdb = WORK / "d44_L_stripped" / "d44_L_stripped.pdb"
    write_renumbered(structure, ["A"], l_pdb)
    l_stripped = run_mpnn("d44_L_stripped", l_pdb, "A", {"A": []})

    rows = []
    for _, record in frame.iterrows():
        for chain, sequence, reference, complex_logp, stripped_logp in (
                ("H", record.Heavy_Sequence, heavy, h_complex, h_stripped),
                ("L", record.Light_Sequence, light, l_complex, l_stripped)):
            if sequence == reference:
                continue
            diffs = [(i, reference[i], sequence[i])
                     for i in range(min(len(reference), len(sequence)))
                     if reference[i] != sequence[i]]
            if len(diffs) != 1 or len(reference) != len(sequence):
                continue
            index, wt, mut = diffs[0]
            wi, mi = ALPHABET.index(wt), ALPHABET.index(mut)
            dc = float(complex_logp[index + 1][mi] - complex_logp[index + 1][wi])
            ds = float(stripped_logp[index + 1][mi] - stripped_logp[index + 1][wi])
            rows.append({
                "pid": "1mlc", "chain": chain, "position": index + 1,
                "wt": wt, "mut": mut,
                "binder": float(record.Binder),
                "dlogp_complex": dc, "dlogp_stripped": ds, "dils": dc - ds,
                "blosum62": float(BLOSUM[wt, mut]),
            })
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "scores.csv", index=False)
    summary = {"n": int(len(scores))}
    for score in ("dlogp_complex", "dils", "dlogp_stripped", "blosum62"):
        rho, p = spearmanr(scores[score], scores["binder"])
        summary[score] = {"rho": float(rho), "p": float(p)}
        print(score, "rho", round(float(rho), 3), "p", round(float(p), 5), flush=True)
    summary["by_chain"] = {}
    for chain, group in scores.groupby("chain"):
        summary["by_chain"][chain] = {
            score: float(spearmanr(group[score], group["binder"]).statistic)
            for score in ("dlogp_complex", "dils", "blosum62")
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
