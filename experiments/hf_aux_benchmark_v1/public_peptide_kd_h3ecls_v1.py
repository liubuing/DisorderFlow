#!/usr/bin/env python
"""H3-ECLS on a public wet-lab antibody-peptide Kd cohort.

Cohort: Zenodo 13120765 (Antibody dataset Kd; SAbDab/TDC), peptide antigens
5-50 aa, 118 with SAbDab H/L/antigen chain mapping; 111 parse to H/L/peptide.
H3 numbering: ANARCI/IMGT (anarci_h3.json).

Endpoint: ECLS advantage vs log10(Kd). Negative Spearman is the desired
direction (higher advantage -> lower Kd -> stronger binding). Development
association only; not a binding prediction claim.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import Atom, Chain, Model, PDBIO, PDBParser, Residue, Structure

ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "results/hf_aux_benchmark_v1/public_peptide_kd/cohort_prepared.json"
ANARCI = ROOT / "results/hf_aux_benchmark_v1/public_peptide_kd/anarci_h3.json"
PDB_DIR = ROOT / "data/hf_aux_benchmark_v1/public_affinity/pdb"
OUT = ROOT / "results/hf_aux_benchmark_v1/public_peptide_kd"
WORK = OUT / "h3_work"
MPNN = ROOT / "ProteinMPNN" / "protein_mpnn_run.py"
WEIGHTS = ROOT / "ProteinMPNN" / "vanilla_model_weights"
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}


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
    result_path = out_dir / "conditional_probs_only" / f"{name}.npz"
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
    return -sum(float(logp[i + 1][ALPHABET.index(aa)])
                for i, aa in zip(indices, sequence)) / len(sequence)


def shuffles(sequence, seed, trials):
    if len(sequence) <= 8:
        from itertools import permutations
        unique = sorted({''.join(p) for p in permutations(sequence)} - {sequence})
        if len(unique) <= trials:
            return unique
        return random.Random(seed).sample(unique, trials)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shuffles", type=int, default=200)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    cohort = json.loads(COHORT.read_text(encoding="utf-8"))
    raw_anarci = json.loads(ANARCI.read_text(encoding="utf-8"))
    anarci = {value["sequence"]: value for value in raw_anarci.values()
              if value.get("status") == "ok"}
    if args.limit:
        cohort = cohort[: args.limit]
    results = []
    for row in cohort:
        pid = row["pid"]
        path = PDB_DIR / f"{pid}.pdb"
        if not path.exists():
            continue
        try:
            structure = PDBParser(QUIET=True).get_structure(pid, str(path))
        except Exception as exc:  # noqa: BLE001
            print(f"parse failed {pid}: {exc}", flush=True)
            continue
        heavy, light, pep = row["Hchain"], row["Lchain"], row["antigen_chain"]
        hres = chain_residues(structure, heavy)
        lres = chain_residues(structure, light)
        pres = chain_residues(structure, pep)
        if len(hres) < 50 or len(lres) < 50 or len(pres) < 5:
            continue
        hseq = "".join(aa for _, aa in hres)
        entry = anarci.get(hseq)
        if not entry or entry.get("status") != "ok" or len(entry.get("indices", [])) < 5:
            print(f"no ANARCI H3 {pid}", flush=True)
            continue
        indices = entry["indices"]
        native = entry["h3"]
        complex_name = f"{pid}_{heavy}_{pep}"
        stripped_name = f"{pid}_{heavy}"
        complex_pdb = WORK / complex_name / f"{complex_name}_complex.pdb"
        stripped_pdb = WORK / stripped_name / f"{stripped_name}_stripped.pdb"
        try:
            lengths = write_renumbered(structure, [heavy, light, pep], complex_pdb)
            write_renumbered(structure, [heavy], stripped_pdb)
        except Exception as exc:  # noqa: BLE001
            print(f"write failed {pid}: {exc}", flush=True)
            continue
        try:
            complex_logp = run_mpnn(
                f"{complex_name}_complex", complex_pdb, heavy,
                {heavy: [], light: list(range(1, lengths[light] + 1)),
                 pep: list(range(1, lengths[pep] + 1))})
            stripped_logp = run_mpnn(
                f"{stripped_name}_stripped", stripped_pdb, heavy, {heavy: []})
        except Exception as exc:  # noqa: BLE001
            print(f"mpnn failed {pid}: {exc}", flush=True)
            continue
        native_ecls = nll(complex_logp, native, indices) - nll(stripped_logp, native, indices)
        controls = []
        for candidate in shuffles(native, 20261112 + len(results), args.shuffles):
            controls.append(nll(complex_logp, candidate, indices)
                            - nll(stripped_logp, candidate, indices))
        advantage = float(np.mean(controls) - native_ecls)
        results.append({
            "pid": pid, "n_h3": len(native), "h3": native,
            "kd": row["kd"], "logKd": float(np.log10(row["kd"])),
            "affinity_method": row["affinity_method"],
            "delta_g": row["delta_g"], "advantage": advantage,
            "native_ecls": float(native_ecls),
            "shuffle_mean_ecls": float(np.mean(controls)),
        })
        print(f"{pid} n={len(native)} Kd={row['kd']:.2e} adv={advantage:.4f}", flush=True)

    frame = pd.DataFrame(results)
    frame.to_csv(OUT / "h3ecls_kd.csv", index=False)
    if len(frame) >= 5:
        from scipy.stats import spearmanr
        valid = frame["logKd"].notna()
        rho, p = spearmanr(frame.loc[valid, "advantage"], frame.loc[valid, "logKd"])
        rng = np.random.default_rng(20261112)
        vals = []
        sub = frame.loc[valid]
        for _ in range(10000):
            idx = rng.integers(0, len(sub), len(sub))
            s = sub.iloc[idx]
            if s["logKd"].nunique() > 2:
                vals.append(spearmanr(s["advantage"], s["logKd"]).statistic)
        by_method = {}
        for method, group in frame.loc[valid].groupby("affinity_method"):
            if len(group) >= 4:
                r, pp = spearmanr(group["advantage"], group["logKd"])
                by_method[str(method)] = {"n": int(len(group)), "rho": float(r), "p": float(pp)}
        summary = {
            "n": int(len(frame)),
            "n_with_kd": int(valid.sum()),
            "rho_advantage_vs_logKd": float(rho),
            "p": float(p),
            "bootstrap_ci95": [float(np.percentile(vals, 2.5)),
                               float(np.percentile(vals, 97.5))] if vals else None,
            "by_method": by_method,
            "claim_boundary": "public wet-lab Kd association; development; not a binding predictor",
        }
    else:
        summary = {"n": int(len(frame)), "claim_boundary": "insufficient data"}
    (OUT / "h3ecls_kd_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
