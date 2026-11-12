#!/usr/bin/env python
"""SAbDab2-Epi peptide-side likelihood shift pilot (PCLS).

For a peptide-antibody complex:
  ECLS_peptide(s) = NLL_complex(s) - NLL_antibody_stripped(s)
  advantage       = mean ECLS(shuffled peptide) - ECLS(native peptide)

The peptide sequence is shuffled with identical composition. Positive advantage
means the antibody context favors the native peptide over composition-matched
shuffles. This is a peptide-side analog of the frozen H3-side ECLS; it is a
development pilot, not a replacement or extension of the frozen ECLS result.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from Bio.PDB import Atom, Chain, MMCIFParser, Model, PDBIO, Residue, Structure

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/hf_aux_benchmark_v1/sabdab2_epi"
OUT = ROOT / "results/hf_aux_benchmark_v1/sabdab2_epi"
WORK = OUT / "work"
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
        completed = subprocess.run(command, capture_output=True, text=True, timeout=900)
        if completed.returncode or not result_path.exists():
            raise RuntimeError(f"ProteinMPNN failed for {name}: {completed.stderr[-1000:]}")
    data = np.load(result_path)
    log_p = data["log_p"][0]
    rows = np.where(data["design_mask"] > 0)[0]
    return {index + 1: log_p[row] for index, row in enumerate(rows)}


def nll(logp, sequence):
    return -sum(float(logp[index + 1][ALPHABET.index(aa)])
                for index, aa in enumerate(sequence)) / len(sequence)


def shuffles(sequence, seed, trials):
    """Composition-matched unique shuffles; exhaustive for short peptides."""
    from itertools import permutations
    if len(sequence) <= 8:
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
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--shuffles", type=int, default=100)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    rows = []
    for split in ("val", "test"):
        for line in (BASE / f"{split}.jsonl").open(encoding="utf-8"):
            d = json.loads(line)
            h = d.get("heavy_chain") or {}
            l = d.get("light_chain") or {}
            peptides = [ch for ch in d.get("antigen_chains", [])
                        if ch.get("molecule_type") == "PEPTIDE"
                        and 5 <= len(ch.get("sequence", "")) <= 50]
            if peptides and h.get("sequence") and l.get("sequence") and d.get("structure"):
                rows.append((split, d, peptides[0]))
    # one record per peptide cluster, then limit
    seen = set()
    selected = []
    for split, record, peptide in rows:
        cluster = peptide.get("cluster") or record["pdb_id"]
        if cluster in seen:
            continue
        seen.add(cluster)
        selected.append((split, record, peptide))
        if len(selected) >= args.limit:
            break
    print(f"selected {len(selected)} complexes", flush=True)

    results = []
    for split, record, peptide in selected:
        member = Path(record["structure"]["structure"]).name
        path = BASE / "extracted" / member
        if not path.exists():
            continue
        try:
            structure = MMCIFParser(QUIET=True).get_structure(record["pdb_id"], str(path))
        except Exception as exc:  # noqa: BLE001
            print(f"parse failed {record['pdb_id']}: {exc}", flush=True)
            continue
        heavy = record["heavy_chain"]["chain_id"]
        light = record["light_chain"]["chain_id"]
        pep_chain = peptide["chain_id"]
        if not all(chain_residues(structure, chain) for chain in (heavy, light, pep_chain)):
            print(f"missing chains {record['pdb_id']}", flush=True)
            continue
        native = "".join(aa for _, aa in chain_residues(structure, pep_chain))
        if len(native) < 5:
            continue
        name = f"{record['pdb_id']}_{pep_chain}"
        complex_pdb = WORK / name / f"{name}_complex.pdb"
        stripped_pdb = WORK / name / f"{name}_stripped.pdb"
        try:
            lengths = write_renumbered(structure, [heavy, light, pep_chain], complex_pdb)
            write_renumbered(structure, [pep_chain], stripped_pdb)
        except Exception as exc:  # noqa: BLE001
            print(f"write failed {record['pdb_id']}: {exc}", flush=True)
            continue
        try:
            complex_logp = run_mpnn(
                f"{name}_complex", complex_pdb, pep_chain,
                {heavy: list(range(1, lengths[heavy] + 1)),
                 light: list(range(1, lengths[light] + 1)),
                 pep_chain: []})
            stripped_logp = run_mpnn(
                f"{name}_stripped", stripped_pdb, pep_chain, {pep_chain: []})
        except Exception as exc:  # noqa: BLE001
            print(f"mpnn failed {record['pdb_id']}: {exc}", flush=True)
            continue
        native_ecls = nll(complex_logp, native) - nll(stripped_logp, native)
        control_values = []
        for candidate in shuffles(native, 20261111 + len(results), args.shuffles):
            control_values.append(nll(complex_logp, candidate) - nll(stripped_logp, candidate))
        advantage = float(np.mean(control_values) - native_ecls)
        results.append({
            "pdb_id": record["pdb_id"], "split": split,
            "peptide_cluster": peptide.get("cluster"),
            "peptide_chain": pep_chain, "n_residues": len(native),
            "native_ecls": float(native_ecls),
            "shuffle_mean_ecls": float(np.mean(control_values)),
            "advantage": advantage,
            "shuffle_sd": float(np.std(control_values)),
        })
        print(f"{record['pdb_id']} {peptide.get('cluster')} n={len(native)} "
              f"advantage={advantage:.4f}", flush=True)

    import pandas as pd
    frame = pd.DataFrame(results)
    frame.to_csv(OUT / "pcls_pilot.csv", index=False)
    summary = {
        "n_complexes": int(len(frame)),
        "n_positive": int((frame["advantage"] > 0).sum()) if len(frame) else 0,
        "mean_advantage": float(frame["advantage"].mean()) if len(frame) else None,
        "median_advantage": float(frame["advantage"].median()) if len(frame) else None,
        "fraction_positive": float((frame["advantage"] > 0).mean()) if len(frame) else None,
        "claim_boundary": "peptide-side development pilot; not the frozen H3 ECLS; not binding",
    }
    (OUT / "pcls_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
