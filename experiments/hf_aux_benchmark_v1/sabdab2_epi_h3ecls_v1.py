#!/usr/bin/env python
"""SAbDab2-Epi H3-side ECLS pilot using a reference-aligned H3.

Reference: 4HIX heavy chain, H3 VRYDHYSGSSDY at 1-based positions 96-107.
Each target heavy chain is aligned to the reference; the reference H3 columns
are transferred to the target. Then the frozen ECLS definition is applied:

  ECLS(s) = NLL_complex(s) - NLL_antibody_stripped(s)
  advantage = mean ECLS(composition-matched shuffled H3) - ECLS(native H3)

This is a development replication on SAbDab2-Epi val/test structures, not the
frozen 31/15 temporal final. It is not a binding or affinity claim.
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
from Bio.Align import PairwiseAligner
from Bio.PDB import Atom, Chain, MMCIFParser, Model, PDBIO, Residue, Structure

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/hf_aux_benchmark_v1/sabdab2_epi"
OUT = ROOT / "results/hf_aux_benchmark_v1/sabdab2_epi"
WORK = OUT / "h3_work"
ANARCI = OUT / "anarci_h3.json"
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
REFERENCE_PDB = ROOT / "data/anti_abeta_refs/4HIX.pdb"
REFERENCE_H3 = "VRYDHYSGSSDY"
REFERENCE_H3_START = 95  # 0-based in the 4HIX H chain


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


def reference_sequence():
    from Bio.PDB import PDBParser
    structure = PDBParser(QUIET=True).get_structure("ref", str(REFERENCE_PDB))
    for chain in next(structure.get_models()):
        sequence = "".join(aa for _, aa in chain_residues(structure, chain.id))
        if REFERENCE_H3 in sequence:
            return sequence
    raise RuntimeError("reference H3 not found")


def transfer_h3(reference, target):
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -2.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(reference, target)[0]
    mapping = {}
    for (r0, r1), (t0, t1) in zip(alignment.aligned[0], alignment.aligned[1]):
        for offset in range(min(r1 - r0, t1 - t0)):
            mapping[r0 + offset] = t0 + offset
    indices = [mapping[i] for i in range(REFERENCE_H3_START, REFERENCE_H3_START + len(REFERENCE_H3))
               if i in mapping]
    return indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shuffles", type=int, default=200)
    parser.add_argument("--all", action="store_true", help="process all eligible complexes")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    reference = reference_sequence()
    print(f"reference length {len(reference)}", flush=True)
    anarci_map = {}
    if ANARCI.exists():
        raw = json.loads(ANARCI.read_text(encoding="utf-8"))
        anarci_map = {value["sequence"]: value for value in raw.values()
                      if value.get("status") == "ok"}
        print(f"ANARCI entries {len(anarci_map)}", flush=True)

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
    seen = set()
    selected = []
    for split, record, peptide in rows:
        cluster = (record.get("clusters") or {}).get("cdrh3") or record["pdb_id"]
        if not args.all and cluster in seen:
            continue
        seen.add(cluster)
        selected.append((split, record, peptide))
        if args.limit and len(selected) >= args.limit:
            break
    print(f"selected {len(selected)} complexes", flush=True)

    results = []
    for split, record, peptide in selected:
        path = BASE / "extracted" / Path(record["structure"]["structure"]).name
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
        heavy_residues = chain_residues(structure, heavy)
        light_residues = chain_residues(structure, light)
        peptide_residues = chain_residues(structure, pep_chain)
        if len(heavy_residues) < 50 or not light_residues or not peptide_residues:
            continue
        if len(heavy_residues) + len(light_residues) + len(peptide_residues) > 600:
            print(f"skip large {record['pdb_id']} "
                  f"({len(heavy_residues)}+{len(light_residues)}+{len(peptide_residues)})",
                  flush=True)
            continue
        heavy_sequence = "".join(aa for _, aa in heavy_residues)
        entry = anarci_map.get(heavy_sequence)
        if entry and len(entry.get("indices", [])) >= 5:
            indices = entry["indices"]
            native = entry["h3"]
            h3_source = "anarci_imgt"
        else:
            indices = transfer_h3(reference, heavy_sequence)
            native = "".join(heavy_sequence[i] for i in indices)
            h3_source = "reference"
        if len(indices) < 5:
            print(f"h3 mapping failed {record['pdb_id']}", flush=True)
            continue
        complex_name = f"{record['pdb_id']}_{heavy}_{pep_chain}"
        stripped_name = f"{record['pdb_id']}_{heavy}"
        complex_pdb = WORK / complex_name / f"{complex_name}_complex.pdb"
        stripped_pdb = WORK / stripped_name / f"{stripped_name}_stripped.pdb"
        try:
            lengths = write_renumbered(structure, [heavy, light, pep_chain], complex_pdb)
            write_renumbered(structure, [heavy], stripped_pdb)
        except Exception as exc:  # noqa: BLE001
            print(f"write failed {record['pdb_id']}: {exc}", flush=True)
            continue
        try:
            complex_logp = run_mpnn(
                f"{complex_name}_complex", complex_pdb, heavy,
                {heavy: [], light: list(range(1, lengths[light] + 1)),
                 pep_chain: list(range(1, lengths[pep_chain] + 1))})
            stripped_logp = run_mpnn(
                f"{stripped_name}_stripped", stripped_pdb, heavy, {heavy: []})
        except Exception as exc:  # noqa: BLE001
            print(f"mpnn failed {record['pdb_id']}: {exc}", flush=True)
            continue
        native_ecls = nll(complex_logp, native, indices) - nll(stripped_logp, native, indices)
        controls = []
        for candidate in shuffles(native, 20261111 + len(results), args.shuffles):
            controls.append(nll(complex_logp, candidate, indices)
                            - nll(stripped_logp, candidate, indices))
        advantage = float(np.mean(controls) - native_ecls)
        results.append({
            "pdb_id": record["pdb_id"], "split": split,
            "peptide_chain": pep_chain,
            "cdrh3_cluster": (record.get("clusters") or {}).get("cdrh3"),
            "n_h3": len(native), "h3": native, "h3_source": h3_source,
            "native_ecls": float(native_ecls),
            "shuffle_mean_ecls": float(np.mean(controls)),
            "advantage": advantage,
            "shuffle_sd": float(np.std(controls)),
        })
        print(f"{record['pdb_id']} {record.get('clusters',{}).get('cdrh3')} "
              f"n={len(native)} advantage={advantage:.4f}", flush=True)

    frame = pd.DataFrame(results)
    prefix = "h3ecls_full" if args.all else "h3ecls_pilot"
    frame.to_csv(OUT / f"{prefix}.csv", index=False)
    summary = {
        "n_complexes": int(len(frame)),
        "n_positive": int((frame["advantage"] > 0).sum()) if len(frame) else 0,
        "mean_advantage": float(frame["advantage"].mean()) if len(frame) else None,
        "median_advantage": float(frame["advantage"].median()) if len(frame) else None,
        "fraction_positive": float((frame["advantage"] > 0).mean()) if len(frame) else None,
        "n_anarci": int((frame["h3_source"] == "anarci_imgt").sum()) if len(frame) else 0,
        "n_reference": int((frame["h3_source"] == "reference").sum()) if len(frame) else 0,
        "by_split": {
            split: {
                "n": int(len(group)),
                "mean_advantage": float(group["advantage"].mean()),
                "fraction_positive": float((group["advantage"] > 0).mean()),
            }
            for split, group in frame.groupby("split")
        } if len(frame) else {},
        "claim_boundary": "ANARCI/IMGT H3 replication on SAbDab2-Epi val/test; not frozen 31/15; not binding",
    }
    (OUT / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
