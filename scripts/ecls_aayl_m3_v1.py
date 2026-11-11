#!/usr/bin/env python3
"""Score AAYL H3 variants with the M3 from-scratch checkpoint.

Builds complex/apo backbone-only PDBs from the parent AAYL structures, injects
each variant H3 sequence, runs ProteinMPNN conditional probabilities with the
M3 checkpoint, and correlates ECLS with the AlphaSeq-estimated Kd endpoint.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import gemmi
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "data" / "aayl_original_retrospective_v1" / "cohort.json"
STRUCT_DIR = ROOT / "data" / "abbibench_structural_audit_v1" / "structures"
MPNN = ROOT / "ProteinMPNN" / "protein_mpnn_run.py"
WEIGHTS = ROOT / "results" / "ecls_cross_arch_v1" / "m3_full10" / "model_weights"
OUT = ROOT / "results" / "ecls_cross_arch_v1" / "aayl_m3"
AA1_TO_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
BACKBONE = ("N", "CA", "C", "O")
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def read_chain(chain):
    sequence = []
    coords = []
    for residue in chain:
        info = gemmi.find_tabulated_residue(residue.name)
        code = info.one_letter_code.upper() if info else ""
        if code not in AA1_TO_3:
            continue
        atoms = {}
        for atom in residue:
            if atom.name in BACKBONE and atom.altloc in ("", "A", "\x00"):
                atoms.setdefault(atom.name, atom.pos)
        if len(atoms) != 4:
            raise ValueError(f"incomplete backbone in {chain.name}:{residue.seqid}")
        sequence.append(code)
        coords.append([atoms[name] for name in BACKBONE])
    return "".join(sequence), np.asarray(
        [[[p.x, p.y, p.z] for p in row] for row in coords], dtype=float)


def write_backbone(path: Path, chains, include_antigen=True):
    lines = []
    serial = 1
    for chain_id, (sequence, coords) in chains.items():
        if chain_id == "A" and not include_antigen:
            continue
        for index, aa in enumerate(sequence, 1):
            for atom_index, atom_name in enumerate(BACKBONE):
                x, y, z = coords[index - 1, atom_index]
                lines.append(
                    f"ATOM  {serial:5d} {atom_name:^4s} {AA1_TO_3[aa]:>3s} "
                    f"{chain_id:1s}{index:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                    f"  1.00  0.00          {atom_name[0]:>2s}\n")
                serial += 1
        lines.append("TER\n")
    lines.append("END\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="ascii")


def run_mpnn(pdb, h3_indices, out_dir, name, heavy_length, seed=2309):
    out_dir.mkdir(parents=True, exist_ok=True)
    fixed = [p for p in range(1, heavy_length + 1) if p - 1 not in set(h3_indices)]
    fixed_path = out_dir / "fixed.jsonl"
    pdb_arg = pdb.as_posix()
    fixed_path.write_text(json.dumps({name: {"B": fixed}}) + "\n")
    command = [
        sys.executable, str(MPNN),
        "--pdb_path", pdb_arg,
        "--pdb_path_chains", "B",
        "--fixed_positions_jsonl", fixed_path.as_posix(),
        "--path_to_model_weights", WEIGHTS.as_posix() + "/",
        "--model_name", "m3",
        "--conditional_probs_only", "1",
        "--conditional_probs_only_backbone", "1",
        "--num_seq_per_target", "1",
        "--batch_size", "1",
        "--seed", str(seed),
        "--suppress_print", "1",
        "--out_folder", out_dir.as_posix(),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
    if completed.returncode:
        raise RuntimeError(completed.stderr[-2000:])
    result_path = out_dir / "conditional_probs_only" / f"{name}.npz"
    result = np.load(result_path)
    return result["log_p"][0]


def h3_nll(logp, h3_indices, sequence):
    return -sum(
        float(logp[index, MPNN_ALPHABET.index(aa)])
        for index, aa in zip(h3_indices, sequence)) / len(sequence)


def rankdata(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    sorted_vals = np.asarray(values, dtype=float)[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom else float("nan")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-variants", type=int, default=None)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    out = args.out if args.out.is_absolute() else ROOT / args.out
    work = out / "work"
    cohort = json.loads(COHORT.read_text(encoding="utf-8"))
    rows = []
    for family, info in cohort["structures"].items():
        pdb_path = STRUCT_DIR / Path(info["file"]).name
        structure = gemmi.read_structure(str(pdb_path))
        model = structure[0]
        chains = {}
        for chain_id in ["A", info["heavy_chain"], "C"]:
            chains[chain_id] = read_chain(model[chain_id])
        heavy_id = info["heavy_chain"]
        h3_indices = info["h3_indices_zero_based_observed_heavy"]
        parent_h3 = info["cdr_h3_sequence"]
        heavy_sequence, heavy_coords = chains[heavy_id]
        if heavy_sequence[h3_indices[0]:h3_indices[-1] + 1] != parent_h3:
            raise ValueError(f"{family} parent H3 mismatch")
        variants = [v for v in cohort["eligible"] if v["family"] == family]
        if args.max_variants:
            variants = variants[:args.max_variants]
        for index, variant in enumerate(variants, 1):
            h3 = variant["h3_sequence"]
            if len(h3) != len(parent_h3):
                continue
            mutant_heavy = list(heavy_sequence)
            for offset, index_h3 in enumerate(h3_indices):
                mutant_heavy[index_h3] = h3[offset]
            mutant_chains = dict(chains)
            mutant_chains[heavy_id] = ("".join(mutant_heavy), heavy_coords)
            name = variant["poi"]
            complex_pdb = work / family / "complex" / f"{name}.pdb"
            apo_pdb = work / family / "apo" / f"{name}.pdb"
            write_backbone(complex_pdb, mutant_chains, include_antigen=True)
            write_backbone(apo_pdb, mutant_chains, include_antigen=False)
            complex_logp = run_mpnn(
                complex_pdb, h3_indices, work / family / "complex" / "mpnn", name,
                len(heavy_sequence))
            apo_logp = run_mpnn(
                apo_pdb, h3_indices, work / family / "apo" / "mpnn", name,
                len(heavy_sequence))
            ecls = h3_nll(complex_logp, h3_indices, h3) - h3_nll(
                apo_logp, h3_indices, h3)
            rows.append({
                "family": family,
                "poi": name,
                "h3_sequence": h3,
                "endpoint": float(variant["endpoint"]),
                "ecls": float(ecls),
            })
            if index % 20 == 0:
                print(f"{family} {index}/{len(variants)}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scores.json").write_text(
        json.dumps({"records": rows}, indent=2) + "\n", encoding="utf-8")
    report = {}
    for family in sorted({row["family"] for row in rows}):
        subset = [row for row in rows if row["family"] == family]
        rho = spearman([row["ecls"] for row in subset],
                       [row["endpoint"] for row in subset])
        report[family] = {"n": len(subset), "spearman_ecls_endpoint": rho}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n",
                                     encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
