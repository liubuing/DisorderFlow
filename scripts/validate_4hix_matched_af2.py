#!/usr/bin/env python
"""AF2 multi-seed validation for pre-AF2-selected matched 4HIX candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from Bio.PDB import PDBParser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PDB_PATH = ROOT / "data/anti_abeta_refs/4HIX.pdb"
BFN_RESULTS = ROOT / "results/ablation/4hix_bfn_matched_v1.json"
MPNN_RESULTS = ROOT / "results/ablation/4hix_mpnn_matched_v1.json"
NATIVE_H3 = "VRYDHYSGSSDY"
AF2_SEEDS = [5101, 5111, 5121]
AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def chain_sequences():
    model = PDBParser(QUIET=True).get_structure("4hix", str(PDB_PATH))[0]
    return {
        chain.id: "".join(
            AA3_TO_1[residue.resname] for residue in chain if residue.resname in AA3_TO_1
        )
        for chain in model
        if chain.id in {"A", "H", "L"}
    }


def selected_candidates(top_n):
    bfn = json.loads(BFN_RESULTS.read_text())
    mpnn = json.loads(MPNN_RESULTS.read_text())
    selected = []
    for arm in bfn["arms"]:
        rows = sorted(
            arm["results"], key=lambda row: (row["ppl"], row["seed"], row["sample_index"])
        )
        for rank, row in enumerate(rows[:top_n], 1):
            selected.append(
                {
                    "id": f"{arm['name']}_r{rank}",
                    "arm": arm["name"],
                    "h3": row["sequence"],
                    "pre_af2_rank": rank,
                    "selection_metric": "ppl",
                    "selection_value": row["ppl"],
                }
            )
    rows = sorted(mpnn["results"], key=lambda row: (row["score"], row["seed"], row["sample_index"]))
    for rank, row in enumerate(rows[:top_n], 1):
        selected.append(
            {
                "id": f"proteinmpnn_r{rank}",
                "arm": "proteinmpnn",
                "h3": row["sequence"],
                "pre_af2_rank": rank,
                "selection_metric": "proteinmpnn_score",
                "selection_value": row["score"],
            }
        )
    selected.append(
        {
            "id": "native",
            "arm": "native",
            "h3": NATIVE_H3,
            "pre_af2_rank": 1,
            "selection_metric": "control",
            "selection_value": None,
        }
    )
    shuffled = list(NATIVE_H3)
    random.Random(5303).shuffle(shuffled)
    selected.append(
        {
            "id": "composition_shuffle",
            "arm": "shuffle",
            "h3": "".join(shuffled),
            "pre_af2_rank": 1,
            "selection_metric": "control",
            "selection_value": None,
        }
    )
    return selected


def full_heavy(native_heavy, h3):
    if len(h3) != len(NATIVE_H3):
        raise ValueError(f"Invalid H3 length: {h3}")
    return native_heavy[:95] + h3 + native_heavy[107:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=AF2_SEEDS)
    parser.add_argument("--recycles", type=int, default=3)
    parser.add_argument("--out-dir", default=str(ROOT / "results/ablation/4hix_matched_af2_v1"))
    args = parser.parse_args()

    from modules.af2_jax_runner import run_multimer_prediction

    sequences = chain_sequences()
    if sequences["H"][95:107] != NATIVE_H3 or sequences["A"] != "DAEFRH":
        raise ValueError("4HIX sequence or H3 mapping changed")
    candidates = selected_candidates(args.top_n)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    pdb_dir = out_dir / "pdb"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for candidate in candidates:
        heavy = full_heavy(sequences["H"], candidate["h3"])
        for seed in args.seeds:
            result = run_multimer_prediction(
                f"{heavy}:{sequences['L']}",
                sequences["A"],
                num_recycle=args.recycles,
                jax_random_seed=seed,
                return_structure=True,
            )
            row = {
                **candidate,
                "seed": seed,
                "success": bool(result.get("success")),
                "elapsed": result.get("elapsed"),
                "plddt": result.get("plddt"),
                "ptm": result.get("ptm"),
                "iptm": result.get("iptm"),
                "interface_pae": result.get("interface_pae"),
                "max_pae": result.get("max_pae"),
                "error": result.get("error"),
            }
            if result.get("success") and result.get("pdb"):
                pdb_path = pdb_dir / f"{candidate['id']}_seed{seed}.pdb"
                pdb_path.write_text(result["pdb"], encoding="ascii")
                row["pdb"] = str(pdb_path.relative_to(ROOT))
                row["pdb_sha256"] = hashlib.sha256(pdb_path.read_bytes()).hexdigest()
            rows.append(row)
            print(candidate["id"], seed, row["success"], row["iptm"], flush=True)

    output = {
        "schema_version": 1,
        "status": "single_scaffold_multi_seed_diagnostic_not_binding_validation",
        "selection_frozen_before_af2": True,
        "selection_rule": "top_n_within_arm_by_pre_af2_native_score",
        "top_n": args.top_n,
        "seeds": args.seeds,
        "recycles": args.recycles,
        "candidates": candidates,
        "results": rows,
    }
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
