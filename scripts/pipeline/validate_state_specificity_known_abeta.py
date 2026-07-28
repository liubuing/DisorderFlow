#!/usr/bin/env python
"""Validate state-specific scorer on known anti-A-beta structures.

This is a sanity check, not a binding proof. It extracts the native paratope
residues that contact the peptide chain in a reference antibody-peptide PDB,
then compares their state-specific score against scrambled composition-matched
controls.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "modules"))

from state_specificity_scorer import load_preset, score_candidate  # noqa: E402


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Known anti-A-beta scorer sanity check")
    parser.add_argument("--preset", default="abeta_core")
    parser.add_argument("--pdb", action="append", default=[],
                        help="Reference PDB path. Can be repeated.")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--contact-cutoff", type=float, default=8.0)
    parser.add_argument("--n-scramble", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def chain_residues(pdb_path: Path):
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    chains = {}
    for model in structure:
        for chain in model:
            residues = []
            for res in chain:
                aa = AA3_TO_1.get(res.resname.strip())
                if not aa:
                    continue
                atom = res["CB"] if "CB" in res else res["CA"] if "CA" in res else None
                if atom is None:
                    continue
                residues.append({
                    "chain": chain.id.strip() or "_",
                    "resid": res.id[1],
                    "aa": aa,
                    "coord": atom.get_coord(),
                })
            if residues:
                chains[chain.id.strip() or "_"] = residues
        break
    return chains


def infer_peptide_chain(chains):
    # A-beta peptide is normally the shortest protein chain in these complexes.
    return min(chains, key=lambda cid: len(chains[cid]))


def extract_contact_paratope(pdb_path: Path, cutoff: float):
    import numpy as np
    chains = chain_residues(pdb_path)
    if len(chains) < 2:
        raise ValueError(f"Need at least two protein chains in {pdb_path}")
    peptide_chain = infer_peptide_chain(chains)
    pep_coords = np.array([r["coord"] for r in chains[peptide_chain]])
    contact_res = []
    for cid, residues in chains.items():
        if cid == peptide_chain:
            continue
        for r in residues:
            d = np.linalg.norm(pep_coords - r["coord"], axis=1).min()
            if d <= cutoff:
                contact_res.append({**r, "min_distance": float(d)})
    contact_res.sort(key=lambda r: (r["chain"], r["resid"]))
    paratope = "".join(r["aa"] for r in contact_res)
    peptide = "".join(r["aa"] for r in chains[peptide_chain])
    return {
        "pdb": str(pdb_path),
        "peptide_chain": peptide_chain,
        "peptide_sequence": peptide,
        "paratope_sequence": paratope,
        "contact_residues": [
            {"chain": r["chain"], "resid": r["resid"], "aa": r["aa"],
             "min_distance": round(r["min_distance"], 3)}
            for r in contact_res
        ],
        "chain_lengths": {cid: len(res) for cid, res in chains.items()},
    }


def scrambled_controls(seq: str, n: int, seed: int):
    rng = random.Random(seed)
    out = []
    seen = {seq}
    chars = list(seq)
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        c = chars[:]
        rng.shuffle(c)
        s = "".join(c)
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def summarize(native_score, scramble_scores):
    vals = [s["state_specific_score"] for s in scramble_scores]
    gaps = [s["specificity_gap"] for s in scramble_scores]
    better_or_equal = sum(1 for v in vals if v >= native_score["state_specific_score"])
    gap_better_or_equal = sum(1 for v in gaps if v >= native_score["specificity_gap"])
    n = max(1, len(vals))
    return {
        "native_score": native_score["state_specific_score"],
        "native_gap": native_score["specificity_gap"],
        "scramble_mean_score": round(sum(vals) / n, 4),
        "scramble_max_score": round(max(vals) if vals else 0.0, 4),
        "scramble_mean_gap": round(sum(gaps) / n, 4),
        "scramble_max_gap": round(max(gaps) if gaps else 0.0, 4),
        "native_score_percentile": round(1.0 - better_or_equal / n, 4),
        "native_gap_percentile": round(1.0 - gap_better_or_equal / n, 4),
        "n_scramble": len(vals),
        "native_passes_filters": native_score["passes_filters"],
        "native_failures": native_score["filter_failures"],
    }


def main():
    args = parse_args()
    spec = load_preset(args.preset)
    pdbs = [Path(p) for p in args.pdb]
    if not pdbs:
        pdbs = [
            PROJECT_ROOT / "data" / "anti_abeta_refs" / "4HIX.pdb",
            PROJECT_ROOT / "data" / "anti_abeta_refs" / "5CSZ.pdb",
        ]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    details = []
    for pdb in pdbs:
        extracted = extract_contact_paratope(pdb, args.contact_cutoff)
        native = score_candidate({
            "candidate_id": f"{pdb.stem}_native_paratope",
            "cdr_h3": extracted["paratope_sequence"],
            "source_pdb": str(pdb),
            "generator": "native_contact_paratope",
        }, spec)
        scrambles = []
        for i, seq in enumerate(scrambled_controls(extracted["paratope_sequence"], args.n_scramble, args.seed), 1):
            scrambles.append(score_candidate({
                "candidate_id": f"{pdb.stem}_scramble_{i:03d}",
                "cdr_h3": seq,
                "source_pdb": str(pdb),
                "generator": "scrambled_contact_paratope",
            }, spec))
        summary = summarize(native, scrambles)
        rows.append({
            "pdb": pdb.stem,
            "peptide_chain": extracted["peptide_chain"],
            "peptide_sequence": extracted["peptide_sequence"],
            "paratope_sequence": extracted["paratope_sequence"],
            "n_contact_residues": len(extracted["contact_residues"]),
            **summary,
        })
        details.append({
            "extracted": extracted,
            "native": native,
            "scrambles": scrambles,
            "summary": summary,
        })

    with open(out_dir / "known_abeta_state_specificity_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "known_abeta_state_specificity_details.json", "w", encoding="utf-8") as f:
        json.dump({"spec": spec.__dict__, "results": details}, f, indent=2)
    with open(out_dir / "known_abeta_state_specificity_report.md", "w", encoding="utf-8") as f:
        f.write("# Known Anti-A-beta State-Specificity Sanity Check\n\n")
        f.write(f"Preset: `{args.preset}`; positive epitope: `{spec.epitope_seq}`; state: `{spec.positive_state}`\n\n")
        f.write("| PDB | Peptide Chain | Peptide Seq | Native Paratope | Native Score | Scramble Mean | Native Percentile | Native Gap | Scramble Gap Mean | Gap Percentile | Pass |\n")
        f.write("|---|---|---|---|---:|---:|---:|---:|---:|---:|---|\n")
        for r in rows:
            f.write(
                f"| {r['pdb']} | {r['peptide_chain']} | `{r['peptide_sequence']}` | `{r['paratope_sequence']}` | "
                f"{r['native_score']} | {r['scramble_mean_score']} | {r['native_score_percentile']} | "
                f"{r['native_gap']} | {r['scramble_mean_gap']} | {r['native_gap_percentile']} | {r['native_passes_filters']} |\n"
            )
        f.write("\nInterpretation: percentile near 1.0 means native scores above nearly all scrambled controls.\n")

    print(f"Wrote sanity-check results to {out_dir}")
    for r in rows:
        print(
            f"{r['pdb']}: native_score={r['native_score']} scramble_mean={r['scramble_mean_score']} "
            f"percentile={r['native_score_percentile']} native_gap={r['native_gap']} "
            f"gap_percentile={r['native_gap_percentile']} pass={r['native_passes_filters']}"
        )


if __name__ == "__main__":
    main()
