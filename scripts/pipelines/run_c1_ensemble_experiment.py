#!/usr/bin/env python
"""C1 H1 experiment: ensemble-robust contact ranking vs single-conformation.

Scores the native antibody CDR against K target conformations and compares
robust (p25) contact separation against scrambled-CDR controls with
single-conformation (reference-only) contact separation. Tests preregistration
``publication/idp_rd_c1_preregistration.json`` gate g1/g2 (H1 only; no AF2).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from Bio.PDB import PDBIO, PDBParser

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "modules"))

from ensemble_contact import per_conformation_contacts, robust_contacts  # noqa: E402

ABETA_CONFS = [
    "data/abeta_conformations/pdbs/abeta42_seed0_42.pdb",
    "data/abeta_conformations/pdbs/abeta42_seed1_42.pdb",
    "data/abeta_conformations/pdbs/abeta42_seed2_42.pdb",
    "data/abeta_conformations/pdbs/abeta42_seed3_42.pdb",
    "data/abeta_conformations/pdbs/abeta42_seed4_42.pdb",
]
# Chothia VH CDR regions (4HIX solanezumab heavy chain)
DEFAULT_CDR = [(26, 32), (52, 56), (95, 102)]


def iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def scramble_cdr(antibody_pdb: str, cdr_regions, seed: int) -> str:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", antibody_pdb)
    model = structure[0]
    cdr_atoms = []
    for chain in model:
        for residue in chain:
            if residue.id[0] != " ":
                continue
            for atom in residue:
                cdr_atoms.append(list(atom.get_coord()))
    generator = random.Random(seed)
    generator.shuffle(cdr_atoms)
    # Reassign shuffled coordinates onto CDR residues only.
    index = 0
    for chain in model:
        for residue in chain:
            if residue.id[0] != " ":
                continue
            if any(start <= residue.id[1] <= end for start, end in cdr_regions):
                for atom in residue:
                    if index < len(cdr_atoms):
                        atom.set_coord(np.asarray(cdr_atoms[index], dtype=float))
                        index += 1
    handle = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
    io = PDBIO()
    io.set_structure(structure)
    io.save(handle.name)
    handle.close()
    return handle.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--antibody", default="idp_benchmark_results/4HIX_H_A.pdb")
    parser.add_argument("--conformations", nargs="+", default=ABETA_CONFS)
    parser.add_argument("--epitope", nargs=2, type=int, default=[16, 24])
    parser.add_argument("--ab-chain", default="H")
    parser.add_argument("--antigen-chain", default="P")
    parser.add_argument("--n-scrambles", type=int, default=20)
    parser.add_argument("--contact-cutoff", type=float, default=8.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    epitope = list(range(args.epitope[0], args.epitope[1] + 1))

    native_per = per_conformation_contacts(
        args.antibody, args.conformations, epitope,
        ab_chain=args.ab_chain, antigen_chain=args.antigen_chain,
        cdr_regions=DEFAULT_CDR, contact_cutoff=args.contact_cutoff)
    native = robust_contacts(native_per, "p25")

    scramble_robust = []
    scramble_single = []
    for seed in range(args.n_scrambles):
        scrambled = scramble_cdr(args.antibody, DEFAULT_CDR, seed)
        per = per_conformation_contacts(
            scrambled, args.conformations, epitope,
            ab_chain=args.ab_chain, antigen_chain=args.antigen_chain,
            cdr_regions=DEFAULT_CDR, contact_cutoff=args.contact_cutoff)
        robust = robust_contacts(per, "p25")
        scramble_robust.append(robust["robust"] if robust["robust"] is not None else -1)
        scramble_single.append(per[0] if per[0] is not None else -1)
        Path(scrambled).unlink(missing_ok=True)

    native_single = native_per[0] if native_per[0] is not None else -1

    def separation(native_value, control_values):
        controls = np.asarray([v for v in control_values], dtype=float)
        return {
            "native": native_value,
            "control_mean": float(controls.mean()),
            "control_min": float(controls.min()),
            "control_max": float(controls.max()),
            "separation": float(native_value - controls.mean()),
            "native_better_than_fraction": float((controls < native_value).mean()),
        }

    robust_sep = separation(native["robust"], scramble_robust)
    single_sep = separation(native_single, scramble_single)

    report = {
        "schema_version": "idp_c1_h1_experiment_v1",
        "classification": "development_only",
        "generated_at_utc": iso_utc(),
        "preregistration": "publication/idp_rd_c1_preregistration.json",
        "hypothesis": "h1_ensemble_robust_contact",
        "native_per_conf": native["per_conf"],
        "native_robust": native["robust"],
        "robust_vs_scrambles": robust_sep,
        "single_conformation_vs_scrambles": single_sep,
        "gate": {
            "g2_ensemble_improvement": (
                robust_sep["native_better_than_fraction"]
                >= single_sep["native_better_than_fraction"]),
            "n_scrambles": args.n_scrambles,
        },
        "claim_boundary": (
            "H1 contact-robustness check only; no AF2 confidence, no binder claim"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
