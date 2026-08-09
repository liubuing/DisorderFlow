#!/usr/bin/env python
"""AF2 counterfactual alanine scan of native 4HIX contacting H3 residues."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.af2_jax_runner import run_multimer_prediction  # noqa: E402
from modules.h3_interface_contacts import extract_h3_interface  # noqa: E402
from scripts.validate_4hix_matched_af2 import (  # noqa: E402
    NATIVE_H3,
    PDB_PATH,
    chain_sequences,
    full_heavy,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[5101, 5111, 5121])
    parser.add_argument("--recycles", type=int, default=3)
    parser.add_argument("--out-dir", default=str(ROOT / "results/ablation/4hix_alanine_af2_v1"))
    args = parser.parse_args()
    interface = extract_h3_interface(
        str(PDB_PATH),
        "H",
        "A",
        light_chain="L",
        expected_h3_sequence=NATIVE_H3,
        expected_peptide_sequence="DAEFRH",
    )
    contact_indices = sorted({contact.h3_index for contact in interface["contacts"]})
    candidates = []
    for index in contact_indices:
        sequence = list(NATIVE_H3)
        sequence[index] = "A"
        candidates.append(
            {
                "id": f"native_{NATIVE_H3[index]}{index + 1}A",
                "arm": "computational_alanine",
                "h3": "".join(sequence),
                "native_h3_index_zero_based": index,
                "native_residue": NATIVE_H3[index],
            }
        )
    sequences = chain_sequences()
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
                "pre_af2_rank": None,
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
                path = pdb_dir / f"{candidate['id']}_seed{seed}.pdb"
                path.write_text(result["pdb"], encoding="ascii")
                row["pdb"] = str(path.relative_to(ROOT))
                row["pdb_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(row)
            print(candidate["id"], seed, row["iptm"], flush=True)
    output = {
        "schema_version": 1,
        "status": "computational_counterfactual_not_experimental_mutation_evidence",
        "contact_definition": "native heavy-atom distance <=4.5 A",
        "contact_h3_indices_zero_based": contact_indices,
        "seeds": args.seeds,
        "recycles": args.recycles,
        "candidates": candidates,
        "results": rows,
    }
    (out_dir / "results.json").write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
