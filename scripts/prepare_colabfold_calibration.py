#!/usr/bin/env python3
"""Prepare native three-chain anti-A-beta controls for ColabFold calibration."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "modules"))

from idp_antibody_design import _extract_sequence_from_pdb  # noqa: E402


def main():
    controls = [
        ("4HIX_native_DAEFRH", "data/anti_abeta_refs/4HIX.pdb", "H", "L", "DAEFRH"),
        ("5CSZ_native_DAEFRHDSGY", "data/anti_abeta_refs/5CSZ.pdb", "A", "B", "DAEFRHDSGY"),
    ]
    output = ROOT / "results/v5_1_candidates/abeta42_colabfold_calibration/controls.fasta"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="ascii") as handle:
        for name, pdb_path, heavy_chain, light_chain, peptide in controls:
            heavy = _extract_sequence_from_pdb(pdb_path, heavy_chain)
            light = _extract_sequence_from_pdb(pdb_path, light_chain)
            if not heavy or not light:
                raise RuntimeError(f"Missing native antibody chain for {name}")
            handle.write(f">{name}\n{heavy}:{light}:{peptide}\n")
            (output.parent / f"{name}.fasta").write_text(
                f">{name}\n{heavy}:{light}:{peptide}\n", encoding="ascii")
    print(output)


if __name__ == "__main__":
    main()
