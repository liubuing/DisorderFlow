"""Minimal standardized-backbone writer for inverse-folding scoring.

Ported (simplified, behaviour-preserving) from modules/h3_interface_contacts.py
in the DisorderFlow repository (write_standardized_backbone / _chain_residues /
_resolved_heavy_atoms). Selected chains are written with contiguous 1-based
numbering, N/CA/C/O atoms only, hetero residues and hydrogens skipped, and
alternate locations resolved by (occupancy, blank altloc, 'A' altloc).

The peptide-stripped ("apo") state is the same writer called without the
peptide chain: the retained chains are bit-identical to the complex state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from Bio.PDB import PDBParser

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
AA1_TO_3 = {value: key for key, value in AA3_TO_1.items()}
BACKBONE_ATOMS = ("N", "CA", "C", "O")


def _resolved_heavy_atoms(residue):
    """Resolved non-hydrogen atoms; best conformer per name."""
    selected = {}
    atoms = residue.get_unpacked_list() if residue.is_disordered() else list(residue)
    for atom in atoms:
        name = atom.get_name().strip()
        element = (atom.element or name[:1]).strip().upper()
        if element in {"H", "D"} or name.startswith(("H", "D")):
            continue
        occupancy = atom.get_occupancy()
        occupancy = float(occupancy) if occupancy is not None else 0.0
        altloc = atom.get_altloc().strip()
        priority = (occupancy, altloc == "", altloc == "A")
        previous = selected.get(name)
        if previous is None or priority > previous[0]:
            coord = tuple(float(value) for value in atom.get_coord())
            selected[name] = (priority, coord)
    return {name: selected[name][1] for name in selected}


def _chain_residues(chain, chain_id: str):
    """Standard amino-acid residues with resolved coordinates, ordered."""
    residues = []
    for residue in chain:
        aa = AA3_TO_1.get(residue.resname.strip())
        if not aa:
            continue
        atoms = _resolved_heavy_atoms(residue)
        if not atoms:
            continue
        residues.append({
            "chain": chain_id,
            "resseq": int(residue.id[1]),
            "icode": residue.id[2].strip(),
            "aa": aa,
            "atoms": atoms,
        })
    residues.sort(key=lambda item: (item["resseq"], item["icode"]))
    return residues


def write_standardized_backbone(
    pdb_path: str,
    output_path: str,
    chains: Sequence[str],
    *,
    model_index: int = 0,
) -> dict:
    """Write selected chains with contiguous numbering; return a manifest."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)
    models = list(structure)
    if model_index < 0 or model_index >= len(models):
        raise ValueError(f"Model index {model_index} is unavailable in {pdb_path}")
    model = models[model_index]
    available = {chain.id.strip() or "_": chain for chain in model}
    missing = [chain for chain in chains if chain not in available]
    if missing:
        raise ValueError(f"Missing configured chains in {pdb_path}: {missing}")
    if len(set(chains)) != len(chains):
        raise ValueError("Chain roles must be unique")

    lines = []
    serial = 1
    manifest = {}
    for chain_id in chains:
        residues = _chain_residues(available[chain_id], chain_id)
        if not residues:
            raise ValueError(f"No standard residues resolved in chain {chain_id}")
        sequence = []
        residue_rows = []
        for new_resseq, residue in enumerate(residues, 1):
            atoms = residue["atoms"]
            missing_backbone = [name for name in BACKBONE_ATOMS if name not in atoms]
            if missing_backbone:
                label = f"{chain_id}{residue['resseq']}{residue['icode']}".strip()
                raise ValueError(f"Incomplete backbone at {label}: {missing_backbone}")
            for atom_name in BACKBONE_ATOMS:
                x, y, z = atoms[atom_name]
                element = atom_name[0]
                lines.append(
                    f"ATOM  {serial:5d} {atom_name:^4s} {AA1_TO_3[residue['aa']]:>3s} "
                    f"{chain_id:1s}{new_resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                    f"  1.00  0.00          {element:>2s}\n")
                serial += 1
            sequence.append(residue["aa"])
            residue_rows.append({
                "original_resseq": residue["resseq"],
                "original_icode": residue["icode"],
                "standardized_resseq": new_resseq,
            })
        lines.append("TER\n")
        manifest[chain_id] = {
            "sequence": "".join(sequence),
            "length": len(sequence),
            "residue_mapping": residue_rows,
        }
    lines.append("END\n")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(lines), encoding="ascii")
    return {
        "schema_version": 1,
        "source_pdb": str(pdb_path),
        "output_pdb": str(output),
        "model_index": model_index,
        "chains": manifest,
    }


def locate_h3_indices(heavy_sequence: str, h3_sequence: str,
                      h3_start: Optional[int] = None) -> list[int]:
    """0-based indices of ``h3_sequence`` in the standardized heavy chain.

    With ``h3_start`` (1-based standardized resseq) the position is asserted;
    otherwise a unique contiguous match is required.
    """
    if not h3_sequence:
        raise ValueError("empty H3 sequence")
    if h3_start is not None:
        if h3_start < 1 or h3_start - 1 + len(h3_sequence) > len(heavy_sequence):
            raise ValueError("h3-start out of range")
        window = heavy_sequence[h3_start - 1:h3_start - 1 + len(h3_sequence)]
        if window != h3_sequence:
            raise ValueError(
                f"heavy chain at h3-start {h3_start} is {window!r}, not {h3_sequence!r}")
        return list(range(h3_start - 1, h3_start - 1 + len(h3_sequence)))
    matches = [
        index for index in range(len(heavy_sequence) - len(h3_sequence) + 1)
        if heavy_sequence[index:index + len(h3_sequence)] == h3_sequence
    ]
    if not matches:
        raise ValueError("H3 sequence not found in standardized heavy chain")
    if len(matches) > 1:
        raise ValueError(
            f"H3 sequence matches {len(matches)} positions; pass --h3-start")
    start = matches[0]
    return list(range(start, start + len(h3_sequence)))
