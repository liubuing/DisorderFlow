"""Auditable CDR-H3 to peptide contact extraction for benchmark diagnostics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from Bio.PDB import PDBParser


AA = "ACDEFGHIKLMNPQRSTVWY"
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
AA1_TO_3 = {value: key for key, value in AA3_TO_1.items()}
BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O", "OXT"})
AROMATIC = frozenset("FWY")
HYDROPHOBIC = frozenset("AILMFWYV")
POSITIVE = frozenset("KRH")
NEGATIVE = frozenset("DE")
POLAR = frozenset("STNQYH")


@dataclass(frozen=True, order=True)
class ResidueID:
    chain: str
    resseq: int
    icode: str = ""


@dataclass(frozen=True)
class InterfaceContact:
    h3_index: int
    h3_residue: ResidueID
    h3_aa: str
    epitope_index: int
    epitope_residue: ResidueID
    epitope_aa: str
    min_heavy_atom_distance: float
    heavy_atom_pair: Tuple[str, str]
    min_sidechain_distance: Optional[float]
    sidechain_atom_pair: Optional[Tuple[str, str]]


@dataclass(frozen=True)
class _ResidueGeometry:
    residue_id: ResidueID
    aa: str
    atoms: Tuple[Tuple[str, Tuple[float, float, float]], ...]


def extract_h3_interface(
    pdb_path: str,
    heavy_chain: str,
    peptide_chain: str,
    *,
    light_chain: Optional[str] = None,
    expected_h3_sequence: Optional[str] = None,
    expected_peptide_sequence: Optional[str] = None,
    contact_cutoff: float = 4.5,
    sidechain_cutoff: float = 5.0,
    model_index: int = 0,
) -> Dict:
    """Extract an anchor-bounded H3 interface using explicit chain roles."""
    if contact_cutoff <= 0 or sidechain_cutoff <= 0:
        raise ValueError("Contact cutoffs must be positive")
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)
    models = list(structure)
    if model_index < 0 or model_index >= len(models):
        raise ValueError(f"Model index {model_index} is unavailable in {pdb_path}")
    model = models[model_index]
    available = {chain.id.strip() or "_": chain for chain in model}
    required = [heavy_chain, peptide_chain]
    if light_chain:
        required.append(light_chain)
    missing = [chain for chain in required if chain not in available]
    if missing:
        raise ValueError(f"Missing configured chains in {pdb_path}: {missing}")

    heavy = _chain_residues(available[heavy_chain], heavy_chain)
    peptide = _chain_residues(available[peptide_chain], peptide_chain)
    if not heavy or not peptide:
        raise ValueError(f"Configured chains have no resolved amino acids in {pdb_path}")
    h3, anchors = infer_anchor_bounded_h3(heavy)
    h3_sequence = "".join(residue.aa for residue in h3)
    peptide_sequence = "".join(residue.aa for residue in peptide)
    if expected_h3_sequence and h3_sequence != expected_h3_sequence:
        raise ValueError(
            f"H3 sequence mismatch for {pdb_path}: {h3_sequence} != {expected_h3_sequence}")
    if expected_peptide_sequence and peptide_sequence != expected_peptide_sequence:
        raise ValueError(
            f"Peptide sequence mismatch for {pdb_path}: "
            f"{peptide_sequence} != {expected_peptide_sequence}")

    contacts = []
    for h3_index, h3_residue in enumerate(h3):
        for epitope_index, epitope_residue in enumerate(peptide):
            heavy_distance, heavy_pair = _minimum_atom_distance(
                h3_residue.atoms, epitope_residue.atoms)
            if heavy_distance > contact_cutoff:
                continue
            h3_sidechain = tuple(
                atom for atom in h3_residue.atoms if atom[0] not in BACKBONE_ATOMS)
            epitope_sidechain = tuple(
                atom for atom in epitope_residue.atoms if atom[0] not in BACKBONE_ATOMS)
            sidechain_distance = None
            sidechain_pair = None
            if h3_sidechain and epitope_sidechain:
                candidate_distance, candidate_pair = _minimum_atom_distance(
                    h3_sidechain, epitope_sidechain)
                if candidate_distance <= sidechain_cutoff:
                    sidechain_distance = candidate_distance
                    sidechain_pair = candidate_pair
            contacts.append(InterfaceContact(
                h3_index=h3_index,
                h3_residue=h3_residue.residue_id,
                h3_aa=h3_residue.aa,
                epitope_index=epitope_index,
                epitope_residue=epitope_residue.residue_id,
                epitope_aa=epitope_residue.aa,
                min_heavy_atom_distance=round(heavy_distance, 4),
                heavy_atom_pair=heavy_pair,
                min_sidechain_distance=(
                    round(sidechain_distance, 4) if sidechain_distance is not None else None),
                sidechain_atom_pair=sidechain_pair,
            ))
    contacts.sort(key=lambda contact: (
        contact.h3_index, contact.epitope_index, contact.min_heavy_atom_distance))
    return {
        "schema_version": 1,
        "pdb_path": str(pdb_path),
        "model_index": model_index,
        "chain_roles": {
            "heavy": heavy_chain,
            "light": light_chain,
            "peptide": peptide_chain,
        },
        "contact_cutoff": contact_cutoff,
        "sidechain_cutoff": sidechain_cutoff,
        "h3_definition": "between conserved C and WG anchors; anchors excluded",
        "h3_anchor_residues": [asdict(anchor.residue_id) for anchor in anchors],
        "h3_sequence": h3_sequence,
        "h3_residues": [asdict(residue.residue_id) for residue in h3],
        "peptide_sequence": peptide_sequence,
        "peptide_residues": [asdict(residue.residue_id) for residue in peptide],
        "contacts": contacts,
    }


def infer_anchor_bounded_h3(
    heavy_residues: Sequence[_ResidueGeometry],
) -> Tuple[List[_ResidueGeometry], Tuple[_ResidueGeometry, _ResidueGeometry]]:
    """Find the variable-domain C...WG motif and return residues between it."""
    sequence = "".join(residue.aa for residue in heavy_residues)
    candidates = []
    upper = min(len(sequence) - 1, 150)
    for c_index in range(70, upper):
        if sequence[c_index] != "C":
            continue
        for w_index in range(c_index + 4, min(c_index + 36, len(sequence) - 1)):
            if sequence[w_index:w_index + 2] == "WG":
                candidates.append((c_index, w_index))
                break
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one variable-domain C...WG H3 anchor pair, found {len(candidates)}")
    c_index, w_index = candidates[0]
    h3 = list(heavy_residues[c_index + 1:w_index])
    if not 3 <= len(h3) <= 30:
        raise ValueError(f"Anchor-bounded H3 length is implausible: {len(h3)}")
    return h3, (heavy_residues[c_index], heavy_residues[w_index])


def score_h3_sequence(sequence: str, interface: Dict) -> Dict:
    """Score sidechain chemistry without any native-sequence similarity term."""
    sequence = str(sequence).strip().upper()
    native_length = len(interface["h3_sequence"])
    if len(sequence) != native_length or any(aa not in AA for aa in sequence):
        raise ValueError(f"Expected a valid {native_length}-residue H3 sequence")
    contacts: Iterable[InterfaceContact] = interface["contacts"]
    weighted_score = 0.0
    total_weight = 0.0
    eligible_contacts = 0
    position_totals = [0.0] * native_length
    position_weights = [0.0] * native_length
    for contact in contacts:
        if contact.min_sidechain_distance is None:
            continue
        weight = math.exp(-max(0.0, contact.min_sidechain_distance - 3.0) / 1.5)
        compatibility = _pair_compatibility(
            sequence[contact.h3_index], contact.epitope_aa)
        weighted_score += weight * compatibility
        total_weight += weight
        eligible_contacts += 1
        position_totals[contact.h3_index] += weight * compatibility
        position_weights[contact.h3_index] += weight
    score = weighted_score / total_weight if total_weight else 0.0
    return {
        "sequence": sequence,
        "chemistry_score": round(score, 6),
        "n_geometry_contacts": len(interface["contacts"]),
        "n_sidechain_contacts": eligible_contacts,
        "position_scores": [
            round(total / weight, 6) if weight else None
            for total, weight in zip(position_totals, position_weights)
        ],
    }


def serialize_interface(interface: Dict) -> Dict:
    """Convert an interface map to JSON-compatible primitives."""
    return {
        **{key: value for key, value in interface.items() if key != "contacts"},
        "contacts": [
            {
                **asdict(contact),
                "h3_residue": asdict(contact.h3_residue),
                "epitope_residue": asdict(contact.epitope_residue),
            }
            for contact in interface["contacts"]
        ],
    }


def write_standardized_backbone(
    pdb_path: str,
    output_path: str,
    chains: Sequence[str],
    *,
    model_index: int = 0,
) -> Dict:
    """Write selected chains with contiguous numbering for inverse-folding tools."""
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
        raise ValueError("Standardized chain roles must be unique")

    lines = []
    serial = 1
    manifest = {}
    for chain_id in chains:
        residues = _chain_residues(available[chain_id], chain_id)
        residue_rows = []
        sequence = []
        for new_resseq, residue in enumerate(residues, 1):
            atoms = dict(residue.atoms)
            missing_backbone = [name for name in ("N", "CA", "C", "O") if name not in atoms]
            if missing_backbone:
                raise ValueError(
                    f"Incomplete backbone at {residue.residue_id}: {missing_backbone}")
            for atom_name in ("N", "CA", "C", "O"):
                x, y, z = atoms[atom_name]
                element = atom_name[0]
                lines.append(
                    f"ATOM  {serial:5d} {atom_name:^4s} {AA1_TO_3[residue.aa]:>3s} "
                    f"{chain_id:1s}{new_resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                    f"  1.00  0.00          {element:>2s}\n")
                serial += 1
            sequence.append(residue.aa)
            residue_rows.append({
                "original": asdict(residue.residue_id),
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


def _chain_residues(chain, chain_id: str) -> List[_ResidueGeometry]:
    residues = []
    for residue in chain:
        aa = AA3_TO_1.get(residue.resname.strip())
        if not aa:
            continue
        atoms = _resolved_heavy_atoms(residue)
        if not atoms:
            continue
        residues.append(_ResidueGeometry(
            residue_id=ResidueID(chain_id, int(residue.id[1]), residue.id[2].strip()),
            aa=aa,
            atoms=atoms,
        ))
    residues.sort(key=lambda residue: (
        residue.residue_id.resseq, residue.residue_id.icode))
    return residues


def _resolved_heavy_atoms(residue) -> Tuple[Tuple[str, Tuple[float, float, float]], ...]:
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
    return tuple((name, selected[name][1]) for name in sorted(selected))


def _minimum_atom_distance(atoms_a, atoms_b):
    best_distance = float("inf")
    best_pair = None
    for name_a, coord_a in atoms_a:
        for name_b, coord_b in atoms_b:
            distance = math.dist(coord_a, coord_b)
            if distance < best_distance:
                best_distance = distance
                best_pair = (name_a, name_b)
    if best_pair is None:
        raise ValueError("Cannot compute a distance without resolved heavy atoms")
    return best_distance, best_pair


def _pair_compatibility(h3_aa: str, epitope_aa: str) -> float:
    score = 0.0
    if h3_aa in HYDROPHOBIC and epitope_aa in HYDROPHOBIC:
        score += 0.35
    if h3_aa in AROMATIC and epitope_aa in AROMATIC:
        score += 0.25
    if ((h3_aa in POSITIVE and epitope_aa in NEGATIVE)
            or (h3_aa in NEGATIVE and epitope_aa in POSITIVE)):
        score += 0.45
    if h3_aa in POLAR and epitope_aa in POLAR:
        score += 0.20
    if h3_aa == "C":
        score -= 0.20
    return max(-0.20, min(1.0, score))
