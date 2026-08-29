#!/usr/bin/env python
"""Curate exact-new components from metadata and coordinate contact fingerprints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from Bio.PDB import MMCIFParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa

ROOT = Path(__file__).resolve().parents[1]

# One antibody lineage per exact-new structure. These are metadata-derived chain roles.
SELECTION = {
    "12ER": ("teclistamab", ["D", "E"], ["C"], "BCMA"),
    "12ES": ("linvoseltamab", ["D", "E"], ["C"], "BCMA"),
    "22HG": ("S2H97", ["I", "M"], ["A"], "SARS_COV_2_RBD"),
    "26PF": ("S54", ["A", "B"], ["C"], "EBV_GP350"),
    "32NZ": ("H90", ["A"], ["B"], "CD44"),
    "9W89": ("10D5", ["B"], ["A"], "CHIKV_E"),
    "9W8A": ("7H9", ["B"], ["A"], "CHIKV_E"),
    "9XHY": ("C092", ["H", "L"], ["E"], "SARS_COV_2_RBD"),
    "9XI4": ("BD56-104", ["H", "L"], ["E"], "SARS_COV_2_RBD"),
    "9YZ7": ("MOD225", ["B"], ["A"], "SARS_COV_2_RBD"),
    "9YZ8": ("MOD203", ["B"], ["A"], "SARS_COV_2_RBD"),
    "9YZ9": ("MOD239", ["B", "D"], ["A", "C"], "SARS_COV_2_RBD"),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contact_fingerprint(path, antibody_chains, antigen_chains, cutoff=5.0):
    structure = next(iter(MMCIFParser(QUIET=True).get_structure(path.stem, str(path))))
    model = structure
    antibody_atoms = [
        atom
        for chain in model
        if chain.id in antibody_chains
        for residue in chain
        for atom in residue
        if atom.element != "H"
    ]
    if not antibody_atoms:
        raise ValueError(f"No antibody atoms found in {path}")
    search = NeighborSearch(antibody_atoms)
    contacts = set()
    for chain in model:
        if chain.id not in antigen_chains:
            continue
        for residue in chain:
            if not is_aa(residue, standard=True):
                continue
            residue_atoms = [atom for atom in residue if atom.element != "H"]
            if any(search.search(atom.coord, cutoff) for atom in residue_atoms):
                contacts.add((residue.id[1], residue.resname))
    if not contacts:
        raise ValueError(f"No antibody-antigen contacts found in {path}")
    return sorted(contacts)


def jaccard(left, right):
    left, right = set(left), set(right)
    return len(left & right) / len(left | right) if left | right else 0.0


def curate(coordinate_dir, metadata_path, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite curation: {output}")
    metadata = json.loads(metadata_path.read_text(encoding="ascii"))
    available = {row["entry_id"] for row in metadata["records"]}
    rows = []
    for entry_id, (lineage, antibody, antigen, target) in SELECTION.items():
        if entry_id not in available:
            raise ValueError(f"Selected entry is absent from metadata: {entry_id}")
        path = coordinate_dir / f"{entry_id}.cif"
        if not path.is_file():
            raise FileNotFoundError(path)
        contacts = contact_fingerprint(path, antibody, antigen)
        rows.append({
            "component_id": entry_id,
            "entry_id": entry_id,
            "lineage": lineage,
            "target": target,
            "antibody_chains": antibody,
            "antigen_chains": antigen,
            "coordinate_sha256": sha256(path),
            "contact_residues": contacts,
        })
    clusters = []
    for row in rows:
        assigned = None
        for cluster in clusters:
            if cluster["target"] == row["target"] and jaccard(
                cluster["representative_contacts"], row["contact_residues"]
            ) >= 0.5:
                assigned = cluster
                break
        if assigned is None:
            assigned = {
                "cluster_id": f"{row['target']}_cluster_{len(clusters) + 1}",
                "target": row["target"],
                "representative_contacts": row["contact_residues"],
                "component_ids": [],
            }
            clusters.append(assigned)
        assigned["component_ids"].append(row["component_id"])
        row["epitope_cluster"] = assigned["cluster_id"]
    lineage_count = len({row["lineage"] for row in rows})
    target_count = len({row["target"] for row in rows})
    cluster_counts = {cluster["cluster_id"]: len(cluster["component_ids"]) for cluster in clusters}
    payload = {
        "schema_version": 1,
        "status": "cohort_candidate_curated" if len(rows) >= 12 and max(cluster_counts.values()) <= 2 else "cohort_blocked",
        "classification": "exact_new_coordinate_contact_curation",
        "metadata": str(metadata_path),
        "metadata_sha256": sha256(metadata_path),
        "component_count": len(rows),
        "independent_lineage_count": lineage_count,
        "target_count": target_count,
        "cluster_counts": cluster_counts,
        "components": rows,
        "clusters": clusters,
        "decision": "freeze_cohort_before_candidate_access" if len(rows) >= 12 and max(cluster_counts.values()) <= 2 else "do_not_freeze_cohort",
        "claim_boundary": (
            "coordinate curation for a general antibody multi-conformer method; "
            "targets are not an IDP-specific confirmation panel; no candidate or "
            "performance result"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = curate(args.coordinates, args.metadata, args.output)
    print(json.dumps({key: value for key, value in result.items() if key not in {"components", "clusters"}}, indent=2))


if __name__ == "__main__":
    main()
