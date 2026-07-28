#!/usr/bin/env python
"""Template-transfer antibody poses onto full-length IDP conformations."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from state_contact_scorer import AA3_TO_1, Contact


def kabsch_transform(moving, target):
    moving = np.asarray(moving, dtype=float)
    target = np.asarray(target, dtype=float)
    if moving.shape != target.shape or moving.ndim != 2 or moving.shape[1] != 3:
        raise ValueError("Kabsch inputs must have matching (N, 3) shapes")
    if len(moving) < 3:
        raise ValueError("At least three matched coordinates are required")
    moving_center = moving.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (moving - moving_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation = target_center - moving_center @ rotation
    fitted = moving @ rotation + translation
    rmsd = float(np.sqrt(np.mean(np.sum((fitted - target) ** 2, axis=1))))
    return rotation, translation, rmsd


def chain_residues(pdb_path, chain_id):
    residues = []
    lookup = {}
    for line in Path(pdb_path).read_text(encoding="utf-8", errors="strict").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or line[21:22].strip() != chain_id:
            continue
        residue_name = line[17:20].strip().upper()
        if residue_name not in AA3_TO_1:
            continue
        key = (int(line[22:26]), line[26:27].strip(), residue_name)
        if key not in lookup:
            lookup[key] = {"resid": key[0], "icode": key[1], "aa": AA3_TO_1[residue_name], "atoms": {}}
            residues.append(lookup[key])
        atom_name = line[12:16].strip()
        altloc = line[16:17].strip()
        if altloc not in ("", "A") or atom_name in lookup[key]["atoms"]:
            continue
        lookup[key]["atoms"][atom_name] = np.array([
            float(line[30:38]), float(line[38:46]), float(line[46:54])
        ])
    return residues


def ca_coordinates(pdb_path, chain_id):
    residues = chain_residues(pdb_path, chain_id)
    if not residues or any("CA" not in residue["atoms"] for residue in residues):
        raise ValueError(f"Chain {chain_id!r} lacks a complete CA trace in {pdb_path}")
    return "".join(row["aa"] for row in residues), np.array([row["atoms"]["CA"] for row in residues])


def transfer_pose(template_pdb, target_pdb, output_pdb, peptide_chain, target_chain,
                  heavy_chain, light_chain, abeta_start, output_antigen_chain="P"):
    template_sequence, template_ca = ca_coordinates(template_pdb, peptide_chain)
    target_sequence, target_ca = ca_coordinates(target_pdb, target_chain)
    start = int(abeta_start) - 1
    end = start + len(template_sequence)
    if target_sequence[start:end] != template_sequence:
        raise ValueError(
            f"Target residues {start + 1}-{end} do not match template peptide "
            f"{template_sequence!r}: found {target_sequence[start:end]!r}"
        )
    rotation, translation, rmsd = kabsch_transform(target_ca[start:end], template_ca)

    template_lines = Path(template_pdb).read_text(encoding="utf-8", errors="strict").splitlines()
    target_lines = Path(target_pdb).read_text(encoding="utf-8", errors="strict").splitlines()
    output = []
    serial = 1
    for line in template_lines:
        if line.startswith(("ATOM  ", "HETATM")) and line[21:22].strip() in {heavy_chain, light_chain}:
            output.append(f"{line[:6]}{serial:5d}{line[11:]}")
            serial += 1
    for line in target_lines:
        if not line.startswith(("ATOM  ", "HETATM")) or line[21:22].strip() != target_chain:
            continue
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        x, y, z = xyz @ rotation + translation
        rewritten = f"{line[:6]}{serial:5d}{line[11:21]}{output_antigen_chain}{line[22:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
        output.append(rewritten)
        serial += 1
    output.append("END")
    output_path = Path(output_pdb)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output) + "\n", encoding="ascii")
    return {
        "epitope_fit_rmsd": rmsd,
        "template_peptide_sequence": template_sequence,
        "target_sequence": target_sequence,
        "target_region": [start + 1, end],
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
    }


def fixed_paratope_contact_map(pdb_path, template_contact_map, peptide_chain="P", cutoff=8.0):
    peptide = chain_residues(pdb_path, peptide_chain)
    if not peptide:
        raise ValueError(f"Peptide chain {peptide_chain!r} not found in {pdb_path}")
    peptide_atoms = [row["atoms"].get("CB", row["atoms"].get("CA")) for row in peptide]
    paratope = []
    for expected in template_contact_map["paratope_residues"]:
        matches = [
            row for row in chain_residues(pdb_path, expected["chain"])
            if row["resid"] == expected["resid"] and row["aa"] == expected["aa"]
        ]
        if len(matches) != 1:
            raise ValueError(f"Paratope residue missing or ambiguous: {expected}")
        paratope.append(matches[0])

    contacts = []
    for pidx, residue in enumerate(paratope):
        pcoord = residue["atoms"].get("CB", residue["atoms"].get("CA"))
        for eidx, (epitope_residue, ecoord) in enumerate(zip(peptide, peptide_atoms)):
            distance = float(np.linalg.norm(pcoord - ecoord))
            if distance <= cutoff:
                contacts.append(Contact(
                    pidx, template_contact_map["paratope_residues"][pidx]["chain"],
                    residue["resid"], residue["aa"], eidx, peptide_chain,
                    epitope_residue["resid"], epitope_residue["aa"], distance,
                ))
    out = dict(template_contact_map)
    out.update({
        "pdb_path": str(pdb_path),
        "peptide_chain": peptide_chain,
        "peptide_sequence": "".join(row["aa"] for row in peptide),
        "contacts": contacts,
        "cutoff": cutoff,
    })
    return out


def pose_geometry_audit(pdb_path, antibody_chains, antigen_chain="P"):
    antibody_atoms = []
    for chain in antibody_chains:
        for residue in chain_residues(pdb_path, chain):
            antibody_atoms.extend(residue["atoms"].values())
    antigen_atoms = [atom for residue in chain_residues(pdb_path, antigen_chain) for atom in residue["atoms"].values()]
    if not antibody_atoms or not antigen_atoms:
        raise ValueError("Pose must contain both antibody and antigen atoms")
    antibody = np.array(antibody_atoms)
    antigen = np.array(antigen_atoms)
    distances = np.linalg.norm(antibody[:, None, :] - antigen[None, :, :], axis=-1)
    return {
        "minimum_heavy_atom_distance": float(distances.min()),
        "severe_clash_pairs_lt_1_5A": int((distances < 1.5).sum()),
        "close_atom_pairs_lt_5A": int((distances < 5.0).sum()),
    }


def resolve_pose_clashes(pdb_path, antibody_chains, antigen_chain="P",
                         max_severe_clashes=0, max_shift=8.0, step=0.1):
    """Apply the smallest rigid antibody retreat that passes the clash gate."""
    path = Path(pdb_path)
    lines = path.read_text(encoding="ascii").splitlines()
    antibody_indices = []
    antibody_coords = []
    antigen_coords = []
    for index, line in enumerate(lines):
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        chain = line[21:22].strip()
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        if chain in antibody_chains:
            antibody_indices.append(index)
            antibody_coords.append(xyz)
        elif chain == antigen_chain:
            antigen_coords.append(xyz)
    antibody = np.array(antibody_coords)
    antigen = np.array(antigen_coords)
    shifted = antibody.copy()
    delta = np.zeros(3, dtype=float)
    for _ in range(int(max_shift / step) + 1):
        distances = np.linalg.norm(shifted[:, None, :] - antigen[None, :, :], axis=-1)
        if int((distances < 1.5).sum()) <= max_severe_clashes:
            break
        clash_antibody, clash_antigen = np.where(distances < 2.5)
        vectors = shifted[clash_antibody] - antigen[clash_antigen]
        weights = 1.0 / np.maximum(distances[clash_antibody, clash_antigen], 0.1) ** 2
        direction = (vectors * weights[:, None]).sum(axis=0)
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            direction = shifted.mean(axis=0) - antigen.mean(axis=0)
            norm = np.linalg.norm(direction)
        increment = direction / max(norm, 1e-8) * step
        shifted += increment
        delta += increment
    selected_shift = float(np.linalg.norm(delta))
    if selected_shift:
        for row_index, line_index in enumerate(antibody_indices):
            x, y, z = antibody[row_index] + delta
            line = lines[line_index]
            lines[line_index] = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
        path.write_text("\n".join(lines) + "\n", encoding="ascii")
    direction = delta / selected_shift if selected_shift else np.zeros(3)
    return {"rigid_retreat_angstrom": selected_shift, "retreat_direction": direction.tolist()}


def refine_pose_local(pdb_path, antibody_chains, antigen_chain="P",
                      max_translation=4.0, max_rotation_degrees=20.0, seed=0,
                      attempts=3):
    """Search a small six-degree-of-freedom neighborhood for a clash-free pose."""
    from scipy.optimize import differential_evolution
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    path = Path(pdb_path)
    lines = path.read_text(encoding="ascii").splitlines()
    antibody_indices = []
    antibody_coords = []
    antigen_coords = []
    for index, line in enumerate(lines):
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        chain = line[21:22].strip()
        if chain in antibody_chains:
            antibody_indices.append(index)
            antibody_coords.append(xyz)
        elif chain == antigen_chain:
            antigen_coords.append(xyz)
    antibody = np.asarray(antibody_coords)
    antigen = np.asarray(antigen_coords)
    center = antigen.mean(axis=0)
    antibody_tree = cKDTree(antibody)

    def inverse_antigen(parameters):
        rotation = Rotation.from_rotvec(np.radians(parameters[3:])).as_matrix()
        translation = np.asarray(parameters[:3])
        return (antigen - center - translation) @ rotation + center

    def objective(parameters):
        moved_antigen = inverse_antigen(parameters)
        severe_neighbors = antibody_tree.query_ball_point(moved_antigen, 1.5)
        severe = sum(len(rows) for rows in severe_neighbors)
        penetration = 0.0
        for antigen_index, neighbors in enumerate(severe_neighbors):
            if neighbors:
                distances = np.linalg.norm(
                    antibody[np.asarray(neighbors)] - moved_antigen[antigen_index], axis=1
                )
                penetration += float(np.square(1.5 - distances).sum())
        close = sum(len(rows) for rows in antibody_tree.query_ball_point(moved_antigen, 5.0))
        motion = np.linalg.norm(parameters[:3]) + 0.05 * np.linalg.norm(parameters[3:])
        # Lexicographic priorities encoded with separated scales: remove severe
        # overlap, retain an interface, then minimize deviation from the template.
        return severe * 10000.0 + penetration * 1000.0 - min(close, 500) * 2.0 + motion

    bounds = [(-max_translation, max_translation)] * 3 + [
        (-max_rotation_degrees, max_rotation_degrees)
    ] * 3
    results = [
        differential_evolution(
            objective, bounds, seed=seed + attempt * 1009, popsize=10,
            maxiter=70, polish=True, workers=1, updating="immediate",
            atol=0.01, tol=0.001,
        )
        for attempt in range(attempts)
    ]
    result = min(results, key=lambda row: row.fun)
    parameters = result.x
    rotation = Rotation.from_rotvec(np.radians(parameters[3:])).as_matrix()
    translation = np.asarray(parameters[:3])
    transformed = (antibody - center) @ rotation.T + center + translation
    for row_index, line_index in enumerate(antibody_indices):
        x, y, z = transformed[row_index]
        line = lines[line_index]
        lines[line_index] = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return {
        "local_refinement_translation": translation.tolist(),
        "local_refinement_translation_norm": float(np.linalg.norm(translation)),
        "local_refinement_rotation_degrees": parameters[3:].tolist(),
        "local_refinement_rotation_norm_degrees": float(np.linalg.norm(parameters[3:])),
        "local_refinement_objective": float(result.fun),
        "local_refinement_success": bool(result.success),
        "local_refinement_attempts": attempts,
    }


def refine_pose_contacts(pdb_path, antibody_chains, template_contact_map,
                         antigen_chain="P", max_translation=4.0,
                         max_rotation_degrees=20.0, seed=0, attempts=3):
    """Search a local rigid neighborhood for zero clashes and fixed-paratope contacts."""
    from scipy.optimize import differential_evolution
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    path = Path(pdb_path)
    lines = path.read_text(encoding="ascii").splitlines()
    antibody_indices = []
    antibody_coords = []
    antigen_heavy = []
    for index, line in enumerate(lines):
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        element = line[76:78].strip() or line[12:16].strip()[0]
        if element.upper() == "H":
            continue
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        chain = line[21:22].strip()
        if chain in antibody_chains:
            antibody_indices.append(index)
            antibody_coords.append(xyz)
        elif chain == antigen_chain:
            antigen_heavy.append(xyz)
    antibody = np.asarray(antibody_coords)
    antigen = np.asarray(antigen_heavy)
    center = antigen.mean(axis=0)

    paratope_coords = []
    for expected in template_contact_map["paratope_residues"]:
        matches = [
            row for row in chain_residues(path, expected["chain"])
            if row["resid"] == expected["resid"] and row["aa"] == expected["aa"]
        ]
        if len(matches) != 1:
            raise ValueError(f"Paratope residue missing or ambiguous: {expected}")
        paratope_coords.append(matches[0]["atoms"].get("CB", matches[0]["atoms"]["CA"]))
    paratope = np.asarray(paratope_coords)
    antigen_contact = np.asarray([
        row["atoms"].get("CB", row["atoms"].get("CA"))
        for row in chain_residues(path, antigen_chain)
    ])

    def transform(coords, parameters):
        rotation = Rotation.from_rotvec(np.radians(parameters[3:])).as_matrix()
        return (coords - center) @ rotation.T + center + np.asarray(parameters[:3])

    def objective(parameters):
        moved = transform(antibody, parameters)
        moved_paratope = transform(paratope, parameters)
        severe = sum(len(rows) for rows in cKDTree(antigen).query_ball_point(moved, 1.5))
        contacts = sum(
            len(rows) for rows in cKDTree(antigen_contact).query_ball_point(moved_paratope, 8.0)
        )
        motion = np.linalg.norm(parameters[:3]) + 0.05 * np.linalg.norm(parameters[3:])
        return severe * 100000.0 - contacts * 100.0 + motion

    bounds = [(-max_translation, max_translation)] * 3 + [
        (-max_rotation_degrees, max_rotation_degrees)
    ] * 3
    results = [
        differential_evolution(
            objective, bounds, seed=seed + attempt * 1009, popsize=10,
            maxiter=70, polish=True, workers=1, updating="immediate",
            atol=0.01, tol=0.001,
        )
        for attempt in range(attempts)
    ]
    result = min(results, key=lambda row: row.fun)
    parameters = result.x
    transformed = transform(antibody, parameters)
    for row_index, line_index in enumerate(antibody_indices):
        x, y, z = transformed[row_index]
        line = lines[line_index]
        lines[line_index] = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return {
        "local_refinement_translation": parameters[:3].tolist(),
        "local_refinement_translation_norm": float(np.linalg.norm(parameters[:3])),
        "local_refinement_rotation_degrees": parameters[3:].tolist(),
        "local_refinement_rotation_norm_degrees": float(np.linalg.norm(parameters[3:])),
        "local_refinement_objective": float(result.fun),
        "local_refinement_success": bool(result.success),
        "local_refinement_attempts": attempts,
        "local_refinement_mode": "fixed_paratope_contact_aware",
    }
