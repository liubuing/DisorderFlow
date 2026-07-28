#!/usr/bin/env python
"""Restrained OpenMM side-chain/interface minimization for fold-passing Fv models."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from openmm import CustomExternalForce, LangevinMiddleIntegrator, Platform, unit
from openmm.app import ForceField, HBonds, Modeller, NoCutoff, PDBFile, Simulation
from pdbfixer import PDBFixer


BACKBONE = {"N", "CA", "C", "O"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-evidence", default="outputs/abeta_multiconf_fold_evidence_v1/fv_colabfold_evidence.csv")
    parser.add_argument("--result-dir", default="outputs/abeta_multiconf_integrated_evidence_v1/colabfold_fv_msa")
    parser.add_argument("--out", default="outputs/abeta_multiconf_fv_repack_v1")
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--interface-cutoff", type=float, default=8.0)
    parser.add_argument("--backbone-restraint", type=float, default=1000.0)
    parser.add_argument("--noninterface-restraint", type=float, default=250.0)
    parser.add_argument("--max-backbone-rmsd", type=float, default=0.75)
    parser.add_argument("--max-interface-clashes", type=int, default=0)
    parser.add_argument("--min-contact-retention", type=float, default=0.75)
    parser.add_argument("--platform", choices=["CPU", "OpenCL"], default="CPU")
    return parser.parse_args()


def load_fold_pass(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["fold_sidecheck_status"] == "fold_sidecheck_pass"]


def find_model(result_dir, construct_id):
    matches = sorted(Path(result_dir).glob(f"{construct_id}_*_unrelaxed_rank_001_*.pdb"))
    if len(matches) != 1:
        raise ValueError(f"Expected one rank-1 model for {construct_id}, found {len(matches)}")
    return matches[0]


def heavy_atom_records(topology, positions):
    records = []
    xyz = positions.value_in_unit(unit.angstrom)
    for atom in topology.atoms():
        if atom.element is None or atom.element.symbol == "H":
            continue
        records.append({
            "index": atom.index,
            "name": atom.name,
            "chain": atom.residue.chain.id,
            "residue": atom.residue.id,
            "coord": np.asarray(xyz[atom.index], dtype=float),
        })
    return records


def interface_residues(topology, positions, cutoff):
    records = heavy_atom_records(topology, positions)
    chains = sorted({row["chain"] for row in records})
    if len(chains) != 2:
        raise ValueError(f"Expected two Fv chains, found {chains}")
    left = [row for row in records if row["chain"] == chains[0]]
    right = [row for row in records if row["chain"] == chains[1]]
    distances = np.linalg.norm(
        np.asarray([row["coord"] for row in left])[:, None, :]
        - np.asarray([row["coord"] for row in right])[None, :, :], axis=-1,
    )
    left_indices, right_indices = np.where(distances <= cutoff)
    residues = {(left[index]["chain"], left[index]["residue"]) for index in left_indices}
    residues.update((right[index]["chain"], right[index]["residue"]) for index in right_indices)
    return residues


def interface_metrics(topology, positions):
    records = heavy_atom_records(topology, positions)
    chains = sorted({row["chain"] for row in records})
    left = [row for row in records if row["chain"] == chains[0]]
    right = [row for row in records if row["chain"] == chains[1]]
    distances = np.linalg.norm(
        np.asarray([row["coord"] for row in left])[:, None, :]
        - np.asarray([row["coord"] for row in right])[None, :, :], axis=-1,
    )
    return {
        "chains": chains,
        "minimum_heavy_atom_distance": float(distances.min()),
        "severe_clash_pairs_lt_1_5A": int((distances < 1.5).sum()),
        "close_atom_pairs_lt_5A": int((distances < 5.0).sum()),
        "contact_atom_pairs_lt_8A": int((distances < 8.0).sum()),
    }


def structure_geometry_metrics(topology, positions):
    records = heavy_atom_records(topology, positions)
    coordinates = np.asarray([row["coord"] for row in records])
    residue_order = {}
    residues_by_chain = {}
    for chain in topology.chains():
        residues = list(chain.residues())
        residues_by_chain[chain.id] = residues
        for index, residue in enumerate(residues):
            residue_order[(chain.id, residue.id)] = index

    from scipy.spatial import cKDTree
    close_pairs = cKDTree(coordinates).query_pairs(1.5)
    nonlocal_clashes = []
    for left_index, right_index in close_pairs:
        left = records[left_index]
        right = records[right_index]
        left_key = (left["chain"], left["residue"])
        right_key = (right["chain"], right["residue"])
        if left_key == right_key:
            continue
        if left["chain"] == right["chain"] and abs(
            residue_order[left_key] - residue_order[right_key]
        ) <= 1:
            continue
        nonlocal_clashes.append(float(np.linalg.norm(left["coord"] - right["coord"])))

    xyz = positions.value_in_unit(unit.angstrom)
    ca_distances = []
    peptide_cn_distances = []
    for residues in residues_by_chain.values():
        for left, right in zip(residues, residues[1:]):
            left_atoms = {atom.name: atom for atom in left.atoms()}
            right_atoms = {atom.name: atom for atom in right.atoms()}
            if "CA" in left_atoms and "CA" in right_atoms:
                ca_distances.append(float(np.linalg.norm(
                    np.asarray(xyz[left_atoms["CA"].index])
                    - np.asarray(xyz[right_atoms["CA"].index])
                )))
            if "C" in left_atoms and "N" in right_atoms:
                peptide_cn_distances.append(float(np.linalg.norm(
                    np.asarray(xyz[left_atoms["C"].index])
                    - np.asarray(xyz[right_atoms["N"].index])
                )))
    return {
        "nonlocal_heavy_atom_clashes_lt_1_5A": len(nonlocal_clashes),
        "minimum_nonlocal_heavy_atom_distance": min(nonlocal_clashes, default=None),
        "consecutive_ca_distance_min": min(ca_distances),
        "consecutive_ca_distance_max": max(ca_distances),
        "peptide_cn_distance_min": min(peptide_cn_distances),
        "peptide_cn_distance_max": max(peptide_cn_distances),
        "ca_geometry_outliers": sum(not 3.5 <= value <= 4.1 for value in ca_distances),
        "peptide_cn_geometry_outliers": sum(
            not 1.15 <= value <= 1.55 for value in peptide_cn_distances
        ),
    }


def backbone_coordinates(topology, positions):
    xyz = positions.value_in_unit(unit.angstrom)
    return {
        (atom.residue.chain.id, atom.residue.id, atom.name): np.asarray(xyz[atom.index], dtype=float)
        for atom in topology.atoms() if atom.name in BACKBONE
    }


def coordinate_rmsd(before, after):
    keys = sorted(set(before) & set(after))
    if not keys:
        raise ValueError("No matching backbone atoms")
    delta = np.asarray([after[key] - before[key] for key in keys])
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def missing_atom_count(fixer):
    return sum(len(atoms) for atoms in fixer.missingAtoms.values()) + sum(
        len(atoms) for atoms in fixer.missingTerminals.values()
    )


def minimize_model(input_path, output_path, args):
    original = PDBFile(str(input_path))
    before_interface = interface_metrics(original.topology, original.positions)
    before_backbone = backbone_coordinates(original.topology, original.positions)
    before_geometry = structure_geometry_metrics(original.topology, original.positions)

    fixer = PDBFixer(filename=str(input_path))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    rebuilt_atoms = missing_atom_count(fixer)
    fixer.addMissingAtoms(seed=0)
    forcefield = ForceField("amber14-all.xml", "implicit/gbn2.xml")
    modeller = Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(forcefield, pH=7.4)
    interface = interface_residues(modeller.topology, modeller.positions, args.interface_cutoff)

    system = forcefield.createSystem(
        modeller.topology, nonbondedMethod=NoCutoff, constraints=HBonds,
    )
    restraint = CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.addPerParticleParameter("k")
    for parameter in ("x0", "y0", "z0"):
        restraint.addPerParticleParameter(parameter)
    position_nm = modeller.positions.value_in_unit(unit.nanometer)
    restrained_backbone = 0
    restrained_noninterface = 0
    for atom in modeller.topology.atoms():
        if atom.element is None or atom.element.symbol == "H":
            continue
        key = (atom.residue.chain.id, atom.residue.id)
        stiffness = None
        if atom.name in BACKBONE:
            stiffness = args.backbone_restraint
            restrained_backbone += 1
        elif key not in interface:
            stiffness = args.noninterface_restraint
            restrained_noninterface += 1
        if stiffness is not None:
            xyz = position_nm[atom.index]
            restraint.addParticle(atom.index, [stiffness, xyz.x, xyz.y, xyz.z])
    system.addForce(restraint)

    integrator = LangevinMiddleIntegrator(
        300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds,
    )
    platform = Platform.getPlatformByName(args.platform)
    properties = {"Precision": "mixed"} if args.platform == "OpenCL" else {}
    simulation = Simulation(modeller.topology, system, integrator, platform, properties)
    simulation.context.setPositions(modeller.positions)
    initial = simulation.context.getState(getEnergy=True, getPositions=True)
    initial_energy = initial.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    simulation.minimizeEnergy(
        tolerance=10.0 * unit.kilojoule_per_mole / unit.nanometer,
        maxIterations=args.max_iterations,
    )
    final = simulation.context.getState(getEnergy=True, getPositions=True)
    final_energy = final.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="ascii") as handle:
        PDBFile.writeFile(modeller.topology, final.getPositions(), handle, keepIds=True)

    after_interface = interface_metrics(modeller.topology, final.getPositions())
    after_backbone = backbone_coordinates(modeller.topology, final.getPositions())
    after_geometry = structure_geometry_metrics(modeller.topology, final.getPositions())
    backbone_rmsd = coordinate_rmsd(before_backbone, after_backbone)
    contact_retention = (
        after_interface["contact_atom_pairs_lt_8A"]
        / max(1, before_interface["contact_atom_pairs_lt_8A"])
    )
    checks = {
        "finite_energy": math.isfinite(initial_energy) and math.isfinite(final_energy),
        "final_energy_negative": final_energy < 0.0,
        "backbone_preserved": backbone_rmsd <= args.max_backbone_rmsd,
        "zero_interface_severe_clashes": (
            after_interface["severe_clash_pairs_lt_1_5A"] <= args.max_interface_clashes
        ),
        "interface_retained": contact_retention >= args.min_contact_retention,
        "zero_nonlocal_heavy_atom_clashes": (
            after_geometry["nonlocal_heavy_atom_clashes_lt_1_5A"] == 0
        ),
        "ca_geometry_not_worsened": (
            after_geometry["ca_geometry_outliers"] <= before_geometry["ca_geometry_outliers"]
        ),
        "peptide_geometry_not_worsened": (
            after_geometry["peptide_cn_geometry_outliers"]
            <= before_geometry["peptide_cn_geometry_outliers"]
        ),
    }
    return {
        "input_pdb": str(input_path),
        "repacked_pdb": str(output_path),
        "method": "PDBFixer side-chain completion plus restrained OpenMM implicit-solvent minimization",
        "forcefield": "amber14-all + implicit/gbn2",
        "platform": platform.getName(),
        "max_iterations": args.max_iterations,
        "rebuilt_missing_heavy_atoms": rebuilt_atoms,
        "interface_residues": len(interface),
        "restrained_backbone_atoms": restrained_backbone,
        "restrained_noninterface_sidechain_atoms": restrained_noninterface,
        "initial_energy_kj_mol": initial_energy,
        "final_energy_kj_mol": final_energy,
        "energy_delta_kj_mol": final_energy - initial_energy,
        "energy_interpretation": (
            "Initial hydrogen-added energy is diagnostic only; pass uses finite negative final energy."
        ),
        "backbone_rmsd_angstrom": backbone_rmsd,
        "interface_contact_retention": contact_retention,
        "before_interface": before_interface,
        "after_interface": after_interface,
        "before_geometry": before_geometry,
        "after_geometry": after_geometry,
        "checks": checks,
        "status": "repack_pass" if all(checks.values()) else "repack_review",
    }


def main():
    args = parse_args()
    rows = load_fold_pass(args.fold_evidence)
    if not rows:
        raise SystemExit("No fold-passing candidates found")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for row in rows:
        construct_id = row["construct_id"]
        input_path = find_model(args.result_dir, construct_id)
        result = minimize_model(input_path, out_dir / f"{construct_id}_fv_repacked.pdb", args)
        result.update({
            "construct_id": construct_id,
            "reference_pdb": row["reference_pdb"],
            "candidate_id": row["candidate_id"],
            "fold_mean_plddt": float(row["mean_plddt"]),
            "fold_ptm": float(row["ptm"]),
            "fold_iptm": float(row["iptm"]),
        })
        results.append(result)
        print(
            f"{construct_id}: {result['status']} dE={result['energy_delta_kj_mol']:.1f} "
            f"backbone_RMSD={result['backbone_rmsd_angstrom']:.3f} "
            f"clashes={result['after_interface']['severe_clash_pairs_lt_1_5A']}",
            flush=True,
        )

    summary = {
        "schema_version": "fv.repack.v1",
        "status": "pass" if all(row["status"] == "repack_pass" for row in results) else "partial",
        "requested": len(rows),
        "completed": len(results),
        "passed": sum(row["status"] == "repack_pass" for row in results),
        "method_boundary": "restrained force-field minimization; not exhaustive Rosetta-style rotamer packing",
        "thresholds": {
            "max_backbone_rmsd": args.max_backbone_rmsd,
            "max_interface_clashes": args.max_interface_clashes,
            "min_contact_retention": args.min_contact_retention,
        },
        "records": results,
    }
    with open(out_dir / "fv_repack_audit.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(out_dir / "fv_repack_report.md", "w", encoding="utf-8") as handle:
        handle.write("# Fv Side-Chain and Interface Repack Audit\n\n")
        handle.write(f"Status: `{summary['status']}`; passed: {summary['passed']}/{len(results)}.\n\n")
        handle.write("This is restrained OpenMM minimization, not exhaustive rotamer packing.\n\n")
        handle.write("| Construct | Status | dE (kJ/mol) | Backbone RMSD | Contact retention | Interface clashes | Global clashes | CA outliers | Peptide outliers |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in results:
            handle.write(
                f"| {row['construct_id']} | {row['status']} | {row['energy_delta_kj_mol']:.1f} | "
                f"{row['backbone_rmsd_angstrom']:.3f} | {row['interface_contact_retention']:.3f} | "
                f"{row['after_interface']['severe_clash_pairs_lt_1_5A']} | "
                f"{row['after_geometry']['nonlocal_heavy_atom_clashes_lt_1_5A']} | "
                f"{row['after_geometry']['ca_geometry_outliers']} | "
                f"{row['after_geometry']['peptide_cn_geometry_outliers']} |\n"
            )


if __name__ == "__main__":
    main()
