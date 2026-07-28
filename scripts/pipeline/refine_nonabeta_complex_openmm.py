#!/usr/bin/env python
"""Restrained local induced-fit minimization of pose-conditioned VH:VL:IDP models."""
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


def io_path(path):
    resolved = str(Path(path).resolve())
    return f"\\\\?\\{resolved}" if len(resolved) >= 248 else resolved


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", default="outputs/non_abeta_idp_initial_guess_evidence_v1/initial_guess_evidence.csv")
    parser.add_argument("--out", default="outputs/non_abeta_idp_complex_refinement_v1")
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--interface-cutoff", type=float, default=8.0)
    parser.add_argument("--framework-restraint", type=float, default=1000.0)
    parser.add_argument("--interface-backbone-restraint", type=float, default=25.0)
    parser.add_argument("--platform", choices=["CPU", "OpenCL"], default="CPU")
    return parser.parse_args()


def heavy_atoms(topology, positions):
    xyz = positions.value_in_unit(unit.angstrom)
    return [
        (atom, np.asarray(xyz[atom.index], dtype=float))
        for atom in topology.atoms()
        if atom.element is not None and atom.element.symbol != "H"
    ]


def interface_residues(topology, positions, cutoff):
    records = heavy_atoms(topology, positions)
    antibody = [(atom, xyz) for atom, xyz in records if atom.residue.chain.id in {"A", "B"}]
    antigen = [(atom, xyz) for atom, xyz in records if atom.residue.chain.id == "C"]
    distances = np.linalg.norm(
        np.asarray([xyz for _, xyz in antibody])[:, None, :]
        - np.asarray([xyz for _, xyz in antigen])[None, :, :], axis=-1,
    )
    left, right = np.where(distances <= cutoff)
    residues = {(antibody[index][0].residue.chain.id, antibody[index][0].residue.id) for index in left}
    residues.update((antigen[index][0].residue.chain.id, antigen[index][0].residue.id) for index in right)
    return residues


def interface_metrics(topology, positions):
    records = heavy_atoms(topology, positions)
    antibody = [xyz for atom, xyz in records if atom.residue.chain.id in {"A", "B"}]
    antigen = [xyz for atom, xyz in records if atom.residue.chain.id == "C"]
    distances = np.linalg.norm(np.asarray(antibody)[:, None, :] - np.asarray(antigen)[None, :, :], axis=-1)
    return {
        "minimum_heavy_atom_distance": float(distances.min()),
        "severe_clash_pairs_lt_1_5A": int((distances < 1.5).sum()),
        "close_atom_pairs_lt_5A": int((distances < 5.0).sum()),
        "contact_atom_pairs_lt_8A": int((distances < 8.0).sum()),
    }


def coordinates(topology, positions, selected):
    xyz = positions.value_in_unit(unit.angstrom)
    return {
        (atom.residue.chain.id, atom.residue.id, atom.name): np.asarray(xyz[atom.index], dtype=float)
        for atom in topology.atoms() if selected(atom)
    }


def coordinate_rmsd(before, after):
    keys = sorted(set(before) & set(after))
    delta = np.asarray([after[key] - before[key] for key in keys])
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def model_path(score_json):
    score_path = Path(score_json)
    prefix, ranked = score_path.name.split("_scores_rank_", 1)
    rank, suffix = ranked.split("_", 1)
    return score_path.parent / f"{prefix}_unrelaxed_rank_{rank}_{suffix.replace('.json', '.pdb')}"


def minimize(input_path, output_path, args):
    original = PDBFile(io_path(input_path))
    before_metrics = interface_metrics(original.topology, original.positions)
    before_backbone = coordinates(original.topology, original.positions, lambda atom: atom.name in BACKBONE)

    fixer = PDBFixer(filename=io_path(input_path))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=0)
    forcefield = ForceField("amber14-all.xml", "implicit/gbn2.xml")
    modeller = Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(forcefield, pH=7.4)
    interface = interface_residues(modeller.topology, modeller.positions, args.interface_cutoff)
    before_interface_backbone = coordinates(
        modeller.topology, modeller.positions,
        lambda atom: atom.name in BACKBONE and (atom.residue.chain.id, atom.residue.id) in interface,
    )

    system = forcefield.createSystem(modeller.topology, nonbondedMethod=NoCutoff, constraints=HBonds)
    restraint = CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.addPerParticleParameter("k")
    for parameter in ("x0", "y0", "z0"):
        restraint.addPerParticleParameter(parameter)
    xyz_nm = modeller.positions.value_in_unit(unit.nanometer)
    for atom in modeller.topology.atoms():
        if atom.element is None or atom.element.symbol == "H":
            continue
        key = (atom.residue.chain.id, atom.residue.id)
        stiffness = None
        if key in interface and atom.name in BACKBONE:
            stiffness = args.interface_backbone_restraint
        elif key not in interface:
            stiffness = args.framework_restraint
        if stiffness is not None:
            xyz = xyz_nm[atom.index]
            restraint.addParticle(atom.index, [stiffness, xyz.x, xyz.y, xyz.z])
    system.addForce(restraint)
    integrator = LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    platform = Platform.getPlatformByName(args.platform)
    properties = {"Precision": "mixed"} if args.platform == "OpenCL" else {}
    simulation = Simulation(modeller.topology, system, integrator, platform, properties)
    simulation.context.setPositions(modeller.positions)
    initial = simulation.context.getState(getEnergy=True)
    simulation.minimizeEnergy(
        tolerance=10 * unit.kilojoule_per_mole / unit.nanometer,
        maxIterations=args.max_iterations,
    )
    final = simulation.context.getState(getEnergy=True, getPositions=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="ascii") as handle:
        PDBFile.writeFile(modeller.topology, final.getPositions(), handle, keepIds=True)
    after_metrics = interface_metrics(modeller.topology, final.getPositions())
    after_backbone = coordinates(modeller.topology, final.getPositions(), lambda atom: atom.name in BACKBONE)
    after_interface_backbone = coordinates(
        modeller.topology, final.getPositions(),
        lambda atom: atom.name in BACKBONE and (atom.residue.chain.id, atom.residue.id) in interface,
    )
    return {
        "initial_energy_kj_mol": initial.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
        "final_energy_kj_mol": final.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
        "interface_residues": len(interface),
        "global_backbone_rmsd": coordinate_rmsd(before_backbone, after_backbone),
        "interface_backbone_rmsd": coordinate_rmsd(before_interface_backbone, after_interface_backbone),
        "contact_retention": after_metrics["contact_atom_pairs_lt_8A"] / max(before_metrics["contact_atom_pairs_lt_8A"], 1),
        **{f"before_{key}": value for key, value in before_metrics.items()},
        **{f"after_{key}": value for key, value in after_metrics.items()},
    }


def main():
    args = parse_args()
    project = Path(__file__).resolve().parents[2]
    with open(project / args.evidence, newline="", encoding="utf-8") as handle:
        evidence = list(csv.DictReader(handle))
    out = project / args.out
    rows = []
    for index, source in enumerate(evidence, 1):
        input_path = project / model_path(source["score_json"])
        output_path = out / "models" / source["construct_id"] / f"conformer{source['conformer']}_seed{source['seed']}.pdb"
        print(f"[{index}/{len(evidence)}] {source['construct_id']} c{source['conformer']} s{source['seed']}", flush=True)
        metrics = minimize(input_path, output_path, args)
        passed = (
            math.isfinite(metrics["final_energy_kj_mol"])
            and metrics["final_energy_kj_mol"] < metrics["initial_energy_kj_mol"]
            and metrics["after_severe_clash_pairs_lt_1_5A"] == 0
            and metrics["contact_retention"] >= 0.75
            and metrics["global_backbone_rmsd"] <= 0.75
        )
        rows.append({
            "construct_id": source["construct_id"], "target": source["target"],
            "construct_type": source["construct_type"], "conformer": source["conformer"],
            "seed": source["seed"], **{key: round(value, 5) if isinstance(value, float) else value for key, value in metrics.items()},
            "refinement_status": "refinement_pass" if passed else "refinement_review",
            "refined_pdb": output_path.relative_to(project).as_posix(),
        })
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "complex_refinement_audit.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "schema_version": "nonabeta.complex_refinement.v1",
        "models": len(rows),
        "passing_models": sum(row["refinement_status"] == "refinement_pass" for row in rows),
        "status": "pass" if all(row["refinement_status"] == "refinement_pass" for row in rows) else "partial",
        "method_boundary": "Restrained local OpenMM minimization; not exhaustive backbone or VH/VL orientation sampling.",
    }
    with open(out / "complex_refinement_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Complex refinement: {summary}")


if __name__ == "__main__":
    main()
