#!/usr/bin/env python
"""Generate a restrained T1 peptide ensemble around a deposited complex pose."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from openmm import CustomExternalForce, LangevinMiddleIntegrator, Platform, unit
from openmm.app import ForceField, HBonds, Modeller, NoCutoff, PDBFile, Simulation
from pdbfixer import PDBFixer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.peptide_conformer_ensemble import (  # noqa: E402
    BACKBONE,
    antibody_aligned_rmsd,
    conformer_passes,
    contact_retention,
    ensemble_dispersion,
    nonlocal_clash_count,
    residue_contact_pairs,
)


def heavy_atom_records(topology, positions):
    coordinates = positions.value_in_unit(unit.angstrom)
    records = []
    for chain in topology.chains():
        for residue_index, residue in enumerate(chain.residues()):
            for atom in residue.atoms():
                if atom.element is None or atom.element.symbol == "H":
                    continue
                records.append({
                    "atom_index": atom.index,
                    "atom_name": atom.name,
                    "chain": chain.id,
                    "residue_index": residue_index,
                    "residue_key": (chain.id, residue.id),
                    "coord": np.asarray(coordinates[atom.index], dtype=float),
                })
    return records


def selected_coordinates(records, selector):
    selected = [record for record in records if selector(record)]
    return np.asarray([record["coord"] for record in selected], dtype=float)


def peptide_geometry(topology, positions, peptide_chain):
    coordinates = positions.value_in_unit(unit.angstrom)
    chain = next(chain for chain in topology.chains() if chain.id == peptide_chain)
    residues = list(chain.residues())
    ca_distances = []
    cn_distances = []
    for left, right in zip(residues, residues[1:]):
        left_atoms = {atom.name: atom for atom in left.atoms()}
        right_atoms = {atom.name: atom for atom in right.atoms()}
        if "CA" in left_atoms and "CA" in right_atoms:
            ca_distances.append(float(np.linalg.norm(
                np.asarray(coordinates[left_atoms["CA"].index])
                - np.asarray(coordinates[right_atoms["CA"].index]))))
        if "C" in left_atoms and "N" in right_atoms:
            cn_distances.append(float(np.linalg.norm(
                np.asarray(coordinates[left_atoms["C"].index])
                - np.asarray(coordinates[right_atoms["N"].index]))))
    return {
        "ca_geometry_outliers": sum(not 3.5 <= value <= 4.1 for value in ca_distances),
        "cn_geometry_outliers": sum(not 1.15 <= value <= 1.55 for value in cn_distances),
        "ca_distance_range": [min(ca_distances), max(ca_distances)],
        "cn_distance_range": [min(cn_distances), max(cn_distances)],
    }


def prepare_system(pdb_path, antibody_chains, peptide_chain, settings):
    fixer = PDBFixer(filename=str(Path(pdb_path).resolve()))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    rebuilt_atoms = sum(len(value) for value in fixer.missingAtoms.values()) + sum(
        len(value) for value in fixer.missingTerminals.values())
    fixer.addMissingAtoms(seed=0)
    forcefield = ForceField("amber14-all.xml", "implicit/gbn2.xml")
    modeller = Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(forcefield, pH=float(settings["ph"]))
    chains = {chain.id for chain in modeller.topology.chains()}
    expected = set(antibody_chains) | {peptide_chain}
    if not expected <= chains:
        raise ValueError(f"Prepared topology lacks required chains: {expected - chains}")
    system = forcefield.createSystem(
        modeller.topology, nonbondedMethod=NoCutoff, constraints=HBonds)
    restraint = CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.addPerParticleParameter("k")
    for parameter in ("x0", "y0", "z0"):
        restraint.addPerParticleParameter(parameter)
    coordinates = modeller.positions.value_in_unit(unit.nanometer)
    restrained_antibody = 0
    restrained_peptide = 0
    for atom in modeller.topology.atoms():
        if atom.element is None or atom.element.symbol == "H":
            continue
        stiffness = None
        if atom.residue.chain.id in antibody_chains:
            stiffness = float(settings["antibody_heavy_atom_restraint"])
            restrained_antibody += 1
        elif atom.residue.chain.id == peptide_chain and atom.name in BACKBONE:
            stiffness = float(settings["peptide_backbone_restraint"])
            restrained_peptide += 1
        if stiffness is not None:
            coordinate = coordinates[atom.index]
            restraint.addParticle(atom.index, [
                stiffness, coordinate.x, coordinate.y, coordinate.z])
    system.addForce(restraint)
    return modeller, system, {
        "rebuilt_missing_atoms": rebuilt_atoms,
        "restrained_antibody_heavy_atoms": restrained_antibody,
        "restrained_peptide_backbone_atoms": restrained_peptide,
    }


def sample_replica(modeller, system, seed, settings, output_path,
                   antibody_chains, peptide_chain, reference):
    integrator = LangevinMiddleIntegrator(
        float(settings["temperature_kelvin"]) * unit.kelvin,
        float(settings["friction_per_ps"]) / unit.picosecond,
        float(settings["timestep_fs"]) * unit.femtoseconds,
    )
    integrator.setRandomNumberSeed(int(seed))
    platform = Platform.getPlatformByName(settings["platform"])
    simulation = Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(
        tolerance=float(settings["minimization_tolerance_kj_mol_nm"])
        * unit.kilojoule_per_mole / unit.nanometer,
        maxIterations=int(settings["minimization_iterations"]))
    simulation.context.setVelocitiesToTemperature(
        float(settings["temperature_kelvin"]) * unit.kelvin, int(seed))
    simulation.step(int(settings["equilibration_steps"]))
    simulation.step(int(settings["production_steps"]))
    simulation.minimizeEnergy(
        tolerance=float(settings["minimization_tolerance_kj_mol_nm"])
        * unit.kilojoule_per_mole / unit.nanometer,
        maxIterations=int(settings["snapshot_minimization_iterations"]))
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    positions = state.getPositions()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii") as handle:
        PDBFile.writeFile(modeller.topology, positions, handle, keepIds=True)
    records = heavy_atom_records(modeller.topology, positions)
    antibody_backbone = selected_coordinates(records, lambda row: (
        row["chain"] in antibody_chains and row["atom_name"] in BACKBONE))
    peptide_backbone = selected_coordinates(records, lambda row: (
        row["chain"] == peptide_chain and row["atom_name"] in BACKBONE))
    antibody_rmsd, peptide_rmsd = antibody_aligned_rmsd(
        reference["antibody_backbone"], antibody_backbone,
        reference["peptide_backbone"], peptide_backbone)
    observed_contacts = residue_contact_pairs(
        records,
        lambda row: row["chain"] in antibody_chains,
        lambda row: row["chain"] == peptide_chain,
        cutoff=float(settings["contact_cutoff_angstrom"]),
    )
    geometry = peptide_geometry(modeller.topology, positions, peptide_chain)
    metrics = {
        "potential_energy_kj_mol": float(
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)),
        "antibody_backbone_rmsd": antibody_rmsd,
        "peptide_backbone_rmsd": peptide_rmsd,
        "peptide_internal_rmsd": float(ensemble_dispersion([
            reference["peptide_backbone"], peptide_backbone
        ])["mean_pairwise_rmsd"]),
        "native_contact_retention": contact_retention(
            reference["contacts"], observed_contacts),
        "n_native_contacts": len(reference["contacts"]),
        "n_observed_contacts": len(observed_contacts),
        "nonlocal_clashes_lt_1_5A": nonlocal_clash_count(records),
        "reference_ca_geometry_outliers": reference["geometry"]["ca_geometry_outliers"],
        "reference_cn_geometry_outliers": reference["geometry"]["cn_geometry_outliers"],
        **geometry,
    }
    checks, passed = conformer_passes(metrics, settings["quality_control"])
    return {
        "seed": int(seed),
        "pdb": str(output_path),
        "metrics": metrics,
        "checks": checks,
        "status": "accepted" if passed else "rejected",
        "peptide_backbone": peptide_backbone,
    }


def generate_ensemble(pdb_path, output_dir, config, record_id="target"):
    settings = config["sampling"]
    antibody_chains = list(config["chains"]["antibody"])
    peptide_chain = config["chains"]["peptide"]
    modeller, system, preparation = prepare_system(
        pdb_path, antibody_chains, peptide_chain, settings)
    reference_records = heavy_atom_records(modeller.topology, modeller.positions)
    reference = {
        "antibody_backbone": selected_coordinates(reference_records, lambda row: (
            row["chain"] in antibody_chains and row["atom_name"] in BACKBONE)),
        "peptide_backbone": selected_coordinates(reference_records, lambda row: (
            row["chain"] == peptide_chain and row["atom_name"] in BACKBONE)),
        "contacts": residue_contact_pairs(
            reference_records,
            lambda row: row["chain"] in antibody_chains,
            lambda row: row["chain"] == peptide_chain,
            cutoff=float(settings["contact_cutoff_angstrom"])),
        "geometry": peptide_geometry(
            modeller.topology, modeller.positions, peptide_chain),
    }
    if not reference["contacts"]:
        raise ValueError("Prepared reference contains no antibody-peptide contacts")
    output_dir = Path(output_dir)
    conformers = []
    for seed in config["sampling"]["seeds"]:
        replica_path = output_dir / "replicas" / f"seed_{seed}.json"
        if replica_path.exists():
            result = json.loads(replica_path.read_text(encoding="utf-8"))
        else:
            result = sample_replica(
                modeller, system, seed, settings,
                output_dir / "conformers" / f"{record_id}_seed_{seed}.pdb",
                antibody_chains, peptide_chain, reference)
            serializable = {key: value for key, value in result.items()
                            if key != "peptide_backbone"}
            replica_path.parent.mkdir(parents=True, exist_ok=True)
            replica_path.write_text(
                json.dumps(serializable, indent=2) + "\n", encoding="ascii")
        conformers.append(result)
        print(f"{record_id} seed {seed}: {result['status']}", flush=True)
    accepted_coordinates = []
    for result in conformers:
        if result["status"] != "accepted":
            continue
        if "peptide_backbone" in result:
            accepted_coordinates.append(result["peptide_backbone"])
        else:
            cached = PDBFile(str(Path(result["pdb"]).resolve()))
            cached_records = heavy_atom_records(cached.topology, cached.positions)
            accepted_coordinates.append(selected_coordinates(
                cached_records, lambda row: (
                    row["chain"] == peptide_chain and row["atom_name"] in BACKBONE)))
    for result in conformers:
        result.pop("peptide_backbone", None)
    output = {
        "schema_version": 1,
        "ensemble_class": "T1 deposited-pose local neighborhood",
        "claim_boundary": (
            "Computational local conformational dispersion around a deposited pose; "
            "not an equilibrium ensemble or blind conformer prediction."),
        "record_id": record_id,
        "source_pdb": str(pdb_path),
        "preparation": preparation,
        "attempted": len(conformers),
        "accepted": len(accepted_coordinates),
        "acceptance_fraction": len(accepted_coordinates) / len(conformers),
        "dispersion": ensemble_dispersion(accepted_coordinates),
        "conformers": conformers,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ensemble_audit.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="ascii")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--record-id", default="target")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = generate_ensemble(args.pdb, args.out_dir, config, args.record_id)
    print(json.dumps({
        "attempted": output["attempted"], "accepted": output["accepted"],
        "dispersion": output["dispersion"], "out_dir": args.out_dir,
    }, indent=2))


if __name__ == "__main__":
    main()
