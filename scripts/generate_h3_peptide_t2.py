#!/usr/bin/env python
"""Generate T2 moderate-perturbation peptide recovery ensembles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from openmm import CustomBondForce, CustomExternalForce, LangevinMiddleIntegrator, Platform, unit
from openmm.app import ForceField, HBonds, Modeller, NoCutoff, PDBFile, Simulation
from pdbfixer import PDBFixer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.peptide_conformer_ensemble import (  # noqa: E402
    BACKBONE, antibody_aligned_rmsd, contact_retention, nonlocal_clash_count,
    residue_contact_pairs,
)
from modules.peptide_t2_recovery import recovery_metrics, split_contact_pairs  # noqa: E402
from scripts.generate_h3_peptide_ensemble import (  # noqa: E402
    heavy_atom_records, peptide_geometry, selected_coordinates,
)


def prepare(pdb_path, settings):
    fixer = PDBFixer(filename=str(Path(pdb_path).resolve()))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=int(settings["pdbfixer_seed"]))
    forcefield = ForceField("amber14-all.xml", "implicit/gbn2.xml")
    modeller = Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(forcefield, pH=float(settings["ph"]))
    return modeller, forcefield


def representative_atoms(topology):
    lookup = {}
    for chain in topology.chains():
        for residue in chain.residues():
            atoms = {atom.name: atom.index for atom in residue.atoms()}
            if "CA" in atoms:
                lookup[(chain.id, residue.id)] = atoms["CA"]
    return lookup


def build_system(modeller, forcefield, antibody_chains, supplied_pairs,
                 settings, perturbation=False):
    system = forcefield.createSystem(
        modeller.topology, nonbondedMethod=NoCutoff, constraints=HBonds)
    coordinates = modeller.positions.value_in_unit(unit.nanometer)
    antibody_force = CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    antibody_force.addPerParticleParameter("k")
    for parameter in ("x0", "y0", "z0"):
        antibody_force.addPerParticleParameter(parameter)
    for atom in modeller.topology.atoms():
        if atom.element is None or atom.element.symbol == "H":
            continue
        if atom.residue.chain.id in antibody_chains:
            xyz = coordinates[atom.index]
            antibody_force.addParticle(atom.index, [
                float(settings["antibody_heavy_atom_restraint"]), xyz.x, xyz.y, xyz.z])
    system.addForce(antibody_force)

    atom_lookup = representative_atoms(modeller.topology)
    selected_pairs = sorted(supplied_pairs, key=repr)
    if perturbation:
        selected_pairs = selected_pairs[:1]
        upper_bound = float(settings["perturbation_anchor_upper_bound_nm"])
        stiffness = float(settings["perturbation_anchor_stiffness"])
    else:
        upper_bound = float(settings["recovery_contact_upper_bound_nm"])
        stiffness = float(settings["recovery_contact_stiffness"])
    restraint = CustomBondForce("0.5*k*step(r-rmax)*(r-rmax)^2")
    restraint.addPerBondParameter("k")
    restraint.addPerBondParameter("rmax")
    added = 0
    for antibody_key, peptide_key in selected_pairs:
        if antibody_key not in atom_lookup or peptide_key not in atom_lookup:
            continue
        restraint.addBond(
            atom_lookup[antibody_key], atom_lookup[peptide_key],
            [stiffness, upper_bound])
        added += 1
    if added == 0:
        raise ValueError("No residue-level T2 restraints could be instantiated")
    system.addForce(restraint)
    return system, added


def run_stage(modeller, system, positions, seed, temperature, steps, settings):
    integrator = LangevinMiddleIntegrator(
        float(temperature) * unit.kelvin,
        float(settings["friction_per_ps"]) / unit.picosecond,
        float(settings["timestep_fs"]) * unit.femtoseconds)
    integrator.setRandomNumberSeed(int(seed))
    platform = Platform.getPlatformByName(settings["platform"])
    simulation = Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy(
        tolerance=float(settings["minimization_tolerance_kj_mol_nm"])
        * unit.kilojoule_per_mole / unit.nanometer,
        maxIterations=int(settings["minimization_iterations"]))
    simulation.context.setVelocitiesToTemperature(float(temperature) * unit.kelvin, int(seed))
    simulation.step(int(steps))
    simulation.minimizeEnergy(
        tolerance=float(settings["minimization_tolerance_kj_mol_nm"])
        * unit.kilojoule_per_mole / unit.nanometer,
        maxIterations=int(settings["snapshot_minimization_iterations"]))
    return simulation.context.getState(getEnergy=True, getPositions=True)


def frame_metrics(topology, positions, reference, antibody_chains, peptide_chain,
                  all_contacts, held_out_contacts, settings):
    records = heavy_atom_records(topology, positions)
    antibody = selected_coordinates(records, lambda row: (
        row["chain"] in antibody_chains and row["atom_name"] in BACKBONE))
    peptide = selected_coordinates(records, lambda row: (
        row["chain"] == peptide_chain and row["atom_name"] in BACKBONE))
    antibody_rmsd, peptide_rmsd = antibody_aligned_rmsd(
        reference["antibody"], antibody, reference["peptide"], peptide)
    observed = residue_contact_pairs(
        records, lambda row: (
            row["chain"] in antibody_chains and row["atom_name"] == "CA"),
        lambda row: row["chain"] == peptide_chain and row["atom_name"] == "CA",
        cutoff=float(settings["contact_cutoff_angstrom"]))
    geometry = peptide_geometry(topology, positions, peptide_chain)
    return {
        "antibody_backbone_rmsd": antibody_rmsd,
        "peptide_backbone_rmsd": peptide_rmsd,
        "all_contact_retention": contact_retention(all_contacts, observed),
        "held_out_contact_retention": contact_retention(held_out_contacts, observed),
        "nonlocal_clashes_lt_1_5A": nonlocal_clash_count(records),
        **geometry,
    }


def generate_t2(pdb_path, output_dir, config, record_id):
    settings = config["sampling"]
    antibody_chains = list(config["chains"]["antibody"])
    peptide_chain = config["chains"]["peptide"]
    source = PDBFile(str(Path(pdb_path).resolve()))
    source_records = heavy_atom_records(source.topology, source.positions)
    all_contacts = residue_contact_pairs(
        source_records,
        lambda row: row["chain"] in antibody_chains and row["atom_name"] == "CA",
        lambda row: row["chain"] == peptide_chain and row["atom_name"] == "CA",
        cutoff=float(settings["contact_cutoff_angstrom"]))
    modeller, forcefield = prepare(pdb_path, settings)
    native_records = heavy_atom_records(modeller.topology, modeller.positions)
    supplied, held_out = split_contact_pairs(
        all_contacts, record_id, float(settings["supplied_contact_fraction"]))
    reference = {
        "antibody": selected_coordinates(native_records, lambda row: (
            row["chain"] in antibody_chains and row["atom_name"] in BACKBONE)),
        "peptide": selected_coordinates(native_records, lambda row: (
            row["chain"] == peptide_chain and row["atom_name"] in BACKBONE)),
        "geometry": peptide_geometry(modeller.topology, modeller.positions, peptide_chain),
    }
    perturb_system, perturb_restraints = build_system(
        modeller, forcefield, antibody_chains, supplied, settings, perturbation=True)
    recovery_system, recovery_restraints = build_system(
        modeller, forcefield, antibody_chains, supplied, settings, perturbation=False)
    output_dir = Path(output_dir)
    replicas = []
    for seed in settings["seeds"]:
        cache = output_dir / "replicas" / f"seed_{seed}.json"
        if cache.exists():
            replicas.append(json.loads(cache.read_text(encoding="utf-8")))
            continue
        try:
            perturbed = run_stage(
                modeller, perturb_system, modeller.positions, int(seed),
                float(settings["perturbation_temperature_kelvin"]),
                int(settings["perturbation_steps"]), settings)
        except Exception as error:  # noqa: BLE001
            result = {
                "seed": int(seed), "initial_in_t2_tier": False,
                "initial_pdb": None, "initial_metrics": None,
                "final_energy_kj_mol": None, "final_metrics": None,
                "recovery": None, "status": "rejected", "pdb": None,
                "error": f"perturbation_stage_failed: {error}",
            }
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
            replicas.append(result)
            print(f"{record_id} seed {seed}: rejected", flush=True)
            continue
        initial_metrics = frame_metrics(
            modeller.topology, perturbed.getPositions(), reference, antibody_chains,
            peptide_chain, all_contacts, held_out, settings)
        initial_pdb = output_dir / "perturbed" / f"{record_id}_seed_{seed}.pdb"
        initial_pdb.parent.mkdir(parents=True, exist_ok=True)
        with initial_pdb.open("w", encoding="ascii") as handle:
            PDBFile.writeFile(
                modeller.topology, perturbed.getPositions(), handle, keepIds=True)
        in_tier = (
            float(settings["initial_rmsd_min_angstrom"])
            <= initial_metrics["peptide_backbone_rmsd"]
            <= float(settings["initial_rmsd_max_angstrom"]))
        if in_tier:
            try:
                recovered = run_stage(
                    modeller, recovery_system, perturbed.getPositions(), int(seed) + 10000,
                    float(settings["recovery_temperature_kelvin"]),
                    int(settings["recovery_steps"]), settings)
                final_positions = recovered.getPositions()
                final_energy = float(recovered.getPotentialEnergy().value_in_unit(
                    unit.kilojoule_per_mole))
                final_metrics = frame_metrics(
                    modeller.topology, final_positions, reference, antibody_chains,
                    peptide_chain, all_contacts, held_out, settings)
                recovery = recovery_metrics(
                    initial_metrics["peptide_backbone_rmsd"],
                    final_metrics["peptide_backbone_rmsd"],
                    initial_metrics["held_out_contact_retention"],
                    final_metrics["held_out_contact_retention"])
                accepted = bool(
                    np.isfinite(final_energy)
                    and final_metrics["antibody_backbone_rmsd"] <= float(
                        settings["quality_control"]["max_antibody_rmsd"])
                    and final_metrics["peptide_backbone_rmsd"] <= float(
                        settings["quality_control"]["max_final_peptide_rmsd"])
                    and final_metrics["nonlocal_clashes_lt_1_5A"] == 0
                    and final_metrics["ca_geometry_outliers"] <= reference["geometry"][
                        "ca_geometry_outliers"]
                    and final_metrics["cn_geometry_outliers"] <= reference["geometry"][
                        "cn_geometry_outliers"])
                output_pdb = output_dir / "conformers" / f"{record_id}_seed_{seed}.pdb"
                output_pdb.parent.mkdir(parents=True, exist_ok=True)
                with output_pdb.open("w", encoding="ascii") as handle:
                    PDBFile.writeFile(modeller.topology, final_positions, handle, keepIds=True)
                recovery_error = None
            except Exception as error:  # noqa: BLE001
                final_energy = None
                final_metrics = None
                recovery = None
                accepted = False
                output_pdb = None
                recovery_error = f"recovery_stage_failed: {error}"
        else:
            final_energy = None
            final_metrics = None
            recovery = None
            accepted = False
            output_pdb = None
            recovery_error = None
        result = {
            "seed": int(seed), "initial_in_t2_tier": in_tier,
            "initial_pdb": str(initial_pdb),
            "initial_metrics": initial_metrics,
            "final_energy_kj_mol": final_energy,
            "final_metrics": final_metrics, "recovery": recovery,
            "status": "accepted" if accepted else "rejected",
            "pdb": str(output_pdb) if output_pdb else None,
            "error": recovery_error,
        }
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
        replicas.append(result)
        print(f"{record_id} seed {seed}: {result['status']}", flush=True)
    accepted = [row for row in replicas if row["status"] == "accepted"]
    output = {
        "schema_version": 1,
        "ensemble_class": "T2 moderate perturbation with broad residue-contact restraints",
        "record_id": record_id, "source_pdb": str(pdb_path),
        "n_native_contacts": len(all_contacts),
        "n_supplied_contacts": len(supplied),
        "n_held_out_contacts": len(held_out),
        "supplied_contacts": sorted(supplied, key=repr),
        "held_out_contacts": sorted(held_out, key=repr),
        "perturbation_restraints": perturb_restraints,
        "recovery_restraints": recovery_restraints,
        "attempted": len(replicas), "accepted": len(accepted),
        "acceptance_fraction": len(accepted) / len(replicas),
        "replicas": replicas,
        "claim_boundary": (
            "Local recovery from moderate perturbations using native-derived broad "
            "residue contacts; not blind docking or equilibrium sampling."),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "t2_audit.json").write_text(
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
    output = generate_t2(args.pdb, args.out_dir, config, args.record_id)
    print(json.dumps({
        "attempted": output["attempted"], "accepted": output["accepted"],
        "supplied": output["n_supplied_contacts"],
        "held_out": output["n_held_out_contacts"],
    }, indent=2))


if __name__ == "__main__":
    main()
