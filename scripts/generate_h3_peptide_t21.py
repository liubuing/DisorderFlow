#!/usr/bin/env python
"""T2.1: Deterministic torsion perturbation recovery with 4-arm controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import random as _random
import sys
from pathlib import Path

import numpy as np
from openmm import CustomBondForce, CustomExternalForce, LangevinMiddleIntegrator, Platform, unit
from openmm.app import CutoffNonPeriodic, ForceField, HBonds, Modeller, PDBFile, Simulation
from pdbfixer import PDBFixer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.peptide_conformer_ensemble import (  # noqa: E402
    BACKBONE, antibody_aligned_rmsd, contact_retention, nonlocal_clash_count,
    residue_contact_pairs,
)
from modules.peptide_t2_recovery import split_contact_pairs  # noqa: E402
from modules.peptide_torsion_perturb import search_target_rmsd  # noqa: E402
from scripts.generate_h3_peptide_ensemble import (  # noqa: E402
    heavy_atom_records, peptide_geometry, selected_coordinates,
)

_AA3_TO_1 = {v: k for k, v in {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}.items()}
CUTOFF = 1.0
CUTOFF_UNIT = CUTOFF * unit.nanometer


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


def _base_system(modeller, forcefield):
    return forcefield.createSystem(
        modeller.topology, nonbondedMethod=CutoffNonPeriodic,
        nonbondedCutoff=CUTOFF_UNIT, constraints=HBonds)


def _add_antibody_restraint(system, modeller, antibody_chains, settings):
    coords = modeller.positions.value_in_unit(unit.nanometer)
    force = CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    for param in ("k", "x0", "y0", "z0"):
        force.addPerParticleParameter(param)
    stiffness = float(settings["antibody_heavy_atom_restraint"])
    for atom in modeller.topology.atoms():
        if atom.element is None or atom.element.symbol == "H":
            continue
        if atom.residue.chain.id in antibody_chains:
            xyz = coords[atom.index]
            force.addParticle(atom.index, [stiffness, xyz.x, xyz.y, xyz.z])
    system.addForce(force)


def _add_contact_restraint(system, modeller, contact_pairs, stiffness, upper_bound):
    atom_map = {}
    for chain in modeller.topology.chains():
        for residue in chain.residues():
            atoms = {a.name: a.index for a in residue.atoms()}
            if "CA" in atoms:
                atom_map[(chain.id, residue.id)] = atoms["CA"]
    restraint = CustomBondForce("0.5*k*step(r-rmax)*(r-rmax)^2")
    restraint.addPerBondParameter("k")
    restraint.addPerBondParameter("rmax")
    added = 0
    for ab_key, pep_key in sorted(contact_pairs, key=repr):
        if ab_key not in atom_map or pep_key not in atom_map:
            continue
        restraint.addBond(atom_map[ab_key], atom_map[pep_key],
                          [float(stiffness), float(upper_bound)])
        added += 1
    if added == 0:
        raise ValueError("No contact restraints instantiated")
    system.addForce(restraint)
    return added


def _add_peptide_position_restraint(system, modeller, peptide_chain, atom_indices, stiffness):
    coords = modeller.positions.value_in_unit(unit.nanometer)
    force = CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    for param in ("k", "x0", "y0", "z0"):
        force.addPerParticleParameter(param)
    for idx in atom_indices:
        xyz = coords[idx]
        force.addParticle(idx, [float(stiffness), xyz.x, xyz.y, xyz.z])
    system.addForce(force)
    return len(atom_indices)


def _peptide_ca_atoms(modeller, peptide_chain):
    result = []
    for atom in modeller.topology.atoms():
        if atom.element is not None and atom.element.symbol != "H":
            if atom.residue.chain.id == peptide_chain and atom.name == "CA":
                result.append(atom.index)
    return result


def _all_heavy_atoms(modeller):
    return [atom.index for atom in modeller.topology.atoms()
            if atom.element is not None and atom.element.symbol != "H"]


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
    antibody = selected_coordinates(records, lambda r: (
        r["chain"] in antibody_chains and r["atom_name"] in BACKBONE))
    peptide = selected_coordinates(records, lambda r: (
        r["chain"] == peptide_chain and r["atom_name"] in BACKBONE))
    antibody_rmsd, peptide_rmsd = antibody_aligned_rmsd(
        reference["antibody"], antibody, reference["peptide"], peptide)
    observed = residue_contact_pairs(
        records,
        lambda r: r["chain"] in antibody_chains and r["atom_name"] == "CA",
        lambda r: r["chain"] == peptide_chain and r["atom_name"] == "CA",
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


def check_accepted(final_energy, metrics, reference, settings):
    qc = settings["quality_control"]
    if not np.isfinite(final_energy):
        return False
    if metrics["antibody_backbone_rmsd"] > float(qc["max_antibody_rmsd"]):
        return False
    if metrics["peptide_backbone_rmsd"] > float(qc["max_final_peptide_rmsd"]):
        return False
    if metrics["nonlocal_clashes_lt_1_5A"] != 0:
        return False
    if not qc.get("relax_geometry_outlier_comparison", False):
        ref_geo = reference["geometry"]
        if metrics["ca_geometry_outliers"] > ref_geo["ca_geometry_outliers"]:
            return False
        if metrics["cn_geometry_outliers"] > ref_geo["cn_geometry_outliers"]:
            return False
    return True


def run_recovery_arm(modeller, forcefield, positions, recovery_system, reference,
                     antibody_chains, peptide_chain, all_contacts, held_out_contacts,
                     seed, settings, arm_label, record_id):
    results = []
    for idx, arm_seed in enumerate(settings["seeds"]):
        label = f"[{record_id}] {arm_label} seed {arm_seed} ({idx + 1}/{len(settings['seeds'])})"
        print(f"  {label}", flush=True)
        try:
            state = run_stage(
                modeller, recovery_system, positions, int(arm_seed) + 10000,
                float(settings["recovery_temperature_kelvin"]),
                int(settings["recovery_steps"]), settings)
            final_positions = state.getPositions()
            final_energy = float(state.getPotentialEnergy().value_in_unit(
                unit.kilojoule_per_mole))
            final_metrics = frame_metrics(
                modeller.topology, final_positions, reference, antibody_chains,
                peptide_chain, all_contacts, held_out_contacts, settings)
            accepted = check_accepted(final_energy, final_metrics, reference, settings)
            results.append({
                "arm": arm_label, "seed": int(arm_seed), "accepted": accepted,
                "final_energy_kj_mol": final_energy, "final_metrics": final_metrics,
            })
        except Exception as error:  # noqa: BLE001
            results.append({
                "arm": arm_label, "seed": int(arm_seed), "accepted": False,
                "final_energy_kj_mol": None, "final_metrics": None,
                "error": str(error),
            })
    return results


def _format_arm(arm_results):
    if isinstance(arm_results, list) and arm_results and isinstance(arm_results[0], dict):
        accepted = [row for row in arm_results if row.get("accepted")]
        return {
            "attempted": len(arm_results), "accepted": len(accepted),
            "acceptance_fraction": len(accepted) / max(1, len(arm_results)),
            "replicas": arm_results,
        }
    return arm_results


def generate_t2_1(pdb_path, output_dir, config, record_id, model, antibody_chains, peptide_chain):
    settings = config["sampling"]
    tp = config["torsion_perturbation"]
    modeller, forcefield = prepare(pdb_path, settings)
    native_records = heavy_atom_records(modeller.topology, modeller.positions)

    all_contacts = residue_contact_pairs(
        native_records,
        lambda r: r["chain"] in antibody_chains and r["atom_name"] == "CA",
        lambda r: r["chain"] == peptide_chain and r["atom_name"] == "CA",
        cutoff=float(settings["contact_cutoff_angstrom"]))
    supplied, held_out = split_contact_pairs(
        all_contacts, record_id, float(settings["supplied_contact_fraction"]))

    reference = {
        "antibody": selected_coordinates(native_records, lambda r: (
            r["chain"] in antibody_chains and r["atom_name"] in BACKBONE)),
        "peptide": selected_coordinates(native_records, lambda r: (
            r["chain"] == peptide_chain and r["atom_name"] in BACKBONE)),
        "geometry": peptide_geometry(modeller.topology, modeller.positions, peptide_chain),
    }

    chain = model[peptide_chain]
    sequence = "".join(
        _AA3_TO_1.get(residue.get_resname(), "X") for residue in chain
        if residue.id[0] == " " and "CA" in residue)
    seed = int(hashlib.sha256(
        f"{record_id}|{tp['perturbation_seed_prefix']}".encode()).hexdigest()[:8], 16)
    selected, trace = search_target_rmsd(
        model, antibody_chains, peptide_chain, sequence, seed,
        target_min=float(tp["rmsd_tier_min_angstrom"]),
        target_max=float(tp["rmsd_tier_max_angstrom"]),
        target_center=float(tp["rmsd_target_center"]),
        maximum_scale_degrees=float(tp["maximum_scale_degrees"]),
        grid_step_degrees=float(tp["grid_step_degrees"]))

    out = Path(output_dir)
    base = {
        "schema_version": 1,
        "ensemble_class": "T2.1 deterministic torsion perturbation with controls",
        "record_id": record_id,
        "torsion_hit_tier": selected is not None,
        "torsion_trace": trace,
        "n_native_contacts": len(all_contacts),
        "n_supplied_contacts": len(supplied),
        "n_held_out_contacts": len(held_out),
        "supplied_contacts": sorted(supplied, key=repr),
        "held_out_contacts": sorted(held_out, key=repr),
        "arms": {},
    }

    if selected is None:
        return base

    base.update({
        "torsion_scale_degrees": selected["scale_degrees"],
        "initial_antibody_rmsd": selected["antibody_rmsd"],
        "initial_peptide_rmsd": selected["peptide_rmsd"],
        "torsion_applied": selected["applied"],
    })

    perturbed_pdb = out / "perturbed" / f"{record_id}.pdb"
    perturbed_pdb.parent.mkdir(parents=True, exist_ok=True)
    from Bio.PDB import PDBIO
    io = PDBIO()
    io.set_structure(selected["model"])
    io.save(str(perturbed_pdb))

    perturbed_modeller, _ = prepare(perturbed_pdb, settings)
    perturbed_positions = perturbed_modeller.positions
    base["initial_metrics"] = frame_metrics(
        perturbed_modeller.topology, perturbed_positions, reference,
        antibody_chains, peptide_chain, all_contacts, held_out, settings)

    def _arm(label, system_fn):
        try:
            system = system_fn()
            results = run_recovery_arm(
                perturbed_modeller, forcefield, perturbed_positions, system, reference,
                antibody_chains, peptide_chain, all_contacts, held_out,
                seed, settings, label, record_id)
            base["arms"][label] = results
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            base["arms"][label] = {"error": str(exc), "attempted": 0, "accepted": 0}

    _arm("supplied_contacts", lambda: _make_supplied_system(
        perturbed_modeller, forcefield, antibody_chains, supplied, settings))
    _arm("all_contacts", lambda: _make_supplied_system(
        perturbed_modeller, forcefield, antibody_chains, all_contacts, settings))
    _arm("random_restraints", lambda: _make_random_system(
        perturbed_modeller, forcefield, antibody_chains, peptide_chain,
        len(supplied), seed, settings))
    _arm("null_structural", lambda: _make_null_system(
        perturbed_modeller, forcefield, antibody_chains, settings))

    for key in list(base["arms"].keys()):
        base["arms"][key] = _format_arm(base["arms"][key])

    return base


def _make_supplied_system(modeller, forcefield, antibody_chains, contact_pairs, settings):
    system = _base_system(modeller, forcefield)
    _add_antibody_restraint(system, modeller, antibody_chains, settings)
    _add_contact_restraint(
        system, modeller, contact_pairs,
        settings["recovery_contact_stiffness"],
        settings["recovery_contact_upper_bound_nm"])
    return system


def _make_random_system(modeller, forcefield, antibody_chains, peptide_chain,
                        count, seed, settings):
    system = _base_system(modeller, forcefield)
    _add_antibody_restraint(system, modeller, antibody_chains, settings)
    peptide_atoms = _peptide_ca_atoms(modeller, peptide_chain)
    rng = _random.Random(int(seed) + 7777)
    selected = rng.sample(peptide_atoms, min(count, len(peptide_atoms)))
    _add_peptide_position_restraint(
        system, modeller, peptide_chain, selected,
        settings.get("peptide_backbone_restraint", 25.0))
    return system


def _make_null_system(modeller, forcefield, antibody_chains, settings):
    system = _base_system(modeller, forcefield)
    _add_antibody_restraint(system, modeller, antibody_chains, settings)
    return system


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--record-id", default="target")
    args = parser.parse_args()
    import yaml
    from Bio.PDB import PDBParser
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    bp = PDBParser(QUIET=True)
    model = bp.get_structure("complex", args.pdb)[0]
    ab = [c for c in config["chains"]["antibody"] if c in {ch.id for ch in model}]
    result = generate_t2_1(
        args.pdb, args.out_dir, config, args.record_id, model,
        ab, config["chains"]["peptide"])
    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    (od / "t2.1_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    summary = {k: {"accepted": v["accepted"], "attempted": v["attempted"]}
               for k, v in result.get("arms", {}).items()
               if isinstance(v, dict) and "accepted" in v}
    print(json.dumps({
        "record_id": result["record_id"],
        "torsion_hit_tier": result["torsion_hit_tier"],
        "n_supplied": result.get("n_supplied_contacts", 0),
        "n_held_out": result.get("n_held_out_contacts", 0),
        "arms": summary,
    }, indent=2))


if __name__ == "__main__":
    main()
