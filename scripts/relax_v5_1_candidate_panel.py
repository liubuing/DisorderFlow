#!/usr/bin/env python3
"""Restrained OpenMM relaxation for calibrated v5.1 candidate structures."""

import argparse
import json
from pathlib import Path

import numpy as np
from openmm import CustomExternalForce, LangevinMiddleIntegrator, Platform, unit
from openmm.app import ForceField, HBonds, Modeller, NoCutoff, PDBFile, Simulation
from pdbfixer import PDBFixer

ROOT = Path(__file__).resolve().parent.parent


def ca_coordinates(topology, positions):
    coordinates = []
    for atom in topology.atoms():
        if atom.name == "CA":
            value = positions[atom.index].value_in_unit(unit.angstrom)
            coordinates.append([value.x, value.y, value.z])
    return np.asarray(coordinates, dtype=np.float64)


def relax(record, output_dir, max_iterations, platform_name):
    input_path = ROOT / record["structure_pdb"]
    fixer = PDBFixer(filename=str(input_path))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    forcefield = ForceField("amber14-all.xml", "implicit/gbn2.xml")
    modeller = Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(forcefield, pH=7.4)
    system = forcefield.createSystem(
        modeller.topology, nonbondedMethod=NoCutoff, constraints=HBonds)

    restraint = CustomExternalForce(
        "0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.addGlobalParameter("k", 1000.0 * unit.kilojoule_per_mole / unit.nanometer**2)
    for parameter in ("x0", "y0", "z0"):
        restraint.addPerParticleParameter(parameter)
    for atom in modeller.topology.atoms():
        if atom.name not in {"N", "CA", "C", "O"}:
            continue
        position = modeller.positions[atom.index].value_in_unit(unit.nanometer)
        restraint.addParticle(atom.index, [position.x, position.y, position.z])
    system.addForce(restraint)

    integrator = LangevinMiddleIntegrator(
        300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
    if platform_name:
        platform = Platform.getPlatformByName(platform_name)
        properties = {"Precision": "mixed"} if platform_name == "OpenCL" else {}
    else:
        try:
            platform = Platform.getPlatformByName("OpenCL")
            properties = {"Precision": "mixed"}
        except Exception:  # noqa: BLE001
            platform = Platform.getPlatformByName("CPU")
            properties = {}
    simulation = Simulation(
        modeller.topology, system, integrator, platform, properties)
    simulation.context.setPositions(modeller.positions)
    initial = simulation.context.getState(getEnergy=True, getPositions=True)
    initial_energy = initial.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    initial_ca = ca_coordinates(modeller.topology, initial.getPositions())
    simulation.minimizeEnergy(
        tolerance=10.0 * unit.kilojoule_per_mole / unit.nanometer,
        maxIterations=max_iterations)
    final = simulation.context.getState(getEnergy=True, getPositions=True)
    final_energy = final.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    final_ca = ca_coordinates(modeller.topology, final.getPositions())
    ca_rmsd = float(np.sqrt(np.mean(np.sum((final_ca - initial_ca) ** 2, axis=1))))

    output_path = output_dir / f"{record['construct_id']}_relaxed.pdb"
    with output_path.open("w", encoding="ascii") as handle:
        PDBFile.writeFile(modeller.topology, final.getPositions(), handle, keepIds=True)
    return {
        "construct_id": record["construct_id"],
        "input_pdb": str(input_path.relative_to(ROOT)),
        "relaxed_pdb": str(output_path.relative_to(ROOT)),
        "platform": platform.getName(),
        "max_iterations": max_iterations,
        "initial_energy_kj_mol": initial_energy,
        "final_energy_kj_mol": final_energy,
        "energy_delta_kj_mol": final_energy - initial_energy,
        "ca_rmsd_angstrom": ca_rmsd,
        "relax_pass": final_energy < 0.0 and ca_rmsd <= 1.5,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=6)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--platform", choices=["CPU", "OpenCL"])
    parser.add_argument("--construct-id")
    parser.add_argument(
        "--screen",
        default="results/v5_1_candidates/abeta42_colabfold_candidates/calibrated_screen.json")
    parser.add_argument("--output-dir", default="results/v5_1_candidates/abeta42_relaxed")
    args = parser.parse_args()
    screen = json.loads((ROOT / args.screen).read_text(encoding="utf-8"))
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    failures = []
    selected = screen["records"][:args.top]
    if args.construct_id:
        selected = [record for record in screen["records"] if record["construct_id"] == args.construct_id]
    for record in selected:
        try:
            result = relax(record, output_dir, args.max_iterations, args.platform)
            records.append(result)
            print(
                f"{result['construct_id']}: dE={result['energy_delta_kj_mol']:.1f} "
                f"CA_RMSD={result['ca_rmsd_angstrom']:.3f} pass={result['relax_pass']}",
                flush=True)
        except Exception as error:  # noqa: BLE001
            failures.append({"construct_id": record["construct_id"], "error": str(error)})
            print(f"{record['construct_id']}: FAILED {error}", flush=True)
    report_path = output_dir / "relaxation_report.json"
    if args.construct_id and report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        records = [
            record for record in existing.get("records", [])
            if record["construct_id"] != args.construct_id
        ] + records
        failures = [
            failure for failure in existing.get("failures", [])
            if failure["construct_id"] != args.construct_id
        ] + failures
    report = {
        "method": "OpenMM 8.5 restrained implicit-solvent minimization",
        "forcefield": "amber14-all + implicit/gbn2",
        "backbone_restraint_k_kj_mol_nm2": 1000.0,
        "n_requested": len(records) + len(failures),
        "n_completed": len(records),
        "n_pass": sum(record["relax_pass"] for record in records),
        "records": records,
        "failures": failures,
    }
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in ("records", "failures")}, indent=2))


if __name__ == "__main__":
    main()
