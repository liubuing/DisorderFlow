#!/usr/bin/env python
"""Prepare and audit six outcome-unexposed IDP antibody ensemble components."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from Bio.PDB import PDBIO, PDBParser, Select
from openmm import CustomExternalForce, LangevinMiddleIntegrator, Platform, unit
from openmm.app import ForceField, HBonds, Modeller, NoCutoff, PDBFile, Simulation
from pdbfixer import PDBFixer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "modules"))

from modules.ensemble_pose_transfer import (  # noqa: E402
    fixed_paratope_contact_map,
    pose_geometry_audit,
    refine_pose_local,
    resolve_pose_clashes,
    transfer_pose,
)
from modules.state_contact_scorer import extract_contact_map  # noqa: E402
from scripts.generate_h3_peptide_ensemble import peptide_geometry  # noqa: E402
from scripts.pipeline.prepare_nonabeta_idp_ensembles import (  # noqa: E402
    farthest_models,
    kabsch_rmsd,
    sequence_residues,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ReferenceSelect(Select):
    def __init__(self, antibody_chains, antigen_chain, antigen_residue_ids):
        self.antibody_chains = set(antibody_chains)
        self.antigen_chain = antigen_chain
        self.antigen_residue_ids = set(antigen_residue_ids)

    def accept_chain(self, chain):
        return chain.id in self.antibody_chains or chain.id == self.antigen_chain

    def accept_residue(self, residue):
        return residue.get_parent().id in self.antibody_chains or (
            residue.get_parent().id == self.antigen_chain
            and residue.id in self.antigen_residue_ids
        )


class EpitopeSelect(Select):
    def __init__(self, chain_id, residue_ids):
        self.chain_id = chain_id
        self.residue_ids = set(residue_ids)

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        return residue.id in self.residue_ids


def trim_reference(component, output_path):
    source = ROOT / component["reference_pdb"]
    parser = PDBParser(QUIET=True)
    model = next(iter(parser.get_structure(component["component_id"], str(source))))
    sequence, residues = sequence_residues(model[component["antigen_chain"]])
    start = sequence.find(component["epitope"])
    if start < 0:
        raise ValueError(f"{component['component_id']} lacks observed epitope {component['epitope']}")
    epitope_residues = residues[start:start + len(component["epitope"])]
    antibody_chains = [component["heavy_chain"]]
    if component.get("light_chain"):
        antibody_chains.append(component["light_chain"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(model)
    io.save(str(output_path), ReferenceSelect(
        antibody_chains,
        component["antigen_chain"],
        [residue.id for residue in epitope_residues],
    ))
    return {
        "path": output_path.relative_to(ROOT).as_posix(),
        "sha256": sha256(output_path),
        "antibody_chains": antibody_chains,
        "epitope_residue_ids": [
            [residue.id[0], residue.id[1], residue.id[2].strip()]
            for residue in epitope_residues
        ],
    }


def write_epitope_model(model, chain_id, epitope, output_path):
    sequence, residues = sequence_residues(model[chain_id])
    start = sequence.find(epitope)
    if start < 0:
        raise ValueError(f"Source model lacks epitope {epitope}")
    selected = residues[start:start + len(epitope)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(model)
    io.save(str(output_path), EpitopeSelect(chain_id, [residue.id for residue in selected]))
    return selected


def prepare_source_conformers(component, output_dir):
    ensemble = component["ensemble"]
    parser = PDBParser(QUIET=True)
    model_records = []
    if ensemble["type"] == "pdb_files":
        for source_index, relative in enumerate(ensemble["paths"], 1):
            source = ROOT / relative
            model = next(iter(parser.get_structure(f"source_{source_index}", str(source))))
            output = output_dir / f"source_{source_index}.pdb"
            residues = write_epitope_model(model, ensemble["chain"], component["epitope"], output)
            model_records.append((source_index, output, residues))
        return model_records

    source = ROOT / ensemble["source_pdb"]
    structure = parser.get_structure(component["component_id"], str(source))
    coordinates = []
    staged = []
    for model_index, model in enumerate(structure, 1):
        sequence, residues = sequence_residues(model[ensemble["chain"]])
        start = sequence.find(component["epitope"])
        if start < 0:
            raise ValueError(f"{component['component_id']} source model {model_index} lacks epitope")
        selected = residues[start:start + len(component["epitope"])]
        coordinates.append(np.asarray([residue["CA"].coord for residue in selected]))
        staged.append((model_index, model, selected))
    order, _ = farthest_models(coordinates, len(coordinates))
    for rank, zero_index in enumerate(order):
        model_index, model, selected = staged[zero_index]
        output = output_dir / f"rank_{rank}_model_{model_index}.pdb"
        output.parent.mkdir(parents=True, exist_ok=True)
        io = PDBIO()
        io.set_structure(model)
        io.save(str(output), EpitopeSelect(
            ensemble["chain"], [residue.id for residue in selected]
        ))
        model_records.append((model_index, output, selected))
    return model_records


def rewrite_antigen_chain(source, output, source_chain, output_chain="P"):
    lines = []
    for line in Path(source).read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith(("ATOM  ", "HETATM")) and line[21] == source_chain:
            line = f"{line[:21]}{output_chain}{line[22:]}"
        lines.append(line)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(lines), encoding="ascii")


def peptide_backbone_coordinates(topology, positions, peptide_chain):
    coordinates = positions.value_in_unit(unit.angstrom)
    return np.asarray([
        coordinates[atom.index]
        for atom in topology.atoms()
        if atom.residue.chain.id == peptide_chain
        and atom.name in {"N", "CA", "C", "O"}
    ], dtype=float)


def generate_peptide_only_ensemble(template, component, component_dir, hydrogens_platform=None):
    ensemble = component["ensemble"]
    settings = ensemble["sampling"]
    peptide_chain = component["antigen_chain"]
    parser = PDBParser(QUIET=True)
    model = next(iter(parser.get_structure("peptide", str(template))))
    peptide_input = component_dir / "synthetic_ensemble" / "peptide_reference.pdb"
    peptide_input.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(model)
    residues = [residue.id for residue in model[peptide_chain]]
    io.save(str(peptide_input), EpitopeSelect(peptide_chain, residues))

    fixer = PDBFixer(filename=str(peptide_input))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=0)
    forcefield = ForceField("amber14-all.xml", "implicit/gbn2.xml")
    modeller = Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(
        forcefield, pH=float(settings["ph"]), platform=hydrogens_platform
    )
    system = forcefield.createSystem(
        modeller.topology, nonbondedMethod=NoCutoff, constraints=HBonds
    )
    restraint = CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.addPerParticleParameter("k")
    for name in ("x0", "y0", "z0"):
        restraint.addPerParticleParameter(name)
    initial_nm = modeller.positions.value_in_unit(unit.nanometer)
    for atom in modeller.topology.atoms():
        if atom.name not in {"N", "CA", "C", "O"}:
            continue
        xyz = initial_nm[atom.index]
        restraint.addParticle(atom.index, [
            float(settings["peptide_backbone_restraint"]), xyz.x, xyz.y, xyz.z
        ])
    system.addForce(restraint)
    reference = peptide_backbone_coordinates(
        modeller.topology, modeller.positions, peptide_chain
    )
    rows = []
    for seed in ensemble["seeds"]:
        output = component_dir / "synthetic_ensemble" / "conformers" / f"seed_{seed}.pdb"
        integrator = LangevinMiddleIntegrator(
            float(settings["temperature_kelvin"]) * unit.kelvin,
            float(settings["friction_per_ps"]) / unit.picosecond,
            float(settings["timestep_fs"]) * unit.femtoseconds,
        )
        integrator.setRandomNumberSeed(int(seed))
        simulation = Simulation(
            modeller.topology,
            system,
            integrator,
            Platform.getPlatformByName(settings["platform"]),
        )
        simulation.context.setPositions(modeller.positions)
        simulation.minimizeEnergy(
            tolerance=float(settings["minimization_tolerance_kj_mol_nm"])
            * unit.kilojoule_per_mole / unit.nanometer,
            maxIterations=int(settings["minimization_iterations"]),
        )
        simulation.context.setVelocitiesToTemperature(
            float(settings["temperature_kelvin"]) * unit.kelvin, int(seed)
        )
        simulation.step(int(settings["equilibration_steps"]))
        simulation.step(int(settings["production_steps"]))
        simulation.minimizeEnergy(
            tolerance=float(settings["minimization_tolerance_kj_mol_nm"])
            * unit.kilojoule_per_mole / unit.nanometer,
            maxIterations=int(settings["snapshot_minimization_iterations"]),
        )
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        positions = state.getPositions()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="ascii") as handle:
            PDBFile.writeFile(modeller.topology, positions, handle, keepIds=True)
        observed = peptide_backbone_coordinates(modeller.topology, positions, peptide_chain)
        rmsd = kabsch_rmsd(reference, observed)
        geometry = peptide_geometry(modeller.topology, positions, peptide_chain)
        qc = settings["quality_control"]
        checks = {
            "minimum_rmsd": rmsd >= float(qc["min_peptide_rmsd"]),
            "maximum_rmsd": rmsd <= float(qc["max_peptide_rmsd"]),
            "ca_geometry": geometry["ca_geometry_outliers"]
            <= int(qc["max_ca_geometry_outliers"]),
            "cn_geometry": geometry["cn_geometry_outliers"]
            <= int(qc["max_cn_geometry_outliers"]),
        }
        rows.append({
            "seed": int(seed),
            "pdb": str(output),
            "sha256": sha256(output),
            "peptide_internal_rmsd": rmsd,
            "potential_energy_kj_mol": float(
                state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            ),
            **geometry,
            "checks": checks,
            "status": "accepted" if all(checks.values()) else "rejected",
        })
        print(f"{component['component_id']} seed {seed}: {rows[-1]['status']}", flush=True)
    audit = {
        "schema_version": 1,
        "ensemble_class": "peptide-only restrained OpenMM local neighborhood",
        "claim_boundary": "Local computational dispersion, not an equilibrium ensemble.",
        "sampling_mode": ensemble["sampling_mode"],
        "attempted": len(rows),
        "accepted": sum(row["status"] == "accepted" for row in rows),
        "conformers": rows,
    }
    (component_dir / "synthetic_ensemble" / "ensemble_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="ascii"
    )
    return audit


def audit_pose(component, pose, template_map, requirements, transfer=None):
    antibody_chains = [component["heavy_chain"]]
    if component.get("light_chain"):
        antibody_chains.append(component["light_chain"])
    retreat = resolve_pose_clashes(
        pose,
        antibody_chains,
        max_severe_clashes=int(requirements["maximum_severe_clashes_lt_1_5A"]),
    )
    geometry = pose_geometry_audit(pose, antibody_chains)
    contact_map = fixed_paratope_contact_map(pose, template_map)
    refinement = {
        "local_refinement_applied": False,
        "local_refinement_translation_norm": 0.0,
        "local_refinement_rotation_norm_degrees": 0.0,
    }
    if (
        geometry["severe_clash_pairs_lt_1_5A"]
        > int(requirements["maximum_severe_clashes_lt_1_5A"])
        or len(contact_map["contacts"]) < int(requirements["minimum_fixed_paratope_contacts"])
    ):
        refined = refine_pose_local(pose, antibody_chains, seed=sum(map(ord, component["component_id"])))
        refinement = {"local_refinement_applied": True, **refined}
        resolve_pose_clashes(
            pose,
            antibody_chains,
            max_severe_clashes=int(requirements["maximum_severe_clashes_lt_1_5A"]),
        )
        geometry = pose_geometry_audit(pose, antibody_chains)
        contact_map = fixed_paratope_contact_map(pose, template_map)
    checks = {
        "exact_epitope": contact_map["peptide_sequence"] == component["epitope"],
        "zero_severe_clashes": geometry["severe_clash_pairs_lt_1_5A"]
        <= int(requirements["maximum_severe_clashes_lt_1_5A"]),
        "minimum_contacts": len(contact_map["contacts"])
        >= int(requirements["minimum_fixed_paratope_contacts"]),
        "bounded_translation": refinement["local_refinement_translation_norm"]
        <= float(requirements["maximum_refinement_translation_angstrom"]),
        "bounded_rotation": refinement["local_refinement_rotation_norm_degrees"]
        <= float(requirements["maximum_refinement_rotation_degrees"]),
        "epitope_fit": transfer is None or transfer["epitope_fit_rmsd"]
        <= float(requirements["maximum_epitope_fit_rmsd_angstrom"]),
    }
    return {
        "pose_pdb": pose.relative_to(ROOT).as_posix(),
        "pose_sha256": sha256(pose),
        "epitope_fit_rmsd": transfer["epitope_fit_rmsd"] if transfer else 0.0,
        **retreat,
        **refinement,
        **geometry,
        "n_contacts": len(contact_map["contacts"]),
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
    }


def prepare_component(component, config, output_root):
    component_dir = output_root / component["component_id"]
    trimmed = trim_reference(component, component_dir / "trimmed_reference.pdb")
    template = ROOT / trimmed["path"]
    template_map = extract_contact_map(str(template), peptide_chain=component["antigen_chain"])
    if template_map["peptide_sequence"] != component["epitope"]:
        raise ValueError(f"{component['component_id']} template epitope mismatch")
    poses = []
    source_rows = []
    if component["ensemble"]["type"] == "synthetic_local_openmm":
        ensemble = generate_peptide_only_ensemble(template, component, component_dir)
        accepted = [row for row in ensemble["conformers"] if row["status"] == "accepted"]
        for rank, row in enumerate(accepted):
            pose = component_dir / "poses" / f"pose_{rank}.pdb"
            transfer = transfer_pose(
                template,
                row["pdb"],
                pose,
                component["antigen_chain"],
                component["antigen_chain"],
                component["heavy_chain"],
                component.get("light_chain") or "",
                1,
            )
            source_rows.append({
                "source_model": row["seed"],
                "source_pdb": str(row["pdb"]),
                "source_sha256": row["sha256"],
            })
            poses.append(audit_pose(
                component, pose, template_map, config["requirements"], transfer=transfer
            ))
    else:
        conformers = prepare_source_conformers(component, component_dir / "source_conformers")
        for rank, (model_index, source, _) in enumerate(conformers):
            pose = component_dir / "poses" / f"pose_{rank}.pdb"
            transfer = transfer_pose(
                template,
                source,
                pose,
                component["antigen_chain"],
                component["ensemble"]["chain"],
                component["heavy_chain"],
                component.get("light_chain") or "",
                1,
            )
            source_rows.append({
                "source_model": model_index,
                "source_pdb": source.relative_to(ROOT).as_posix(),
                "source_sha256": sha256(source),
            })
            poses.append(audit_pose(
                component, pose, template_map, config["requirements"], transfer=transfer
            ))

    selected = [row for row in poses if row["status"] == "pass"][: int(
        config["requirements"]["selected_pose_conformers_per_component"]
    )]
    selected_paths = {row["pose_pdb"] for row in selected}
    for row in poses:
        row["selected_for_panel"] = row["pose_pdb"] in selected_paths
    return {
        **component,
        "trimmed_reference_pdb": trimmed["path"],
        "trimmed_reference_sha256": trimmed["sha256"],
        "ensemble_provenance": component["ensemble"]["provenance"],
        "source_conformers": source_rows,
        "poses": poses,
        "passing_poses": sum(row["status"] == "pass" for row in poses),
        "selected_poses": len(selected),
        "status": "pass" if len(selected) == int(
            config["requirements"]["selected_pose_conformers_per_component"]
        ) else "fail",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/benchmarks/idp_ensemble_expansion_6ref_v1.yml"
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for component in config["components"]:
        reference = ROOT / component["reference_pdb"]
        if sha256(reference) != component["reference_sha256"]:
            raise ValueError(f"Reference hash mismatch: {reference}")
        source = component["ensemble"].get("source_pdb")
        if source and sha256(ROOT / source) != component["ensemble"]["source_sha256"]:
            raise ValueError(f"Source hash mismatch: {source}")

    output_path = ROOT / (args.out or config["output"]["preparation"])
    output_root = output_path.parent / "work"
    rows = []
    for component in config["components"]:
        row = prepare_component(component, config, output_root)
        rows.append(row)
        output = {
            "schema_version": 1,
            "status": "pass" if all(item["status"] == "pass" for item in rows)
            and len(rows) == len(config["components"]) else "partial",
            "classification": config["classification"],
            "config": args.config,
            "config_sha256": sha256(config_path),
            "components": rows,
            "claim_boundary": config["claim_boundary"],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
        print(
            f"Expansion preparation {component['component_id']}: "
            f"status={row['status']} selected={row['selected_poses']}",
            flush=True,
        )
    if output["status"] == "pass":
        manifest_path = ROOT / config["output"]["component_manifest"]
        manifest = {
            "schema_version": 1,
            "status": "frozen_after_pose_gate_before_candidate_generation",
            "classification": config["classification"],
            "preparation": output_path.relative_to(ROOT).as_posix(),
            "preparation_sha256": sha256(output_path),
            "components": [
                {
                    "component_id": row["component_id"],
                    "target": row["target"],
                    "disease": row["disease"],
                    "antibody": row["antibody"],
                    "reference_path": row["trimmed_reference_pdb"],
                    "reference_sha256": row["trimmed_reference_sha256"],
                    "heavy_chain": row["heavy_chain"],
                    "light_chain": row.get("light_chain"),
                    "antigen_chain": row["antigen_chain"],
                    "epitope": row["epitope"],
                    "h3_positions_1_indexed": row["h3_positions_1_indexed"],
                    "native_h3": row["native_h3"],
                    "ensemble_provenance": row["ensemble_provenance"],
                    "pose_paths": [
                        pose["pose_pdb"] for pose in row["poses"]
                        if pose["selected_for_panel"]
                    ],
                }
                for row in rows
            ],
            "claim_boundary": config["claim_boundary"],
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    if output["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
