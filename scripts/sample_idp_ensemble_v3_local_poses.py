#!/usr/bin/env python
"""Restrained local interface sampling (priority 3) for single-structure v3 components.

Follows configs/benchmarks/idp_ensemble_v3_pose_protocol.yml: OpenMM minimization
plus 300 K Langevin dynamics with antibody-framework CA restraints (1000
kJ/mol/nm^2) and antigen CA restraints (100 kJ/mol/nm^2). Every sampled pose is
audited against its experimental reference (clashes < 1.8 A, H3 backbone
completeness, framework CA RMSD <= 1.0 A, interface CA RMSD <= 3.0 A).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from openmm import CustomExternalForce, LangevinMiddleIntegrator, Platform, unit
from openmm.app import ForceField, Modeller, PDBFile, Simulation
from pdbfixer import PDBFixer as Fixer

ROOT = Path(__file__).resolve().parents[1]

SEEDS = [13501, 13511]
DYNAMICS_PS = 2.5
TIMESTEP_FS = 2.0
FRICTION_PER_PS = 1.0
TEMPERATURE_K = 300.0
MINIMIZATION_STEPS = 400
FRAMEWORK_K = 1000.0
ANTIGEN_K = 100.0
CLASH_ANGSTROM = 1.8
FRAMEWORK_RMSD_MAX = 1.0
INTERFACE_RMSD_MAX = 3.0


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_pose_atoms(pdb_path):
    """Return ordered residues per chain with heavy atoms for audit math."""
    structure = PDBParser(QUIET=True).get_structure(Path(pdb_path).stem, str(pdb_path))
    model = next(iter(structure))
    chains = {}
    for chain in model:
        residues = []
        for residue in chain:
            name = "MET" if residue.resname == "MSE" else residue.resname
            if not is_aa(residue, standard=True) or "CA" not in residue:
                continue
            atoms = {
                atom.name: np.asarray(atom.coord, dtype=float)
                for atom in residue
                if atom.element != "H"
            }
            residues.append({
                "resid": residue.id[1],
                "name": name,
                "atoms": atoms,
            })
        if residues:
            chains[chain.id] = residues
    return chains


STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE",
}
SEGMENT_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def segment_pose_pdb(pose_path, scratch_path):
    """Split a pose PDB at residue gaps; return per-segment original letters.

    Solvent and ligand records are dropped so that only standard polymer
    residues remain, which keeps residue numbering monotonic per chain.
    """
    lines = Path(pose_path).read_text(encoding="ascii").splitlines()
    kept = []
    last_resid = {}
    segment_letters = []
    current_letter = {}
    for line in lines:
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        residue_name = line[17:20].strip().upper()
        if residue_name not in STANDARD_RESIDUES:
            continue
        chain = line[21:22].strip() or "_"
        try:
            resid = int(line[22:26])
        except ValueError:
            continue
        if chain not in current_letter or (
            chain in last_resid and resid - last_resid[chain] > 1
        ):
            current_letter[chain] = SEGMENT_LETTERS[len(segment_letters)]
            segment_letters.append(line[21:22])
        last_resid[chain] = resid
        kept.append(f"{line[:21]}{current_letter[chain]}{line[22:]}")
    kept.append("END")
    scratch_path.write_text("\n".join(kept) + "\n", encoding="ascii")
    return segment_letters


def build_system(segmented_path, antibody_chains, antigen_chains, ordered_letters):
    fixer = Fixer(str(segmented_path))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    letter_by_chain_id = {
        chain.id: ordered_letters[index]
        for index, chain in enumerate(fixer.topology.chains())
        if index < len(ordered_letters)
    }
    if len(letter_by_chain_id) != len(ordered_letters):
        raise ValueError("Segmented chain count differs from detected segments")
    modeller = Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(ForceField("amber14-all.xml"), pH=7.0)
    for chain in modeller.topology.chains():
        if chain.id not in letter_by_chain_id:
            raise ValueError("Chain identity lost while adding hydrogens")
    force_field = ForceField("amber14-all.xml")
    system = force_field.createSystem(modeller.topology, nonbondedCutoff=1.6 * unit.nanometer, constraints=None)

    framework = CustomExternalForce(
        "0.5*k_f*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
    )
    framework.addGlobalParameter("k_f", FRAMEWORK_K * unit.kilojoule_per_mole / unit.nanometer ** 2)
    framework.addPerParticleParameter("x0")
    framework.addPerParticleParameter("y0")
    framework.addPerParticleParameter("z0")
    antigen_restraint = CustomExternalForce(
        "0.5*k_a*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
    )
    antigen_restraint.addGlobalParameter("k_a", ANTIGEN_K * unit.kilojoule_per_mole / unit.nanometer ** 2)
    antigen_restraint.addPerParticleParameter("x0")
    antigen_restraint.addPerParticleParameter("y0")
    antigen_restraint.addPerParticleParameter("z0")

    antibody_set = set(antibody_chains)
    antigen_set = set(antigen_chains)
    positions = modeller.positions
    for atom in modeller.topology.atoms():
        if atom.name != "CA":
            continue
        chain_id = letter_by_chain_id.get(atom.residue.chain.id)
        if chain_id is None:
            continue
        position = positions[atom.index]
        parameters = (position[0], position[1], position[2])
        if chain_id in antibody_set:
            framework.addParticle(atom.index, parameters)
        elif chain_id in antigen_set:
            antigen_restraint.addParticle(atom.index, parameters)
    system.addForce(framework)
    system.addForce(antigen_restraint)
    return modeller, system, letter_by_chain_id


def write_heavy_pdb(topology, positions, letter_by_chain_id, output_path):
    """Write heavy atoms with original chain letters restored."""
    lines = []
    serial = 1
    nanometer = 10.0
    for atom in topology.atoms():
        if atom.element is not None and atom.element.symbol.lower() == "h":
            continue
        chain_id = letter_by_chain_id.get(atom.residue.chain.id, "X")
        insertion = (atom.residue.insertionCode or " ").strip() or " "
        coordinate = positions[atom.index]
        lines.append(
            f"ATOM  {serial:5d} {atom.name:<4s}"
            f"{atom.residue.name[:3]:>4s} {chain_id:1s}"
            f"{atom.residue.id:>4s}{insertion:1s}   "
            f"{coordinate.x * nanometer:8.3f}{coordinate.y * nanometer:8.3f}"
            f"{coordinate.z * nanometer:8.3f}"
            f"  1.00  0.00          {atom.element.symbol.upper():>2s}"
        )
        serial += 1
    lines.append("END")
    output_path.write_text("\n".join(lines) + "\n", encoding="ascii")


def sample_component(component_id, reference_pose, antibody_chains, antigen_chains,
                     output_dir, seeds):
    output_dir.mkdir(parents=True, exist_ok=True)
    segmented = output_dir / "input_segmented.pdb"
    ordered_letters = segment_pose_pdb(reference_pose, segmented)
    if not set(antibody_chains + antigen_chains) <= set(ordered_letters):
        raise ValueError(f"{component_id}: expected chains missing after segmentation")
    modeller, system, letter_by_chain_id = build_system(
        segmented, antibody_chains, antigen_chains, ordered_letters
    )
    platform = Platform.getPlatformByName("CPU")
    rows = []
    for seed in seeds:
        integrator = LangevinMiddleIntegrator(
            TEMPERATURE_K * unit.kelvin, FRICTION_PER_PS / unit.picosecond,
            TIMESTEP_FS * unit.femtosecond,
        )
        integrator.setRandomNumberSeed(seed)
        simulation = Simulation(modeller.topology, system, integrator, platform)
        simulation.context.setPositions(modeller.positions)
        simulation.minimizeEnergy(maxIterations=MINIMIZATION_STEPS)
        simulation.integrator.step(int(DYNAMICS_PS * 1000 / TIMESTEP_FS))
        state = simulation.context.getState(getPositions=True)
        pose_path = output_dir / f"pose_s{seed}_{component_id}.pdb"
        write_heavy_pdb(modeller.topology, state.getPositions(), letter_by_chain_id, pose_path)
        rows.append({
            "pose_id": f"sampled_s{seed}_{component_id}",
            "seed": seed,
            "path": str(pose_path.resolve().relative_to(ROOT).as_posix()),
            "coordinate_sha256": sha256(pose_path),
        })
        del simulation, integrator
    return rows


def audit_pose(reference_chains, sampled_chains, antibody_chains, antigen_chains,
               h3_sequence, contact_residues):
    heavy_chain_id = antibody_chains[0]
    reference_heavy = reference_chains[heavy_chain_id]
    sampled_heavy = sampled_chains[heavy_chain_id]
    reference_sequence = "".join(
        {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
         "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
         "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
         "TYR": "Y", "VAL": "V"}[row["name"]] for row in reference_heavy
    )
    h3_start = reference_sequence.find(h3_sequence)
    if h3_start < 0:
        raise ValueError("H3 sequence not found in reference heavy chain")
    h3_span = set(range(h3_start, h3_start + len(h3_sequence)))

    def matched_ca_pairs(chain_id, reference_rows, sampled_rows, selection):
        pairs = []
        for index in selection:
            if index >= len(reference_rows) or index >= len(sampled_rows):
                continue
            if reference_rows[index]["name"] != sampled_rows[index]["name"]:
                continue
            if "CA" not in reference_rows[index]["atoms"] or "CA" not in sampled_rows[index]["atoms"]:
                continue
            pairs.append((reference_rows[index]["atoms"]["CA"], sampled_rows[index]["atoms"]["CA"]))
        return pairs

    if len(reference_heavy) != len(sampled_heavy):
        raise ValueError("Residue count changed during sampling")
    framework_pairs = matched_ca_pairs(
        heavy_chain_id, reference_heavy, sampled_heavy,
        [index for index in range(len(reference_heavy)) if index not in h3_span],
    )
    interface_pairs = matched_ca_pairs(
        heavy_chain_id, reference_heavy, sampled_heavy, sorted(h3_span)
    )
    contact_keys = {(resid, name) for resid, name in contact_residues}
    for chain_id in antigen_chains:
        reference_chain = reference_chains.get(chain_id, [])
        sampled_chain = sampled_chains.get(chain_id, [])
        if len(reference_chain) != len(sampled_chain):
            continue
        selection = [
            index for index, row in enumerate(reference_chain)
            if (row["resid"], row["name"]) in contact_keys
        ]
        interface_pairs.extend(matched_ca_pairs(chain_id, reference_chain, sampled_chain, selection))

    def kabsch_rmsd(pairs):
        if not pairs:
            return 0.0
        mobile = np.stack([sampled for _, sampled in pairs])
        target = np.stack([reference for reference, _ in pairs])
        mobile_center = mobile.mean(axis=0)
        target_center = target.mean(axis=0)
        covariance = (mobile - mobile_center).T @ (target - target_center)
        u, _, vt = np.linalg.svd(covariance)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        fitted = (mobile - mobile_center) @ rotation + target_center
        return float(np.sqrt(np.mean(np.sum((fitted - target) ** 2, axis=1))))

    framework_rmsd = kabsch_rmsd(framework_pairs)
    interface_rmsd = kabsch_rmsd(interface_pairs)

    h3_backbone_missing = 0
    for index in sorted(h3_span):
        row = sampled_heavy[index]
        h3_backbone_missing += sum(
            atom not in row["atoms"] for atom in ("N", "CA", "C", "O")
        )

    clash_pairs = 0
    sampled_atoms = []
    for chain_id, residues in sampled_chains.items():
        for row in residues:
            for atom in row["atoms"].values():
                sampled_atoms.append((chain_id, row["resid"], atom))
    from scipy.spatial import cKDTree

    coords = np.stack([atom for _, _, atom in sampled_atoms])
    tree = cKDTree(coords)
    for left, right in tree.query_pairs(CLASH_ANGSTROM, output_type="ndarray"):
        chain_l, resid_l, _ = sampled_atoms[left]
        chain_r, resid_r, _ = sampled_atoms[right]
        if chain_l != chain_r:
            clash_pairs += 1
        elif abs(resid_l - resid_r) > 1:
            clash_pairs += 1

    checks = {
        "heavy_atom_clashes_lt_1_8a": clash_pairs <= 0,
        "h3_backbone_complete": h3_backbone_missing <= 0,
        "framework_ca_rmsd_ok": framework_rmsd <= FRAMEWORK_RMSD_MAX,
        "interface_ca_rmsd_ok": interface_rmsd <= INTERFACE_RMSD_MAX,
    }
    return {
        "heavy_atom_clashes_lt_1_8a": clash_pairs,
        "h3_backbone_missing_atoms": h3_backbone_missing,
        "framework_ca_rmsd_angstrom": round(framework_rmsd, 4),
        "interface_ca_rmsd_angstrom": round(interface_rmsd, 4),
        "checks": checks,
        "accepted": all(checks.values()),
    }


def sample(admission_path, readiness_path, experimental_manifest, output_dir):
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite sampled pose panel: {output_dir}")
    admission = json.loads(admission_path.read_text(encoding="ascii"))
    readiness = json.loads(readiness_path.read_text(encoding="ascii"))
    experimental = json.loads(experimental_manifest.read_text(encoding="ascii"))
    readiness_rows = {
        row["component_id"]: row for row in readiness["components"]
    }
    experimental_rows = {
        row["component_id"]: row for row in experimental["components"]
    }
    output_dir.mkdir(parents=True)
    components = []
    try:
        for component in admission["components"]:
            component_id = component["component_id"]
            ready = readiness_rows[component_id]
            if ready["multi_pose_ready"]:
                continue
            h3_sequence = ready.get("h3_sequence")
            if not h3_sequence:
                raise ValueError(f"{component_id} lacks H3 localization")
            reference_pose_row = experimental_rows[component_id]["poses"][0]
            reference_pose = ROOT / reference_pose_row["path"]
            antibody_chains = reference_pose_row["antibody_chains"]
            antigen_chains = reference_pose_row["antigen_chains"]
            reference_chains = load_pose_atoms(reference_pose)
            sampled = sample_component(
                component_id, reference_pose, antibody_chains, antigen_chains,
                output_dir / component_id, SEEDS,
            )
            rows = []
            for row in sampled:
                audit = audit_pose(
                    reference_chains, load_pose_atoms(ROOT / row["path"]),
                    antibody_chains, antigen_chains, h3_sequence,
                    component["contact_residues"],
                )
                rows.append({
                    **row,
                    "entry_id": component_id,
                    "source": "restrained_local_interface_sampling",
                    "source_priority": 3,
                    "antibody_chains": antibody_chains,
                    "antigen_chains": antigen_chains,
                    "sampling": {
                        "seeds_kj_per_mol_per_nm2": {
                            "antibody_framework_ca": FRAMEWORK_K,
                            "antigen_ca": ANTIGEN_K,
                        },
                        "temperature_kelvin": TEMPERATURE_K,
                        "dynamics_ps": DYNAMICS_PS,
                        "timestep_fs": TIMESTEP_FS,
                        "minimization_steps": MINIMIZATION_STEPS,
                        "force_field": "amber14-all.xml",
                    },
                    **audit,
                })
            components.append({
                "component_id": component_id,
                "poses": rows,
                "accepted_pose_count": sum(row["accepted"] for row in rows),
            })
        manifest = {
            "schema_version": 1,
            "status": "sampled_pose_panel_complete",
            "classification": "v3_priority3_restrained_local_sampling",
            "admission": str(admission_path),
            "admission_sha256": sha256(admission_path),
            "seeds": SEEDS,
            "component_count": len(components),
            "components": components,
            "claim_boundary": (
                "restrained local sampling near experimental structures for a "
                "general antibody multi-conformer panel; no candidate or "
                "performance result"
            ),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="ascii"
        )
        return manifest
    except Exception:
        for path in sorted(output_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        output_dir.rmdir()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--experimental-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = sample(
        args.admission, args.readiness, args.experimental_manifest, args.output_dir
    )
    print(json.dumps({
        "status": result["status"],
        "accepted_pose_counts": {
            row["component_id"]: row["accepted_pose_count"] for row in result["components"]
        },
    }, indent=2))


if __name__ == "__main__":
    main()
