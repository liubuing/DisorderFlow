import sys, json, yaml, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from generate_h3_peptide_t21 import (
    prepare, heavy_atom_records, selected_coordinates, BACKBONE,
    residue_contact_pairs, peptide_geometry,
    build_null_system, run_recovery_arm,
)
from modules.peptide_t2_recovery import split_contact_pairs
from Bio.PDB import PDBParser
from openmm import CustomExternalForce, unit
from openmm.app import CutoffNonPeriodic, ForceField, HBonds

# Load existing perturbed PDB from completed record
record_id = "pdb_00007rqr_A_B"
work_dir = Path(ROOT / f"results/publication/h3_t2.1_temporal_final_v1/work/{record_id}")
perturbed_pdb = work_dir / "t2.1" / "perturbed" / f"{record_id}.pdb"
config = yaml.safe_load(Path(ROOT / "configs/benchmarks/peptide_h3_t2.1_debug_v1.yml").read_text())

settings = config["sampling"]
antibody_chains = [c for c in config["chains"]["antibody"]]
peptide_chain = config["chains"]["peptide"]

modeller, forcefield = prepare(perturbed_pdb, settings)
native_records = heavy_atom_records(modeller.topology, modeller.positions)
positions = modeller.positions
positions_nm = positions.value_in_unit(unit.nanometer)

all_contacts = residue_contact_pairs(
    native_records,
    lambda row: row["chain"] in antibody_chains and row["atom_name"] == "CA",
    lambda row: row["chain"] == peptide_chain and row["atom_name"] == "CA",
    cutoff=8.0)

source_pdb = work_dir / "deposited_complex.pdb"
source_modeller, _ = prepare(source_pdb, settings)
source_records = heavy_atom_records(source_modeller.topology, source_modeller.positions)

supplied, held_out = split_contact_pairs(all_contacts, record_id, 0.5)
reference = {
    "antibody": selected_coordinates(source_records, lambda row: (
        row["chain"] in antibody_chains and row["atom_name"] in BACKBONE)),
    "peptide": selected_coordinates(source_records, lambda row: (
        row["chain"] == peptide_chain and row["atom_name"] in BACKBONE)),
    "geometry": peptide_geometry(source_modeller.topology, source_modeller.positions, peptide_chain),
}

seed = 42
import hashlib
seed = int(hashlib.sha256(f"{record_id}|1009".encode()).hexdigest()[:8], 16)

# Test random arm
print("=== Testing random_restraints arm ===")
try:
    random_rest_system = forcefield.createSystem(
        modeller.topology, nonbondedMethod=CutoffNonPeriodic,
        nonbondedCutoff=1.0*unit.nanometer, constraints=HBonds)
    antibody_force = CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    antibody_force.addPerParticleParameter("k")
    antibody_force.addPerParticleParameter("x0")
    antibody_force.addPerParticleParameter("y0")
    antibody_force.addPerParticleParameter("z0")
    for atom in modeller.topology.atoms():
        if atom.element is None or atom.element.symbol == "H":
            continue
        if atom.residue.chain.id in antibody_chains:
            xyz = positions_nm[atom.index]
            antibody_force.addParticle(atom.index, [
                float(settings["antibody_heavy_atom_restraint"]),
                xyz.x, xyz.y, xyz.z])
    random_rest_system.addForce(antibody_force)
    peptide_restraint = CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    peptide_restraint.addPerParticleParameter("k")
    peptide_restraint.addPerParticleParameter("x0")
    peptide_restraint.addPerParticleParameter("y0")
    peptide_restraint.addPerParticleParameter("z0")
    import random as rmod
    rng = rmod.Random(int(seed) + 7777)
    peptide_atoms = []
    for atom in modeller.topology.atoms():
        if atom.element is None or atom.element.symbol == "H":
            continue
        if atom.residue.chain.id == peptide_chain and atom.name == "CA":
            peptide_atoms.append(atom.index)
    print(f"  peptide CA atoms: {len(peptide_atoms)}, supplied: {len(supplied)}")
    count = min(len(supplied), len(peptide_atoms))
    selected = rng.sample(peptide_atoms, count)
    for atom_idx in selected:
        xyz = positions_nm[atom_idx]
        peptide_restraint.addParticle(atom_idx, [25.0, xyz.x, xyz.y, xyz.z])
    random_rest_system.addForce(peptide_restraint)
    
    results = run_recovery_arm(
        modeller, forcefield, positions, random_rest_system, reference,
        antibody_chains, peptide_chain, all_contacts, held_out,
        seed, settings, "random_restraints", record_id)
    accepted = sum(1 for r in results if r.get("accepted"))
    print(f"  random_restraints: {accepted}/{len(results)} accepted")
except Exception as e:
    print(f"  random_restraints FAILED: {e}")
    traceback.print_exc()

# Test null arm
print("=== Testing null_structural arm ===")
try:
    null_system = build_null_system(modeller, forcefield, antibody_chains, settings)
    results = run_recovery_arm(
        modeller, forcefield, positions, null_system, reference,
        antibody_chains, peptide_chain, all_contacts, held_out,
        seed, settings, "null_structural", record_id)
    accepted = sum(1 for r in results if r.get("accepted"))
    print(f"  null_structural: {accepted}/{len(results)} accepted")
except Exception as e:
    print(f"  null_structural FAILED: {e}")
    traceback.print_exc()

print("DONE")
