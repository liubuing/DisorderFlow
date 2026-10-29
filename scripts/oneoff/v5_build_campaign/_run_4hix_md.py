"""Extract Fv-only from 4HIX and run T1 ensemble MD on CPU."""
# 角色：v5 构建活动的一次性脚本——从 4HIX 中切出仅含 Fv（VH/VL）与肽 A 的结构，
# 再写出 T1 集系综 MD（OpenMM/PDBFixer，CPU 平台）的缩减配置 JSON，
# 交由 generate_h3_peptide_ensemble.py 实际执行（本脚本只准备，不运行 MD）。
import sys, os, json, time
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select

class FvSelect(Select):
    """Keep only Fv residues (VH ~1-119, VL ~1-113) + peptide (A)."""
    def accept_residue(self, residue):
        return residue.id[0] == ' '
    def accept_chain(self, chain):
        return chain.id in ('H', 'L', 'A')

p = PDBParser(QUIET=True)
s = p.get_structure('x', 'idp_design_results/4hix_clean.pdb')

# Fv-only: trim VH to ~residues up to position 119 (before CH1 starts)
# For 4HIX, VH ends around residue 119 based on the sequence
# VL ends around 113
io = PDBIO()
io.set_structure(s)
fv_path = 'idp_design_results/4hix_fv.pdb'
io.save(fv_path, FvSelect())

# Count residues
from Bio.PDB import PDBParser as P2
p2 = P2(QUIET=True)
s2 = p2.get_structure('y', fv_path)
for chain in s2[0]:
    n = sum(1 for r in chain if r.id[0] == ' ')
    print(f"Chain {chain.id}: {n} residues")
print(f"Saved: {fv_path}")

# Now run T1 MD with reduced config
config = {
    "schema_version": 1,
    "status": "4hix_fv_t1_ensemble",
    "chains": {"antibody": ["H", "L"], "peptide": "A"},
    "sampling": {
        "platform": "CPU",
        "ph": 7.4,
        "pdbfixer_seed": 3527,
        "friction_per_ps": 1.0,
        "timestep_fs": 2.0,
        "minimization_tolerance_kj_mol_nm": 10.0,
        "minimization_iterations": 200,
        "snapshot_minimization_iterations": 100,
        "antibody_heavy_atom_restraint": 5000.0,   # 抗体 heavy-atom 位置约束（kJ/mol/nm^2），强约束保持骨架
        "peptide_backbone_restraint": 25.0,        # 肽骨架弱约束，允许肽在界面处弛豫
        "contact_cutoff_angstrom": 4.5,            # 接触判定的重原子距离阈值
        "seeds": [2701, 2711, 2721],               # 三个等距种子做 T1 集系综
        "temperature_kelvin": 300.0,
        "equilibration_steps": 50,
        "production_steps": 100,
        "quality_control": {
            # QC：抗体 RMSD 需 <=0.5Å（约束下应基本不动）；肽 RMSD 需在
            # [0.05, 3.0]Å 之间（过小说明未弛豫，过大说明脱离）；接触保持率 >=0.5。
            "max_antibody_rmsd": 0.5,
            "min_peptide_rmsd": 0.05,
            "max_peptide_rmsd": 3.0,
            "min_contact_retention": 0.5,
        }
    }
}

config_path = 'idp_design_results/4hix_fv_ensemble_config.json'
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"Config: {config_path}")
print(f"Run: python scripts/generate_h3_peptide_ensemble.py --pdb {fv_path} --config {config_path} --out-dir idp_design_results/4hix_fv_ensemble --record-id 4HIX_Fv")
