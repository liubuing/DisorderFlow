import os
#!/usr/bin/env python3
"""Generate CDR designs with V17i Pair Routing model for AF2 validation.

V17i uses pair_feat routing — run_bfn_design doesn't work with it.
This script directly uses AntibodyBFN.sample() to generate designs
with proper pair_feat encoding, then grafts CDRs into scaffold PDBs
for AF2 batch validation.

Usage:
    python run_v17i_af2_design.py [--ckpt CHECKPOINT] [--samples N]
Output: oc_validation_results/oc_v17i_phase1_complex.json
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..')); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))
import torch, yaml, numpy as np
from easydict import EasyDict

AA = 'ACDEFGHIKLMNPQRSTVWY'

# Config
CKPT = sys.argv[1] if len(sys.argv) > 1 else 'logs/v17c_zeroinit/bfn_v17c_zeroinit_xpu_2026_06_26__23_22_18/checkpoints/3800.pt'
N_SAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 20
SCAFFOLDS = [
    ('data/misfolding_targets/3STB.pdb', 'A'),  # best scaffold from V15 results
    ('data/misfolding_targets/5IMK.pdb', 'B'),
]
ANTIGENS = [
    ('data/abeta_conformations/pdbs/abeta42_seed0_42.pdb', 'P'),
    ('data/abeta_conformations/pdbs/abeta42_seed2_42.pdb', 'P'),
    ('data/abeta_conformations/pdbs/abeta42_seed4_42.pdb', 'P'),
]
CDR_SPEC = 'B:26-33,B:51-58,B:97-113'  # fallback default
SCAFFOLD_CDRS = {
    '5IMK.pdb': 'B:26-33,B:51-58,B:97-113',
    '3STB.pdb': 'A:26-33,A:51-58,A:97-113',
}

from disorderflow.models.bfn_model import AntibodyBFN
from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.train import recursive_to
from disorderflow.utils.transforms import get_transform
from disorderflow.datasets.protein import preprocess_protein_structure
from disorderflow.utils.misc import seed_all

# Load model
cfg = EasyDict(yaml.safe_load(open('configs/train/bfn_v17i_pairrouting_xpu.yml', encoding='utf-8')))
model = AntibodyBFN(cfg.model)
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model'], strict=False); model.eval()
print(f'Loaded V17i step {ckpt["iteration"]}, val {ckpt.get("avg_val_loss","?"):.4f}')

# Parse CDR regions
def parse_regions(spec):
    regions = {}
    for part in spec.split(','):
        chain, rng = part.split(':')
        s, e = map(int, rng.split('-'))
        regions.setdefault(chain, []).extend(range(s-1, e))
    return {k: sorted(set(v)) for k, v in regions.items()}

all_designs = []

for scaffold_pdb, scaff_chain in SCAFFOLDS:
    cdr_spec = SCAFFOLD_CDRS.get(os.path.basename(scaffold_pdb), CDR_SPEC)
    regions_dict = parse_regions(cdr_spec)
    tx = get_transform([
        {'type': 'mask_region', 'regions': regions_dict},
        {'type': 'merge_protein'},
        {'type': 'patch_protein'},
    ])
    for antigen_pdb, ag_chain in ANTIGENS:
        label = f'{os.path.basename(scaffold_pdb).replace(".pdb","")}_{os.path.basename(antigen_pdb).replace(".pdb","")}'
        print(f'\n{label}...')

        # Preprocess both and merge chains
        scaff_struct = preprocess_protein_structure(scaffold_pdb, chain_ids=[scaff_chain])
        ag_struct = preprocess_protein_structure(antigen_pdb, chain_ids=[ag_chain])
        if scaff_struct is None or ag_struct is None:
            print(f'  SKIP: preprocess failed')
            continue

        # Merge scaffold + antigen chains into one structure
        merged = {
            'id': f'{scaff_chain}+{ag_chain}',
            'chains': scaff_struct['chains'] + ag_struct['chains'],
            'num_chains': scaff_struct['num_chains'] + ag_struct['num_chains'],
            'all_chain_ids': scaff_struct['all_chain_ids'] + ag_struct['all_chain_ids'],
        }

        for si in range(N_SAMPLES):
            seed_all(42 + si)
            batch = recursive_to(PaddingCollate()([tx(merged)]), 'cpu')
            gen_mask = batch['generate_flag'][0].bool()
            if gen_mask.sum() == 0:
                print(f'  SKIP: no CDR mask')
                continue

            with torch.no_grad():
                traj = model.sample(batch, sample_opt={
                    'deterministic': False, 'num_recycles': 3})

            pred_logits = traj['pred_logits'][0, gen_mask.cpu()]
            cdr_seq = ''.join(AA[a] for a in pred_logits.argmax(dim=-1).tolist())
            iptm = float(traj.get('iptm', torch.tensor([0])).mean())
            plddt = float(traj.get('plddt', torch.zeros(1, gen_mask.sum()))[0].mean())

            all_designs.append({
                'scaffold': scaffold_pdb,
                'antigen': antigen_pdb,
                'cdr_seq': cdr_seq,
                'bfn_iptm': iptm,
                'bfn_plddt': plddt,
                'sample': si,
            })

            if (si + 1) % 5 == 0:
                print(f'  {si+1}/{N_SAMPLES} designs generated')

print(f'\nTotal: {len(all_designs)} designs ({len(SCAFFOLDS)} scaffolds x {len(ANTIGENS)} antigens x {N_SAMPLES} samples)')

# Save
os.makedirs('oc_validation_results', exist_ok=True)
out_path = 'oc_validation_results/oc_v17i_phase1_complex.json'
with open(out_path, 'w') as f:
    json.dump(all_designs, f, indent=2)
print(f'Saved to {out_path}')

# Print top designs by BFN ipTM
sorted_d = sorted(all_designs, key=lambda d: d['bfn_iptm'], reverse=True)
print(f'\n=== Top-5 by BFN ipTM ===')
for i, d in enumerate(sorted_d[:5]):
    print(f'#{i+1}: iptm={d["bfn_iptm"]:.4f} plddt={d["bfn_plddt"]:.3f} '
          f'CDR={d["cdr_seq"][:60]}...')

# Now generate grafted PDBs for AF2
print(f'\nGenerating grafted PDBs for AF2...')
from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain, Residue, Atom
import numpy as np

def graft_cdrs_to_pdb(scaffold_pdb, scaffold_chain, cdr_seq, cdr_spec, out_path):
    """Graft designed CDR sequence into scaffold PDB, save as new PDB."""
    parser = PDBParser(QUIET=True)
    s = parser.get_structure('s', scaffold_pdb)

    # Get scaffold sequence
    scaff_seq = []
    for res in s[0][scaffold_chain]:
        if 'CA' in res:
            scaff_seq.append(res.resname)

    # Parse CDR regions
    cdr_regions = []
    for part in cdr_spec.split(','):
        chain, rng = part.split(':')
        s_idx, e_idx = map(int, rng.split('-'))
        if chain == scaffold_chain:
            cdr_regions.append((s_idx - 1, e_idx))

    # Graft designed CDRs
    new_seq = list(scaff_seq)
    pos = 0
    for start, end in cdr_regions:
        for j in range(start, end):
            if pos < len(cdr_seq):
                aa_char = cdr_seq[pos]
                aa3 = {'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','E':'GLU',
                       'Q':'GLN','G':'GLY','H':'HIS','I':'ILE','L':'LEU','K':'LYS',
                       'M':'MET','F':'PHE','P':'PRO','S':'SER','T':'THR','W':'TRP',
                       'Y':'TYR','V':'VAL'}.get(aa_char, 'GLY')
                try:
                    s[0][scaffold_chain][j+1].resname = aa3
                except KeyError:
                    pass
                pos += 1

    io = PDBIO()
    io.set_structure(s)
    io.save(out_path)

af2_dir = 'oc_validation_results/_complexes_v17i'
os.makedirs(af2_dir, exist_ok=True)
for d in all_designs:  # ALL designs, not just top-10
    scaffold = d['scaffold']
    antigen = d['antigen']
    scaff_name = os.path.basename(scaffold).replace('.pdb', '')
    ag_name = os.path.basename(antigen).replace('.pdb', '')
    out_pdb = os.path.join(af2_dir, f'{scaff_name}_{ag_name}_s{d["sample"]:02d}.pdb')

    scaff_chain = 'B' if '5IMK' in scaffold else 'A'
    cdr_spec_scaff = SCAFFOLD_CDRS.get(os.path.basename(scaffold), CDR_SPEC)
    graft_cdrs_to_pdb(scaffold, scaff_chain, d['cdr_seq'], cdr_spec_scaff, out_pdb)
    d['grafted_pdb'] = out_pdb

# Update JSON with PDB paths
with open(out_path, 'w') as f:
    json.dump(all_designs, f, indent=2)

print(f'Top-10 grafted PDBs saved to {af2_dir}/')
print(f'\nNext: run AF2 on these PDBs with antigen context.')
print(f'  python af2_wsl_batch.py {af2_dir}/ --antigen {ANTIGENS[0][0]}')
