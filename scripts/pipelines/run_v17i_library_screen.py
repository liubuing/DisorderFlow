import os
#!/usr/bin/env python3
"""§4.2 CDR Library Screening: generate diverse designs, rank, AF2 validate.

Break the CDR diversity ceiling by:
1. Multiple random seeds + disorder-guided sampling for exploration
2. Composite scoring: BFN ipTM + pLDDT - disorder_penalty + diversity_bonus
3. Sequence clustering to de-duplicate near-identical designs
4. Top-20 sent to AF2 for final ranking

Usage:
    python run_v17i_library_screen.py [--samples 200] [--top 20]
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..')); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))
import torch, yaml, numpy as np
from easydict import EasyDict
from collections import defaultdict

AA = 'ACDEFGHIKLMNPQRSTVWY'

CKPT = 'logs/v17c_zeroinit/bfn_v17c_zeroinit_xpu_2026_06_26__23_22_18/checkpoints/4600.pt'
N_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 100
TOP_K = int(sys.argv[2]) if len(sys.argv) > 2 else 20

SCAFFOLDS = [
    ('data/misfolding_targets/3STB.pdb', 'A', 'A:26-33,A:51-58,A:97-113'),
    ('data/misfolding_targets/5IMK.pdb', 'B', 'B:26-33,B:51-58,B:97-113'),
]
ANTIGENS = [
    'data/abeta_conformations/pdbs/abeta42_seed0_42.pdb',
    'data/abeta_conformations/pdbs/abeta42_seed2_42.pdb',
    'data/abeta_conformations/pdbs/abeta42_seed4_42.pdb',
]

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
print(f'V17i step {ckpt["iteration"]} val {ckpt.get("avg_val_loss","?"):.4f}')

def parse_regions(spec):
    regions = {}
    for part in spec.split(','):
        chain, rng = part.split(':')
        s, e = map(int, rng.split('-'))
        regions.setdefault(chain, []).extend(range(s-1, e))
    return {k: sorted(set(v)) for k, v in regions.items()}

all_designs = []
for scaffold_pdb, scaff_chain, cdr_spec in SCAFFOLDS:
    regions_dict = parse_regions(cdr_spec)
    tx = get_transform([
        {'type': 'mask_region', 'regions': regions_dict},
        {'type': 'merge_protein'},
        {'type': 'patch_protein'},
    ])
    for antigen_pdb in ANTIGENS:
        scaff_struct = preprocess_protein_structure(scaffold_pdb, chain_ids=[scaff_chain])
        ag_struct = preprocess_protein_structure(antigen_pdb, chain_ids=['P'])
        if scaff_struct is None or ag_struct is None: continue

        merged = {
            'id': f'{scaff_chain}+P',
            'chains': scaff_struct['chains'] + ag_struct['chains'],
            'num_chains': scaff_struct['num_chains'] + ag_struct['num_chains'],
            'all_chain_ids': scaff_struct['all_chain_ids'] + ag_struct['all_chain_ids'],
        }

        samples_per = N_SAMPLES // (len(SCAFFOLDS) * len(ANTIGENS))
        for si in range(samples_per):
            seed_all(42 + si * 137 + hash(scaffold_pdb) % 1000)
            batch = recursive_to(PaddingCollate()([tx(merged)]), 'cpu')
            gen_mask = batch['generate_flag'][0].bool()
            if gen_mask.sum() == 0: continue

            with torch.no_grad():
                # Use disorder-guided sampling for diversity (§A from plan)
                traj = model.sample(batch, sample_opt={
                    'deterministic': False,
                    'num_recycles': 3,
                    'disorder_guided': True,
                    'disorder_guided_strength': 2.0,  # aggressive exploration
                })

            pred_logits = traj['pred_logits'][0, gen_mask.cpu()]
            cdr_seq = ''.join(AA[a] for a in pred_logits.argmax(dim=-1).tolist())
            iptm = float(traj.get('iptm', torch.tensor([0])).mean())
            plddt = float(traj.get('plddt', torch.zeros(1, gen_mask.sum()))[0].mean())
            pred_entropy = float(traj.get('pred_entropy', torch.tensor([0])).mean())

            all_designs.append({
                'scaffold': os.path.basename(scaffold_pdb),
                'antigen': os.path.basename(antigen_pdb),
                'cdr_seq': cdr_seq,
                'bfn_iptm': iptm,
                'bfn_plddt': plddt,
                'entropy': pred_entropy,
                'seed': si,
            })
        print(f'{os.path.basename(scaffold_pdb)}+{os.path.basename(antigen_pdb)}: {samples_per} designs')

# ---- Inject known anti-Aβ CDR templates (§2.4) ----
KNOWN_TEMPLATES = [
    # 5CSZ aducanumab-like Fab: H1=GFTFSSYA, H2=INASGTR, H3=ARGKGYV (Chothia)
    {'cdr': 'GFTFSSYAINSASGTRTARYCARGRGYV', 'source': '5CSZ_H'},
    # 4HIX another anti-Aβ: H1=GFTFSSYA, H2=ISGSGGS, H3=AKDRYSGY
    {'cdr': 'GFTFSSYAVSGSGGSTTYAKDRYSGY', 'source': '4HIX_H'},
]
for t in KNOWN_TEMPLATES:
    all_designs.append({
        'scaffold': 'template', 'antigen': 'template',
        'cdr_seq': t['cdr'],
        'bfn_iptm': 0.0, 'bfn_plddt': 0.0, 'entropy': 0.0,
        'seed': -1,
    })

print(f'\nGenerated {len(all_designs)} designs (+{len(KNOWN_TEMPLATES)} known templates)')

# ---- Diversity scoring + de-duplication ----
# Cluster by sequence identity
def seq_identity(s1, s2):
    if len(s1) != len(s2): return 0
    return sum(a==b for a,b in zip(s1,s2)) / len(s1)

# Simple greedy clustering: keep best BFN iptm per cluster
clusters = []
for d in sorted(all_designs, key=lambda x: x['bfn_iptm'], reverse=True):
    is_new = True
    for c in clusters:
        if seq_identity(d['cdr_seq'], c['cdr_seq']) > 0.85:
            is_new = False
            break
    if is_new:
        clusters.append(d)

print(f'After dedup (85% identity): {len(clusters)} unique designs')

# Composite score: ipTM + pLDDT + entropy bonus (more entropy = more diverse)
for d in clusters:
    # Higher entropy = reward exploration
    d['composite'] = (0.4 * d['bfn_iptm'] + 0.3 * d['bfn_plddt'] +
                      0.15 * min(d['entropy'], 2.0) / 2.0 +
                      0.15 * (1.0 - abs(d['bfn_iptm'] - 0.12)))  # proximity to sweet spot

clusters.sort(key=lambda x: x['composite'], reverse=True)
top = clusters[:TOP_K]

print(f'\n=== Top-{TOP_K} diverse designs ===')
for i, d in enumerate(top):
    print(f'#{i+1}: composite={d["composite"]:.4f} iptm={d["bfn_iptm"]:.4f} '
          f'plddt={d["bfn_plddt"]:.3f} entropy={d["entropy"]:.3f} '
          f'{d["scaffold"]}+{d["antigen"]}')
    print(f'    CDR: {d["cdr_seq"][:80]}')

# Save
os.makedirs('oc_validation_results', exist_ok=True)
with open('oc_validation_results/oc_v17i_library_top20.json', 'w') as f:
    json.dump(top, f, indent=2)
print(f'\nSaved top-20 to oc_validation_results/oc_v17i_library_top20.json')

# Generate grafted PDBs for AF2
af2_dir = 'oc_validation_results/_complexes_v17i_library'
os.makedirs(af2_dir, exist_ok=True)

# Map back to original scaffold/antigen paths
scaffold_map = {os.path.basename(s): (s, c) for s, c, _ in SCAFFOLDS}
antigen_map = {os.path.basename(a): a for a in ANTIGENS}

from Bio.PDB import PDBParser, PDBIO
def graft_cdrs(scaffold_pdb, scaffold_chain, cdr_seq, cdr_spec, out_path):
    parser = PDBParser(QUIET=True)
    s = parser.get_structure('s', scaffold_pdb)
    aa3 = {'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','E':'GLU','Q':'GLN',
           'G':'GLY','H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE',
           'P':'PRO','S':'SER','T':'THR','W':'TRP','Y':'TYR','V':'VAL'}
    cdr_regions = []
    for part in cdr_spec.split(','):
        chain, rng = part.split(':')
        s_idx, e_idx = map(int, rng.split('-'))
        if chain == scaffold_chain:
            cdr_regions.append((s_idx - 1, e_idx))
    pos = 0
    for start, end in cdr_regions:
        for j in range(start, end):
            if pos < len(cdr_seq):
                aa_char = cdr_seq[pos]
                try:
                    s[0][scaffold_chain][j+1].resname = aa3.get(aa_char, 'GLY')
                except KeyError: pass
                pos += 1
    io = PDBIO(); io.set_structure(s); io.save(out_path)

for i, d in enumerate(top):
    if d['scaffold'] == 'template':
        # Graft templates onto 3STB (best scaffold)
        for scaff_path, scaff_chain, cdr_spec in SCAFFOLDS:
            if '3STB' in scaff_path:
                out_pdb = os.path.join(af2_dir, f'3STB_template_{d["source"]}.pdb')
                graft_cdrs(scaff_path, scaff_chain, d['cdr_seq'], cdr_spec, out_pdb)
                d['grafted_pdb'] = out_pdb
                d['scaffold'] = '3STB.pdb'
                break
    else:
        scaff_path, scaff_chain = scaffold_map[d['scaffold']]
        cdr_spec = next(cs for s, c, cs in SCAFFOLDS if os.path.basename(s) == d['scaffold'])
        out_pdb = os.path.join(af2_dir, f'{d["scaffold"].replace(".pdb","")}_{d["antigen"].replace(".pdb","")}_top{i+1:02d}.pdb')
        graft_cdrs(scaff_path, scaff_chain, d['cdr_seq'], cdr_spec, out_pdb)
        d['grafted_pdb'] = out_pdb

print(f'Grafted PDBs saved to {af2_dir}/')
print(f'\nAF2 run: modify run_v17i_af2.py INPUT path to oc_validation_results/oc_v17i_library_top20.json')
