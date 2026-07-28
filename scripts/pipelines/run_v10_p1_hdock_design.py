import os
#!/usr/bin/env python3
"""V10 P1: WAY4 candidate HDOCK docking verification.

Generates CDR designs with V19 at different disorder levels, grafts them
into the 4HIX antibody scaffold, runs HDOCK docking against the native
Abeta16-23 peptide, and compares scores to native/scrambled baselines.
"""
import sys, os, pickle, time, json, subprocess, shutil, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..')); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))
import numpy as np
import torch, yaml

# Config
V19_CKPT = 'logs/bfn_v19_diversity_xpu_2026_07_02__14_23_57_v19_final/checkpoints/best.pt'
ABETA_REF = 'data/anti_abeta_refs/4HIX.pdb'
HDOCK_DIR = 'HDOCKlite-v1.1'
HDOCK_LD = 'LD_LIBRARY_PATH=.'
N_CANDIDATES = 10   # CDR designs per condition
N_SEEDS = 3          # sampling seeds per design
AA = 'ARNDCQEGHILKMFPSTWYV'

os.makedirs('hdock_design_results', exist_ok=True)

print("=" * 60)
print("V10 P1: WAY4 Candidate HDOCK Docking Verification")
print("=" * 60)

# Load model
import bfn_loader
from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.train import recursive_to
from disorderflow.utils.transforms import get_transform
from disorderflow.utils.protein import parsers, constants
from disorderflow.datasets.sabdab import _label_heavy_chain_cdr, _label_light_chain_cdr
from Bio.PDB import PDBParser, PDBIO

app = yaml.safe_load(open('app_config.yaml', encoding='utf-8'))
orig_ckpt = app['models']['bfn']['checkpoint']
app['models']['bfn']['checkpoint'] = V19_CKPT
yaml.dump(app, open('app_config.yaml', 'w', encoding='utf-8'))
bfn_loader._bfn_model = None; bfn_loader._bfn_config = None
model, _ = bfn_loader.load_bfn('cpu')
app['models']['bfn']['checkpoint'] = orig_ckpt
yaml.dump(app, open('app_config.yaml', 'w', encoding='utf-8'))
print("V19 model loaded")

# Parse 4HIX structure and identify CDRs
parser = PDBParser(QUIET=True)
structure = parser.get_structure('4HIX', ABETA_REF)[0]
hd, hm = parsers.parse_biopython_structure(structure['H'], max_resseq=113)
hd, hm = _label_heavy_chain_cdr(hd, hm)
ld, lm = parsers.parse_biopython_structure(structure['L'], max_resseq=106)
ld, lm = _label_light_chain_cdr(ld, lm)

def get_cdr_info(data, cdr_types):
    """Get CDR positions and native sequences."""
    info = {}
    for ct in cdr_types:
        mask = (data['cdr_flag'] == ct)
        if mask.any():
            indices = mask.nonzero(as_tuple=True)[0]
            aa = data['aa'][indices]
            seq = ''.join([AA[a.item()] if 0 <= a.item() < 20 else 'X' for a in aa])
            info[ct] = {'indices': indices, 'seq': seq, 'len': len(indices)}
    return info

h_cdr_info = get_cdr_info(hd, [1, 2, 3])
l_cdr_info = get_cdr_info(ld, [4, 5, 6])
print("CDR info extracted")

# Build the merged antibody structure for BFN sampling
transform = get_transform([
    {'type': 'mask_multiple_cdrs'},
    {'type': 'merge_chains'},
    {'type': 'patch_around_anchor'}
])

# Parse antigen (A-beta peptide)
ag_data, _ = parsers.parse_biopython_structure(structure['A'])
structure_dict = {'heavy': hd, 'light': ld, 'antigen': ag_data}

def sample_cdr_sequences(model, structure_dict, epitope_disorder_val, n_samples, n_seeds):
    """Sample CDR sequences from the model at a given disorder level."""
    sequences = []
    for i in range(n_samples):
        batch_data = transform(structure_dict)
        batch = recursive_to(PaddingCollate()([batch_data]), 'cpu')
        L = batch['aa'].shape[1]

        # Set epitope disorder profile
        d = np.ones(L) * epitope_disorder_val
        batch['epitope_disorder_profile'] = torch.tensor(d, dtype=torch.float32).unsqueeze(0)
        batch['mask_antigen'] = torch.zeros(1, L).bool()
        batch['mask_antigen'][0, -len(ag_data['aa']):] = True
        batch['mask'] = torch.ones(1, L).bool()

        for seed in range(n_seeds):
            torch.manual_seed(i * 100 + seed)
            with torch.no_grad():
                try:
                    traj = model.sample(batch, sample_opt={
                        'deterministic': False, 'num_recycles': 1})
                except Exception:
                    continue
            gen_flag = batch['generate_flag'][0].bool()
            if not gen_flag.any():
                continue
            logits = traj['pred_logits'][0, gen_flag]
            pred_aa = logits.argmax(dim=-1).cpu().numpy()
            seq = ''.join(AA[a] for a in pred_aa)

            # Split into H and L CDRs based on generate_flag positions
            gen_positions = gen_flag.nonzero(as_tuple=True)[0]
            # Determine which positions belong to heavy vs light
            # Heavy comes first in merged structure
            h_len = len(hd['aa'])
            l_len = len(ld['aa']) if ld else 0
            h_gen_mask = gen_positions < h_len
            l_gen_mask = gen_positions >= h_len

            h_cdr_seq = ''.join([AA[logits[j].argmax().item()] for j in range(len(gen_positions)) if h_gen_mask[j]])
            l_cdr_seq = ''.join([AA[logits[j].argmax().item()] for j in range(len(gen_positions)) if l_gen_mask[j]])

            sequences.append({
                'h_cdr_seq': h_cdr_seq if h_cdr_seq else 'N/A',
                'l_cdr_seq': l_cdr_seq if l_cdr_seq else 'N/A',
                'full_seq': seq,
                'cdr_len': len(seq),
                'epitope_disorder': epitope_disorder_val,
            })
    return sequences


def graft_cdrs_to_pdb(pdb_path, out_path, h_cdr_seq, l_cdr_seq, h_cdr_info, l_cdr_info):
    """Graft CDR sequences into a PDB by renaming CDR residues.
    Keeps backbone coordinates, only changes residue names.
    """
    parser2 = PDBParser(QUIET=True)
    s = parser2.get_structure('ref', pdb_path)[0]

    # Graft heavy chain CDRs
    h_chain = s['H'] if 'H' in s else None
    if h_chain and h_cdr_seq:
        # Concatenate all H CDRs
        all_h_cdr = ''
        for ct in [1, 2, 3]:
            if ct in h_cdr_info:
                all_h_cdr += h_cdr_info[ct]['seq']
        if len(h_cdr_seq) == len(all_h_cdr):
            # Map designed sequence back to CDR positions
            pos = 0
            for ct in [1, 2, 3]:
                if ct in h_cdr_info:
                    for idx in h_cdr_info[ct]['indices']:
                        resname = AA_to_3(h_cdr_seq[pos]) if pos < len(h_cdr_seq) else 'ALA'
                        try:
                            # Find residue by index
                            residues = list(h_chain.get_residues())
                            if idx.item() < len(residues):
                                residues[idx.item()].resname = resname
                        except Exception:
                            pass
                        pos += 1

    # Graft light chain CDRs
    l_chain = s['L'] if 'L' in s else None
    if l_chain and l_cdr_seq:
        all_l_cdr = ''
        for ct in [4, 5, 6]:
            if ct in l_cdr_info:
                all_l_cdr += l_cdr_info[ct]['seq']
        if len(l_cdr_seq) == len(all_l_cdr):
            pos = 0
            for ct in [4, 5, 6]:
                if ct in l_cdr_info:
                    for idx in l_cdr_info[ct]['indices']:
                        resname = AA_to_3(l_cdr_seq[pos]) if pos < len(l_cdr_seq) else 'ALA'
                        try:
                            residues = list(l_chain.get_residues())
                            if idx.item() < len(residues):
                                residues[idx.item()].resname = resname
                        except Exception:
                            pass
                        pos += 1

    io = PDBIO()
    io.set_structure(s)
    io.save(out_path)


def AA_to_3(aa):
    """One-letter AA code to three-letter."""
    mapping = {'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','Q':'GLN','E':'GLU',
               'G':'GLY','H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE',
               'P':'PRO','S':'SER','T':'THR','W':'TRP','Y':'TYR','V':'VAL'}
    return mapping.get(aa, 'ALA')


def run_hdock(receptor_pdb, ligand_pdb, out_name):
    """Run HDOCK and return top-1 score."""
    cmd = (f'cd {HDOCK_DIR} && {HDOCK_LD} ./hdock {receptor_pdb} {ligand_pdb} '
           f'-out {out_name} 2>&1')
    try:
        result = subprocess.run(['wsl', 'bash', '-c', cmd],
                               capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return None

    out_path = os.path.join(HDOCK_DIR, out_name)
    if not os.path.exists(out_path):
        return None

    scores = []
    with open(out_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 9:
                try:
                    scores.append(float(parts[6]))
                except ValueError:
                    continue
    if not scores:
        return None
    return {'top1': min(scores), 'top5': sorted(scores)[:5],
            'mean': np.mean(scores), 'n': len(scores)}


print("\n--- Generating CDR candidates ---")

# High disorder (0.7) and low disorder (0.05) conditions
conditions = [
    ('high_disorder', 0.7, N_CANDIDATES),
    ('low_disorder', 0.05, N_CANDIDATES),
]

all_candidates = []
for cond_name, epi_d, n in conditions:
    print(f"  Sampling {cond_name} (epi_d={epi_d})...")
    seqs = sample_cdr_sequences(model, structure_dict, epi_d, n, N_SEEDS)
    for s in seqs:
        s['condition'] = cond_name
    all_candidates.extend(seqs)
    # Show diversity
    unique_seqs = set(s['full_seq'] for s in seqs)
    print(f"    Generated {len(seqs)} sequences, {len(unique_seqs)} unique")

# Select top candidates: highest diversity (unique_aa proxy via sequence uniqueness)
# For each condition, deduplicate and pick top
by_cond = {}
for c in all_candidates:
    by_cond.setdefault(c['condition'], []).append(c)

selected = []
for cond, cands in by_cond.items():
    # Deduplicate
    seen = set()
    unique = []
    for c in cands:
        if c['full_seq'] not in seen:
            seen.add(c['full_seq'])
            unique.append(c)
    # Pick top 5 by CDR length (more complete) and diversity
    unique.sort(key=lambda x: len(set(x['full_seq'])), reverse=True)
    selected.extend(unique[:5])

print(f"\nSelected {len(selected)} candidates for HDOCK docking")

# Prepare native and scrambled baselines
# Native: already docked in P0-3
# Scrambled: generate by shuffling native CDR sequences

# Get native CDR sequences
native_h = ''.join([h_cdr_info[ct]['seq'] for ct in [1,2,3]])
native_l = ''.join([l_cdr_info[ct]['seq'] for ct in [4,5,6]])
print(f"Native: H={native_h} L={native_l}")

# Create scrambled
scram_h = list(native_h); random.shuffle(scram_h); scram_h = ''.join(scram_h)
scram_l = list(native_l); random.shuffle(scram_l); scram_l = ''.join(scram_l)
print(f"Scrambled: H={scram_h} L={scram_l}")

# Graft native CDRs into receptor
print("\n--- Grafting CDRs into 4HIX scaffold ---")
os.makedirs('hdock_design_results/receptors', exist_ok=True)

# Native receptor
graft_cdrs_to_pdb(ABETA_REF, 'hdock_design_results/receptors/native.pdb',
                  native_h, native_l, h_cdr_info, l_cdr_info)

# Scrambled receptor
graft_cdrs_to_pdb(ABETA_REF, 'hdock_design_results/receptors/scrambled.pdb',
                  scram_h, scram_l, h_cdr_info, l_cdr_info)

# Design candidates
design_files = []
for i, c in enumerate(selected):
    out_path = f'hdock_design_results/receptors/design_{i:03d}.pdb'
    graft_cdrs_to_pdb(ABETA_REF, out_path,
                      c['h_cdr_seq'], c['l_cdr_seq'],
                      h_cdr_info, l_cdr_info)
    design_files.append({'path': out_path, **c})

# Copy native ligand
shutil.copy('HDOCKlite-v1.1/4HIX_ligand_native.pdb',
            'hdock_design_results/ligand.pdb')

print(f"Grafted {len(design_files) + 2} receptors")

# Run HDOCK on all
print("\n--- Running HDOCK docking (this will take a while) ---")
all_results = []

# Use relative paths from HDOCK directory
for name, receptor_rel in [
    ('native', '../hdock_design_results/receptors/native.pdb'),
    ('scrambled', '../hdock_design_results/receptors/scrambled.pdb'),
]:
    print(f"  Docking {name}...")
    result = run_hdock(receptor_rel, '../hdock_design_results/ligand.pdb',
                       f'p1_{name}.out')
    if result:
        result['name'] = name
        result['condition'] = name
        all_results.append(result)
        print(f"    top1={result['top1']:.2f}")

for i, d in enumerate(design_files):
    receptor_rel = f'../hdock_design_results/receptors/design_{i:03d}.pdb'
    print(f"  Docking design_{i:03d} ({d['condition']})...")
    result = run_hdock(receptor_rel, '../hdock_design_results/ligand.pdb',
                       f'p1_design_{i:03d}.out')
    if result:
        result['name'] = f'design_{i:03d}'
        result['condition'] = d['condition']
        result['cdr_len'] = d['cdr_len']
        result['epitope_disorder'] = d['epitope_disorder']
        all_results.append(result)
        print(f"    top1={result['top1']:.2f}")

# Analysis
print(f"\n{'='*60}")
print("P1 HDOCK Docking Results")
print(f"{'='*60}")

by_cond = {}
for r in all_results:
    by_cond.setdefault(r['condition'], []).append(r['top1'])

print(f"{'Condition':<20} {'N':>4} {'Top1 Mean':>10} {'Top1 Min':>10}")
print("-" * 50)
for cond in ['native', 'high_disorder', 'low_disorder', 'scrambled']:
    scores = by_cond.get(cond, [])
    if scores:
        print(f"{cond:<20} {len(scores):>4} {np.mean(scores):>10.2f} {np.min(scores):>10.2f}")

# Save
output = {
    'probe': 'V10 P1 HDOCK design verification',
    'n_designs': len(design_files),
    'results': all_results,
    'candidates': selected,
}
ts = time.strftime('%Y%m%d_%H%M%S')
out_path = f'idp_design_results/hdock_design_scores_{ts}.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {out_path}")
print("P1 complete.")
