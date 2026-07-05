import os
#!/usr/bin/env python3
"""V10 P1: HDOCK design candidate docking (quick: 5+5 designs + baselines)."""
import sys, os, pickle, time, json, subprocess, shutil, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..')); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))
import numpy as np
import torch, yaml
import bfn_loader
from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.train import recursive_to
from disorderflow.utils.transforms import get_transform
from disorderflow.utils.protein import parsers, constants
from disorderflow.datasets.sabdab import _label_heavy_chain_cdr, _label_light_chain_cdr
from Bio.PDB import PDBParser, PDBIO

AA = 'ARNDCQEGHILKMFPSTWYV'
A3 = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN',
      'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS',
      'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP',
      'Y': 'TYR', 'V': 'VAL'}

V19 = 'logs/bfn_v19_diversity_xpu_2026_07_02__14_23_57_v19_final/checkpoints/best.pt'
AB = 'data/anti_abeta_refs/4HIX.pdb'

print("=" * 50)
print("V10 P1: HDOCK Design Candidate Docking")
print("=" * 50)

# Load model
print("Loading V19...")
app = yaml.safe_load(open('app_config.yaml', encoding='utf-8'))
orig_ckpt = app['models']['bfn']['checkpoint']
app['models']['bfn']['checkpoint'] = V19
yaml.dump(app, open('app_config.yaml', 'w', encoding='utf-8'))
bfn_loader._bfn_model = None; bfn_loader._bfn_config = None
model, _ = bfn_loader.load_bfn('cpu')
app['models']['bfn']['checkpoint'] = orig_ckpt
yaml.dump(app, open('app_config.yaml', 'w', encoding='utf-8'))

# Parse 4HIX
parser = PDBParser(QUIET=True)
s = parser.get_structure('4HIX', AB)[0]
hd, hm = parsers.parse_biopython_structure(s['H'], max_resseq=113)
hd, hm = _label_heavy_chain_cdr(hd, hm)
ld, lm = parsers.parse_biopython_structure(s['L'], max_resseq=106)
ld, lm = _label_light_chain_cdr(ld, lm)
ag_data, _ = parsers.parse_biopython_structure(s['A'])

def get_cdr_info(data, cts):
    info = {}
    for ct in cts:
        m = (data['cdr_flag'] == ct)
        if m.any():
            idx = m.nonzero(as_tuple=True)[0]
            aas = data['aa'][idx]
            seq = ''.join([AA[a.item()] if 0 <= a.item() < 20 else 'X' for a in aas])
            info[ct] = {'idx': idx, 'seq': seq, 'len': len(idx)}
    return info

hci = get_cdr_info(hd, [1, 2, 3])
lci = get_cdr_info(ld, [4, 5, 6])
native_h = ''.join([hci[ct]['seq'] for ct in [1, 2, 3]])
native_l = ''.join([lci[ct]['seq'] for ct in [4, 5, 6]])
print("Native H CDRs: %s" % native_h)
print("Native L CDRs: %s" % native_l)

# Sample CDRs with V19
tr = get_transform([{'type': 'mask_multiple_cdrs'}, {'type': 'merge_chains'},
                     {'type': 'patch_around_anchor'}])
struct = {'heavy': hd, 'light': ld, 'antigen': ag_data}

def sample_cdrs(epi_d, n):
    seqs = []
    for i in range(n):
        bd = tr(struct)
        batch = recursive_to(PaddingCollate()([bd]), 'cpu')
        L = batch['aa'].shape[1]
        al = len(ag_data['aa'])
        d = np.ones(L) * epi_d
        batch['epitope_disorder_profile'] = torch.tensor(d, dtype=torch.float32).unsqueeze(0)
        batch['mask_antigen'] = torch.zeros(1, L).bool()
        batch['mask_antigen'][0, -al:] = True
        batch['mask'] = torch.ones(1, L).bool()
        torch.manual_seed(i * 100 + 42)
        with torch.no_grad():
            try:
                traj = model.sample(batch, sample_opt={'deterministic': False, 'num_recycles': 1})
            except Exception:
                continue
        gf = batch['generate_flag'][0].bool()
        if not gf.any():
            continue
        logits = traj['pred_logits'][0, gf]
        pred_aa = logits.argmax(dim=-1).cpu().numpy()
        seq = ''.join(AA[a] for a in pred_aa)
        seqs.append(seq)
    return seqs

print("Sampling designs...")
hi_seqs = sample_cdrs(0.7, 5)
lo_seqs = sample_cdrs(0.05, 5)
print("  High-disorder: %d seqs" % len(hi_seqs))
print("  Low-disorder: %d seqs" % len(lo_seqs))
for i, s in enumerate(hi_seqs):
    print("    hi_%02d: %s" % (i, s[:25]))
for i, s in enumerate(lo_seqs):
    print("    lo_%02d: %s" % (i, s[:25]))

# Graft CDRs into PDB
def graft(out_path, h_seq, l_seq):
    ps = PDBParser(QUIET=True)
    st = ps.get_structure('ref', AB)[0]
    hc = st['H'] if 'H' in st else None
    if hc and h_seq and len(h_seq) == len(native_h):
        pos = 0
        for ct in [1, 2, 3]:
            if ct in hci:
                for idx in hci[ct]['idx']:
                    if pos < len(h_seq):
                        residues = list(hc.get_residues())
                        if idx.item() < len(residues):
                            residues[idx.item()].resname = A3.get(h_seq[pos], 'ALA')
                    pos += 1
    lc = st['L'] if 'L' in st else None
    if lc and l_seq and len(l_seq) == len(native_l):
        pos = 0
        for ct in [4, 5, 6]:
            if ct in lci:
                for idx in lci[ct]['idx']:
                    if pos < len(l_seq):
                        residues = list(lc.get_residues())
                        if idx.item() < len(residues):
                            residues[idx.item()].resname = A3.get(l_seq[pos], 'ALA')
                    pos += 1
    io2 = PDBIO()
    io2.set_structure(st)
    io2.save(out_path)

# Scramble
scram_h = list(native_h)
random.shuffle(scram_h)
scram_h = ''.join(scram_h)
scram_l = list(native_l)
random.shuffle(scram_l)
scram_l = ''.join(scram_l)

os.makedirs('hdock_design_results/receptors', exist_ok=True)
graft('hdock_design_results/receptors/native.pdb', native_h, native_l)
graft('hdock_design_results/receptors/scrambled.pdb', scram_h, scram_l)
for i, seq in enumerate(hi_seqs):
    graft('hdock_design_results/receptors/hi_%02d.pdb' % i, seq, native_l)
for i, seq in enumerate(lo_seqs):
    graft('hdock_design_results/receptors/lo_%02d.pdb' % i, seq, native_l)
shutil.copy('HDOCKlite-v1.1/4HIX_ligand_native.pdb', 'hdock_design_results/ligand.pdb')
print("Grafted %d receptors" % (2 + len(hi_seqs) + len(lo_seqs)))

# HDOCK docking
def hdock_score(name):
    rec = '../hdock_design_results/receptors/%s.pdb' % name
    lig = '../hdock_design_results/ligand.pdb'
    out = 'p1_%s.out' % name
    cmd = "cd HDOCKlite-v1.1 && LD_LIBRARY_PATH=. ./hdock %s %s -out %s 2>&1" % (rec, lig, out)
    try:
        r = subprocess.run(['wsl', 'bash', '-c', cmd], capture_output=True, text=True, timeout=180)
    except Exception as e:
        print("    HDOCK error: %s" % e)
        return None
    op = 'HDOCKlite-v1.1/%s' % out
    if not os.path.exists(op):
        return None
    scores = []
    with open(op) as f:
        for line in f:
            p = line.strip().split()
            if len(p) == 9:
                try:
                    scores.append(float(p[6]))
                except ValueError:
                    pass
    if not scores:
        return None
    return {'top1': min(scores), 'top5': sorted(scores)[:5],
            'mean': np.mean(scores), 'n': len(scores)}

print("\nRunning HDOCK dockings (~50s each)...")
results = {}
for name in ['native', 'scrambled'] + ['hi_%02d' % i for i in range(5)] + ['lo_%02d' % i for i in range(5)]:
    rec_path = 'hdock_design_results/receptors/%s.pdb' % name
    if not os.path.exists(rec_path):
        continue
    print("  %s..." % name, end=' ', flush=True)
    r = hdock_score(name)
    if r:
        results[name] = r
        print("top1=%.1f (n=%d)" % (r['top1'], r['n']))
    else:
        print("FAILED")

# Summary
print("\n" + "=" * 50)
print("P1 Results")
print("=" * 50)
for cond, names in [
    ('native', ['native']),
    ('scrambled', ['scrambled']),
    ('high_design', ['hi_%02d' % i for i in range(5)]),
    ('low_design', ['lo_%02d' % i for i in range(5)]),
]:
    scores = [results[n]['top1'] for n in names if n in results]
    if scores:
        print("%-15s n=%d top1_mean=%.1f top1_min=%.1f" %
              (cond, len(scores), np.mean(scores), np.min(scores)))

ts = time.strftime('%Y%m%d_%H%M%S')
out_path = 'idp_design_results/hdock_design_scores_%s.json' % ts
with open(out_path, 'w') as f:
    json.dump({'probe': 'V10 P1', 'results': results}, f, indent=2)
print("Saved: %s" % out_path)
print("P1 done.")
