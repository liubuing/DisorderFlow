#!/usr/bin/env python
"""MSA Validation with colabfold: Top 3 complex designs vs VSIG4.

Runs colabfold_batch with --msa-mode mmseqs2_uniref_env for proper MSA,
giving realistic AF2 scores (unlike single-seq JAX mode).
"""
import os, sys, json, subprocess, time

os.chdir('/home/liubuzi/disorderflow-main/disorderflow-main')
sys.path.insert(0, '.')
sys.path.insert(0, 'modules')

AA3TO1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
          'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
          'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

# Read 5IMK sequences
def read_pdb_seq(pdb_path, chain):
    seen = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('ATOM') and line[21] == chain:
                resid = int(line[22:26])
                if resid not in seen:
                    seen[resid] = AA3TO1.get(line[17:20].strip(), 'X')
    return ''.join(seen[i] for i in sorted(seen))

# VSIG4 antigen (chain A) + nanobody scaffold (chain B)
antigen_seq = read_pdb_seq('data/misfolding_targets/5IMK.pdb', 'A')
scaffold_seq = read_pdb_seq('data/misfolding_targets/5IMK.pdb', 'B')

# Load top 3 complex designs
with open('idp_design_results/ad_phase3_demo/fixbb_vs_complex.json') as f:
    data = json.load(f)

# We need the actual sequences from the complex design run
# Let me re-run a quick design to get top sequences
print("=" * 80)
print("  MSA Validation —colabfold with full MSA")
print("=" * 80)
print(f"  Antigen (VSIG4): {len(antigen_seq)} AA")
print(f"  Scaffold (VHH):  {len(scaffold_seq)} AA")
print()

# Get top designs from the complex mode
import torch
from bfn_loader import load_bfn
from disorderflow.datasets.protein import preprocess_protein_structure
from disorderflow.utils.train import recursive_to
from disorderflow.utils.misc import seed_all
from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.transforms import get_transform

AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model, config = load_bfn(DEVICE)
seed_all(42)

cdr_spec = {
    'B': list(range(26, 34)) + list(range(51, 59)) + list(range(97, 114))
}

structure = preprocess_protein_structure('data/misfolding_targets/5IMK.pdb', chain_ids=['A', 'B'])
transform = get_transform([
    {'type': 'mask_region', 'regions': cdr_spec},
    {'type': 'merge_protein'},
    {'type': 'patch_protein'},
])
batch = recursive_to(PaddingCollate()([transform(structure)]), DEVICE)
gen_mask = batch['generate_flag'][0].bool()

top_designs = []
for i in range(5):
    with torch.no_grad():
        traj = model.sample(batch, sample_opt={'deterministic': False, 'num_recycles': 3})
    pred_aa = traj[0][2][0][gen_mask]
    seq = ''.join(AA_LETTERS[a] if a < 20 else 'X' for a in pred_aa.cpu())

    logits = traj['pred_logits'][0][gen_mask]
    lp = torch.log_softmax(logits[..., :20], dim=-1)
    nll = -lp[range(len(pred_aa)), pred_aa].mean()
    ppl = torch.exp(nll).item()

    top_designs.append({
        'seq': seq,
        'plddt': traj['plddt'][0][gen_mask].mean().item(),
        'iptm': traj['iptm'][0].item(),
        'ppl': ppl,
    })

# Graft CDRs
cdr_regions = [(26, 33), (51, 58), (97, 113)]

def graft(scaffold, cdr_seq):
    full = list(scaffold)
    pos = 0
    for start, end in cdr_regions:
        length = end - start + 1
        cdr = cdr_seq[pos:pos+length]
        for j in range(length):
            full[start - 1 + j] = cdr[j]
        pos += length
    return ''.join(full)

# Create output directory and FASTA files
out_dir = 'idp_design_results/ad_phase3_demo/msa_validation'
os.makedirs(out_dir, exist_ok=True)

# Take top 3
for rank, d in enumerate(top_designs[:3]):
    ab_seq = graft(scaffold_seq, d['seq'])

    # Multimer FASTA: antibody chain first, then antigen
    fasta = f">antibody_{rank+1}\n{ab_seq}\n>antigen_VSIG4\n{antigen_seq}\n"

    job_dir = os.path.join(out_dir, f'design_{rank+1}')
    os.makedirs(job_dir, exist_ok=True)
    fasta_path = os.path.join(job_dir, 'multimer.fasta')
    with open(fasta_path, 'w') as f:
        f.write(fasta)

    mutations = sum(1 for j in range(len(scaffold_seq)) if scaffold_seq[j] != ab_seq[j])
    print(f"[Design {rank+1}] BFN pLDDT={d['plddt']:.3f} ipTM={d['iptm']:.3f} PPL={d['ppl']:.1f} | {mutations} mutations")
    print(f"  CDRs: {d['seq'][:8]} / {d['seq'][8:16]} / {d['seq'][16:]}")
    print(f"  FASTA: {fasta_path}")
    print()

# Run colabfold
print("=" * 80)
print("  Running colabfold_batch with MSA...")
print("=" * 80)

colabfold_exe = '/home/liubuzi/disorderflow-main/venv/bin/colabfold_batch'

for rank in range(3):
    job_dir = os.path.join(out_dir, f'design_{rank+1}')
    fasta_path = os.path.join(job_dir, 'multimer.fasta')
    result_dir = os.path.join(job_dir, 'af2_result')

    print(f"\n[Design {rank+1}] colabfold...")
    t0 = time.time()

    cmd = [
        colabfold_exe, fasta_path, result_dir,
        '--model-type', 'alphafold2_multimer_v3',
        '--num-recycle', '3',
        '--num-models', '1',
        '--use-dropout',
        '--random-seed', '42',
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        elapsed = time.time() - t0

        # Parse results
        scores_path = os.path.join(result_dir, 'multimer_scores_rank_001_alphafold2_multimer_v3_seed_042.json')
        if os.path.exists(scores_path):
            with open(scores_path) as f:
                scores = json.load(f)

            print(f"  Completed in {elapsed:.0f}s")
            print(f"  AF2 pLDDT: {scores.get('plddt', 'N/A')}")
            print(f"  AF2 ipTM:  {scores.get('iptm', 'N/A')}")
            print(f"  AF2 pTM:   {scores.get('ptm', 'N/A')}")

            # Parse PAE
            pae_path = os.path.join(result_dir, 'multimer_predicted_aligned_error_v1.json')
            if os.path.exists(pae_path):
                import numpy as np
                pae_data = json.load(open(pae_path))
                pae = np.array(pae_data.get('predicted_aligned_error', [[0]]))
                ab_len = len(graft(scaffold_seq, top_designs[rank]['seq']))
                interface_pae = pae[:ab_len, ab_len:].mean()
                print(f"  iPAE: {interface_pae:.1f}")
        else:
            print(f"  ERROR: No scores file found")
            print(f"  stdout: {result.stdout[-200:]}")
            print(f"  stderr: {result.stderr[-200:]}")

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT (>10min)")
    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\n{'='*80}")
print(f"  MSA Validation Complete")
print(f"  Results: {out_dir}/")
print(f"{'='*80}")
