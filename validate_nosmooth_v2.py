#!/usr/bin/env python
"""Compare nosmooth v2 vs v11ext: CDR design + JAX AF2 validation.

Key question: does higher dist loss (148 vs 116) mean worse actual structure?
"""
import os, sys, json, time

os.chdir('/home/liubuzi/disorderflow-main/disorderflow-main')
sys.path.insert(0, '.')
sys.path.insert(0, 'modules')

import torch
from disorderflow.datasets.protein import preprocess_protein_structure
from disorderflow.utils.train import recursive_to
from disorderflow.utils.transforms import get_transform
from disorderflow.utils.data import PaddingCollate
from bfn_loader import load_bfn
from af2_jax_runner import run_multimer_prediction, AA_TO_IDX

AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_SAMPLES = 5

CHECKPOINTS = {
    'v11ext': './logs/bfn_phase3_v11_extended_2026_05_31__12_09_57/checkpoints/best.pt',
    'nosmooth_v2': './logs/bfn_phase3_v11_nosmooth_v2_2026_06_01__16_08_58/checkpoints/best.pt',
}

CDR_SPEC = {'B': list(range(26, 34)) + list(range(51, 59)) + list(range(97, 114))}
ANTIGEN_CHAIN = 'A'
CDR_SPEC_STR = 'B:26-33,51-58,97-113'

# Load antigen sequence
AA3TO1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
          'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
          'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

def read_pdb_seq(pdb_path, chain):
    seen = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('ATOM') and line[21] == chain:
                resid = int(line[22:26])
                if resid not in seen:
                    seen[resid] = AA3TO1.get(line[17:20].strip(), 'X')
    return ''.join(seen[i] for i in sorted(seen))

antigen_seq = read_pdb_seq('data/misfolding_targets/5IMK.pdb', ANTIGEN_CHAIN)

def load_model(ckpt_path):
    model, config = load_bfn(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    return model

def design_cdrs(model, num_samples):
    structure = preprocess_protein_structure(
        'data/misfolding_targets/5IMK.pdb', chain_ids=[ANTIGEN_CHAIN, 'B'])
    transform = get_transform([
        {'type': 'mask_region', 'regions': CDR_SPEC},
        {'type': 'merge_protein'},
        {'type': 'patch_protein'},
    ])
    batch = recursive_to(PaddingCollate()([transform(structure)]), DEVICE)
    gen_mask = batch['generate_flag'][0].bool()

    results = []
    for _ in range(num_samples):
        with torch.no_grad():
            traj = model.sample(batch, sample_opt={'deterministic': False, 'num_recycles': 3})

        pred_aa = traj[0][2][0][gen_mask]
        seq = ''.join(AA_LETTERS[a] if a < 20 else 'X' for a in pred_aa.cpu())

        logits = traj['pred_logits'][0][gen_mask]
        lp = torch.log_softmax(logits[..., :20], dim=-1)
        nll = -lp[range(len(pred_aa)), pred_aa].mean()
        ppl = torch.exp(nll).item()

        results.append({
            'seq': seq,
            'plddt': traj['plddt'][0][gen_mask].mean().item(),
            'iptm': traj['iptm'][0].item(),
            'ptm': traj.get('ptm', [0.0])[0].item() if 'ptm' in traj else 0.0,
            'pae': traj['pae'][0][gen_mask][:, gen_mask].mean().item(),
            'ppl': ppl,
        })
    return results

# Graft CDR sequence into scaffold
cdr_regions = [(26, 33), (51, 58), (97, 113)]

def read_scaffold():
    seen = {}
    with open('data/misfolding_targets/5IMK.pdb') as f:
        for line in f:
            if line.startswith('ATOM') and line[21] == 'B':
                resid = int(line[22:26])
                if resid not in seen:
                    seen[resid] = AA3TO1.get(line[17:20].strip(), 'X')
    return ''.join(seen[i] for i in sorted(seen))

scaffold_seq = read_scaffold()

def graft(cdr_seq):
    full = list(scaffold_seq)
    pos = 0
    for start, end in cdr_regions:
        length = end - start + 1
        cdr = cdr_seq[pos:pos+length]
        for j in range(length):
            full[start - 1 + j] = cdr[j]
        pos += length
    return ''.join(full)

print("=" * 80)
print("  nosmooth v2 vs v11ext: CDR Design + AF2 Validation")
print("=" * 80)
print(f"  Target: 5IMK VSIG4 (chain A={len(antigen_seq)}aa) + VHH (chain B={len(scaffold_seq)}aa)")
print(f"  CDRs: {CDR_SPEC_STR} ({sum(e-s+1 for s,e in cdr_regions)} residues)")
print(f"  Design mode: Complex (antigen visible)")
print(f"  Samples per model: {NUM_SAMPLES}")
print()

all_results = {}

for name, ckpt_path in CHECKPOINTS.items():
    print(f"\n{'─'*80}")
    print(f"  {name}: {os.path.basename(ckpt_path)}")
    print(f"{'─'*80}")

    model = load_model(ckpt_path)
    designs = design_cdrs(model, NUM_SAMPLES)

    print(f"\n  BFN Internal Metrics:")
    print(f"  {'#':<4} {'pLDDT':<8} {'ipTM':<8} {'PPL':<8} {'CDR (first 30)':<35}")
    print(f"  {'─'*70}")
    for i, d in enumerate(designs):
        print(f"  {i+1:<4} {d['plddt']:.4f}   {d['iptm']:.4f}   {d['ppl']:.1f}     {d['seq'][:30]}")

    # AF2 validation
    print(f"\n  JAX AF2 Multimer Validation (single-seq, 3 recycles):")
    af2_results = []
    for i, d in enumerate(designs):
        full_ab = graft(d['seq'])
        result = run_multimer_prediction(full_ab, antigen_seq, num_recycle=3)
        if result['success']:
            af2_results.append(result)
            ipae = result.get('interface_pae')
            ipae_str = f"iPAE={ipae:.1f}" if ipae is not None else ""
            print(f"  #{i+1}: pLDDT={result['plddt']:.3f}  ipTM={result['iptm']:.3f}  "
                  f"pTM={result['ptm']:.3f}  {ipae_str}  ({result['elapsed']:.0f}s)")
        else:
            print(f"  #{i+1}: FAILED —{result.get('error', '?')[:80]}")

    if af2_results:
        avg = {
            'plddt': sum(r['plddt'] for r in af2_results) / len(af2_results),
            'iptm': sum(r['iptm'] for r in af2_results) / len(af2_results),
            'ptm': sum(r['ptm'] for r in af2_results) / len(af2_results),
        }
        all_results[name] = {
            'bfn_avg_iptm': sum(d['iptm'] for d in designs) / len(designs),
            'bfn_avg_plddt': sum(d['plddt'] for d in designs) / len(designs),
            'bfn_avg_ppl': sum(d['ppl'] for d in designs) / len(designs),
            'af2_avg_plddt': avg['plddt'],
            'af2_avg_iptm': avg['iptm'],
            'af2_avg_ptm': avg['ptm'],
        }

# Summary
if len(all_results) == 2:
    print(f"\n{'='*80}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Metric':<20} {'v11ext':<12} {'nosmooth v2':<12} {'Delta':<12}")
    print(f"  {'─'*56}")
    for metric, label in [('bfn_avg_iptm', 'BFN ipTM'), ('bfn_avg_plddt', 'BFN pLDDT'),
                           ('bfn_avg_ppl', 'BFN PPL'),
                           ('af2_avg_iptm', 'AF2 ipTM'), ('af2_avg_plddt', 'AF2 pLDDT'),
                           ('af2_avg_ptm', 'AF2 pTM')]:
        v1 = all_results['v11ext'][metric]
        v2 = all_results['nosmooth_v2'][metric]
        delta = v2 - v1
        direction = '→ if delta > 0 else '→
        print(f"  {label:<20} {v1:<12.4f} {v2:<12.4f} {delta:+.4f} {direction}")

    os.makedirs('idp_design_results/nosmooth_v2_validation', exist_ok=True)
    with open('idp_design_results/nosmooth_v2_validation/comparison.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  Conclusion:")
    bfn_iptm_delta = all_results['nosmooth_v2']['bfn_avg_iptm'] - all_results['v11ext']['bfn_avg_iptm']
    af2_iptm_delta = all_results['nosmooth_v2']['af2_avg_iptm'] - all_results['v11ext']['af2_avg_iptm']

    if abs(af2_iptm_delta) < 0.02:
        print(f"  AF2 ipTM delta ({af2_iptm_delta:+.4f}) is negligible —")
        print(f"  structure quality is EQUIVALENT despite higher dist loss (148 vs 116).")
        print(f"  148 is the new baseline for label_smoothing=0 training.")
    elif af2_iptm_delta > 0:
        print(f"  nosmooth v2 is BETTER on AF2 ipTM ({af2_iptm_delta:+.4f}) —")
        print(f"  higher dist loss is a numerical artifact, not quality degradation.")
    else:
        print(f"  nosmooth v2 is WORSE on AF2 ipTM ({af2_iptm_delta:+.4f}) —")
        print(f"  dist loss degradation reflects real structure quality loss.")

print(f"\n{'='*80}")
print(f"  DONE")
print(f"{'='*80}")
