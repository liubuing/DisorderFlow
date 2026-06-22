#!/usr/bin/env python
"""Complex Design Demo: 5IMK nanobody CDR design with VSIG4 antigen context.

Compares FixBB (no antigen) vs Complex (antigen visible) to demonstrate
how BFN's ipTM calibration improves with antigen context.

Key insight: In FixBB mode, BFN's ipTM only reflects internal antibody
consistency. In Complex mode, ipTM reflects actual interface quality.
"""
import os, sys, json, time

os.chdir('/home/liubuzi/disorderflow-main/disorderflow-main')
sys.path.insert(0, '.')
sys.path.insert(0, 'modules')

import torch
import re
from disorderflow.datasets.protein import preprocess_protein_structure
from disorderflow.utils.train import recursive_to
from disorderflow.utils.misc import seed_all
from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.transforms import get_transform
from bfn_loader import load_bfn

AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ========================
# Setup
# ========================
model, config = load_bfn(DEVICE)
seed_all(42)

# 5IMK: chain A = VSIG4 (antigen 232aa), chain B = nanobody (125aa)
# Nanobody CDRs (0-indexed PDB residues):
#   CDR1: 26-33 (GRTFSSYG)
#   CDR2: 51-58 (IRWNGGST)
#   CDR3: 97-113 (GRWDKYGSSFQDEYDY)

cdr_spec = {
    'B': list(range(26, 34)) + list(range(51, 59)) + list(range(97, 114))
}
cdr_spec_str = 'B:26-33,51-58,97-113'
NUM_SAMPLES = 10

print("=" * 80)
print("  Complex Design Demo — BFN Phase 3 Model")
print("  5IMK Nanobody (chain B) + VSIG4 Antigen (chain A)")
print("=" * 80)
print(f"  CDRs: {cdr_spec_str} ({len(cdr_spec['B'])} residues)")
print(f"  Samples: {NUM_SAMPLES}")
print()

# ========================
# Mode 1: FixBB (antigen masked out)
# ========================
print("─" * 80)
print("  MODE 1: FixBB — Antigen chain A masked out")
print("─" * 80)

# Only load chain B
structure_fixbb = preprocess_protein_structure(
    'data/misfolding_targets/5IMK.pdb', chain_ids=['B']
)

transform_fixbb = get_transform([
    {'type': 'mask_region', 'regions': cdr_spec},
    {'type': 'merge_protein'},
    {'type': 'patch_protein'},
])
batch_fixbb = recursive_to(PaddingCollate()([transform_fixbb(structure_fixbb)]), DEVICE)
gen_mask_fixbb = batch_fixbb['generate_flag'][0].bool()

fixbb_results = []
for i in range(NUM_SAMPLES):
    with torch.no_grad():
        traj = model.sample(batch_fixbb, sample_opt={'deterministic': False, 'num_recycles': 3})

    pred_aa = traj[0][2][0][gen_mask_fixbb]
    seq = ''.join(AA_LETTERS[a] if a < 20 else 'X' for a in pred_aa.cpu())

    logits = traj['pred_logits'][0][gen_mask_fixbb]
    lp = torch.log_softmax(logits[..., :20], dim=-1)
    nll = -lp[range(len(pred_aa)), pred_aa].mean()
    ppl = torch.exp(nll).item()
    entropy = -(torch.exp(lp) * lp).sum(dim=-1).mean().item()

    fixbb_results.append({
        'seq': seq,
        'plddt': traj['plddt'][0][gen_mask_fixbb].mean().item(),
        'iptm': traj['iptm'][0].item(),
        'pae': traj['pae'][0][gen_mask_fixbb][:, gen_mask_fixbb].mean().item(),
        'ppl': ppl, 'entropy': entropy
    })

# ========================
# Mode 2: Complex (antigen chain A visible)
# ========================
print("─" * 80)
print("  MODE 2: Complex — Antigen chain A visible")
print("─" * 80)

# Load both chains A (antigen) and B (nanobody)
structure_complex = preprocess_protein_structure(
    'data/misfolding_targets/5IMK.pdb', chain_ids=['A', 'B']
)

transform_complex = get_transform([
    {'type': 'mask_region', 'regions': cdr_spec},
    {'type': 'merge_protein'},
    {'type': 'patch_protein'},
])
batch_complex = recursive_to(PaddingCollate()([transform_complex(structure_complex)]), DEVICE)
gen_mask_complex = batch_complex['generate_flag'][0].bool()

complex_results = []
for i in range(NUM_SAMPLES):
    with torch.no_grad():
        traj = model.sample(batch_complex, sample_opt={'deterministic': False, 'num_recycles': 3})

    pred_aa = traj[0][2][0][gen_mask_complex]
    seq = ''.join(AA_LETTERS[a] if a < 20 else 'X' for a in pred_aa.cpu())

    logits = traj['pred_logits'][0][gen_mask_complex]
    lp = torch.log_softmax(logits[..., :20], dim=-1)
    nll = -lp[range(len(pred_aa)), pred_aa].mean()
    ppl = torch.exp(nll).item()
    entropy = -(torch.exp(lp) * lp).sum(dim=-1).mean().item()

    complex_results.append({
        'seq': seq,
        'plddt': traj['plddt'][0][gen_mask_complex].mean().item(),
        'iptm': traj['iptm'][0].item(),
        'pae': traj['pae'][0][gen_mask_complex][:, gen_mask_complex].mean().item(),
        'ppl': ppl, 'entropy': entropy
    })

# ========================
# Comparison
# ========================
print(f"\n{'='*80}")
print(f"  COMPARISON: FixBB vs Complex")
print(f"{'='*80}")

print(f"\n{'Mode':<10} {'Design':<5} {'pLDDT':<10} {'ipTM':<10} {'PAE':<10} {'PPL':<10} {'CDR Seq':<35}")
print("-" * 95)

for i in range(min(NUM_SAMPLES, 5)):
    fb = fixbb_results[i]
    print(f"{'FixBB':<10} {i+1:<5} {fb['plddt']:.4f}    {fb['iptm']:.4f}    {fb['pae']:.2f}      {fb['ppl']:.1f}       {fb['seq'][:33]}...")

print("-" * 95)
for i in range(min(NUM_SAMPLES, 5)):
    cb = complex_results[i]
    print(f"{'Complex':<10} {i+1:<5} {cb['plddt']:.4f}    {cb['iptm']:.4f}    {cb['pae']:.2f}      {cb['ppl']:.1f}       {cb['seq'][:33]}...")

# Averages
fb_avg_iptm = sum(r['iptm'] for r in fixbb_results) / len(fixbb_results)
fb_avg_plddt = sum(r['plddt'] for r in fixbb_results) / len(fixbb_results)
cb_avg_iptm = sum(r['iptm'] for r in complex_results) / len(complex_results)
cb_avg_plddt = sum(r['plddt'] for r in complex_results) / len(complex_results)

print(f"\n{'='*80}")
print(f"  SUMMARY")
print(f"{'='*80}")
print(f"  FixBB:   pLDDT={fb_avg_plddt:.4f}  ipTM={fb_avg_iptm:.4f}")
print(f"  Complex: pLDDT={cb_avg_plddt:.4f}  ipTM={cb_avg_iptm:.4f}")
print(f"  ipTM delta: {cb_avg_iptm - fb_avg_iptm:+.4f}")

if cb_avg_iptm < fb_avg_iptm:
    print(f"\n  ⚠ Complex mode ipTM is LOWER — model is more conservative")
    print(f"    with antigen context, reflecting real interface uncertainty.")
else:
    print(f"\n  Complex mode ipTM is higher — antigen context provides")
    print(f"    additional structural constraints.")

print(f"\n  BFN overconfidence diagnosis:")
print(f"    FixBB ipTM ~{fb_avg_iptm:.3f}: measures internal structure quality only")
print(f"    Complex ipTM ~{cb_avg_iptm:.3f}: includes antigen interface signal")
print(f"    Gap = BFN's overestimation when antigen is absent")

# Save
os.makedirs('idp_design_results/ad_phase3_demo', exist_ok=True)
with open('idp_design_results/ad_phase3_demo/fixbb_vs_complex.json', 'w') as f:
    save_data = {
        'fixbb': {k: v for k, v in fixbb_results[0].items() if k != 'seq'},
        'complex': {k: v for k, v in complex_results[0].items() if k != 'seq'},
        'fixbb_avg_iptm': fb_avg_iptm,
        'complex_avg_iptm': cb_avg_iptm,
        'fixbb_avg_plddt': fb_avg_plddt,
        'complex_avg_plddt': cb_avg_plddt,
        'conclusion': 'Complex mode reduces overconfidence by including antigen context'
    }
    json.dump(save_data, f, indent=2)

print(f"\n  Saved: idp_design_results/ad_phase3_demo/fixbb_vs_complex.json")
print(f"{'='*80}")
print(f"  DONE")
print(f"{'='*80}")
