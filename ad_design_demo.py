#!/usr/bin/env python
"""Alzheimer's Disease Antibody Design Demo —BFN Phase 3 Model.

Target: Abeta42 fibril (2NAO)
Scaffold: 5IMK VHH nanobody
CDRs to design: H1(26-33), H2(51-58), H3(97-113)

Demonstrates BFN's core advantage: simultaneous sequence design + confidence estimation.
"""
import os, sys, json, time

os.chdir('/home/liubuzi/disorderflow-main/disorderflow-main')
sys.path.insert(0, '.')
sys.path.insert(0, 'modules')

import torch
from bfn_loader import run_bfn_design

# ========================
# Design Parameters
# ========================
SCAFFOLD_PDB = 'data/misfolding_targets/5IMK.pdb'
CDR_SPEC = 'B:26-33,51-58,97-113'  # 5IMK nanobody CDRs (chain B)
NUM_SAMPLES = 20
STOCHASTIC = True
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 80)
print("  Alzheimer's Disease Antibody Design —BFN Phase 3 Model")
print("  Target: Abeta42 fibril  |  Scaffold: 5IMK VHH nanobody")
print("=" * 80)
print(f"  CDR regions: {CDR_SPEC}")
print(f"  Samples: {NUM_SAMPLES} (stochastic)")
print(f"  Model: Phase 3 v11 Extended (15000 iters, 3306 complexes)")
print()

# ========================
# Run BFN Design
# ========================
t0 = time.time()
results = run_bfn_design(SCAFFOLD_PDB, CDR_SPEC, NUM_SAMPLES, STOCHASTIC, DEVICE)
elapsed = time.time() - t0

# ========================
# Rank by composite score
# ========================
for r in results:
    plddt = r['plddt']
    iptm = r['iptm']
    ppl = min(r['ppl'], 200)
    pae = max(min(r['pae'], 15), 1)
    entropy = r['entropy']
    # Composite: higher iptm/plddt, lower PPL/PAE/entropy is better
    r['composite'] = (iptm * 0.35 + plddt * 0.25
                      + (1 - ppl/200) * 0.20
                      + (1 - entropy/3) * 0.10
                      + (1 - pae/15) * 0.10)

results.sort(key=lambda x: x['composite'], reverse=True)

# ========================
# Output
# ========================
print(f"\n{'='*80}")
print(f"  RESULTS —{NUM_SAMPLES} designs in {elapsed:.1f}s")
print(f"{'='*80}")

print(f"\n{'Rank':<5} {'Sequence':<38} {'pLDDT':<8} {'ipTM':<8} {'PAE':<8} {'PPL':<8} {'Entropy':<8} {'Score':<8}")
print("-" * 95)
for i, r in enumerate(results):
    label = '鈽? if i < 3 else ' '
    print(f"{label}{i+1:<4} {r['sequence']:<38} {r['plddt']:.3f}   {r['iptm']:.3f}   "
          f"{r['pae']:.1f}     {r['ppl']:.1f}     {r['entropy']:.3f}   {r['composite']:.3f}")

# ========================
# Summary
# ========================
best = results[0]
print(f"\n{'='*80}")
print(f"  TOP DESIGN")
print(f"{'='*80}")
print(f"  Sequence:     {best['sequence']}")
print(f"  pLDDT:        {best['plddt']:.3f}  (per-residue confidence)")
print(f"  ipTM:         {best['iptm']:.3f}  (interface confidence)")
print(f"  PAE:          {best['pae']:.1f}   (predicted alignment error)")
print(f"  PPL:          {best['ppl']:.1f}   (perplexity, lower = more confident)")
print(f"  Entropy:      {best['entropy']:.3f}  (per-residue entropy)")

# Quality assessment
if best['iptm'] > 0.7 and best['plddt'] > 0.8:
    quality = "HIGH —strong interface + confident residues"
elif best['iptm'] > 0.5 and best['plddt'] > 0.6:
    quality = "MEDIUM —decent binding potential"
else:
    quality = "LOW —Abeta42 is intrinsically disordered, confidence naturally lower"

print(f"\n  Quality: {quality}")
print(f"\n  Full CDR mapping (5IMK positions):")
print(f"    CDR1 (26-33): {best['sequence'][:8]}")
print(f"    CDR2 (51-58): {best['sequence'][8:16]}")
print(f"    CDR3 (97-113): {best['sequence'][16:]}")

# Save
os.makedirs('idp_design_results/ad_phase3_demo', exist_ok=True)
with open('idp_design_results/ad_phase3_demo/design_results.json', 'w') as f:
    json.dump(results[:10], f, indent=2)
print(f"\n  Top 10 saved to: idp_design_results/ad_phase3_demo/design_results.json")

# Stats
plddt_vals = [r['plddt'] for r in results]
iptm_vals = [r['iptm'] for r in results]
print(f"\n  Stats across {NUM_SAMPLES} designs:")
print(f"    pLDDT: mean={sum(plddt_vals)/len(plddt_vals):.3f}  "
      f"max={max(plddt_vals):.3f}  min={min(plddt_vals):.3f}")
print(f"    ipTM:  mean={sum(iptm_vals)/len(iptm_vals):.3f}  "
      f"max={max(iptm_vals):.3f}  min={min(iptm_vals):.3f}")

print(f"\n{'='*80}")
print(f"  DESIGN COMPLETE")
print(f"  Next: AF2 multimer validation on top designs")
print(f"{'='*80}")
