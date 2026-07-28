#!/usr/bin/env python
"""Alzheimer's antibody design: BFN on 5IMK nanobody CDRs targeting Abeta42 fibril."""
import os, sys, json, time
os.chdir(r'C:\biological\AntibodyDesignBFN-main\AntibodyDesignBFN-main')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))

import torch
import misfolding_knowledge_base as mkb
import misfolding_pipeline as mfp
import cascade_filter as cf
import target_design_helpers as tdh
from app import load_bfn, run_bfn_protein

print("=" * 80)
print("  Alzheimer's Disease Antibody Design — BFN on 5IMK Nanobody Scaffold")
print("=" * 80)

# ── Load target info ──
target = mkb.get_target('alzheimer_abeta')
scaffold = mkb.get_scaffold('nanobody_5imk')
print(f"\nTarget: {target['disease_cn']}")
print(f"Pathogenic form: {target['pathogenic_form']}")
print(f"Scaffold: {scaffold['name']} ({scaffold['type']})")
print(f"CDRs: {scaffold['cdr_definitions']}")

# ── Target PDB ──
target_pdb = 'data/misfolding_targets/2NAO_model1_A_1-42.pdb'
print(f"\nTarget PDB: {target_pdb}")

# ── Epitope analysis ──
print("\n" + "─" * 80)
print("  STEP 1: Epitope Analysis")
print("─" * 80)

epitope_data = tdh.score_epitope_residues(target_pdb, 'A')
print(f"\nTop 15 epitope residues (by combined score):")
for res in epitope_data[:15]:
    marker = " ★" if res['rank'] <= 5 else ""
    print(f"  #{res['rank']:2d}: {res['chain_id']}{res['resseq']}({res['resname']}) "
          f"score={res['combined_score']:.3f} (SASA={res['sasa']:.1f} "
          f"relSASA={res['rel_sasa']:.3f} hydro={res['hydrophilicity']:.2f} "
          f"protr={res['protrusion']:.3f}){marker}")

# Check against known epitopes
known_epitopes = mkb.get_known_epitope_regions('alzheimer_abeta')
print(f"\nKnown therapeutic epitopes:")
for ep in known_epitopes:
    print(f"  {ep.get('ab', '?'):30s} region {ep['region']} — {ep.get('source', '')}")

# Score epitopes against known regions
print(f"\nValidation against known epitopes:")
for ep in known_epitopes:
    region_start, region_end = ep['region']
    region_residues = [r for r in epitope_data if region_start <= r['resseq'] <= region_end]
    if region_residues:
        avg_rank = sum(r['rank'] for r in region_residues) / len(region_residues)
        avg_score = sum(r['combined_score'] for r in region_residues) / len(region_residues)
        print(f"  {ep['ab']:30s} [{region_start}-{region_end}]: avg rank={avg_rank:.1f}, avg score={avg_score:.3f}")

# Format epitope UI
epitope_ui = tdh.format_epitope_summary_for_ui(epitope_data, top_n=20)
print(f"\n{epitope_ui}")

# ── BFN Design on 5IMK CDRs ──
print("\n" + "─" * 80)
print("  STEP 2: BFN CDR Design on 5IMK Nanobody")
print("─" * 80)

scaffold_pdb = 'data/misfolding_targets/5IMK.pdb'
# Design CDR regions: H1=26-33, H2=51-58, H3=97-113 on chain A
region_spec = "A:26-33,51-58,97-113"
num_samples = 20
stochastic = True

print(f"Scaffold: {scaffold_pdb}")
print(f"Design region: {region_spec}")
print(f"Samples: {num_samples} | Stochastic: {stochastic}")

t0 = time.time()
design_output, fasta_str, results_list = run_bfn_protein(
    scaffold_pdb, region_spec, num_samples, stochastic, eval_mode=True
)
elapsed = time.time() - t0

print(f"\nDesign completed in {elapsed:.1f}s")
print(design_output)

# ── Misfolding Cascade Filter ──
print("\n" + "─" * 80)
print("  STEP 3: Misfolding-Specific Cascade Filtering")
print("─" * 80)

misfolding_filtered, misfolding_report = cf.apply_cascade_misfolding(results_list)
print(misfolding_report)

if misfolding_filtered:
    print(f"\nPassed misfolding cascade: {len(misfolding_filtered)} designs")
    top = misfolding_filtered[0]
    print(f"Top design: {top['sequence']}")
    print(f"  Composite score: {top['composite_score']:.3f}")
    for key in ['iptm', 'plddt', 'ppl', 'entropy', 'pae']:
        if key in top and top[key] is not None:
            print(f"  {key}: {top[key]:.3f}")

    # Save top designs
    os.makedirs('misfolding_results', exist_ok=True)
    with open('misfolding_results/alzheimers_top_designs.json', 'w') as f:
        json.dump(misfolding_filtered[:5], f, indent=2, default=str)
    print("\nTop 5 designs saved to misfolding_results/alzheimers_top_designs.json")
else:
    print("No designs passed misfolding cascade. Using all raw results...")
    misfolding_filtered = results_list

# ── IDP Warning ──
print("\n" + "─" * 80)
print("  STEP 4: IDP Risk Assessment")
print("─" * 80)

if mkb.is_idp_target('alzheimer_abeta'):
    warning = mkb.get_idp_warning('alzheimer_abeta')
    print(f"\nIDP WARNING: {warning}")
    cf_warning = cf.format_idp_score_warning('alzheimer_abeta', warning)
    print(cf_warning)
else:
    print("\nTarget is not flagged as IDP — confidence scores are reliable for structured fibril core")

# ── AF2 Multimer Validation Prep ──
print("\n" + "─" * 80)
print("  STEP 5: AF2 Multimer Validation Prep")
print("─" * 80)

# Build best design sequence
best_design = misfolding_filtered[0]['sequence'] if misfolding_filtered else results_list[0]['sequence']

# Get the full scaffold with designed CDRs
scaffold_seq = scaffold['sequence'].strip()
print(f"Scaffold full length: {len(scaffold_seq)} AA")
print(f"Original CDR1 (26-33): {scaffold_seq[25:33]}")
print(f"Original CDR2 (51-58): {scaffold_seq[50:58]}")
print(f"Original CDR3 (97-113): {scaffold_seq[96:113]}")

# The designed sequence is for the CDR regions only
# Build the full antibody by grafting designed CDRs onto scaffold
cdr_lengths = {
    'H1': (26, 33, 8),   # start, end, length
    'H2': (51, 58, 8),
    'H3': (97, 113, 17),
}
total_cdr_len = sum(v[2] for v in cdr_lengths.values())
print(f"Total CDR design length: {total_cdr_len}")
print(f"Best design sequence ({len(best_design)} AA): {best_design}")

# Graft the designed CDRs
full_ab_seq = list(scaffold_seq)
pos = 0
for name, (start, end, length) in cdr_lengths.items():
    cdr_seq = best_design[pos:pos+length]
    for j in range(length):
        full_ab_seq[start - 1 + j] = cdr_seq[j]
    pos += length
    print(f"  Designed {name} ({start}-{end}): {cdr_seq}")

full_ab_seq_str = ''.join(full_ab_seq)

# Abeta sequence - known from knowledge base / PDB extraction
aa3to1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
          'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
          'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
seen_res = {}
with open(target_pdb) as f:
    for line in f:
        if line.startswith('ATOM') and line[21] == 'A':
            resid = int(line[22:26])
            if resid not in seen_res:
                seen_res[resid] = aa3to1.get(line[17:20].strip(), 'X')

abeta_seq = ''.join(seen_res[i] for i in sorted(seen_res))
print(f"\nAbeta sequence: {abeta_seq} ({len(abeta_seq)} AA)")

# Build multimer FASTA
multimer_fasta = tdh.build_multimer_fasta(full_ab_seq_str, abeta_seq, 'nanobody_5IMK', 'Abeta42')
fasta_path = 'misfolding_results/alzheimers_multimer.fasta'
os.makedirs('misfolding_results', exist_ok=True)
with open(fasta_path, 'w') as f:
    f.write(multimer_fasta)
print(f"\nMultimer FASTA saved to: {fasta_path}")
print(multimer_fasta)

# ── Mutations summary ──
print("─" * 80)
print("  Mutation Summary vs Original 5IMK Scaffold")
print("─" * 80)
mutations = []
for i, (orig, new) in enumerate(zip(scaffold_seq, full_ab_seq_str)):
    if orig != new:
        mutations.append((i+1, orig, new))  # 1-indexed

if mutations:
    print(f"Total mutations: {len(mutations)}")
    for pos, orig, new in mutations:
        print(f"  Pos {pos:3d}: {orig} → {new}")
else:
    print("No mutations (all CDRs identical to scaffold)")

# ── Summary ──
print("\n" + "=" * 80)
print("  DESIGN COMPLETE — Alzheimer's Abeta Nanobody Ready for AF2 Validation")
print("=" * 80)
print(f"""
Design Summary:
  Target:      Alzheimer's Abeta42 fibril (2NAO)
  Scaffold:    5IMK VHH nanobody (anti-fibril optimized)
  CDRs:        H1(26-33) + H2(51-58) + H3(97-113) = {total_cdr_len} residues designed
  Samples:     {num_samples}
  Top BFN seq: {best_design}
  Mutations:   {len(mutations)} from scaffold
  Cascade:     {len(misfolding_filtered) if misfolding_filtered else 0} passed misfolding filter

Next: Run ColabFold AF2 multimer on misfolding_results/alzheimers_multimer.fasta
""")
