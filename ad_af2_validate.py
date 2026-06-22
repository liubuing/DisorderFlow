#!/usr/bin/env python
"""AF2 Multimer Validation: Top AD designs vs Abeta42 fibril.

Validates the top 3 Phase 3 antibody designs against the Abeta42 target
using the JAX-native AlphaFold2 multimer runner.
"""
import os, sys, json, time

os.chdir('/home/liubuzi/disorderflow-main/disorderflow-main')
sys.path.insert(0, '.')
sys.path.insert(0, 'modules')

from af2_jax_runner import validate_antibody_epitope

# ========================
# Load designs
# ========================
with open('idp_design_results/ad_phase3_demo/design_results.json') as f:
    designs = json.load(f)

# ========================
# Sequences
# ========================
# 5IMK nanobody scaffold (chain A, 132 residues)
SCAFFOLD_SEQ = (
    "EVQLVESGGGLVQAGGSLRLSCAAS"  # 1-25  (FR1)
    "GRTFSSYA"                     # 26-33 (CDR1)
    "MGWFRQAPGKEREFVA"            # 34-50 (FR2)
    "GISRSGGSA"                    # 51-58 (CDR2) —wait, actual 5IMK CDR2 is longer
    "YYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYC"  # 59-96 (FR3)
    "AAGGRFDYSYWSRDPY"            # 97-113 (CDR3) —actual 5IMK CDR3
    "WGQGTQVTVSS"                  # 114-124 (FR4)
)

# Actually let me read the real scaffold sequence from the PDB
AA3TO1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
    'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
}

# Read 5IMK sequence
scaffold_pdb = 'data/misfolding_targets/5IMK.pdb'
seen_res = {}
with open(scaffold_pdb) as f:
    for line in f:
        if line.startswith('ATOM') and line[21] == 'A':
            resid = int(line[22:26])
            if resid not in seen_res:
                seen_res[resid] = AA3TO1.get(line[17:20].strip(), 'X')
scaffold_seq = ''.join(seen_res[i] for i in sorted(seen_res))

# Read Abeta42 sequence
target_pdb = 'data/misfolding_targets/2NAO_model1_A_1-42.pdb'
seen_abeta = {}
with open(target_pdb) as f:
    for line in f:
        if line.startswith('ATOM') and line[21] == 'A':
            resid = int(line[22:26])
            if resid not in seen_abeta:
                seen_abeta[resid] = AA3TO1.get(line[17:20].strip(), 'X')
abeta_seq = ''.join(seen_abeta[i] for i in sorted(seen_abeta))

print("=" * 80)
print("  AF2 Multimer Validation —AD Antibody Designs vs Abeta42")
print("=" * 80)
print(f"  Scaffold (5IMK): {len(scaffold_seq)} AA")
print(f"  Target  (Abeta42): {len(abeta_seq)} AA")
print(f"  Designs to validate: {len(designs[:3])}")
print()

# ========================
# CDR Grafting
# ========================
cdr_spec = [(26, 33, 'H1'), (51, 58, 'H2'), (97, 113, 'H3')]

def graft_cdrs(scaffold, cdr_seq, cdr_spec):
    """Graft designed CDRs into scaffold sequence."""
    full = list(scaffold)
    pos = 0
    for start, end, name in cdr_spec:
        length = end - start + 1
        cdr = cdr_seq[pos:pos+length]
        for j in range(length):
            full[start - 1 + j] = cdr[j]
        pos += length
    return ''.join(full)

# ========================
# Run AF2 Validation
# ========================
results = []
for i, d in enumerate(designs[:3]):
    cdr_seq = d['sequence']
    ab_seq = graft_cdrs(scaffold_seq, cdr_seq, cdr_spec)

    # Show mutations vs scaffold
    mutations = [(j+1, scaffold_seq[j], ab_seq[j])
                 for j in range(len(scaffold_seq)) if scaffold_seq[j] != ab_seq[j]]

    print(f"[{i+1}] Design (BFN pLDDT={d['plddt']:.3f} ipTM={d['iptm']:.3f} PPL={d['ppl']:.1f})")
    print(f"    CDRs: {cdr_seq[:8]} / {cdr_seq[8:16]} / {cdr_seq[16:]}")
    print(f"    Mutations: {len(mutations)}")
    print(f"    Running AF2 multimer ({len(ab_seq)}+{len(abeta_seq)} AA)...", end=' ', flush=True)

    t0 = time.time()
    af_result = validate_antibody_epitope(
        ab_seq, abeta_seq, num_recycle=3, verbose=True
    )

    r = {
        'rank': i+1,
        'bfn_plddt': d['plddt'],
        'bfn_iptm': d['iptm'],
        'bfn_ppl': d['ppl'],
        'bfn_pae': d['pae'],
        'cdr_sequence': cdr_seq,
        'full_ab_seq': ab_seq,
        'mutations': len(mutations),
        **af_result
    }
    results.append(r)
    print()

# ========================
# Summary
# ========================
print("=" * 80)
print("  VALIDATION SUMMARY")
print("=" * 80)
print(f"\n{'Rank':<5} {'BFN pLDDT':<10} {'BFN ipTM':<10} {'BFN PPL':<8} "
      f"{'AF2 pLDDT':<10} {'AF2 ipTM':<10} {'iPAE':<8} {'Muts':<5}")
print("-" * 75)

for r in results:
    af2_plddt = r.get('plddt', 0)
    af2_iptm = r.get('iptm', 0)
    ipae = r.get('interface_pae', 99)
    print(f"{r['rank']:<5} {r['bfn_plddt']:.3f}      {r['bfn_iptm']:.3f}      "
          f"{r['bfn_ppl']:.1f}     {af2_plddt:.3f}      {af2_iptm:.3f}      "
          f"{ipae:.1f}     {r['mutations']}")

# Correlation analysis
print(f"\nAF2 vs BFN Confidence Comparison:")
for r in results:
    print(f"  Design {r['rank']}: BFN pLDDT={r['bfn_plddt']:.3f} ipTM={r['bfn_iptm']:.3f} | "
          f"AF2 pLDDT={r.get('plddt', 0):.3f} ipTM={r.get('iptm', 0):.3f} "
          f"iPAE={r.get('interface_pae', 99):.1f}")

# Save
out_dir = 'idp_design_results/ad_phase3_demo/af2_validation'
os.makedirs(out_dir, exist_ok=True)
with open(f'{out_dir}/validation_results.json', 'w') as f:
    json.dump([{k: v for k, v in r.items() if k != 'full_ab_seq'} for r in results], f, indent=2)

print(f"\nResults saved to: {out_dir}/validation_results.json")
print("Done.")
