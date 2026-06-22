#!/usr/bin/env python
"""JAX-native AF2 Multimer validation of BFN complex designs vs VSIG4.

Uses af2_jax_runner (single-seq, multimer v3) since colabfold needs tensorflow
(unavailable on Python 3.14). Scores will be lower than full MSA but provide
relative ranking ability.
"""
import os, sys, json, time

os.chdir('/home/liubuzi/disorderflow-main/disorderflow-main')
sys.path.insert(0, '.')
sys.path.insert(0, 'modules')

from af2_jax_runner import run_multimer_prediction, build_multimer_features, AA_TO_IDX

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

antigen_seq = read_pdb_seq('data/misfolding_targets/5IMK.pdb', 'A')
scaffold_seq = read_pdb_seq('data/misfolding_targets/5IMK.pdb', 'B')

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

def parse_fasta(path):
    seqs = []
    with open(path) as f:
        header, seq = None, ''
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header and seq:
                    seqs.append((header, seq))
                header = line[1:]
                seq = ''
            elif line:
                seq += line
        if header and seq:
            seqs.append((header, seq))
    return seqs

# Top 3 designs from MSA validation run
out_dir = 'idp_design_results/ad_phase3_demo/msa_validation'

print("=" * 80)
print("  JAX AF2 Multimer —Complex Design Validation")
print("  Model: alphafold2_multimer_v3, single-seq, 3 recycles")
print("=" * 80)
print(f"  Antigen: VSIG4 ({len(antigen_seq)} AA)")
print(f"  Scaffold: VHH ({len(scaffold_seq)} AA)")
print()

results = []
for rank in range(3):
    fasta_path = os.path.join(out_dir, f'design_{rank+1}', 'multimer.fasta')
    seqs = parse_fasta(fasta_path)
    ab_seq = seqs[0][1] if seqs[0][0].startswith('antibody') else seqs[1][1]

    print(f"Design {rank+1}: {len(ab_seq)} AA antibody + {len(antigen_seq)} AA antigen")
    print(f"  Antibody: {ab_seq[:50]}...")
    print(f"  Antigen:  {antigen_seq[:50]}...")

    result = run_multimer_prediction(ab_seq, antigen_seq, num_recycle=3)

    if result['success']:
        ipae = result.get('interface_pae')
        print(f"  AF2 pLDDT: {result['plddt']:.3f}  ipTM: {result['iptm']:.3f}  "
              f"pTM: {result['ptm']:.3f}  iPAE: {ipae:.1f}" if ipae else
              f"  AF2 pLDDT: {result['plddt']:.3f}  ipTM: {result['iptm']:.3f}  "
              f"pTM: {result['ptm']:.3f}")
        print(f"  Time: {result['elapsed']:.0f}s")
        results.append(result)
    else:
        print(f"  FAILED: {result.get('error', 'Unknown')}")

    print()

if results:
    avg_plddt = sum(r['plddt'] for r in results) / len(results)
    avg_iptm = sum(r['iptm'] for r in results) / len(results)
    print("=" * 80)
    print("  SUMMARY (JAX AF2 Multimer, single-seq)")
    print(f"  Avg pLDDT: {avg_plddt:.3f}")
    print(f"  Avg ipTM:  {avg_iptm:.3f}")
    print(f"  Avg pTM:   {sum(r['ptm'] for r in results) / len(results):.3f}")
    print()
    print("  NOTE: Single-seq scores are LOWER than full MSA scores.")
    print("  Use these for RELATIVE ranking, not absolute quality assessment.")
    print("=" * 80)

    save_path = os.path.join(out_dir, 'af2_jax_results.json')
    with open(save_path, 'w') as f:
        json.dump({
            'avg_plddt': avg_plddt,
            'avg_iptm': avg_iptm,
            'individual': [{
                'plddt': r['plddt'],
                'iptm': r['iptm'],
                'ptm': r['ptm'],
                'interface_pae': r.get('interface_pae'),
            } for r in results],
            'note': 'Single-seq AF2 multimer v3, 3 recycles. Scores lower than full MSA.',
        }, f, indent=2)
    print(f"\n  Saved: {save_path}")

print(f"\n{'='*80}")
print(f"  DONE")
print(f"{'='*80}")
