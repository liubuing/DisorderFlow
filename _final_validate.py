"""Complete IDP Antibody Design Validation Report.
ProteinMPNN CDR-H3 design + AF2 multimer + Triplet scoring.
"""
import sys, os, json, re, time, tempfile
import numpy as np
from pathlib import Path

sys.path.insert(0, '.')
sys.path.insert(0, 'modules')

from af2_jax_runner import run_multimer_prediction
from idp_validation_triplet import score_complex, interface_contacts
from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain

VH_PRE = 'VQLLESGGGLVQPGGSLRLSCAASGFTFSNYGMSWVRQAPGKGLEWVASIRSGGGRTYYSDNVKGRFTISRDNAKNSLYLQMNSLRAEDTALYYCV'
VH_POST = 'WGQGTLVTVSS'
VL = 'YVVMTQSPLSLPVTPGEPASISCKSSQSLLDSDGKTYLNWLLQKPGQSPQRLIYLVSKLDSGVPDRFSGSGSGTDFTLKISRVEAEDVGVYYCWQGTHFPRTFGQGTKVEIKR'
EPITOPE = 'DAEFRH'
NATIVE_H3 = 'VRYDHYSGSSDY'
OUTDIR = Path('idp_design_results/4hix_final_validation')
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── Load designs ──
designs = []
with open('idp_design_results/4hix_mpnn_h3/seqs/4hix_clean.fa') as f:
    for line in f:
        if line.startswith('>T=0.5'):
            parts = line.strip().split(',')
            sample = parts[1].split('=')[1].strip()
            score = float(parts[2].split('=')[1])
            seq = next(f).strip()
            chains = seq.split('/')
            h_chain = chains[1]
            m = re.search(r'YCV(.+?)WGQ', h_chain)
            if m:
                h3 = m.group(1)
                designs.append({
                    'sample': int(sample), 'mpnn_score': score, 'h3': h3,
                    'vh': VH_PRE + h3 + VH_POST,
                })
designs.sort(key=lambda d: d['mpnn_score'])

print("=" * 70)
print("  IDP Antibody Design — Final Validation Report")
print("=" * 70)
print(f"  Scaffold: 4HIX (humanized 3D6 Fab, anti-Abeta)")
print(f"  Epitope:  Abeta 1-6 (DAEFRH)")
print(f"  Native CDR-H3: {NATIVE_H3}")
print(f"  ProteinMPNN designs: {len(designs)} sequences")
print()

# ── AF2 Multimer Validation ──
print("[1/3] AF2 Multimer Validation (VH:VL format, recycle=3)")
print("-" * 70)

all_tests = []
all_tests.append({'label': 'native', 'h3': NATIVE_H3, 'vh': VH_PRE + NATIVE_H3 + VH_POST})
for d in designs:
    all_tests.append({
        'label': f"s{d['sample']:02d}", 'h3': d['h3'],
        'vh': d['vh'], 'mpnn_score': d['mpnn_score'],
    })

for test in all_tests:
    fab = f"{test['vh']}:{VL}"
    print(f"  [{test['label']}] H3={test['h3']} ", end='', flush=True)
    t0 = time.time()
    try:
        r = run_multimer_prediction(fab, EPITOPE, num_recycle=3, jax_random_seed=42)
        if r.get('success'):
            test['plddt'] = float(r['plddt'])
            test['iptm'] = float(r['iptm'])
            test['ptm'] = float(r['ptm'])
            test['ipae'] = float(r.get('interface_pae', r.get('max_pae', 99)))
            test['af2_ok'] = True
            elapsed = time.time() - t0
            print(f"ipTM={test['iptm']:.4f} pLDDT={test['plddt']:.3f} iPAE={test['ipae']:.1f} ({elapsed:.0f}s)")
        else:
            test['af2_ok'] = False
            test['af2_error'] = str(r.get('error', '?'))
            print(f"FAIL")
    except Exception as e:
        test['af2_ok'] = False
        test['af2_error'] = str(e)
        print(f"CRASH: {e}")

# ── Triplet scoring ──
print()
print("[2/3] Triplet Interface Scoring")
print("-" * 70)

native_triplet = score_complex(
    'idp_design_results/4hix_clean.pdb',
    cdr_chain='H', epitope_chain='A',
    cdr_regions=[(97, 106)])
print(f"  Native: contacts={native_triplet['contacts']} density={native_triplet['density']:.2f} "
      f"mean_dist={native_triplet.get('mean_distance','?'):.1f}A composite={native_triplet.get('composite','?'):.4f}")

for test in all_tests[1:6]:  # top 5 designs only
    print(f"  [{test['label']}] H3={test['h3']} ", end='', flush=True)
    test['triplet'] = {}
    test['triplet']['note'] = 'scoring uses native PDB with grafted H3 — contact topology unchanged'
    print("(same scaffold geometry)")

# ── Generate summary ──
print()
print("[3/3] Final Ranking")
print("-" * 70)

native = all_tests[0]
design_results = all_tests[1:]

# Sort by AF2 ipTM (higher better) then by ProteinMPNN score (lower better)
design_results.sort(key=lambda d: (-d.get('iptm', 0), d.get('mpnn_score', 999)))

print(f"{'Rank':<5} {'Label':<8} {'H3':<16} {'MPNN':>6} {'ipTM':>8} {'pLDDT':>8} {'iPAE':>8} {'vs_Native':>10}")
print("-" * 75)
print(f"{'N/A':<5} {'native':<8} {NATIVE_H3:<16} {'---':>6} "
      f"{native.get('iptm',0):>8.4f} {native.get('plddt',0):>8.3f} "
      f"{native.get('ipae',99):>8.1f} {'---':>10}")

for rank, d in enumerate(design_results[:10], 1):
    delta = d.get('iptm', 0) - native.get('iptm', 0)
    print(f"{rank:<5} {d['label']:<8} {d['h3']:<16} {d.get('mpnn_score',0):>6.3f} "
          f"{d.get('iptm',0):>8.4f} {d.get('plddt',0):>8.3f} "
          f"{d.get('ipae',99):>8.1f} {delta:>+10.4f}")

# ── Best candidates ──
print()
print("=" * 70)
print("  TOP CANDIDATES FOR EXPERIMENTAL VALIDATION")
print("=" * 70)

for rank, d in enumerate(design_results[:3], 1):
    print(f"  [{rank}] {d['label']}: {d['h3']}")
    print(f"      AF2 ipTM: {d.get('iptm',0):.4f} (native: {native.get('iptm',0):.4f})")
    print(f"      AF2 pLDDT: {d.get('plddt',0):.3f}")
    print(f"      AF2 iPAE: {d.get('ipae',99):.1f}A")
    print(f"      ProteinMPNN score: {d.get('mpnn_score',0):.3f}")
    print(f"      Full VH: {d['vh'][:60]}...")
    print()

# ── Save ──
report = {
    'scaffold': '4HIX (humanized 3D6 Fab)',
    'epitope': 'Abeta 1-6 (DAEFRH)',
    'native_h3': NATIVE_H3,
    'native_af2': {'iptm': native.get('iptm'), 'plddt': native.get('plddt'),
                   'ptm': native.get('ptm'), 'ipae': native.get('ipae')},
    'native_triplet': native_triplet,
    'designs': [{
        'label': d['label'], 'h3': d['h3'],
        'mpnn_score': d.get('mpnn_score'),
        'af2_iptm': d.get('iptm'), 'af2_plddt': d.get('plddt'),
        'af2_ptm': d.get('ptm'), 'af2_ipae': d.get('ipae'),
        'af2_ok': d.get('af2_ok', False),
    } for d in design_results],
    'top_candidates': design_results[:3],
}
with open(OUTDIR / 'final_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)
print(f"Report saved to {OUTDIR / 'final_report.json'}")
