"""AF2 multimer validation of top 5 ProteinMPNN-designed CDR-H3 sequences for 4HIX."""
import sys, os, json, re
sys.path.insert(0, '.')
sys.path.insert(0, 'modules')

from af2_validator import validate_sequences
import numpy as np

# 4HIX native sequences
VH_PRE = 'VQLLESGGGLVQPGGSLRLSCAASGFTFSNYGMSWVRQAPGKGLEWVASIRSGGGRTYYSDNVKGRFTISRDNAKNSLYLQMNSLRAEDTALYYCV'
VH_POST = 'WGQGTLVTVSS'
VL = 'YVVMTQSPLSLPVTPGEPASISCKSSQSLLDSDGKTYLNWLLQKPGQSPQRLIYLVSKLDSGVPDRFSGSGSGTDFTLKISRVEAEDVGVYYCWQGTHFPRTFGQGTKVEIKR'
EPITOPE = 'DAEFRH'
NATIVE_H3 = 'VRYDHYSGSSDY'

# Read top designs
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
                designs.append({'sample': int(sample), 'mpnn_score': score, 'h3': h3})

designs.sort(key=lambda d: d['mpnn_score'])

# Build full Fab sequences for top 5
top_n = 5
fab_sequences = []
labels = []
for i, d in enumerate(designs[:top_n]):
    full_vh = VH_PRE + d['h3'] + VH_POST
    fab = full_vh + VL  # VH-VL concatenation for ColabFold
    fab_sequences.append(fab)
    labels.append(f"design_{i+1}_h3={d['h3'][:8]}...")

# Add native as control
native_fab = VH_PRE + NATIVE_H3 + VH_POST + VL
fab_sequences.append(native_fab)
labels.append(f"native_{NATIVE_H3[:8]}...")

print(f"Validating {len(fab_sequences)} designs with AF2 multimer (recycle=1)...")
print(f"Epitope: {EPITOPE}")
print(f"Fab length: ~{len(fab_sequences[0])} aa")
print()

# Run AF2 validation
results = validate_sequences(
    sequences=fab_sequences,
    antigen_sequences=EPITOPE,
    output_dir='idp_design_results/4hix_af2_validation',
    num_recycle=1,
    timeout=3600,
)

print("\n=== AF2 Multimer Results ===")
print(f"{'Design':<25} {'pLDDT':>8} {'ipTM':>8} {'pTM':>8} {'iPAE':>8} {'Success':>8}")
print("-" * 70)

for label, result in zip(labels, results):
    if result.get('success'):
        print(f"{label:<25} {result.get('plddt',0):>8.3f} {result.get('iptm',0):>8.3f} "
              f"{result.get('ptm',0):>8.3f} {result.get('max_pae',0):>8.1f} {'OK':>8}")
    else:
        print(f"{label:<25} {'--':>8} {'--':>8} {'--':>8} {'--':>8} "
              f"FAIL: {result.get('error', 'unknown')[:50]}")

# Save results
output = {
    'native_h3': NATIVE_H3,
    'epitope': EPITOPE,
    'designs': [],
}
for label, result in zip(labels, results):
    output['designs'].append({
        'label': label,
        'plddt': result.get('plddt'),
        'iptm': result.get('iptm'),
        'ptm': result.get('ptm'),
        'max_pae': result.get('max_pae'),
        'success': result.get('success'),
        'error': result.get('error'),
    })
os.makedirs('idp_design_results/4hix_validation', exist_ok=True)
with open('idp_design_results/4hix_validation/af2_results.json', 'w') as f:
    json.dump(output, f, indent=2)
