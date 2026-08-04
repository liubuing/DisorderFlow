"""AF2 validation with CORRECT chain format (VH:VL not VH+VL)."""
import sys, os, time
sys.path.insert(0, 'modules')
from af2_jax_runner import run_multimer_prediction

VH_PRE = 'VQLLESGGGLVQPGGSLRLSCAASGFTFSNYGMSWVRQAPGKGLEWVASIRSGGGRTYYSDNVKGRFTISRDNAKNSLYLQMNSLRAEDTALYYCV'
VH_POST = 'WGQGTLVTVSS'
VL = 'YVVMTQSPLSLPVTPGEPASISCKSSQSLLDSDGKTYLNWLLQKPGQSPQRLIYLVSKLDSGVPDRFSGSGSGTDFTLKISRVEAEDVGVYYCWQGTHFPRTFGQGTKVEIKR'
EPITOPE = 'DAEFRH'
NATIVE_H3 = 'VRYDHYSGSSDY'

VH_NATIVE = VH_PRE + NATIVE_H3 + VH_POST
# CORRECT: separate chains with ':'
FAB_CORRECT = f"{VH_NATIVE}:{VL}"

print("Testing CORRECT chain format (VH:VL with colon separator)")
print(f"VH: {len(VH_NATIVE)} aa, VL: {len(VL)} aa, Epitope: {len(EPITOPE)} aa")
print()

for recycle in [1, 3]:
    print(f"Recycle={recycle}: ", end='', flush=True)
    t0 = time.time()
    try:
        result = run_multimer_prediction(FAB_CORRECT, EPITOPE, num_recycle=recycle, jax_random_seed=42)
        elapsed = time.time() - t0
        if result.get('success'):
            print(f"pLDDT={result.get('plddt',0):.3f} ipTM={result.get('iptm',0):.3f} "
                  f"pTM={result.get('ptm',0):.3f} iPAE={result.get('interface_pae',0):.1f} "
                  f"({elapsed:.0f}s)")
        else:
            print(f"FAIL: {result.get('error','?')}")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"CRASH: {e} ({elapsed:.0f}s)")

print()

# Test top 3 ProteinMPNN designs with CORRECT format
import json, re
NATIVE_H3S = 'VRYDHYSGSSDY'
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

print("=== Top 3 ProteinMPNN Designs + Native (CORRECT chain format) ===")
test_seqs = [(f"native_{NATIVE_H3[:8]}", VH_PRE + NATIVE_H3S + VH_POST)]
for i, d in enumerate(designs[:3]):
    test_seqs.append((f"design_{i+1}_{d['h3'][:8]}", VH_PRE + d['h3'] + VH_POST))

for label, vh_seq in test_seqs:
    fab = f"{vh_seq}:{VL}"
    print(f"  [{label}]: ", end='', flush=True)
    t0 = time.time()
    try:
        result = run_multimer_prediction(fab, EPITOPE, num_recycle=3, jax_random_seed=42)
        elapsed = time.time() - t0
        if result.get('success'):
            print(f"ipTM={result.get('iptm',0):.4f} pLDDT={result.get('plddt',0):.3f} "
                  f"iPAE={result.get('interface_pae',0):.1f} ({elapsed:.0f}s)")
        else:
            print(f"FAIL: {result.get('error','?')}")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"CRASH: {e} ({elapsed:.0f}s)")
