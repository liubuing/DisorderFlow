"""AF2 multimer validation using JAX-native runner (no CLI dependency)."""
import sys, os, json, re, time
sys.path.insert(0, 'modules')

from af2_jax_runner import run_multimer_prediction

VH_PRE = 'VQLLESGGGLVQPGGSLRLSCAASGFTFSNYGMSWVRQAPGKGLEWVASIRSGGGRTYYSDNVKGRFTISRDNAKNSLYLQMNSLRAEDTALYYCV'
VH_POST = 'WGQGTLVTVSS'
VL = 'YVVMTQSPLSLPVTPGEPASISCKSSQSLLDSDGKTYLNWLLQKPGQSPQRLIYLVSKLDSGVPDRFSGSGSGTDFTLKISRVEAEDVGVYYCWQGTHFPRTFGQGTKVEIKR'
EPITOPE = 'DAEFRH'
NATIVE_H3 = 'VRYDHYSGSSDY'

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

# Top 5 + native
top_designs = []
for i, d in enumerate(designs[:5]):
    top_designs.append({
        'label': f'design_{i+1}',
        'h3': d['h3'],
        'mpnn_score': d['mpnn_score'],
        'fab': VH_PRE + d['h3'] + VH_POST + VL,
    })
top_designs.append({
    'label': 'native',
    'h3': NATIVE_H3,
    'mpnn_score': 0,
    'fab': VH_PRE + NATIVE_H3 + VH_POST + VL,
})

print(f"AF2 Multimer: {len(top_designs)} designs, epitope={EPITOPE}")
print(f"Fab length: {len(top_designs[0]['fab'])} aa")
print()

for d in top_designs:
    label = d['label']
    h3 = d['h3']
    fab = d['fab']
    print(f"  [{label}] H3={h3} | running AF2...", end=' ', flush=True)
    t0 = time.time()
    try:
        result = run_multimer_prediction(fab, EPITOPE, num_recycle=1, jax_random_seed=42)
        elapsed = time.time() - t0
        if result.get('success'):
            plddt = result.get('plddt', 0)
            iptm = result.get('iptm', 0)
            ptm = result.get('ptm', 0)
            ipae = result.get('interface_pae', result.get('max_pae', 99))
            print(f"pLDDT={plddt:.3f} ipTM={iptm:.3f} pTM={ptm:.3f} iPAE={ipae:.1f} ({elapsed:.0f}s)")
            d['af2_result'] = {
                'plddt': float(plddt),
                'iptm': float(iptm),
                'ptm': float(ptm),
                'interface_pae': float(ipae),
                'elapsed': elapsed,
            }
        else:
            print(f"FAILED: {result.get('error', 'unknown')}")
            d['af2_result'] = {'error': str(result.get('error', 'unknown'))}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"CRASH: {e} ({elapsed:.0f}s)")
        d['af2_result'] = {'error': str(e)}

print()
print("=== Summary ===")
print(f"{'Design':<12} {'H3':<15} {'MPNN':>6} {'pLDDT':>8} {'ipTM':>8} {'iPAE':>8}")
print("-" * 60)
for d in top_designs:
    r = d.get('af2_result', {})
    plddt = r.get('plddt', '--')
    iptm = r.get('iptm', '--')
    ipae = r.get('interface_pae', '--')
    print(f"{d['label']:<12} {d['h3']:<15} {d['mpnn_score']:>6.3f} "
          f"{str(plddt):>8} {str(iptm):>8} {str(ipae):>8}")

# Save
output = [{
    'label': d['label'], 'h3': d['h3'],
    'mpnn_score': d['mpnn_score'],
    **d.get('af2_result', {}),
} for d in top_designs]
os.makedirs('idp_design_results/4hix_validation', exist_ok=True)
with open('idp_design_results/4hix_validation/af2_jax_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to idp_design_results/4hix_validation/af2_jax_results.json")
