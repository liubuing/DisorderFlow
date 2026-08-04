"""Diagnose AF2 runner: test native 4HIX with different recycle settings."""
import sys, os, time
sys.path.insert(0, 'modules')
from af2_jax_runner import run_multimer_prediction

VH_PRE = 'VQLLESGGGLVQPGGSLRLSCAASGFTFSNYGMSWVRQAPGKGLEWVASIRSGGGRTYYSDNVKGRFTISRDNAKNSLYLQMNSLRAEDTALYYCV'
VH_POST = 'WGQGTLVTVSS'
VL = 'YVVMTQSPLSLPVTPGEPASISCKSSQSLLDSDGKTYLNWLLQKPGQSPQRLIYLVSKLDSGVPDRFSGSGSGTDFTLKISRVEAEDVGVYYCWQGTHFPRTFGQGTKVEIKR'
EPITOPE = 'DAEFRH'
NATIVE_H3 = 'VRYDHYSGSSDY'
FAB = VH_PRE + NATIVE_H3 + VH_POST + VL

print("Diagnosing AF2 multimer on native 4HIX Fab + DAEFRH epitope")
print(f"Fab: {len(FAB)} aa, Epitope: {len(EPITOPE)} aa")
print()

# Test different recycle settings
for recycle in [1, 3, 5]:
    print(f"Recycle={recycle}: ", end='', flush=True)
    t0 = time.time()
    try:
        result = run_multimer_prediction(FAB, EPITOPE, num_recycle=recycle, jax_random_seed=42)
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

# Test with different epitope formats
for epi_label, epi_seq in [
    ("DAEFRH (6aa)", "DAEFRH"),
    ("DAEFRHD (7aa)", "DAEFRHD"),
    ("DAEFRHDSGYEV (12aa)", "DAEFRHDSGYEV"),
]:
    print(f"Epitope={epi_label}: ", end='', flush=True)
    t0 = time.time()
    try:
        result = run_multimer_prediction(FAB, epi_seq, num_recycle=3, jax_random_seed=42)
        elapsed = time.time() - t0
        if result.get('success'):
            print(f"ipTM={result.get('iptm',0):.3f} pLDDT={result.get('plddt',0):.3f} "
                  f"iPAE={result.get('interface_pae',0):.1f} ({elapsed:.0f}s)")
        else:
            print(f"FAIL: {result.get('error','?')}")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"CRASH: {e} ({elapsed:.0f}s)")
