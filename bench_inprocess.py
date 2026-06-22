"""Benchmark in-process AF2: test if XLA cache survives across seeds."""
import sys, os, time
_project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "modules"))
os.chdir(_project_root)

from af2_jax_runner import run_multimer_prediction

seq = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDLELPPLPPSSPSSSPPPPTPPPSSPPSSSPSLPAS"
seq_clean = "".join(aa for aa in seq.upper() if aa in "ARNDCQEGHILKMFPSTWYV")
print(f"Sequence: {len(seq_clean)}aa")

# Seed 1: JIT compile
print("Seed 1 (JIT compile)...", end=" ", flush=True)
t0 = time.time()
r1 = run_multimer_prediction(seq_clean, "", num_recycle=0, jax_random_seed=1, return_structure=True)
dt1 = time.time() - t0
print(f"{dt1:.1f}s success={r1['success']}")

# Seed 3: should reuse
print("Seed 3 (cached?)...", end=" ", flush=True)
t0 = time.time()
r3 = run_multimer_prediction(seq_clean, "", num_recycle=0, jax_random_seed=3, return_structure=True)
dt3 = time.time() - t0
print(f"{dt3:.1f}s success={r3['success']}")

# Seed 5
print("Seed 5 (cached?)...", end=" ", flush=True)
t0 = time.time()
r5 = run_multimer_prediction(seq_clean, "", num_recycle=0, jax_random_seed=5, return_structure=True)
dt5 = time.time() - t0
print(f"{dt5:.1f}s success={r5['success']}")

print(f"\n=== Results ===")
print(f"Seed 1: {dt1:.1f}s  Seed 3: {dt3:.1f}s  Seed 5: {dt5:.1f}s")
if dt3 < dt1 * 0.5:
    print(f"XLA cache WORKS! Speedup: {dt1/dt3:.1f}x after JIT")
    est = dt1 + dt3 * 4
    print(f"Estimated per protein (5 seeds): {est/60:.1f} min")
    print(f"Estimated for 314 proteins: {est*314/3600:.1f} hours")
else:
    print(f"XLA cache NOT working (same speed)")
