"""Benchmark AF2 GPU inference matching build script's subprocess flow."""
import sys, os, time, json, subprocess

_project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(_project_root)

# Use the exact same worker pattern as build_conformation_cf.py
worker_code = '''import sys, json, os, threading
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_enable_fast_min_max=false")
os.chdir({project_root!r})
sys.path.insert(0, {project_root!r})
sys.path.insert(0, {modules_dir!r})
from af2_jax_runner import run_multimer_prediction
import numpy as _np

data = json.loads(sys.stdin.read())
seq = data["seq"]
seed = data["seed"]
nr = data["recycle"]
try:
    r = run_multimer_prediction(seq, "", num_recycle=nr,
                                 jax_random_seed=seed, return_structure=True)
    if r.get("success"):
        pos = r.get("final_atom_positions")
        if pos is not None and pos.shape[0] == len(seq):
            print(json.dumps({{"ok": True, "pos": _np.array(pos).tolist()}}))
            sys.exit(0)
    print(json.dumps({{"ok": False}}))
except Exception as e:
    print(json.dumps({{"ok": False, "error": str(e)[:200]}}))
'''.format(project_root=_project_root, modules_dir=os.path.join(_project_root, "modules"))

# Test with a typical IDP sequence (~250aa, close to build average)
test_seq = "M" + "GGSGGSGGSGGSGGSGGSGGSGGSGGSGGS" * 7  # ~211aa
# Use a real IDP-like: pick one that crashed before but shorter
test_seq = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDLELPPLPPSSPSSSPPPPTPPPSSPPSSSPSLPASLPASLPASLPASLPAS"  # ~140aa

seq_clean = "".join(aa for aa in test_seq.upper() if aa in "ARNDCQEGHILKMFPSTWYV")
print(f"Benchmark: {len(seq_clean)}aa protein, 3 seeds, num_recycle=0")
print(f"Pattern: subprocess per seed (matching build script)")
print(f"{'='*60}")

# Warmup run
print("Warmup seed...", end=" ", flush=True)
t0 = time.time()
payload = json.dumps({"seq": seq_clean, "seed": 1, "recycle": 0})
r = subprocess.run([sys.executable, "-c", worker_code],
                   input=payload, capture_output=True, text=True, timeout=600)
print(f"{time.time()-t0:.1f}s (incl JIT)")

# Timed runs
times = []
for seed in [3, 5, 7]:
    print(f"seed={seed}...", end=" ", flush=True)
    t0 = time.time()
    payload = json.dumps({"seq": seq_clean, "seed": seed, "recycle": 0})
    r = subprocess.run([sys.executable, "-c", worker_code],
                       input=payload, capture_output=True, text=True, timeout=600)
    dt = time.time() - t0
    times.append(dt)
    result = json.loads(r.stdout.strip()) if r.stdout.strip() else {}
    status = "OK" if result.get("ok") else f"FAIL({result.get('error','?')})"
    print(f"{dt:.1f}s {status}")

if times:
    avg = sum(times) / len(times)
    print(f"\n{'='*60}")
    print(f"Average per seed (after warmup): {avg:.1f}s")
    print(f"Estimated per protein (5 seeds): {avg*5/60:.1f} min")
    print(f"Estimated for 314 proteins: {avg*5*314/3600:.1f} hours")
    if avg < 60:
        print(f"Speedup vs CPU (~1800s/protein): {1800/(avg*5):.0f}x")
