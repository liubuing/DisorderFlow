#!/usr/bin/env python
"""Deeper debug: test if crash is seed-specific or resource-related."""
import sys, os, subprocess, json, time, shutil

PROJ = r"C:\biological\disorderflow-main"
SEQ = "MADGQMPFSCHYPSRLRRDPFRDSPLSSRLLDDGFGMDPFPDDLTASWPDWALPRLSSAWPGTLRSGMVPRGPTATARFGVPAEGRTPPPFPGEPWKVCVNVHSFKPEELMVKTKDGYVEVSGKHEEKQQEGGIVSKNFTKKIQLPAEVDPVTVFASLSPEGLLIIEAPQVPPYSTFGESSFNNELPQDSQEVTCT"
# Short control sequence that we know works
SHORT_SEQ = "MDPKDKEKKKNDDKKEKKKNDDKDK"

def run_single(seq, seed, num_recycle=1, label=""):
    """Run a single AF2 prediction in a subprocess."""
    code = f'''
import sys, os, json, time
sys.path.insert(0, {PROJ!r})
sys.path.insert(0, {os.path.join(PROJ, "modules")!r})
os.chdir({PROJ!r})
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
from af2_jax_runner import run_multimer_prediction
seq = {seq!r}
t0 = time.time()
r = run_multimer_prediction(seq, "", num_recycle={num_recycle}, jax_random_seed={seed}, return_structure=True)
elapsed = time.time() - t0
out = {{"success": r.get("success"), "elapsed": elapsed}}
if r.get("success"):
    pos = r.get("final_atom_positions")
    out["shape"] = list(pos.shape) if pos is not None else None
else:
    out["error"] = str(r.get("error", "?"))[:200]
print(json.dumps(out))
'''
    t0 = time.time()
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=900,
            env={**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
        )
        elapsed = time.time() - t0
        if r.returncode != 0:
            tail = r.stderr.strip()[-200:] if r.stderr else "(no stderr)"
            return False, f"CRASH(exit={r.returncode}, {elapsed:.0f}s) | {tail}"
        result = json.loads(r.stdout.strip())
        if result.get("success"):
            return True, f"OK shape={result.get('shape')} ({result.get('elapsed',0):.0f}s)"
        else:
            return False, f"FAIL: {result.get('error','?')} ({elapsed:.0f}s)"
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT ({time.time()-t0:.0f}s)"
    except Exception as e:
        return False, f"EXC: {e}"

def find_jax_cache():
    """Find JAX compilation cache directories."""
    candidates = []
    for d in [
        os.path.expandvars(r"%LOCALAPPDATA%\jax"),
        os.path.expandvars(r"%TEMP%\jax_cache"),
        os.path.expanduser("~/.jax_cache"),
        os.path.join(os.environ.get("TEMP", "/tmp"), "jax_cache"),
        os.path.join(os.environ.get("TEMP", "/tmp"), "jax"),
    ]:
        if os.path.exists(d):
            size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(d) for f in fs)
            candidates.append((d, size))
    return candidates

if __name__ == "__main__":
    print("=" * 60)
    print("Deep Debug: Q9UJY1 AF2 Stack Overflow")
    print("=" * 60)

    # Check JAX cache
    print("\n[1] JAX cache dirs:")
    for d, size in find_jax_cache():
        print(f"    {d} ({size/1024/1024:.0f} MB)")

    # Test 1: run seed=3 directly (no prior seed=1) with recycle=1
    print("\n[2] Direct seed=3 (recycle=1, no warmup):")
    ok, msg = run_single(SEQ, seed=3, num_recycle=1, label="seed3-direct")
    print(f"    {msg}")

    # Test 2: run short sequence first to "warm up" JAX, then Q9UJY1 seed=3
    print("\n[3] Warmup short seq + seed=3 (recycle=1):")
    ok, msg = run_single(SHORT_SEQ, seed=1, num_recycle=1, label="warmup")
    print(f"    Warmup: {msg}")
    if ok:
        ok2, msg2 = run_single(SEQ, seed=3, num_recycle=1, label="seed3-after-warmup")
        print(f"    Q9UJY1 seed=3: {msg2}")

    # Test 3: try seed=5 directly
    print("\n[4] Direct seed=5 (recycle=1, no warmup):")
    ok, msg = run_single(SEQ, seed=5, num_recycle=1, label="seed5-direct")
    print(f"    {msg}")

    # Test 4: try recycle=2
    print("\n[5] Direct seed=3 (recycle=2):")
    ok, msg = run_single(SEQ, seed=3, num_recycle=2, label="seed3-recycle2")
    print(f"    {msg}")

    # Test 5: check if running same seed twice crashes on second run
    print("\n[6] seed=1 twice (recycle=1):")
    ok, msg = run_single(SEQ, seed=1, num_recycle=1, label="seed1-run1")
    print(f"    Run1: {msg}")
    if ok:
        ok2, msg2 = run_single(SEQ, seed=1, num_recycle=1, label="seed1-run2")
        print(f"    Run2: {msg2}")

    print("\nDone.")
