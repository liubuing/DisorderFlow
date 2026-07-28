#!/usr/bin/env python
"""Test if long delays between AF2 subprocesses prevent crashes."""
import sys, os, subprocess, json, time

PROJ = r"C:\biological\disorderflow-main"
SEQ = "MADGQMPFSCHYPSRLRRDPFRDSPLSSRLLDDGFGMDPFPDDLTASWPDWALPRLSSAWPGTLRSGMVPRGPTATARFGVPAEGRTPPPFPGEPWKVCVNVHSFKPEELMVKTKDGYVEVSGKHEEKQQEGGIVSKNFTKKIQLPAEVDPVTVFASLSPEGLLIIEAPQVPPYSTFGESSFNNELPQDSQEVTCT"
SHORT = "MDPKDKEKKKNDDKKEKKKNDDKDK"

def run(seq, seed, nr=1, timeout=900):
    code = f'''
import sys, os, json, time, gc
sys.path.insert(0, {PROJ!r})
sys.path.insert(0, {os.path.join(PROJ, "modules")!r})
os.chdir({PROJ!r})
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
from af2_jax_runner import run_multimer_prediction
import jax
seq = {seq!r}
r = run_multimer_prediction(seq, "", num_recycle={nr}, jax_random_seed={seed}, return_structure=True)
# Try to aggressively clean up
jax.clear_caches()
import gc; gc.collect()
out = {{"success": r.get("success")}}
if r.get("success"):
    pos = r.get("final_atom_positions")
    out["shape"] = list(pos.shape) if pos is not None else None
else:
    out["error"] = str(r.get("error", "?"))[:200]
del r
print(json.dumps(out))
'''
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        return False, f"CRASH(exit={r.returncode})"
    res = json.loads(r.stdout.strip())
    return res.get("success"), f"shape={res.get('shape')}" if res.get("success") else f"FAIL: {res.get('error','?')[:80]}"

if __name__ == "__main__":
    print("=== Delay test: Q9UJY1 seed=3 recycle=1 ===\n")

    # Run 1: short seq first
    print("[1] Short seq seed=1 (warmup)...", end="", flush=True)
    ok, msg = run(SHORT, 1)
    print(f" {msg}")

    # Delay 30s
    print("    Sleeping 60s to let OS reclaim...")
    time.sleep(60)

    # Run 2: Q9UJY1 seed=3
    print("[2] Q9UJY1 seed=3 (after 60s delay)...", end="", flush=True)
    ok, msg = run(SEQ, 3)
    print(f" {msg}")

    # Delay 60s
    print("    Sleeping 120s...")
    time.sleep(120)

    # Run 3: Q9UJY1 seed=5
    print("[3] Q9UJY1 seed=5 (after 120s delay)...", end="", flush=True)
    ok, msg = run(SEQ, 5)
    print(f" {msg}")

    print("\nDone.")
