#!/usr/bin/env python
"""Debug AF2 crash on Q9UJY1 (L=196) —test seed by seed, with subprocess isolation."""
import sys, os, subprocess, json, time

PROJ = r"C:\biological\disorderflow-main"
SEQ = "MADGQMPFSCHYPSRLRRDPFRDSPLSSRLLDDGFGMDPFPDDLTASWPDWALPRLSSAWPGTLRSGMVPRGPTATARFGVPAEGRTPPPFPGEPWKVCVNVHSFKPEELMVKTKDGYVEVSGKHEEKQQEGGIVSKNFTKKIQLPAEVDPVTVFASLSPEGLLIIEAPQVPPYSTFGESSFNNELPQDSQEVTCT"

def run_seed(seed, num_recycle=0):
    """Run a single AF2 seed in a subprocess, return (exit_code, success, shape_or_error)."""
    code = f'''
import sys, os, json
sys.path.insert(0, {PROJ!r})
sys.path.insert(0, {os.path.join(PROJ, "modules")!r})
os.chdir({PROJ!r})
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
from af2_jax_runner import run_multimer_prediction
seq = {SEQ!r}
r = run_multimer_prediction(seq, "", num_recycle={num_recycle}, jax_random_seed={seed}, return_structure=True)
out = {{"success": r.get("success"), "elapsed": r.get("elapsed", 0)}}
if r.get("success"):
    pos = r.get("final_atom_positions")
    out["shape"] = list(pos.shape) if pos is not None else None
else:
    out["error"] = str(r.get("error", "?"))[:200]
print(json.dumps(out))
'''
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=600,
            env={**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
        )
        if r.returncode != 0:
            stderr_tail = r.stderr.strip()[-300:] if r.stderr else ""
            return r.returncode, False, f"CRASH(exit={r.returncode}) stderr: {stderr_tail}"
        result = json.loads(r.stdout.strip())
        if result.get("success"):
            return 0, True, f"shape={result.get('shape')} ({result.get('elapsed',0):.0f}s)"
        else:
            return 0, False, f"FAIL: {result.get('error','?')}"
    except subprocess.TimeoutExpired:
        return -1, False, "TIMEOUT"
    except Exception as e:
        return -2, False, f"EXCEPTION: {e}"

if __name__ == "__main__":
    print(f"Debug: Q9UJY1 L={len(SEQ)}")
    print(f"Python: {sys.executable}")
    print()

    for recycle in [0, 1, 3]:
        print(f"--- num_recycle={recycle} ---")
        for seed in [1, 3, 5, 7, 9]:
            print(f"  seed={seed}...", end="", flush=True)
            t0 = time.time()
            exit_code, success, msg = run_seed(seed, num_recycle=recycle)
            print(f" {msg}")
            if not success:
                print(f"    -> First failure at recycle={recycle} seed={seed}, stopping this recycle")
                break
        else:
            print(f"  All seeds OK with recycle={recycle}!")
        print()
