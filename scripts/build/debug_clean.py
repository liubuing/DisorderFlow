#!/usr/bin/env python
"""Clean system test: run ONE AF2 prediction, verify JAX resource cleanup."""
import sys, os, gc, time

PROJ = r"C:\biological\disorderflow-main"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "modules"))
os.chdir(PROJ)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import numpy as np

SEQ_SHORT = "MDPKDKEKKKNDDKKEKKKNDDKDK"
SEQ_LONG = "MADGQMPFSCHYPSRLRRDPFRDSPLSSRLLDDGFGMDPFPDDLTASWPDWALPRLSSAWPGTLRSGMVPRGPTATARFGVPAEGRTPPPFPGEPWKVCVNVHSFKPEELMVKTKDGYVEVSGKHEEKQQEGGIVSKNFTKKIQLPAEVDPVTVFASLSPEGLLIIEAPQVPPYSTFGESSFNNELPQDSQEVTCT"

from af2_jax_runner import run_multimer_prediction

print("=== Clean system test ===\n")

# Test 1: run long seq first (if system is clean, it should work)
print("[1] Direct Q9UJY1 seed=1 recycle=1...", end="", flush=True)
t0 = time.time()
r = run_multimer_prediction(SEQ_LONG, "", num_recycle=1, jax_random_seed=1, return_structure=True)
elapsed = time.time() - t0
if r.get("success"):
    print(f" OK shape={r['final_atom_positions'].shape} ({elapsed:.0f}s)")
else:
    print(f" FAIL: {r.get('error','?')[:100]} ({elapsed:.0f}s)")

# Force cleanup
jax.clear_caches()
gc.collect()
print("  -> jax.clear_caches() + gc.collect() done")
time.sleep(10)

# Test 2: same call again —if resource leak is within JAX, this should crash
print("[2] Q9UJY1 seed=3 recycle=1 (after cleanup)...", end="", flush=True)
t0 = time.time()
r2 = run_multimer_prediction(SEQ_LONG, "", num_recycle=1, jax_random_seed=3, return_structure=True)
elapsed = time.time() - t0
if r2.get("success"):
    print(f" OK shape={r2['final_atom_positions'].shape} ({elapsed:.0f}s)")
else:
    print(f" FAIL: {r2.get('error','?')[:100]} ({elapsed:.0f}s)")

# Check memory
jax.clear_caches()
gc.collect()
print("  -> cleanup done")
time.sleep(10)

# Test 3: third call
print("[3] Q9UJY1 seed=5 recycle=1 (after cleanup)...", end="", flush=True)
t0 = time.time()
r3 = run_multimer_prediction(SEQ_LONG, "", num_recycle=1, jax_random_seed=5, return_structure=True)
elapsed = time.time() - t0
if r3.get("success"):
    print(f" OK shape={r3['final_atom_positions'].shape} ({elapsed:.0f}s)")
else:
    print(f" FAIL: {r3.get('error','?')[:100]} ({elapsed:.0f}s)")

print("\nDone.")
