"""V5 Step 0: Chai-1 dry-run — API validation + memory probe.

Tests run_inference with minimal sequences, confirms constraint API,
checks GPU memory on RTX 5060 8GB."""
import os, sys, tempfile, time, json
os.chdir("/mnt/c/biological/DisorderFlow")

print("V5 Step 0: Chai-1 API dry-run")
print("=" * 50)

# Test 1: Minimal inference
print("\n[Test 1] Minimal 2-chain inference (10+10aa)...")
fasta = """>protein|antibody
AAAAAAAAAA
>protein|peptide
AAAAAAAAAA
"""
tmp_fasta = "/tmp/v5_test.fasta"
with open(tmp_fasta, "w") as f:
    f.write(fasta)

tmp_out = "/tmp/v5_test_out"
os.makedirs(tmp_out, exist_ok=True)

from chai_lab.chai1 import run_inference
from pathlib import Path

tmp_fasta_p = Path(tmp_fasta)
tmp_out_p = Path(tmp_out)

t0 = time.time()
try:
    run_inference(
        fasta_file=tmp_fasta_p,
        output_dir=tmp_out_p,
        use_msa_server=False,
        use_templates_server=False,
        use_esm_embeddings=True,
        num_trunk_recycles=0,
        num_diffn_timesteps=10,  # minimal for speed
        num_diffn_samples=1,
        device="cuda",
    )
    dt = time.time() - t0
    print(f"  OK: {dt:.0f}s")
    # Check output
    for f in os.listdir(tmp_out):
        if f.endswith(".json"):
            with open(os.path.join(tmp_out, f)) as jf:
                data = json.load(jf)
                keys = list(data.keys()) if isinstance(data, dict) else "list"
                print(f"  Output keys: {keys}")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 2: Check constraint API
print("\n[Test 2] Checking constraint API...")
try:
    from chai_lab import constraints
    print(f"  constraints module: {dir(constraints)[:10]}")
except Exception as e:
    print(f"  constraints module FAIL: {e}")

# Test 3: Memory probe with realistic size
print("\n[Test 3] Memory probe (~240aa antibody + 10aa peptide)...")
ab_seq = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAK"  # ~120aa VH
fasta2 = f">protein|antibody\n{ab_seq}\n>protein|peptide\nKLVFFAED\n"
tmp_fasta2 = "/tmp/v5_test_real.fasta"
with open(tmp_fasta2, "w") as f:
    f.write(fasta2)
tmp_out2 = "/tmp/v5_test_real_out"
os.makedirs(tmp_out2, exist_ok=True)

tmp_fasta2_p = Path(tmp_fasta2)
tmp_out2_p = Path(tmp_out2)

t0 = time.time()
try:
    run_inference(
        fasta_file=tmp_fasta2_p,
        output_dir=tmp_out2_p,
        use_msa_server=False,
        use_templates_server=False,
        use_esm_embeddings=True,
        num_trunk_recycles=0,
        num_diffn_timesteps=10,
        num_diffn_samples=1,
        device="cuda",
    )
    dt = time.time() - t0
    print(f"  OK: {dt:.0f}s (no OOM!)")
    for f in os.listdir(tmp_out2):
        fp = os.path.join(tmp_out2, f)
        if f.endswith(".json"):
            with open(fp) as jf:
                data = json.load(jf)
                for k in ["iptm", "ptm", "plddt"]:
                    if k in str(data)[:500]:
                        print(f"  {k} present in output")
except Exception as e:
    print(f"  FAIL: {e}")

print("\nStep 0 complete.")
