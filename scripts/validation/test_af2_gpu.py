"""Quick AF2 GPU inference test."""
import sys, os, time
_project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "modules"))
os.chdir(_project_root)

print("Testing AF2 GPU inference...")

ab_seq = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAK"
epi_seq = "GGSGGSGGSGGSGGS"

from af2_jax_runner import run_multimer_prediction

t0 = time.time()
result = run_multimer_prediction(
    ab_seq, epi_seq,
    data_dir=os.path.expanduser('~/.cache/colabfold'),
    num_recycle=0,
    jax_random_seed=42,
    return_structure=True,
)
dt = time.time() - t0
print(f"Done in {dt:.1f}s")
print(f"success: {result.get('success')}")
print(f"error: {result.get('error')}")
print(f"elapsed: {result.get('elapsed')}")
tb = result.get('traceback', '')
print(f"--- FULL TRACEBACK ---")
print(tb)
print(f"--- END ---")
