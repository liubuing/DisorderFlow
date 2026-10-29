"""Quick AF2 GPU inference test."""
# GPU 推理最小冒烟：一条固定抗体序列 + 短表位，recycle=0（最快路径），
# 验证 JAX/CUDA/参数文件全链路可用；任何异常都会打印完整 traceback。
import sys, os, time
# 把项目根与 modules/ 加入 sys.path 并 chdir 到根——af2_jax_runner 在 modules/ 下，
# 其内部按相对路径读取配置与参数
_project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "modules"))
os.chdir(_project_root)

print("Testing AF2 GPU inference...")

# 固定测试序列：通用抗体骨架（无 CDR 特异性含义），表位用甘氨酸-丝氨酸连接子占位
ab_seq = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAK"
epi_seq = "GGSGGSGGSGGSGGS"

from af2_jax_runner import run_multimer_prediction

t0 = time.time()
result = run_multimer_prediction(
    ab_seq, epi_seq,
    data_dir=os.path.expanduser('~/.cache/colabfold'),  # ColabFold 参数缓存目录
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
