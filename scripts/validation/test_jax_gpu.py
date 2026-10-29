"""Test JAX GPU availability."""
# 环境自检脚本：确认 JAX 能看到 GPU 并通过一次小规模矩阵乘法冒烟测试。
# 在 DisorderFlow 管线启动 AF2/BFN 等重推理任务前，用于快速排查 GPU 驱动与后端配置。
import jax

# 打印后端与设备清单：若 default_backend 不是 gpu，说明 JAX 回退到了 CPU，需先修复环境再继续。
print("JAX version:", jax.__version__)
print("Devices:", jax.devices())
print("GPU count:", jax.device_count("gpu"))
print("Default backend:", jax.default_backend())

# Quick GPU computation test
# 1000x1000 矩阵乘仅用于验证 GPU 数值通路可用，规模刻意很小以保证秒级完成。
import jax.numpy as jnp
x = jnp.ones((1000, 1000))
y = jnp.dot(x, x)
print(f"GPU matmul test: {x.shape} @ {x.shape} -> sum={y.sum():.1f}")
print("JAX GPU OK!")
