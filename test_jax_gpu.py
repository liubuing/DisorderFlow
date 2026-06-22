"""Test JAX GPU availability."""
import jax

print("JAX version:", jax.__version__)
print("Devices:", jax.devices())
print("GPU count:", jax.device_count("gpu"))
print("Default backend:", jax.default_backend())

# Quick GPU computation test
import jax.numpy as jnp
x = jnp.ones((1000, 1000))
y = jnp.dot(x, x)
print(f"GPU matmul test: {x.shape} @ {x.shape} -> sum={y.sum():.1f}")
print("JAX GPU OK!")
