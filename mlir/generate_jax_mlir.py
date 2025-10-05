import jax
import jax.numpy as jnp
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def simple_mlp(x, w1, b1, w2, b2):
    a1 = jnp.matmul(x, w1) + b1
    h1 = jax.nn.relu(a1)
    out = jnp.matmul(h1, w2) + b2
    return out

# Example input and weights
x = jnp.ones((2, 4), jnp.float32)
w1 = jnp.ones((4, 8), jnp.float32)
b1 = jnp.ones((8,), jnp.float32)
w2 = jnp.ones((8, 3), jnp.float32)
b2 = jnp.ones((3,), jnp.float32)

stablehlo_txt = jax.jit(simple_mlp).lower(x, w1, b1, w2, b2).compiler_ir(dialect="stablehlo")
print(stablehlo_txt)

with open("PoC_jax.mlir", "w") as f:
    f.write(str(stablehlo_txt))

print("Saved PoC_jax.mlir")
