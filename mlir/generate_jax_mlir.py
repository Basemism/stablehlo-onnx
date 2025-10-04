import jax
import jax.numpy as jnp
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def toy(x, y, z):
    return jnp.matmul(jax.nn.relu(x + y), z)

x = jnp.ones((2, 4), jnp.float32)
y = jnp.ones((2, 4), jnp.float32)
z = jnp.ones((4, 3), jnp.float32)

stablehlo_txt = jax.jit(toy).lower(x, y, z).compiler_ir(dialect="stablehlo")
print(stablehlo_txt)

with open("PoC_jax.mlir", "w") as f:
    f.write(str(stablehlo_txt))

print("Saved PoC_jax.mlir")
