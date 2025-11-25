import jax.numpy as jnp
from jax import config
dtype = jnp.float32

config.update("jax_enable_x64", False)