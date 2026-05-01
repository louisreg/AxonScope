import numpy as np
import jax.numpy as jnp
import jax

def vtrap(x, y):
    """Stable implementation of vtrap (from NEURON mod file)."""
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        z = x / y
        out = np.where(np.abs(z) < 1e-6,
                       y * (1.0 - z / 2.0),   # series expansion
                       x / (np.exp(z) - 1.0))
    return out

@jax.jit
def vtrap_jax(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Safe version of x/(exp(x/y)-1) to avoid division by zero, with jax."""
    z = x / y
    small = jnp.abs(z) < 1e-6
    return jnp.where(small, y * (1 - z / 2), x / (jnp.exp(z) - 1))


@jax.jit
def expM1_jax(x: jnp.ndarray, y: float) -> jnp.ndarray:
    """Safe version of x / (exp(x/y) - 1) to avoid division by zero."""
    small = jnp.abs(x / y) < 1e-6
    return jnp.where(small, y * (1 - x / (2 * y)), x / (jnp.exp(x / y) - 1))