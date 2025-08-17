import numpy as np

def vtrap(x, y):
    """Stable implementation of vtrap (from NEURON mod file)."""
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        z = x / y
        out = np.where(np.abs(z) < 1e-6,
                       y * (1.0 - z / 2.0),   # series expansion
                       x / (np.exp(z) - 1.0))
    return out