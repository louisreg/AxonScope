from __future__ import annotations

import os

import jax.numpy as jnp
from jax import config

_dtype_name = os.environ.get("AXONSCOPE_DTYPE", "float32").strip().lower()
_enable_x64 = os.environ.get("AXONSCOPE_ENABLE_X64", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

if _dtype_name in {"float64", "double", "fp64"}:
    dtype = jnp.float64
    dtype_name = "float64"
    _enable_x64 = True
elif _dtype_name in {"float32", "single", "fp32", ""}:
    dtype = jnp.float32
    dtype_name = "float32"
else:
    raise ValueError(
        "AXONSCOPE_DTYPE must be 'float32' or 'float64', "
        f"got {_dtype_name!r}."
    )

enable_x64 = bool(_enable_x64)
config.update("jax_enable_x64", enable_x64)
