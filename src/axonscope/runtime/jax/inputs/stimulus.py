"""JAX-ready scalar stimulus callables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp

from axonscope.stimulation import Stimulus


@dataclass(frozen=True)
class JaxStimulus:
    """JAX-ready stimulus representation used by runtime input code.

    `t` is stored in milliseconds and `y` is stored in the numeric unit already
    required by the consuming solver term.
    """

    t: jnp.ndarray
    y: jnp.ndarray
    mode: Literal["hold", "linear"] = "hold"

    def __call__(self, tq):
        """Evaluate the stimulus at one scalar time in milliseconds."""

        if self.mode == "linear":
            return jnp.interp(tq, self.t, self.y, left=self.y[0], right=self.y[-1])

        idx = jnp.searchsorted(self.t, tq, side="right") - 1
        idx = jnp.clip(idx, 0, self.y.shape[0] - 1)
        return self.y[idx]


def compile_stimulus(
    stimulus: Stimulus,
    dtype_local: jnp.dtype | None = None,
) -> JaxStimulus:
    """Compile a descriptive stimulus to a JAX-ready callable.

    The stimulus is assumed to already be expressed in the physical unit needed
    by its consumer, such as nanoamperes for clamps or amperes for electrodes.
    """

    if dtype_local is None:
        dtype_local = jnp.float32
    return JaxStimulus(
        t=jnp.asarray(stimulus.t, dtype=dtype_local),
        y=jnp.asarray(stimulus.y, dtype=dtype_local),
        mode=stimulus.mode,
    )


__all__ = [
    "JaxStimulus",
    "compile_stimulus",
]
