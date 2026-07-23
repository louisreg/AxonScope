from __future__ import annotations

from axonfleet.runtime.jax.kernels.triton_double_cable import (
    jax_triton_thomas_dependency_skip_reason,
)


def test_jax_triton_dependency_probe_is_import_safe_without_gpu_stack():
    reason = jax_triton_thomas_dependency_skip_reason()

    assert reason is None or isinstance(reason, str)
