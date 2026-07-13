from types import SimpleNamespace

import jax
import numpy as np
import jax.numpy as jnp

from axonscope.runtime.jax.kernels.block_tridiagonal import (
    solve_block_tridiagonal_2x2_scalar,
)
from axonscope.runtime.jax.cable_geometry import (
    apply_diffusion_operator,
    diffusion_operator_coeffs,
)
from axonscope.runtime.jax.kernels.double_cable_linear import (
    assemble_double_cable_linear_system,
    assemble_double_cable_linear_system_xb,
    double_cable_space_from_xb,
    double_cable_space_to_xb,
    prepare_double_cable_linear_system_static_terms,
    prepare_double_cable_linear_system_static_terms_xb,
)


def _uniform_cable_namespace(*, x_um, diffusion_coefficient: float) -> SimpleNamespace:
    diameter_um = 1.0
    Cm_uF_cm2 = 1.0
    radius_cm = 0.5 * diameter_um * 1e-4
    Ra_ohm_cm = radius_cm / (2.0 * diffusion_coefficient * Cm_uF_cm2 * 1e-6 * 1000.0)
    return SimpleNamespace(
        n_compartments=x_um.shape[0],
        h_cm=jnp.diff(x_um) * 1e-4,
        diam_um=jnp.full(x_um.shape, diameter_um, dtype=x_um.dtype),
        Ra_ohm_cm=jnp.full(x_um.shape, Ra_ohm_cm, dtype=x_um.dtype),
        Cm_uF_cm2=jnp.full(x_um.shape, Cm_uF_cm2, dtype=x_um.dtype),
        has_heterogeneous_cable_properties=False,
    )


def _dense_block_tridiagonal_solution(
    *,
    a00,
    a01,
    a10,
    a11,
    off0,
    off1,
    rhs0,
    rhs1,
) -> np.ndarray:
    a00_np = np.asarray(a00)
    n = a00_np.shape[0]
    matrix = np.zeros((2 * n, 2 * n), dtype=a00_np.dtype)
    rhs = np.empty((2 * n,), dtype=a00_np.dtype)
    for i in range(n):
        row = 2 * i
        matrix[row, row] = np.asarray(a00)[i]
        matrix[row, row + 1] = np.asarray(a01)[i]
        matrix[row + 1, row] = np.asarray(a10)[i]
        matrix[row + 1, row + 1] = np.asarray(a11)[i]
        rhs[row] = np.asarray(rhs0)[i]
        rhs[row + 1] = np.asarray(rhs1)[i]
        if i > 0:
            matrix[row, row - 2] = np.asarray(off0)[i - 1]
            matrix[row + 1, row - 1] = np.asarray(off1)[i - 1]
        if i < n - 1:
            matrix[row, row + 2] = np.asarray(off0)[i]
            matrix[row + 1, row + 3] = np.asarray(off1)[i]
    return np.linalg.solve(matrix, rhs).reshape(n, 2)


def test_non_uniform_diffusion_operator_matches_quadratic_second_derivative():
    """
    On V(x) = x^2, the discrete non-uniform operator should recover d2V/dx2 = 2.
    """
    x_um = jnp.array([0.0, 20.0, 55.0, 90.0, 150.0], dtype=jnp.float32)
    x_cm = x_um * 1e-4
    D = 0.3

    axon = _uniform_cable_namespace(x_um=x_um, diffusion_coefficient=D)

    lower, diag, upper = diffusion_operator_coeffs(axon, jnp.float32)
    V = x_cm ** 2
    diffusion = apply_diffusion_operator(V, lower, diag, upper)
    np.testing.assert_allclose(np.asarray(diffusion)[1:-1], 2.0 * D, atol=2e-5, rtol=0.0)


def test_sealed_end_diffusion_operator_keeps_constant_profile_constant():
    """A constant voltage profile must remain diffusion-free everywhere."""
    x_um = jnp.array([0.0, 20.0, 55.0, 90.0, 150.0], dtype=jnp.float32)
    V = jnp.full_like(x_um, -67.5)

    axon = _uniform_cable_namespace(x_um=x_um, diffusion_coefficient=0.3)

    lower, diag, upper = diffusion_operator_coeffs(axon, jnp.float32)
    diffusion = apply_diffusion_operator(V, lower, diag, upper)
    np.testing.assert_allclose(np.asarray(diffusion), 0.0, atol=1e-7, rtol=0.0)


def test_scalar_block_tridiagonal_solver_matches_generic_2x2_solver():
    a00 = jnp.asarray([4.0, 4.2, 4.1, 4.3], dtype=jnp.float32)
    a01 = jnp.asarray([-1.1, -1.0, -1.2, -1.1], dtype=jnp.float32)
    a10 = jnp.asarray([-1.1, -1.0, -1.2, -1.1], dtype=jnp.float32)
    a11 = jnp.asarray([5.0, 5.1, 5.2, 5.3], dtype=jnp.float32)
    off0 = jnp.asarray([-0.15, -0.20, -0.18], dtype=jnp.float32)
    off1 = jnp.asarray([-0.05, -0.07, -0.06], dtype=jnp.float32)
    rhs0 = jnp.asarray([1.0, -0.5, 0.25, 0.75], dtype=jnp.float32)
    rhs1 = jnp.asarray([-0.2, 0.4, 0.8, -0.1], dtype=jnp.float32)

    dense = _dense_block_tridiagonal_solution(
        a00=a00,
        a01=a01,
        a10=a10,
        a11=a11,
        off0=off0,
        off1=off1,
        rhs0=rhs0,
        rhs1=rhs1,
    )
    scalar0, scalar1 = solve_block_tridiagonal_2x2_scalar(
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
    )
    scalar = jnp.stack([scalar0, scalar1], axis=1)

    np.testing.assert_allclose(np.asarray(scalar), dense, rtol=1e-6, atol=1e-6)


def test_double_cable_linear_system_assembly_matches_explicit_formula():
    batch_size = 3
    n = 5
    batch = jnp.arange(batch_size, dtype=jnp.float32)[:, None]
    x = jnp.arange(n, dtype=jnp.float32)[None, :]

    Vi = -70.0 + 0.2 * batch + 0.1 * x
    Ve = 0.5 * batch - 0.05 * x
    Gm_abs = 0.02 + 0.001 * x + 0.0002 * batch
    GE_abs = -1.0 + 0.02 * x
    Iinj_abs = 0.01 * batch + 0.002 * x
    I_outward_abs = 0.003 * x
    I_corr_abs = 0.001 * batch
    extracellular_drive_abs = 0.04 + 0.005 * x

    Cm_abs = jnp.linspace(0.08, 0.12, n, dtype=jnp.float32)
    Cx_abs = jnp.linspace(0.02, 0.03, n, dtype=jnp.float32)
    Gx_abs = jnp.linspace(0.004, 0.006, n, dtype=jnp.float32)
    Gax_i = jnp.linspace(0.2, 0.3, n - 1, dtype=jnp.float32)
    Gax_e = jnp.linspace(0.05, 0.07, n - 1, dtype=jnp.float32)
    left_i = jnp.linspace(0.1, 0.2, n, dtype=jnp.float32)
    right_i = jnp.linspace(0.15, 0.25, n, dtype=jnp.float32)
    left_e = jnp.linspace(0.01, 0.02, n, dtype=jnp.float32)
    right_e = jnp.linspace(0.012, 0.022, n, dtype=jnp.float32)
    area = jnp.linspace(1.0, 1.2, n, dtype=jnp.float32)
    background = jnp.zeros((n,), dtype=jnp.float32)
    dt_ms = jnp.asarray(0.01, dtype=jnp.float32)

    static = prepare_double_cable_linear_system_static_terms(
        area_cm2=area,
        Cm_abs=Cm_abs,
        Cx_abs=Cx_abs,
        Gx_abs=Gx_abs,
        Gax_e=Gax_e,
        Gax_i=Gax_i,
        left_i=left_i,
        right_i=right_i,
        left_e=left_e,
        right_e=right_e,
        I_background=background,
        dt_ms=dt_ms,
        batch_size=batch_size,
        nx=n,
    )
    system = assemble_double_cable_linear_system(
        Vi=Vi,
        Ve=Ve,
        Gm_abs=Gm_abs,
        GE_abs=GE_abs,
        static=static,
        Iinj_abs=Iinj_abs,
        I_outward_abs=I_outward_abs,
        I_corr_abs=I_corr_abs,
        extracellular_drive_abs=extracellular_drive_abs,
    )

    cm_over_dt = Cm_abs[None, :] / dt_ms
    cx_over_dt = Cx_abs[None, :] / dt_ms
    vm = Vi - Ve
    membrane_charge = cm_over_dt * vm
    np.testing.assert_allclose(
        np.asarray(system.a00),
        np.asarray(cm_over_dt + left_i[None, :] + right_i[None, :] + Gm_abs),
        rtol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(system.a01), np.asarray(-(cm_over_dt + Gm_abs)), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(system.a10), np.asarray(system.a01), rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(system.a11),
        np.asarray(cm_over_dt + cx_over_dt + Gx_abs[None, :] + left_e[None, :] + right_e[None, :] + Gm_abs),
        rtol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(system.off0), np.asarray(-Gax_i), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(system.off1), np.asarray(-Gax_e), rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(system.rhs0),
        np.asarray(membrane_charge + GE_abs + Iinj_abs - I_outward_abs - I_corr_abs),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(system.rhs1),
        np.asarray(
            -membrane_charge
            - GE_abs
            + cx_over_dt * Ve
            + extracellular_drive_abs
            + I_outward_abs
            + I_corr_abs
        ),
        rtol=1e-6,
    )


def test_double_cable_linear_system_xb_assembly_matches_batch_first():
    batch_size = 3
    n = 5
    batch = jnp.arange(batch_size, dtype=jnp.float32)[:, None]
    x = jnp.arange(n, dtype=jnp.float32)[None, :]

    static = prepare_double_cable_linear_system_static_terms(
        area_cm2=jnp.linspace(1.0, 1.2, n, dtype=jnp.float32),
        Cm_abs=jnp.linspace(0.08, 0.12, n, dtype=jnp.float32),
        Cx_abs=jnp.linspace(0.02, 0.03, n, dtype=jnp.float32),
        Gx_abs=jnp.linspace(0.004, 0.006, n, dtype=jnp.float32),
        Gax_e=jnp.linspace(0.05, 0.07, n - 1, dtype=jnp.float32),
        Gax_i=jnp.linspace(0.2, 0.3, n - 1, dtype=jnp.float32),
        left_i=jnp.linspace(0.1, 0.2, n, dtype=jnp.float32),
        right_i=jnp.linspace(0.15, 0.25, n, dtype=jnp.float32),
        left_e=jnp.linspace(0.01, 0.02, n, dtype=jnp.float32),
        right_e=jnp.linspace(0.012, 0.022, n, dtype=jnp.float32),
        I_background=jnp.zeros((n,), dtype=jnp.float32),
        dt_ms=jnp.asarray(0.01, dtype=jnp.float32),
        batch_size=batch_size,
        nx=n,
    )
    static_xb = prepare_double_cable_linear_system_static_terms_xb(
        area_cm2=jnp.linspace(1.0, 1.2, n, dtype=jnp.float32),
        Cm_abs=jnp.linspace(0.08, 0.12, n, dtype=jnp.float32),
        Cx_abs=jnp.linspace(0.02, 0.03, n, dtype=jnp.float32),
        Gx_abs=jnp.linspace(0.004, 0.006, n, dtype=jnp.float32),
        Gax_e=jnp.linspace(0.05, 0.07, n - 1, dtype=jnp.float32),
        Gax_i=jnp.linspace(0.2, 0.3, n - 1, dtype=jnp.float32),
        left_i=jnp.linspace(0.1, 0.2, n, dtype=jnp.float32),
        right_i=jnp.linspace(0.15, 0.25, n, dtype=jnp.float32),
        left_e=jnp.linspace(0.01, 0.02, n, dtype=jnp.float32),
        right_e=jnp.linspace(0.012, 0.022, n, dtype=jnp.float32),
        I_background=jnp.zeros((n,), dtype=jnp.float32),
        dt_ms=jnp.asarray(0.01, dtype=jnp.float32),
        batch_size=batch_size,
        nx=n,
    )
    for name in (
        "area",
        "cm_over_dt",
        "cx_over_dt",
        "cx_plus_gx",
        "a00_static",
        "a11_static",
        "background_abs",
        "zero_abs",
    ):
        np.testing.assert_allclose(
            np.asarray(double_cable_space_from_xb(getattr(static_xb, name))),
            np.asarray(getattr(static, name)),
            rtol=1e-6,
        )
    values = {
        "Vi": -70.0 + 0.2 * batch + 0.1 * x,
        "Ve": 0.5 * batch - 0.05 * x,
        "Gm_abs": 0.02 + 0.001 * x + 0.0002 * batch,
        "GE_abs": -1.0 + 0.02 * x,
        "Iinj_abs": 0.01 * batch + 0.002 * x,
        "I_outward_abs": 0.003 * x,
        "I_corr_abs": 0.001 * batch,
        "extracellular_drive_abs": 0.04 + 0.005 * x,
    }

    system = assemble_double_cable_linear_system(static=static, **values)
    system_xb = assemble_double_cable_linear_system_xb(
        static=static_xb,
        **{
            key: double_cable_space_to_xb(value, batch_size=batch_size, nx=n)
            for key, value in values.items()
        },
    )

    for name in system._fields:
        batch_first = getattr(system, name)
        node_first = getattr(system_xb, name)
        if name in {"off0", "off1"}:
            np.testing.assert_allclose(
                np.asarray(node_first),
                np.broadcast_to(
                    np.asarray(batch_first)[:, None],
                    np.asarray(node_first).shape,
                ),
                rtol=1e-6,
            )
        else:
            converted = double_cable_space_from_xb(node_first)
            np.testing.assert_allclose(
                np.asarray(converted),
                np.asarray(batch_first),
                rtol=1e-6,
            )


def test_scalar_block_tridiagonal_solver_handles_single_row_under_jit():
    solve = jax.jit(solve_block_tridiagonal_2x2_scalar)

    scalar0, scalar1 = solve(
        jnp.asarray([4.0], dtype=jnp.float32),
        jnp.asarray([-1.0], dtype=jnp.float32),
        jnp.asarray([-1.0], dtype=jnp.float32),
        jnp.asarray([5.0], dtype=jnp.float32),
        jnp.asarray([], dtype=jnp.float32),
        jnp.asarray([], dtype=jnp.float32),
        jnp.asarray([2.0], dtype=jnp.float32),
        jnp.asarray([3.0], dtype=jnp.float32),
    )

    expected = np.linalg.solve(
        np.asarray([[4.0, -1.0], [-1.0, 5.0]], dtype=np.float32),
        np.asarray([2.0, 3.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray([scalar0[0], scalar1[0]]),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )
