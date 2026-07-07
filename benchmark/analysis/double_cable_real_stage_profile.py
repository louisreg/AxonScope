from __future__ import annotations

import argparse
import csv
import json
import os
import platform as host_platform
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import axonscope as axs
from axonscope.backends.jax.batch_kernels import (
    _initial_double_cable_batch_state,
    _prepare_double_cable_batch_arrays,
    _resolve_double_cable_kernel_block_solver,
)
from axonscope.backends.jax.common import (
    solve_block_tridiagonal_2x2_pcr,
    solve_block_tridiagonal_2x2_pcr_soa,
    solve_block_tridiagonal_2x2_pcr_soa_batched,
    solve_block_tridiagonal_2x2_pcr_soa_batched_nomask,
    solve_block_tridiagonal_2x2_pcr_soa_batched_padded,
    solve_block_tridiagonal_2x2_pcr_soa_batched_shift,
    solve_block_tridiagonal_2x2_pcr_soa_batched_transposed,
    solve_block_tridiagonal_2x2_pcr_soa_hybrid_batched,
    solve_block_tridiagonal_2x2_scalar,
)
from axonscope.backends.jax.input_lowering import (
    lower_double_cable_extracellular_input,
    lower_double_cable_intracellular_input,
)
from axonscope.backends.jax.observer_runtime import (
    init_vm_raster_state,
    update_vm_raster_state_batch_from_tables,
)
from axonscope.backends.jax.recording_lowering import (
    lower_batch_recording_options,
    lower_observers_for_cohort,
)
from axonscope.backends.jax.runtime_preparation import (
    prepare_batch_runtime,
    prepared_cohort_for_group,
)
from axonscope.backends.jax.shape_bucketing import double_cable_kernel_group
from axonscope.dispatcher.plan import build_dispatch_plan
from axonscope.solvers.options import BatchOptions, SolverOptions, resolve_double_cable_block_solver

from benchmark.analysis.double_cable_solver_candidates import (
    solve_block_tridiagonal_2x2_pcr_soa_batched_symmetric,
)
from benchmark.workloads.curve_options import PRESETS
from benchmark.workloads.curve_runtime import _build_pool


REPEAT_FIELDS = (
    "stage",
    "variant",
    "phase",
    "repeat",
    "platform",
    "device",
    "precision",
    "target_nx",
    "actual_nx",
    "n_axons",
    "kernel_group_size",
    "diameters",
    "recording",
    "extracellular_format",
    "elapsed_ms",
    "rss_delta_mib",
    "output_bytes",
)

SUMMARY_FIELDS = (
    "stage",
    "variant",
    "platform",
    "device",
    "precision",
    "target_nx",
    "actual_nx",
    "n_axons",
    "kernel_group_size",
    "diameters",
    "recording",
    "extracellular_format",
    "repeats",
    "mean_ms",
    "min_ms",
    "max_ms",
    "first_run_ms",
    "rss_delta_mib_max",
    "output_bytes",
)


@dataclass(frozen=True)
class StageCase:
    stage: str
    variant: str
    fn: Callable[..., Any]
    args: tuple[Any, ...]


@dataclass(frozen=True)
class RealStageInputs:
    options: dict[str, Any]
    group_metadata: dict[str, Any]
    stage_cases: tuple[StageCase, ...]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Profile real AxonScope double-cable JAX stages using public "
            "benchmark workloads and backend/runtime preparation, without "
            "adding a production solver route."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/results/p11b_double_cable_real_stage_profile"),
    )
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="quick")
    parser.add_argument("--nx", type=int)
    parser.add_argument("--n-axons", type=int)
    parser.add_argument("--tsim", type=float)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--precision", choices=("fp32", "fp64"))
    parser.add_argument(
        "--diameters",
        choices=("same_diameter", "different_diameters"),
        default="different_diameters",
    )
    parser.add_argument(
        "--recording",
        choices=("observer_only", "full_vm", "probe_vm"),
        default="observer_only",
    )
    parser.add_argument("--amplitude-uA", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-chunk-steps", type=int, default=50)
    parser.add_argument(
        "--double-cable-block-solver",
        choices=("auto", "thomas", "pcr", "pcr_soa", "pcr_adaptive"),
        default="auto",
    )
    parser.add_argument(
        "--solver",
        action="append",
        choices=(
            "active_auto",
            "thomas_vmap",
            "pcr_matrix_vmap",
            "pcr_soa_vmap",
            "pcr_soa_batched",
            "pcr_soa_symmetric_batched",
            "pcr_soa_nomask_batched",
            "pcr_soa_shift_batched",
            "pcr_soa_transposed_batched",
            "pcr_soa_padded_batched",
            "pcr_soa_hybrid_batched",
        ),
        help="Block-solve variant to include. Repeat to select several. Defaults to active_auto.",
    )
    parser.add_argument(
        "--one-step-solver",
        action="append",
        choices=(
            "active_auto",
            "pcr_soa",
            "pcr_soa_batched",
            "pcr_soa_shift_batched",
            "pcr_soa_padded_batched",
        ),
        help=(
            "One-step proxy solver variant to include. Repeat to select several. "
            "Defaults to active_auto. Benchmark-only; this does not add a runtime policy."
        ),
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    if args.repeats < 1:
        parser.error("--repeats must be >= 1.")
    if args.warmups < 0:
        parser.error("--warmups must be >= 0.")
    if args.nx is not None and args.nx < 3:
        parser.error("--nx must be >= 3.")
    if args.n_axons is not None and args.n_axons < 1:
        parser.error("--n-axons must be >= 1.")
    if args.time_chunk_steps < 1:
        parser.error("--time-chunk-steps must be >= 1.")

    if args.precision == "fp64":
        jax.config.update("jax_enable_x64", True)
    device = _select_device(args.platform)
    args.output.mkdir(parents=True, exist_ok=True)

    with jax.default_device(device):
        inputs = _prepare_real_stage_inputs(args, device=device)

    metadata = _metadata(args=args, device=device, inputs=inputs)
    _write_json(args.output / "metadata.json", metadata)

    repeat_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    with jax.default_device(device):
        for case in inputs.stage_cases:
            rows, summary = _measure_case(
                case,
                repeats=args.repeats,
                warmups=args.warmups,
                args=args,
                device_name=str(device),
                group_metadata=inputs.group_metadata,
            )
            repeat_rows.extend(rows)
            summary_rows.append(summary)

    repeat_csv = args.output / "real_stage_repeats.csv"
    summary_csv = args.output / "real_stage_summary.csv"
    _write_csv(repeat_csv, REPEAT_FIELDS, repeat_rows)
    _write_csv(summary_csv, SUMMARY_FIELDS, summary_rows)
    if not args.no_plots:
        _write_plots(args.output / "plots", summary_rows)
    _write_report(args.output / "real_stage_report.md", summary_rows, metadata)

    print(f"wrote: {summary_csv}")
    print(f"wrote: {args.output / 'real_stage_report.md'}")
    return 0


def _prepare_real_stage_inputs(args: argparse.Namespace, *, device: Any) -> RealStageInputs:
    preset = PRESETS[args.preset]
    precision = args.precision or preset.precision
    options = {
        "preset": args.preset,
        "source": "point_source_axonscope",
        "tsim": float(args.tsim if args.tsim is not None else preset.tsim),
        "dt": float(args.dt if args.dt is not None else preset.dt),
        "nx": int(args.nx if args.nx is not None else preset.nx),
        "n_axons": int(args.n_axons if args.n_axons is not None else preset.n_axons),
        "precision": precision,
        "recording": args.recording,
        "cable": "double_cable",
        "double_cable_block_solver": args.double_cable_block_solver,
        "population": "single_model",
        "diameters": args.diameters,
        "platform": args.platform,
        "execution_policy": "default",
        "repeats": int(args.repeats),
        "warmups": int(args.warmups),
        "memory_trace": "rss",
        "memory_top_n": 0,
        "profile": False,
        "profile_backend": "auto",
        "profile_output": None,
        "profile_create_perfetto": False,
        "jax_device_memory_profile": False,
        "jax_device_memory_profile_stages": ("kernel.wait",),
        "output": str(args.output),
        "resume": False,
        "case_filter": None,
        "spatial_recording": "probes",
        "observer_criterion": "vm_raster",
        "amplitude_min": 0.0,
        "amplitude_max": float(args.amplitude_uA),
        "amplitude_tolerance": 1e-3,
        "max_iterations": 1,
        "stimulation": "biphasic",
        "seed": int(args.seed),
        "cache_mode": "warm",
        "time_chunk_policy": "explicit",
        "time_chunk_steps": int(args.time_chunk_steps),
        "one_step_solver": tuple(getattr(args, "one_step_solver", None) or ("active_auto",)),
        "amplitude_batch_size": None,
        "retention": "summary_only",
        "amplitude_count": 1,
    }
    amplitudes = np.full(options["n_axons"], float(args.amplitude_uA), dtype=float)
    pool, row_meta = _build_pool(options, amplitudes, curve_context="recruitment")
    plan = build_dispatch_plan(pool)
    double_groups = tuple(group for group in plan.groups if group.mode == "double")
    if not double_groups:
        raise RuntimeError("real double-cable stage profile built no double-cable dispatch group.")
    group = max(double_groups, key=lambda item: item.size)
    kernel_group = double_cable_kernel_group(group)

    runtime = prepare_batch_runtime(
        kernel_group,
        tsim_ms=options["tsim"],
        dt_ms=options["dt"],
        solver_options=SolverOptions(),
        mode="double",
        include_extracellular=True,
        include_area=True,
        backend_context=None,
    )
    cohort = prepared_cohort_for_group(kernel_group)

    observer_defs = _observer_definitions(options) if args.recording == "observer_only" else None
    batch_options = _batch_options(args)
    kernel_options = lower_batch_recording_options(
        kernel_group,
        batch_options,
        observers=observer_defs,
    )
    observer_plan = lower_observers_for_cohort(
        observer_defs,
        cohort=cohort,
        dtype=runtime.membrane.dtype,
        prefer_vm_raster=kernel_options.recording.mode == "none",
    )
    intracellular = lower_double_cable_intracellular_input(
        cohort=cohort,
        runtime=runtime,
    )
    extracellular = lower_double_cable_extracellular_input(
        cohort=cohort,
        runtime=runtime,
        tsim_ms=options["tsim"],
        dt_ms=options["dt"],
        observer_plan=observer_plan,
        kernel_options=kernel_options,
    )
    batch_size = int(cohort.size)
    resolved = resolve_double_cable_block_solver(
        args.double_cable_block_solver,
        platform=args.platform,
    )
    active_solver = _resolve_double_cable_kernel_block_solver(
        resolved,
        batch_size=batch_size,
    )
    arrays = _prepare_double_cable_batch_arrays(
        runtime=runtime,
        batch_size=batch_size,
        output=args.recording,
        variant=active_solver,
        time_chunk_steps=kernel_options.time_chunk_steps,
        factorized_vext=extracellular.factorized is not None,
        observer="vm_raster" if observer_plan is not None else None,
    )
    (
        area_cm2,
        Cm_abs,
        Cx_abs,
        Gx_abs,
        Gax_e,
        Gax_i,
        left_i,
        right_i,
        left_e,
        right_e,
        background,
        shared_coefficients,
    ) = arrays
    Vi, Ve, gates, state = _initial_double_cable_batch_state(
        runtime,
        batch_size=batch_size,
        Veinit_mV=0.0,
    )
    row_indices = jnp.arange(batch_size, dtype=jnp.int32)
    dt_ms = jnp.asarray(options["dt"], dtype=runtime.membrane.dtype)
    Vm = Vi - Ve

    gates_pred = _block_until_ready(
        _real_gate_update(
            runtime.membrane.backend,
            gates,
            Vm,
            row_indices,
            dt_ms,
        )
    )
    Gm_den, GE_den = _block_until_ready(
        _real_membrane_conductance_terms(
            runtime.membrane.backend,
            gates_pred,
            row_indices,
        )
    )
    drive_case = _drive_stage_case(
        extracellular.midpoint,
        extracellular.initial_previous,
        Cx_abs=Cx_abs,
        Gx_abs=Gx_abs,
        batch_size=batch_size,
        nx=runtime.membrane.Nx,
        dt_ms=dt_ms,
        dtype_local=runtime.membrane.dtype,
        extracellular_format=extracellular.format,
    )
    drive = _block_until_ready(drive_case.fn(*drive_case.args))
    Iinj_abs = jnp.zeros_like(_batch_space(area_cm2, batch_size=batch_size, nx=runtime.membrane.Nx))
    I_outward = background
    I_corr = jnp.zeros_like(_batch_space(background, batch_size=batch_size, nx=runtime.membrane.Nx))
    assembled = _block_until_ready(
        _real_assemble_system(
            Vi,
            Ve,
            Gm_den,
            GE_den,
            area_cm2,
            Cm_abs,
            Cx_abs,
            Gx_abs,
            left_i,
            right_i,
            left_e,
            right_e,
            Gax_i,
            Gax_e,
            Iinj_abs,
            I_outward,
            I_corr,
            drive,
            dt_ms,
        )
    )
    (
        area_precomputed,
        cm_over_dt_precomputed,
        cx_over_dt_precomputed,
        a00_static,
        a11_static,
        off_i_precomputed,
        off_e_precomputed,
        I_outward_abs_precomputed,
        I_corr_abs_precomputed,
    ) = _precompute_assembly_inputs(
        Vi=Vi,
        area_cm2=area_cm2,
        Cm_abs=Cm_abs,
        Cx_abs=Cx_abs,
        Gx_abs=Gx_abs,
        left_i=left_i,
        right_i=right_i,
        left_e=left_e,
        right_e=right_e,
        Gax_i=Gax_i,
        Gax_e=Gax_e,
        I_outward_den=I_outward,
        I_corr_den=I_corr,
        dt_ms=dt_ms,
    )
    assembled_precomputed = _block_until_ready(
        _real_assemble_system_precomputed(
            Vi,
            Ve,
            Gm_den,
            GE_den,
            area_precomputed,
            cm_over_dt_precomputed,
            cx_over_dt_precomputed,
            a00_static,
            a11_static,
            off_i_precomputed,
            off_e_precomputed,
            Iinj_abs,
            I_outward_abs_precomputed,
            I_corr_abs_precomputed,
            drive,
        )
    )
    _assert_assembled_close(assembled, assembled_precomputed)
    observer_state = None
    if observer_plan is not None:
        observer_state = init_vm_raster_state(observer_plan, batch_size=batch_size, nt=1)

    requested_solvers = tuple(args.solver or ("active_auto",))
    solver_cases = _solver_cases(
        assembled,
        requested=requested_solvers,
        active_solver=active_solver,
    )
    one_step_solvers = _one_step_solver_names(
        getattr(args, "one_step_solver", None) or ("active_auto",),
        active_solver=active_solver,
    )
    one_step_cases = tuple(
        StageCase(
            "one_step_proxy",
            f"{name}_real",
            _real_one_step_proxy,
            (
                runtime.membrane.backend,
                Vi,
                Ve,
                gates,
                row_indices,
                area_cm2,
                Cm_abs,
                Cx_abs,
                Gx_abs,
                left_i,
                right_i,
                left_e,
                right_e,
                Gax_i,
                Gax_e,
                Iinj_abs,
                I_outward,
                I_corr,
                drive,
                dt_ms,
                name,
            ),
        )
        for name in one_step_solvers
    )
    one_step_precomputed_cases = tuple(
        StageCase(
            "one_step_proxy",
            f"{name}_real_precomputed_static",
            _real_one_step_proxy_precomputed,
            (
                runtime.membrane.backend,
                Vi,
                Ve,
                gates,
                row_indices,
                area_precomputed,
                cm_over_dt_precomputed,
                cx_over_dt_precomputed,
                a00_static,
                a11_static,
                off_i_precomputed,
                off_e_precomputed,
                Iinj_abs,
                I_outward_abs_precomputed,
                I_corr_abs_precomputed,
                drive,
                dt_ms,
                name,
            ),
        )
        for name in one_step_solvers
    )
    stage_cases = [
        StageCase(
            "membrane_gate_update",
            _backend_variant(runtime.membrane.backend),
            _real_gate_update,
            (runtime.membrane.backend, gates, Vm, row_indices, dt_ms),
        ),
        StageCase(
            "membrane_conductance_terms",
            _backend_variant(runtime.membrane.backend),
            _real_membrane_conductance_terms,
            (runtime.membrane.backend, gates_pred, row_indices),
        ),
        drive_case,
        StageCase(
            "system_assembly",
            "real_double_cable",
            _real_assemble_system,
            (
                Vi,
                Ve,
                Gm_den,
                GE_den,
                area_cm2,
                Cm_abs,
                Cx_abs,
                Gx_abs,
                left_i,
                right_i,
                left_e,
                right_e,
                Gax_i,
                Gax_e,
                Iinj_abs,
                I_outward,
                I_corr,
                drive,
                dt_ms,
            ),
        ),
        StageCase(
            "system_assembly",
            "precomputed_static",
            _real_assemble_system_precomputed,
            (
                Vi,
                Ve,
                Gm_den,
                GE_den,
                area_precomputed,
                cm_over_dt_precomputed,
                cx_over_dt_precomputed,
                a00_static,
                a11_static,
                off_i_precomputed,
                off_e_precomputed,
                Iinj_abs,
                I_outward_abs_precomputed,
                I_corr_abs_precomputed,
                drive,
            ),
        ),
        *solver_cases,
        *one_step_cases,
        *one_step_precomputed_cases,
    ]
    if observer_plan is not None and observer_state is not None:
        stage_cases.append(
            StageCase(
                "observer_write",
                "vm_raster_real",
                _real_observer_write,
                (
                    observer_state,
                    Vm,
                    observer_plan.probe_indices,
                    observer_plan.probe_mask,
                    observer_plan.thresholds_mV,
                ),
            )
        )

    group_metadata = {
        "target_nx": int(options["nx"]),
        "actual_nx": int(runtime.membrane.Nx),
        "n_axons": int(options["n_axons"]),
        "kernel_group_size": int(batch_size),
        "public_group_size": int(group.size),
        "diameters": args.diameters,
        "recording": args.recording,
        "extracellular_format": extracellular.format,
        "extracellular_factorized_rank": extracellular.factorized_rank,
        "shared_coefficients": bool(shared_coefficients),
        "active_solver": active_solver,
        "one_step_solvers": one_step_solvers,
        "assembly_variants": ("real_double_cable", "precomputed_static"),
        "resolved_solver": resolved,
        "membrane_backend": _backend_variant(runtime.membrane.backend),
        "membrane_model": type(runtime.membrane.membrane).__name__,
        "uses_generated_model_step": bool(
            getattr(runtime.membrane.membrane, "uses_generated_model_step", False)
        ),
        "stateless_vm_only": bool(
            runtime.membrane.membrane.supports_stateless_vm_only_fast_path()
        ),
        "unique_row_diameters": int(len({round(float(meta["diameter_um"]), 6) for meta in row_meta})),
        "device": str(device),
    }
    return RealStageInputs(
        options=options,
        group_metadata=group_metadata,
        stage_cases=tuple(stage_cases),
    )


def _assert_assembled_close(reference: tuple[Any, ...], candidate: tuple[Any, ...]) -> None:
    for index, (expected, actual) in enumerate(zip(reference, candidate, strict=True)):
        np.testing.assert_allclose(
            np.asarray(actual),
            np.asarray(expected),
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"precomputed assembly output {index} differs from baseline",
        )


def _batch_options(args: argparse.Namespace) -> BatchOptions:
    solver = args.double_cable_block_solver
    if args.recording == "observer_only":
        return BatchOptions.none(
            time_chunk_steps=int(args.time_chunk_steps),
            double_cable_block_solver=solver,
        )
    if args.recording == "full_vm":
        return BatchOptions.full(
            time_chunk_steps=int(args.time_chunk_steps),
            double_cable_block_solver=solver,
        )
    return BatchOptions.probes(
        8,
        time_chunk_steps=int(args.time_chunk_steps),
        double_cable_block_solver=solver,
    )


def _observer_definitions(options: dict[str, Any]) -> tuple[Any, ...]:
    return (
        axs.analysis.Activation(
            threshold=0.0 * axs.mV,
            blanking=min(0.2, float(options["tsim"]) * 0.25) * axs.ms,
            target=axs.positions.ALL,
        ),
    )


@partial(jax.jit, static_argnames=("backend",))
def _real_gate_update(
    backend: Any,
    gates: Any,
    Vm: Any,
    row_indices: Any,
    dt_ms: Any,
) -> Any:
    row_gate_update = getattr(backend, "cn_gate_update_for_row", None)
    if callable(row_gate_update):
        return jax.vmap(
            lambda row_index, gates_row, vm_row: row_gate_update(
                row_index,
                g_prev=gates_row,
                V_mV=vm_row,
                dt=dt_ms,
            )
        )(row_indices, gates, Vm)
    return jax.vmap(
        lambda gates_row, vm_row: backend.cn_gate_update(
            g_prev=gates_row,
            V_mV=vm_row,
            dt=dt_ms,
        )
    )(gates, Vm)


@partial(jax.jit, static_argnames=("backend",))
def _real_membrane_conductance_terms(
    backend: Any,
    gates: Any,
    row_indices: Any,
) -> tuple[Any, Any]:
    row_terms = getattr(backend, "membrane_conductance_terms_for_row", None)
    if callable(row_terms):
        return jax.vmap(
            lambda row_index, gates_row: row_terms(row_index, gates_row)
        )(row_indices, gates)
    return jax.vmap(backend.membrane_conductance_terms)(gates)


def _drive_stage_case(
    midpoint: Any,
    initial_previous: Any,
    *,
    Cx_abs: Any,
    Gx_abs: Any,
    batch_size: int,
    nx: int,
    dt_ms: Any,
    dtype_local: Any,
    extracellular_format: str,
) -> StageCase:
    if hasattr(midpoint, "footprint_mV_per_A"):
        return StageCase(
            "extracellular_rhs_drive",
            extracellular_format,
            _real_factorized_extracellular_drive_for_step,
            (
                midpoint.current_mid_A,
                midpoint.current_initial_previous_A,
                midpoint.footprint_mV_per_A,
                Cx_abs,
                Gx_abs,
                batch_size,
                nx,
                dt_ms,
                dtype_local,
            ),
        )
    return StageCase(
        "extracellular_rhs_drive",
        extracellular_format,
        _real_dense_extracellular_drive_for_step,
        (
            midpoint,
            initial_previous,
            Cx_abs,
            Gx_abs,
            batch_size,
            nx,
            dt_ms,
            dtype_local,
        ),
    )


@partial(jax.jit, static_argnames=("batch_size", "nx", "dtype_local"))
def _real_factorized_extracellular_drive_for_step(
    current_mid_A: Any,
    current_initial_previous_A: Any,
    footprint_mV_per_A: Any,
    Cx_abs: Any,
    Gx_abs: Any,
    batch_size: int,
    nx: int,
    dt_ms: Any,
    dtype_local: Any,
) -> Any:
    cx_over_dt = _batch_space(Cx_abs, batch_size=batch_size, nx=nx) / dt_ms
    gx = _batch_space(Gx_abs, batch_size=batch_size, nx=nx)
    footprint = _batch_space(
        jnp.asarray(footprint_mV_per_A, dtype=dtype_local),
        batch_size=batch_size,
        nx=nx,
    )
    current_mid = jnp.asarray(current_mid_A, dtype=dtype_local)
    current_now = current_mid[0] if current_mid.ndim == 1 else current_mid[:, 0]
    previous = jnp.asarray(current_initial_previous_A, dtype=dtype_local)
    return (
        (cx_over_dt + gx) * _current_to_space(current_now)
        - cx_over_dt * _current_to_space(previous)
    ) * footprint


@partial(jax.jit, static_argnames=("batch_size", "nx", "dtype_local"))
def _real_dense_extracellular_drive_for_step(
    midpoint: Any,
    initial_previous: Any,
    Cx_abs: Any,
    Gx_abs: Any,
    batch_size: int,
    nx: int,
    dt_ms: Any,
    dtype_local: Any,
) -> Any:
    cx_over_dt = _batch_space(Cx_abs, batch_size=batch_size, nx=nx) / dt_ms
    gx = _batch_space(Gx_abs, batch_size=batch_size, nx=nx)
    vext = jnp.asarray(midpoint, dtype=dtype_local)[:, 0, :]
    previous = jnp.asarray(initial_previous, dtype=dtype_local)
    return (cx_over_dt + gx) * vext - cx_over_dt * previous


@jax.jit
def _real_assemble_system(
    Vi: Any,
    Ve: Any,
    Gm_den: Any,
    GE_den: Any,
    area_cm2: Any,
    Cm_abs: Any,
    Cx_abs: Any,
    Gx_abs: Any,
    left_i: Any,
    right_i: Any,
    left_e: Any,
    right_e: Any,
    Gax_i: Any,
    Gax_e: Any,
    Iinj_abs: Any,
    I_outward_den: Any,
    I_corr_den: Any,
    extracellular_drive_abs: Any,
    dt_ms: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    batch_size = int(Vi.shape[0])
    nx = int(Vi.shape[1])
    area = _batch_space(area_cm2, batch_size=batch_size, nx=nx)
    cm_over_dt = _batch_space(Cm_abs, batch_size=batch_size, nx=nx) / dt_ms
    cx_over_dt = _batch_space(Cx_abs, batch_size=batch_size, nx=nx) / dt_ms
    gx = _batch_space(Gx_abs, batch_size=batch_size, nx=nx)
    left_i_batch = _batch_space(left_i, batch_size=batch_size, nx=nx)
    right_i_batch = _batch_space(right_i, batch_size=batch_size, nx=nx)
    left_e_batch = _batch_space(left_e, batch_size=batch_size, nx=nx)
    right_e_batch = _batch_space(right_e, batch_size=batch_size, nx=nx)
    Gm_abs = Gm_den * area
    GE_abs = GE_den * area
    I_outward_abs = _batch_space(I_outward_den, batch_size=batch_size, nx=nx) * area
    I_corr_abs = _batch_space(I_corr_den, batch_size=batch_size, nx=nx) * area
    Vm = Vi - Ve
    a00 = cm_over_dt + Gm_abs + left_i_batch + right_i_batch
    a01 = -(cm_over_dt + Gm_abs)
    a10 = a01
    a11 = cm_over_dt + Gm_abs + cx_over_dt + gx + left_e_batch + right_e_batch
    rhs0 = cm_over_dt * Vm + GE_abs + Iinj_abs - I_outward_abs - I_corr_abs
    rhs1 = (
        -cm_over_dt * Vm
        - GE_abs
        + cx_over_dt * Ve
        + extracellular_drive_abs
        + I_outward_abs
        + I_corr_abs
    )
    return a00, a01, a10, a11, -jnp.asarray(Gax_i), -jnp.asarray(Gax_e), rhs0, rhs1


def _precompute_assembly_inputs(
    *,
    Vi: Any,
    area_cm2: Any,
    Cm_abs: Any,
    Cx_abs: Any,
    Gx_abs: Any,
    left_i: Any,
    right_i: Any,
    left_e: Any,
    right_e: Any,
    Gax_i: Any,
    Gax_e: Any,
    I_outward_den: Any,
    I_corr_den: Any,
    dt_ms: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    batch_size = int(Vi.shape[0])
    nx = int(Vi.shape[1])
    area = _batch_space(area_cm2, batch_size=batch_size, nx=nx)
    cm_over_dt = _batch_space(Cm_abs, batch_size=batch_size, nx=nx) / dt_ms
    cx_over_dt = _batch_space(Cx_abs, batch_size=batch_size, nx=nx) / dt_ms
    gx = _batch_space(Gx_abs, batch_size=batch_size, nx=nx)
    left_i_batch = _batch_space(left_i, batch_size=batch_size, nx=nx)
    right_i_batch = _batch_space(right_i, batch_size=batch_size, nx=nx)
    left_e_batch = _batch_space(left_e, batch_size=batch_size, nx=nx)
    right_e_batch = _batch_space(right_e, batch_size=batch_size, nx=nx)
    return _block_until_ready(
        (
            area,
            cm_over_dt,
            cx_over_dt,
            cm_over_dt + left_i_batch + right_i_batch,
            cm_over_dt + cx_over_dt + gx + left_e_batch + right_e_batch,
            -jnp.asarray(Gax_i),
            -jnp.asarray(Gax_e),
            _batch_space(I_outward_den, batch_size=batch_size, nx=nx) * area,
            _batch_space(I_corr_den, batch_size=batch_size, nx=nx) * area,
        )
    )


@jax.jit
def _real_assemble_system_precomputed(
    Vi: Any,
    Ve: Any,
    Gm_den: Any,
    GE_den: Any,
    area: Any,
    cm_over_dt: Any,
    cx_over_dt: Any,
    a00_static: Any,
    a11_static: Any,
    off_i: Any,
    off_e: Any,
    Iinj_abs: Any,
    I_outward_abs: Any,
    I_corr_abs: Any,
    extracellular_drive_abs: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    Gm_abs = Gm_den * area
    GE_abs = GE_den * area
    Vm = Vi - Ve
    cm_plus_gm = cm_over_dt + Gm_abs
    membrane_charge = cm_over_dt * Vm
    a00 = a00_static + Gm_abs
    a01 = -cm_plus_gm
    a10 = a01
    a11 = a11_static + Gm_abs
    rhs0 = membrane_charge + GE_abs + Iinj_abs - I_outward_abs - I_corr_abs
    rhs1 = (
        -membrane_charge
        - GE_abs
        + cx_over_dt * Ve
        + extracellular_drive_abs
        + I_outward_abs
        + I_corr_abs
    )
    return a00, a01, a10, a11, off_i, off_e, rhs0, rhs1


@partial(jax.jit, static_argnames=("backend", "solver"))
def _real_one_step_proxy(
    backend: Any,
    Vi: Any,
    Ve: Any,
    gates: Any,
    row_indices: Any,
    area_cm2: Any,
    Cm_abs: Any,
    Cx_abs: Any,
    Gx_abs: Any,
    left_i: Any,
    right_i: Any,
    left_e: Any,
    right_e: Any,
    Gax_i: Any,
    Gax_e: Any,
    Iinj_abs: Any,
    I_outward_den: Any,
    I_corr_den: Any,
    extracellular_drive_abs: Any,
    dt_ms: Any,
    solver: str,
) -> tuple[Any, Any, Any]:
    Vm = Vi - Ve
    gates_new = _real_gate_update(backend, gates, Vm, row_indices, dt_ms)
    Gm_den, GE_den = _real_membrane_conductance_terms(backend, gates_new, row_indices)
    assembled = _real_assemble_system(
        Vi,
        Ve,
        Gm_den,
        GE_den,
        area_cm2,
        Cm_abs,
        Cx_abs,
        Gx_abs,
        left_i,
        right_i,
        left_e,
        right_e,
        Gax_i,
        Gax_e,
        Iinj_abs,
        I_outward_den,
        I_corr_den,
        extracellular_drive_abs,
        dt_ms,
    )
    Vi_new, Ve_new = _solve_by_name(solver, assembled)
    return Vi_new, Ve_new, gates_new


@partial(jax.jit, static_argnames=("backend", "solver"))
def _real_one_step_proxy_precomputed(
    backend: Any,
    Vi: Any,
    Ve: Any,
    gates: Any,
    row_indices: Any,
    area: Any,
    cm_over_dt: Any,
    cx_over_dt: Any,
    a00_static: Any,
    a11_static: Any,
    off_i: Any,
    off_e: Any,
    Iinj_abs: Any,
    I_outward_abs: Any,
    I_corr_abs: Any,
    extracellular_drive_abs: Any,
    dt_ms: Any,
    solver: str,
) -> tuple[Any, Any, Any]:
    Vm = Vi - Ve
    gates_new = _real_gate_update(backend, gates, Vm, row_indices, dt_ms)
    Gm_den, GE_den = _real_membrane_conductance_terms(backend, gates_new, row_indices)
    assembled = _real_assemble_system_precomputed(
        Vi,
        Ve,
        Gm_den,
        GE_den,
        area,
        cm_over_dt,
        cx_over_dt,
        a00_static,
        a11_static,
        off_i,
        off_e,
        Iinj_abs,
        I_outward_abs,
        I_corr_abs,
        extracellular_drive_abs,
    )
    Vi_new, Ve_new = _solve_by_name(solver, assembled)
    return Vi_new, Ve_new, gates_new


@jax.jit
def _solve_thomas_vmap(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
) -> tuple[Any, Any]:
    a00, a01, a10, a11, off0, off1 = _broadcast_solver_coefficients(
        a00, a01, a10, a11, off0, off1, rhs0=rhs0
    )
    return jax.vmap(solve_block_tridiagonal_2x2_scalar)(
        a00, a01, a10, a11, off0, off1, rhs0, rhs1
    )


@jax.jit
def _solve_pcr_matrix_vmap(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
) -> tuple[Any, Any]:
    a00, a01, a10, a11, off0, off1 = _broadcast_solver_coefficients(
        a00, a01, a10, a11, off0, off1, rhs0=rhs0
    )
    return jax.vmap(solve_block_tridiagonal_2x2_pcr)(
        a00, a01, a10, a11, off0, off1, rhs0, rhs1
    )


@jax.jit
def _solve_pcr_soa_vmap(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
) -> tuple[Any, Any]:
    a00, a01, a10, a11, off0, off1 = _broadcast_solver_coefficients(
        a00, a01, a10, a11, off0, off1, rhs0=rhs0
    )
    return jax.vmap(solve_block_tridiagonal_2x2_pcr_soa)(
        a00, a01, a10, a11, off0, off1, rhs0, rhs1
    )


@jax.jit
def _solve_pcr_soa_batched(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
) -> tuple[Any, Any]:
    return solve_block_tridiagonal_2x2_pcr_soa_batched(
        a00, a01, a10, a11, off0, off1, rhs0, rhs1
    )


@jax.jit
def _solve_pcr_soa_symmetric_batched(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
) -> tuple[Any, Any]:
    return solve_block_tridiagonal_2x2_pcr_soa_batched_symmetric(
        a00, a01, a10, a11, off0, off1, rhs0, rhs1
    )


@jax.jit
def _solve_pcr_soa_nomask_batched(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
) -> tuple[Any, Any]:
    return solve_block_tridiagonal_2x2_pcr_soa_batched_nomask(
        a00, a01, a10, a11, off0, off1, rhs0, rhs1
    )


@jax.jit
def _solve_pcr_soa_shift_batched(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
) -> tuple[Any, Any]:
    return solve_block_tridiagonal_2x2_pcr_soa_batched_shift(
        a00, a01, a10, a11, off0, off1, rhs0, rhs1
    )


@jax.jit
def _solve_pcr_soa_transposed_batched(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
) -> tuple[Any, Any]:
    return solve_block_tridiagonal_2x2_pcr_soa_batched_transposed(
        a00, a01, a10, a11, off0, off1, rhs0, rhs1
    )


@jax.jit
def _solve_pcr_soa_padded_batched(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
) -> tuple[Any, Any]:
    return solve_block_tridiagonal_2x2_pcr_soa_batched_padded(
        a00, a01, a10, a11, off0, off1, rhs0, rhs1
    )


@jax.jit
def _solve_pcr_soa_hybrid_batched(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
) -> tuple[Any, Any]:
    return solve_block_tridiagonal_2x2_pcr_soa_hybrid_batched(
        a00, a01, a10, a11, off0, off1, rhs0, rhs1
    )


@jax.jit
def _real_observer_write(
    state: Any,
    Vm: Any,
    probe_indices: Any,
    probe_mask: Any,
    thresholds_mV: Any,
) -> Any:
    return update_vm_raster_state_batch_from_tables(
        state,
        vm_mV=Vm,
        step_index=jnp.asarray(0, dtype=jnp.int32),
        probe_indices=probe_indices,
        probe_mask=probe_mask,
        thresholds_mV=thresholds_mV,
    )


def _broadcast_solver_coefficients(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    *,
    rhs0: Any,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    batch_size = int(rhs0.shape[0])
    nx = int(rhs0.shape[1])
    return (
        _batch_space(a00, batch_size=batch_size, nx=nx),
        _batch_space(a01, batch_size=batch_size, nx=nx),
        _batch_space(a10, batch_size=batch_size, nx=nx),
        _batch_space(a11, batch_size=batch_size, nx=nx),
        _batch_edge(off0, batch_size=batch_size, nx=nx),
        _batch_edge(off1, batch_size=batch_size, nx=nx),
    )


def _solver_cases(
    assembled: tuple[Any, Any, Any, Any, Any, Any, Any, Any],
    *,
    requested: Sequence[str],
    active_solver: str,
) -> tuple[StageCase, ...]:
    out: list[StageCase] = []
    names: list[str] = []
    for name in requested:
        resolved = active_solver if name == "active_auto" else name
        if resolved == "pcr_adaptive":
            resolved = "pcr_soa_batched"
        if resolved == "pcr_soa":
            resolved = "pcr_soa_batched"
        if resolved == "pcr":
            resolved = "pcr_matrix_vmap"
        if resolved == "thomas":
            resolved = "thomas_vmap"
        if resolved not in names:
            names.append(resolved)
    mapping = {
        "thomas_vmap": _solve_thomas_vmap,
        "pcr_matrix_vmap": _solve_pcr_matrix_vmap,
        "pcr_soa_vmap": _solve_pcr_soa_vmap,
        "pcr_soa_batched": _solve_pcr_soa_batched,
        "pcr_soa_symmetric_batched": _solve_pcr_soa_symmetric_batched,
        "pcr_soa_nomask_batched": _solve_pcr_soa_nomask_batched,
        "pcr_soa_shift_batched": _solve_pcr_soa_shift_batched,
        "pcr_soa_transposed_batched": _solve_pcr_soa_transposed_batched,
        "pcr_soa_padded_batched": _solve_pcr_soa_padded_batched,
        "pcr_soa_hybrid_batched": _solve_pcr_soa_hybrid_batched,
    }
    for name in names:
        out.append(StageCase("block_solve", name, mapping[name], assembled))
    return tuple(out)


def _one_step_solver_names(
    requested: Sequence[str],
    *,
    active_solver: str,
) -> tuple[str, ...]:
    names: list[str] = []
    for name in requested:
        resolved = active_solver if name == "active_auto" else name
        if resolved == "pcr_adaptive":
            resolved = "pcr_soa"
        if resolved not in names:
            names.append(resolved)
    return tuple(names)


def _solve_by_name(
    solver: str,
    assembled: tuple[Any, Any, Any, Any, Any, Any, Any, Any],
) -> tuple[Any, Any]:
    if solver == "thomas":
        return _solve_thomas_vmap(*assembled)
    if solver == "pcr":
        return _solve_pcr_matrix_vmap(*assembled)
    if solver in {"pcr_soa", "pcr_adaptive", "pcr_soa_batched"}:
        return _solve_pcr_soa_batched(*assembled)
    if solver == "pcr_soa_shift_batched":
        return _solve_pcr_soa_shift_batched(*assembled)
    if solver == "pcr_soa_padded_batched":
        return _solve_pcr_soa_padded_batched(*assembled)
    raise ValueError(f"unsupported one-step solver: {solver!r}")


def _batch_space(values: Any, *, batch_size: int, nx: int) -> Any:
    arr = jnp.asarray(values)
    if arr.ndim == 0:
        return jnp.broadcast_to(arr, (batch_size, nx))
    if arr.ndim == 1:
        return jnp.broadcast_to(arr[None, :], (batch_size, nx))
    return arr


def _batch_edge(values: Any, *, batch_size: int, nx: int) -> Any:
    arr = jnp.asarray(values)
    edge_count = max(int(nx) - 1, 0)
    if arr.ndim == 1:
        return jnp.broadcast_to(arr[None, :], (batch_size, edge_count))
    return arr


def _current_to_space(value: Any) -> Any:
    arr = jnp.asarray(value)
    if arr.ndim == 0:
        return arr
    return arr[:, None]


def _measure_case(
    case: StageCase,
    *,
    repeats: int,
    warmups: int,
    args: argparse.Namespace,
    device_name: str,
    group_metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rss_start = _rss_mib()
    start = perf_counter()
    first_out = _block_until_ready(case.fn(*case.args))
    first_run_ms = (perf_counter() - start) * 1000.0
    rss_end = _rss_mib()
    output_bytes = _output_nbytes(first_out)
    rows.append(
        _repeat_row(
            case=case,
            phase="first_run",
            repeat=0,
            args=args,
            device_name=device_name,
            group_metadata=group_metadata,
            elapsed_ms=first_run_ms,
            rss_delta_mib=_delta(rss_start, rss_end),
            output_bytes=output_bytes,
        )
    )
    for index in range(warmups):
        start = perf_counter()
        _block_until_ready(case.fn(*case.args))
        rows.append(
            _repeat_row(
                case=case,
                phase="warmup",
                repeat=index,
                args=args,
                device_name=device_name,
                group_metadata=group_metadata,
                elapsed_ms=(perf_counter() - start) * 1000.0,
                rss_delta_mib=None,
                output_bytes=output_bytes,
            )
        )
    measured: list[float] = []
    rss_deltas: list[float] = []
    for index in range(repeats):
        rss_start = _rss_mib()
        start = perf_counter()
        out = _block_until_ready(case.fn(*case.args))
        elapsed_ms = (perf_counter() - start) * 1000.0
        rss_end = _rss_mib()
        output_bytes = _output_nbytes(out)
        rss_delta = _delta(rss_start, rss_end)
        if rss_delta is not None:
            rss_deltas.append(rss_delta)
        measured.append(elapsed_ms)
        rows.append(
            _repeat_row(
                case=case,
                phase="measured",
                repeat=index,
                args=args,
                device_name=device_name,
                group_metadata=group_metadata,
                elapsed_ms=elapsed_ms,
                rss_delta_mib=rss_delta,
                output_bytes=output_bytes,
            )
        )
    summary = {
        **_row_context(args=args, device_name=device_name, group_metadata=group_metadata),
        "stage": case.stage,
        "variant": case.variant,
        "repeats": repeats,
        "mean_ms": sum(measured) / len(measured),
        "min_ms": min(measured),
        "max_ms": max(measured),
        "first_run_ms": first_run_ms,
        "rss_delta_mib_max": max(rss_deltas) if rss_deltas else None,
        "output_bytes": output_bytes,
    }
    return rows, summary


def _repeat_row(
    *,
    case: StageCase,
    phase: str,
    repeat: int,
    args: argparse.Namespace,
    device_name: str,
    group_metadata: dict[str, Any],
    elapsed_ms: float,
    rss_delta_mib: float | None,
    output_bytes: int,
) -> dict[str, Any]:
    return {
        **_row_context(args=args, device_name=device_name, group_metadata=group_metadata),
        "stage": case.stage,
        "variant": case.variant,
        "phase": phase,
        "repeat": repeat,
        "elapsed_ms": elapsed_ms,
        "rss_delta_mib": rss_delta_mib,
        "output_bytes": output_bytes,
    }


def _row_context(
    *,
    args: argparse.Namespace,
    device_name: str,
    group_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "platform": args.platform,
        "device": device_name,
        "precision": args.precision or PRESETS[args.preset].precision,
        "target_nx": group_metadata["target_nx"],
        "actual_nx": group_metadata["actual_nx"],
        "n_axons": group_metadata["n_axons"],
        "kernel_group_size": group_metadata["kernel_group_size"],
        "diameters": args.diameters,
        "recording": args.recording,
        "extracellular_format": group_metadata["extracellular_format"],
    }


def _block_until_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(_block_until_ready(item) for item in value)
    if isinstance(value, list):
        return [_block_until_ready(item) for item in value]
    block = getattr(value, "block_until_ready", None)
    if callable(block):
        block()
    return value


def _output_nbytes(value: Any) -> int:
    if isinstance(value, (tuple, list)):
        return sum(_output_nbytes(item) for item in value)
    nbytes = getattr(value, "nbytes", None)
    return int(nbytes or 0)


def _select_device(platform_name: str) -> Any:
    devices = jax.devices(platform_name)
    if not devices:
        raise RuntimeError(f"No JAX {platform_name} device is available.")
    return devices[0]


def _rss_mib() -> float | None:
    try:
        import psutil
    except Exception:
        return None
    process = psutil.Process(os.getpid())
    return float(process.memory_info().rss) / (1024.0 * 1024.0)


def _delta(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return end - start


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")


def _metadata(
    *,
    args: argparse.Namespace,
    device: Any,
    inputs: RealStageInputs,
) -> dict[str, Any]:
    return {
        "script": "benchmark/analysis/double_cable_real_stage_profile.py",
        "purpose": "Real AxonScope double-cable compiler/runtime stage cartography.",
        "platform": args.platform,
        "device": str(device),
        "jax_version": jax.__version__,
        "python": host_platform.python_version(),
        "host": {
            "system": host_platform.system(),
            "release": host_platform.release(),
            "machine": host_platform.machine(),
            "processor": host_platform.processor(),
        },
        "git": _git_metadata(),
        "options": inputs.options,
        "group": inputs.group_metadata,
        "limitations": [
            "Benchmark-only profiler; it does not add runtime policy or solver routing.",
            "Stages use real public double-cable axons and current backend preparation, but measure one-step proxies unless stated otherwise.",
            "Use curve benchmarks for end-to-end workflow speed claims.",
        ],
    }


def _git_metadata() -> dict[str, Any]:
    def run_git(*cmd: str) -> str | None:
        try:
            result = subprocess.run(
                ("git", *cmd),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except Exception:
            return None
        return result.stdout.strip()

    status = run_git("status", "--short")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_short": status,
    }


def _backend_variant(backend: Any) -> str:
    return type(backend).__name__


def _write_report(path: Path, rows: Sequence[dict[str, Any]], metadata: dict[str, Any]) -> None:
    stage_rows = sorted(rows, key=lambda item: (item["stage"], item["variant"]))
    block_rows = [row for row in rows if row["stage"] == "block_solve"]
    fastest_block = min(block_rows, key=lambda row: float(row["mean_ms"])) if block_rows else None
    lines = [
        "# Real Double-Cable Stage Profile",
        "",
        "Benchmark-only cartography for real AxonScope double-cable JAX stages.",
        "This report does not choose runtime policy.",
        "",
        "## Context",
        "",
        f"- Platform: `{metadata['platform']}`",
        f"- Device: `{metadata['device']}`",
        f"- JAX: `{metadata['jax_version']}`",
        f"- Git commit: `{metadata['git'].get('commit')}`",
        f"- Git dirty: `{metadata['git'].get('dirty')}`",
        f"- Target Nx: `{metadata['group']['target_nx']}`",
        f"- Actual kernel Nx: `{metadata['group']['actual_nx']}`",
        f"- Kernel group size: `{metadata['group']['kernel_group_size']}`",
        f"- Extracellular input: `{metadata['group']['extracellular_format']}`",
        f"- Active solver: `{metadata['group']['active_solver']}`",
        f"- Membrane backend: `{metadata['group']['membrane_backend']}`",
        "",
    ]
    if fastest_block is not None:
        lines.extend(
            [
                "## Fastest Measured Block Solve",
                "",
                (
                    f"`{fastest_block['variant']}` at "
                    f"{float(fastest_block['mean_ms']):.3f} ms mean "
                    f"({float(fastest_block['max_ms']):.3f} ms max)."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Stage Means",
            "",
            "| stage | variant | mean ms | first run ms | max ms | output KiB |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stage_rows:
        lines.append(
            "| {stage} | {variant} | {mean_ms:.3f} | {first_run_ms:.3f} | {max_ms:.3f} | {output_kib:.1f} |".format(
                output_kib=float(row["output_bytes"]) / 1024.0,
                **row,
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `membrane_gate_update` and `membrane_conductance_terms` use the real prepared membrane backend.",
            "- `system_assembly` uses real prepared double-cable coefficients and real extracellular forcing for one step.",
            "- `system_assembly/precomputed_static` precomposes static diagonal terms and absolute currents as a benchmark-only diagnostic.",
            "- `block_solve` runs selected solver functions on the real assembled system.",
            "- `one_step_proxy` fuses gate update, conductance terms, assembly, and the selected benchmark one-step block solver for one step; `_precomputed_static` variants use the same precomposed assembly inputs.",
            "- `observer_write` is present only for observer-only VmRaster runs.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plots(output_dir: Path, rows: Sequence[dict[str, Any]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "PLOTS_SKIPPED.txt").write_text(str(exc) + "\n", encoding="utf-8")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: float(row["mean_ms"]), reverse=True)
    labels = [f"{row['stage']}\n{row['variant']}" for row in ordered]
    values = [float(row["mean_ms"]) for row in ordered]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(labels, values, color="#5b758f")
    ax.set_ylabel("mean measured time (ms)")
    ax.set_title("Real double-cable stage profile")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "real_stage_means.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
