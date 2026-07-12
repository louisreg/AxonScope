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
from axonscope.runtime.jax.batch_kernels import (
    _initial_double_cable_batch_state,
    _prepare_double_cable_batch_arrays,
    _resolve_double_cable_kernel_block_solver,
)
from axonscope.runtime.jax.common import (
    double_cable_block_residual_norm,
    solve_block_tridiagonal_2x2_pcr,
    solve_block_tridiagonal_2x2_pcr_soa,
    solve_block_tridiagonal_2x2_pcr_soa_batched,
    solve_block_tridiagonal_2x2_pcr_soa_batched_nomask,
    solve_block_tridiagonal_2x2_pcr_soa_batched_padded,
    solve_block_tridiagonal_2x2_pcr_soa_batched_shift,
    solve_block_tridiagonal_2x2_pcr_soa_batched_transposed,
    solve_block_tridiagonal_2x2_pcr_soa_hybrid_batched,
    solve_block_tridiagonal_2x2_scalar,
    solve_block_tridiagonal_2x2_scalar_batched,
)
from axonscope.runtime.jax.input_lowering import (
    lower_double_cable_extracellular_input,
    lower_double_cable_intracellular_input,
)
from axonscope.runtime.jax.jax_triton_double_cable import (
    solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_batched,
    solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_loop_xb,
    solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_loop_batched,
)
from axonscope.runtime.jax.observer_runtime import (
    init_vm_raster_state,
    update_vm_raster_state_batch_from_tables,
)
from axonscope.runtime.jax.recording_lowering import (
    lower_batch_recording_options,
    lower_observers_for_cohort,
)
from axonscope.runtime.jax.runtime_preparation import (
    prepare_batch_runtime,
    prepared_cohort_for_group,
)
from axonscope.runtime.jax.shape_bucketing import double_cable_kernel_group
from axonscope.dispatcher.plan import build_dispatch_plan
from axonscope.solvers.options import BatchOptions, SolverOptions

from benchmark.analysis.double_cable_solver_candidates import (
    solve_block_tridiagonal_2x2_pcr_soa_batched_symmetric,
)
from benchmark.workloads.curve_options import PRESETS
from benchmark.workloads.curve_runtime import _build_pool


REPEAT_FIELDS = (
    "stage",
    "variant",
    "layout",
    "block_b",
    "stage_group",
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
    "membrane_backend",
    "membrane_model",
    "membrane_gates_max",
    "membrane_channels_max",
    "membrane_backend_branches",
    "membrane_gated_compartments",
    "membrane_leak_compartments",
    "elapsed_ms",
    "rss_delta_mib",
    "output_bytes",
)

_GPU_PLATFORMS = frozenset({"cuda", "gpu", "metal", "rocm"})


def _resolve_profile_block_solver(solver: str, *, platform: str) -> str:
    if solver == "auto":
        return "pcr_adaptive" if platform.lower() in _GPU_PLATFORMS else "thomas"
    if solver in {"thomas", "pcr", "pcr_soa", "pcr_adaptive"}:
        return solver
    raise ValueError(
        "double-cable stage profile solver must be auto, thomas, pcr, "
        "pcr_soa, or pcr_adaptive."
    )

SUMMARY_FIELDS = (
    "stage",
    "variant",
    "layout",
    "block_b",
    "stage_group",
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
    "membrane_backend",
    "membrane_model",
    "membrane_gates_max",
    "membrane_channels_max",
    "membrane_backend_branches",
    "membrane_gated_compartments",
    "membrane_leak_compartments",
    "repeats",
    "mean_ms",
    "min_ms",
    "max_ms",
    "first_run_ms",
    "rss_delta_mib_max",
    "output_bytes",
)

VALIDATION_FIELDS = (
    "stage",
    "variant",
    "reference_variant",
    "layout",
    "block_b",
    "stage_group",
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
    "membrane_backend",
    "membrane_model",
    "membrane_gates_max",
    "membrane_channels_max",
    "membrane_backend_branches",
    "membrane_gated_compartments",
    "membrane_leak_compartments",
    "max_abs_diff",
    "max_rel_diff",
    "reference_max_abs",
    "max_abs_vm_diff",
    "max_rel_vm_diff",
    "max_residual_norm",
    "median_residual_norm",
    "all_finite",
    "passes_tolerance",
    "status",
    "atol",
    "rtol",
    "residual_tolerance",
    "notes",
)


@dataclass(frozen=True)
class StageCase:
    stage: str
    variant: str
    fn: Callable[..., Any]
    args: tuple[Any, ...]
    layout: str | None = None
    block_b: int | None = None
    validation_args: tuple[Any, ...] | None = None


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
        "--jax-triton-block-b",
        type=int,
        nargs="+",
        help=(
            "Tile widths for benchmark-only jax-triton tiled Thomas variants. "
            "Defaults preserve each variant's historical setting."
        ),
    )
    parser.add_argument(
        "--solver",
        action="append",
        choices=(
            "active_auto",
            "thomas_vmap",
            "thomas_batched_scan",
            "jax_triton_tiled_thomas",
            "jax_triton_tiled_thomas_loop",
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
            "thomas_batched_scan",
            "jax_triton_tiled_thomas",
            "jax_triton_tiled_thomas_loop",
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
    parser.add_argument(
        "--validate-solvers",
        action="store_true",
        help=(
            "After timing measurements, compare selected solver outputs against "
            "a trusted reference and write real_stage_validation.csv. This is "
            "benchmark-only and deliberately runs after measurement so cold "
            "timing is not hidden by validation."
        ),
    )
    parser.add_argument(
        "--validation-atol",
        type=float,
        help="Absolute tolerance for --validate-solvers. Defaults by precision.",
    )
    parser.add_argument(
        "--validation-rtol",
        type=float,
        help="Relative tolerance for --validate-solvers. Defaults by precision.",
    )
    parser.add_argument(
        "--validation-residual-tolerance",
        type=float,
        help=(
            "Maximum relative residual allowed for block-solve validation. "
            "Defaults by precision."
        ),
    )
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
    if args.jax_triton_block_b and any(value < 1 for value in args.jax_triton_block_b):
        parser.error("--jax-triton-block-b values must be >= 1.")
    if args.validation_atol is not None and args.validation_atol < 0.0:
        parser.error("--validation-atol must be >= 0.")
    if args.validation_rtol is not None and args.validation_rtol < 0.0:
        parser.error("--validation-rtol must be >= 0.")
    if (
        args.validation_residual_tolerance is not None
        and args.validation_residual_tolerance < 0.0
    ):
        parser.error("--validation-residual-tolerance must be >= 0.")

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
    validation_rows: list[dict[str, Any]] = []
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
        if args.validate_solvers:
            validation_rows = _validate_solver_cases(
                inputs.stage_cases,
                args=args,
                device_name=str(device),
                group_metadata=inputs.group_metadata,
            )

    repeat_csv = args.output / "real_stage_repeats.csv"
    summary_csv = args.output / "real_stage_summary.csv"
    _write_csv(repeat_csv, REPEAT_FIELDS, repeat_rows)
    _write_csv(summary_csv, SUMMARY_FIELDS, summary_rows)
    if args.validate_solvers:
        _write_csv(args.output / "real_stage_validation.csv", VALIDATION_FIELDS, validation_rows)
    if not args.no_plots:
        _write_plots(args.output / "plots", summary_rows)
    _write_report(
        args.output / "real_stage_report.md",
        summary_rows,
        metadata,
        validation_rows=validation_rows,
    )

    print(f"wrote: {summary_csv}")
    if args.validate_solvers:
        print(f"wrote: {args.output / 'real_stage_validation.csv'}")
    print(f"wrote: {args.output / 'real_stage_report.md'}")
    if any(row["status"] != "pass" for row in validation_rows):
        print("solver validation failed; inspect real_stage_validation.csv", file=sys.stderr)
        return 2
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
        "jax_triton_block_b": tuple(args.jax_triton_block_b or ()),
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
        "validate_solvers": bool(args.validate_solvers),
        "validation_atol": _validation_atol(args),
        "validation_rtol": _validation_rtol(args),
        "validation_residual_tolerance": _validation_residual_tolerance(args),
        "amplitude_batch_size": None,
        "retention": "summary_only",
        "amplitude_count": 1,
    }
    amplitudes = np.full(options["n_axons"], float(args.amplitude_uA), dtype=float)
    pool, row_meta, _update_handles, _shared_stimulus, _stimulus_cache = _build_pool(
        options,
        amplitudes,
        curve_context="recruitment",
    )
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
        runtime_context=None,
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
    )
    batch_size = int(cohort.size)
    resolved = _resolve_profile_block_solver(
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
    gated_leak_cases: tuple[StageCase, ...] = ()
    if _is_gated_leak_stack_backend(runtime.membrane.backend):
        gated_gm, gated_ge = _block_until_ready(
            _real_gated_only_conductance_terms(
                runtime.membrane.backend,
                gates_pred,
            )
        )
        gated_leak_cases = (
            StageCase(
                "membrane_gate_update_gated_only",
                _backend_variant(runtime.membrane.backend),
                _real_gated_only_gate_update,
                (runtime.membrane.backend, gates, Vm, dt_ms),
            ),
            StageCase(
                "membrane_conductance_terms_gated_only",
                _backend_variant(runtime.membrane.backend),
                _real_gated_only_conductance_terms,
                (runtime.membrane.backend, gates_pred),
            ),
            StageCase(
                "membrane_conductance_terms_mask_mix",
                _backend_variant(runtime.membrane.backend),
                _real_gated_leak_conductance_mix,
                (runtime.membrane.backend, gates_pred, gated_gm, gated_ge),
            ),
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
    (
        area_precomputed_xb,
        cm_over_dt_precomputed_xb,
        cx_over_dt_precomputed_xb,
        a00_static_xb,
        a11_static_xb,
        off_i_precomputed_xb,
        off_e_precomputed_xb,
        I_outward_abs_precomputed_xb,
        I_corr_abs_precomputed_xb,
    ) = _precompute_assembly_inputs_xb(
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
    assembled_precomputed_xb = _block_until_ready(
        _real_assemble_system_precomputed_xb(
            _space_to_xb(Vi),
            _space_to_xb(Ve),
            _space_to_xb(Gm_den),
            _space_to_xb(GE_den),
            area_precomputed_xb,
            cm_over_dt_precomputed_xb,
            cx_over_dt_precomputed_xb,
            a00_static_xb,
            a11_static_xb,
            off_i_precomputed_xb,
            off_e_precomputed_xb,
            _space_to_xb(Iinj_abs),
            I_outward_abs_precomputed_xb,
            I_corr_abs_precomputed_xb,
            _space_to_xb(drive),
        )
    )
    _assert_assembled_close(assembled_precomputed, _assembled_from_xb(assembled_precomputed_xb))
    observer_state = None
    if observer_plan is not None:
        observer_state = init_vm_raster_state(observer_plan, batch_size=batch_size, nt=1)

    requested_solvers = tuple(args.solver or ("active_auto",))
    solver_cases = _solver_cases(
        assembled,
        assembled_xb=assembled_precomputed_xb,
        requested=requested_solvers,
        active_solver=active_solver,
        jax_triton_block_bs=_jax_triton_block_bs(args),
    )
    one_step_solvers = _expand_jax_triton_one_step_solvers(
        _one_step_solver_names(
            getattr(args, "one_step_solver", None) or ("active_auto",),
            active_solver=active_solver,
        ),
        jax_triton_block_bs=_jax_triton_block_bs(args),
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
            layout=_solver_layout(name),
            block_b=_solver_block_b(name),
        )
        for name in one_step_solvers
    )
    one_step_without_solve_cases = (
        StageCase(
            "one_step_without_solve",
            "real_materialized",
            _real_one_step_without_solve,
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
            ),
        ),
        StageCase(
            "one_step_without_solve",
            "precomputed_static_materialized",
            _real_one_step_without_solve_precomputed,
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
            ),
        ),
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
            layout=_solver_layout(name),
            block_b=_solver_block_b(name),
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
        *gated_leak_cases,
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
            layout="BX",
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
            layout="BX",
        ),
        StageCase(
            "system_assembly",
            "precomputed_static_xb",
            _real_assemble_system_precomputed_xb,
            (
                _space_to_xb(Vi),
                _space_to_xb(Ve),
                _space_to_xb(Gm_den),
                _space_to_xb(GE_den),
                area_precomputed_xb,
                cm_over_dt_precomputed_xb,
                cx_over_dt_precomputed_xb,
                a00_static_xb,
                a11_static_xb,
                off_i_precomputed_xb,
                off_e_precomputed_xb,
                _space_to_xb(Iinj_abs),
                I_outward_abs_precomputed_xb,
                I_corr_abs_precomputed_xb,
                _space_to_xb(drive),
            ),
            layout="XB",
        ),
        *solver_cases,
        *one_step_without_solve_cases,
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

    membrane_metadata = _membrane_backend_metadata(runtime.membrane.backend, gates)
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
        "assembly_variants": ("real_double_cable", "precomputed_static", "precomputed_static_xb"),
        "resolved_solver": resolved,
        "membrane_backend": _backend_variant(runtime.membrane.backend),
        "membrane_model": type(runtime.membrane.membrane).__name__,
        **membrane_metadata,
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


def _space_to_xb(values: Any) -> Any:
    arr = jnp.asarray(values)
    return jnp.swapaxes(arr, 0, 1) if arr.ndim == 2 else arr


def _edge_to_xb(values: Any) -> Any:
    arr = jnp.asarray(values)
    return jnp.swapaxes(arr, 0, 1) if arr.ndim == 2 else arr


def _space_from_xb(values: Any) -> Any:
    arr = jnp.asarray(values)
    return jnp.swapaxes(arr, 0, 1) if arr.ndim == 2 else arr


def _edge_from_xb(values: Any) -> Any:
    arr = jnp.asarray(values)
    return jnp.swapaxes(arr, 0, 1) if arr.ndim == 2 else arr


def _assembled_from_xb(
    assembled: tuple[Any, Any, Any, Any, Any, Any, Any, Any],
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    a00, a01, a10, a11, off0, off1, rhs0, rhs1 = assembled
    return (
        _space_from_xb(a00),
        _space_from_xb(a01),
        _space_from_xb(a10),
        _space_from_xb(a11),
        _edge_from_xb(off0),
        _edge_from_xb(off1),
        _space_from_xb(rhs0),
        _space_from_xb(rhs1),
    )


def _batch_options(args: argparse.Namespace) -> BatchOptions:
    if args.recording == "observer_only":
        return BatchOptions.none(time_chunk_steps=int(args.time_chunk_steps))
    if args.recording == "full_vm":
        return BatchOptions.full(time_chunk_steps=int(args.time_chunk_steps))
    return BatchOptions.probes(
        8,
        time_chunk_steps=int(args.time_chunk_steps),
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


def _is_gated_leak_stack_backend(backend: Any) -> bool:
    return all(
        hasattr(backend, attr)
        for attr in (
            "gated_model",
            "gated_gate_count",
            "_gated_mask_col",
            "_leak_g_col",
            "_leak_ge_col",
        )
    )


@partial(jax.jit, static_argnames=("backend",))
def _real_gated_only_gate_update(
    backend: Any,
    gates: Any,
    Vm: Any,
    dt_ms: Any,
) -> Any:
    gated_gate_count = int(backend.gated_gate_count)
    return jax.vmap(
        lambda gates_row, vm_row: backend.gated_model.cn_gate_update(
            g_prev=gates_row[:, :gated_gate_count],
            V_mV=vm_row,
            dt=dt_ms,
        )
    )(gates, Vm)


@partial(jax.jit, static_argnames=("backend",))
def _real_gated_only_conductance_terms(
    backend: Any,
    gates: Any,
) -> tuple[Any, Any]:
    gated_gate_count = int(backend.gated_gate_count)
    return jax.vmap(
        lambda gates_row: backend.gated_model.membrane_conductance_terms(
            gates_row[:, :gated_gate_count]
        )
    )(gates)


@partial(jax.jit, static_argnames=("backend",))
def _real_gated_leak_conductance_mix(
    backend: Any,
    gates: Any,
    gated_gm: Any,
    gated_ge: Any,
) -> tuple[Any, Any]:
    gated_mask = gates[:, :, backend._gated_mask_col]
    leak_gm = gates[:, :, backend._leak_g_col]
    leak_ge = gates[:, :, backend._leak_ge_col]
    return (
        gated_mask * gated_gm + (1.0 - gated_mask) * leak_gm,
        gated_mask * gated_ge + (1.0 - gated_mask) * leak_ge,
    )


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


def _precompute_assembly_inputs_xb(
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
    (
        area,
        cm_over_dt,
        cx_over_dt,
        a00_static,
        a11_static,
        off_i,
        off_e,
        I_outward_abs,
        I_corr_abs,
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
        I_outward_den=I_outward_den,
        I_corr_den=I_corr_den,
        dt_ms=dt_ms,
    )
    return _block_until_ready(
        (
            _space_to_xb(area),
            _space_to_xb(cm_over_dt),
            _space_to_xb(cx_over_dt),
            _space_to_xb(a00_static),
            _space_to_xb(a11_static),
            _edge_to_xb(off_i),
            _edge_to_xb(off_e),
            _space_to_xb(I_outward_abs),
            _space_to_xb(I_corr_abs),
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


@jax.jit
def _real_assemble_system_precomputed_xb(
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


@partial(jax.jit, static_argnames=("backend",))
def _real_one_step_without_solve(
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
) -> tuple[Any, tuple[Any, Any, Any, Any, Any, Any, Any, Any]]:
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
    return gates_new, assembled


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


@partial(jax.jit, static_argnames=("backend",))
def _real_one_step_without_solve_precomputed(
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
) -> tuple[Any, tuple[Any, Any, Any, Any, Any, Any, Any, Any]]:
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
    return gates_new, assembled


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
def _solve_thomas_batched_scan(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
) -> tuple[Any, Any]:
    return solve_block_tridiagonal_2x2_scalar_batched(
        a00, a01, a10, a11, off0, off1, rhs0, rhs1
    )


@jax.jit
def _solve_jax_triton_tiled_thomas(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
) -> tuple[Any, Any]:
    return solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_batched(
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
        block_b=128,
    )


@jax.jit
def _solve_jax_triton_tiled_thomas_loop(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
) -> tuple[Any, Any]:
    return solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_loop_batched(
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
        block_b=64,
    )


def _solve_jax_triton_tiled_thomas_with_block_b(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
    *,
    block_b: int,
) -> tuple[Any, Any]:
    return solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_batched(
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
        block_b=int(block_b),
    )


def _solve_jax_triton_tiled_thomas_loop_with_block_b(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
    *,
    block_b: int,
) -> tuple[Any, Any]:
    return solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_loop_batched(
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
        block_b=int(block_b),
    )


def _solve_jax_triton_tiled_thomas_loop_xb_to_bx(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
    *,
    block_b: int,
) -> tuple[Any, Any]:
    out0_xb, out1_xb = solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_loop_xb(
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
        block_b=int(block_b),
    )
    return jnp.swapaxes(out0_xb, 0, 1), jnp.swapaxes(out1_xb, 0, 1)


def _make_block_b_solver(fn: Callable[..., Any], *, block_b: int) -> Callable[..., Any]:
    return jax.jit(partial(fn, block_b=int(block_b)))


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
    assembled_xb: tuple[Any, Any, Any, Any, Any, Any, Any, Any],
    requested: Sequence[str],
    active_solver: str,
    jax_triton_block_bs: Sequence[int],
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
        "thomas_batched_scan": _solve_thomas_batched_scan,
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
        if name == "jax_triton_tiled_thomas":
            block_bs = tuple(jax_triton_block_bs or (128,))
            for block_b in block_bs:
                out.append(
                    StageCase(
                        "block_solve",
                        f"{name}_b{int(block_b)}",
                        _make_block_b_solver(
                            _solve_jax_triton_tiled_thomas_with_block_b,
                            block_b=int(block_b),
                        ),
                        assembled,
                        layout="BX_WRAPPER",
                        block_b=int(block_b),
                    )
                )
            continue
        if name == "jax_triton_tiled_thomas_loop":
            block_bs = tuple(jax_triton_block_bs or (64,))
            for block_b in block_bs:
                out.append(
                    StageCase(
                        "block_solve",
                        f"{name}_b{int(block_b)}",
                        _make_block_b_solver(
                            _solve_jax_triton_tiled_thomas_loop_with_block_b,
                            block_b=int(block_b),
                        ),
                        assembled,
                        layout="BX_WRAPPER",
                        block_b=int(block_b),
                    )
                )
                out.append(
                    StageCase(
                        "block_solve",
                        f"{name}_xb_b{int(block_b)}",
                        _make_block_b_solver(
                            _solve_jax_triton_tiled_thomas_loop_xb_to_bx,
                            block_b=int(block_b),
                        ),
                        assembled_xb,
                        layout="XB_DIRECT",
                        block_b=int(block_b),
                        validation_args=assembled,
                    )
                )
            continue
        out.append(StageCase("block_solve", name, mapping[name], assembled))
    return tuple(out)


def _jax_triton_block_bs(args: argparse.Namespace) -> tuple[int, ...]:
    values = tuple(int(value) for value in (args.jax_triton_block_b or ()))
    return tuple(dict.fromkeys(values))


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


def _expand_jax_triton_one_step_solvers(
    names: Sequence[str],
    *,
    jax_triton_block_bs: Sequence[int],
) -> tuple[str, ...]:
    expanded: list[str] = []
    for name in names:
        if name == "jax_triton_tiled_thomas":
            expanded.extend(f"{name}_b{int(block_b)}" for block_b in (jax_triton_block_bs or (128,)))
            continue
        if name == "jax_triton_tiled_thomas_loop":
            expanded.extend(f"{name}_b{int(block_b)}" for block_b in (jax_triton_block_bs or (64,)))
            continue
        expanded.append(name)
    return tuple(dict.fromkeys(expanded))


def _split_jax_triton_solver_block_b(solver: str) -> tuple[str, int | None]:
    prefix = "jax_triton_tiled_thomas_loop_b"
    if solver.startswith(prefix):
        return "jax_triton_tiled_thomas_loop", int(solver[len(prefix) :])
    prefix = "jax_triton_tiled_thomas_b"
    if solver.startswith(prefix):
        return "jax_triton_tiled_thomas", int(solver[len(prefix) :])
    return solver, None


def _solver_layout(solver: str) -> str | None:
    base, _ = _split_jax_triton_solver_block_b(solver)
    if base in {"jax_triton_tiled_thomas", "jax_triton_tiled_thomas_loop"}:
        return "BX_WRAPPER"
    return None


def _solver_block_b(solver: str) -> int | None:
    _, block_b = _split_jax_triton_solver_block_b(solver)
    return block_b


def _solve_by_name(
    solver: str,
    assembled: tuple[Any, Any, Any, Any, Any, Any, Any, Any],
) -> tuple[Any, Any]:
    solver, block_b = _split_jax_triton_solver_block_b(solver)
    if solver == "thomas":
        return _solve_thomas_vmap(*assembled)
    if solver == "thomas_batched_scan":
        return _solve_thomas_batched_scan(*assembled)
    if solver == "jax_triton_tiled_thomas":
        if block_b is not None:
            return _solve_jax_triton_tiled_thomas_with_block_b(*assembled, block_b=block_b)
        return _solve_jax_triton_tiled_thomas(*assembled)
    if solver == "jax_triton_tiled_thomas_loop":
        if block_b is not None:
            return _solve_jax_triton_tiled_thomas_loop_with_block_b(*assembled, block_b=block_b)
        return _solve_jax_triton_tiled_thomas_loop(*assembled)
    if solver == "pcr":
        return _solve_pcr_matrix_vmap(*assembled)
    if solver in {"pcr_soa", "pcr_adaptive", "pcr_soa_batched"}:
        return _solve_pcr_soa_batched(*assembled)
    if solver == "pcr_soa_shift_batched":
        return _solve_pcr_soa_shift_batched(*assembled)
    if solver == "pcr_soa_padded_batched":
        return _solve_pcr_soa_padded_batched(*assembled)
    raise ValueError(f"unsupported one-step solver: {solver!r}")


def _validate_solver_cases(
    stage_cases: Sequence[StageCase],
    *,
    args: argparse.Namespace,
    device_name: str,
    group_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    block_cases = [case for case in stage_cases if case.stage == "block_solve"]
    one_step_cases = [case for case in stage_cases if case.stage == "one_step_proxy"]
    if block_cases:
        rows.extend(
            _validate_block_solve_cases(
                block_cases,
                args=args,
                device_name=device_name,
                group_metadata=group_metadata,
            )
        )
    if one_step_cases:
        rows.extend(
            _validate_one_step_cases(
                one_step_cases,
                args=args,
                device_name=device_name,
                group_metadata=group_metadata,
            )
        )
    if not rows:
        raise RuntimeError("--validate-solvers found no block_solve or one_step_proxy cases.")
    return rows


def _validate_block_solve_cases(
    cases: Sequence[StageCase],
    *,
    args: argparse.Namespace,
    device_name: str,
    group_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    reference = _select_validation_reference_case(
        cases,
        preferred=(
            "thomas_batched_scan",
            "pcr_soa_batched",
            "pcr_matrix_vmap",
            "thomas_vmap",
        ),
        context="block_solve",
    )
    reference_out = _block_until_ready(reference.fn(*reference.args))
    rows: list[dict[str, Any]] = []
    for case in cases:
        out = _block_until_ready(case.fn(*case.args))
        stats = _comparison_stats(out, reference_out)
        residual_args = case.validation_args if case.validation_args is not None else case.args
        max_residual, median_residual = _residual_stats(residual_args, out)
        rows.append(
            _validation_row(
                case=case,
                reference_variant=reference.variant,
                args=args,
                device_name=device_name,
                group_metadata=group_metadata,
                stats=stats,
                max_residual_norm=max_residual,
                median_residual_norm=median_residual,
            )
        )
    return rows


def _validate_one_step_cases(
    cases: Sequence[StageCase],
    *,
    args: argparse.Namespace,
    device_name: str,
    group_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: dict[str, list[StageCase]] = {}
    for case in cases:
        groups.setdefault(_one_step_validation_group(case.variant), []).append(case)
    for group_name in sorted(groups):
        group_cases = groups[group_name]
        reference = _select_validation_reference_case(
            group_cases,
            preferred=(
                f"thomas_batched_scan_{group_name}",
                f"pcr_soa_batched_{group_name}",
                f"pcr_soa_{group_name}",
                f"pcr_matrix_vmap_{group_name}",
                f"thomas_vmap_{group_name}",
            ),
            context=f"one_step_proxy/{group_name}",
        )
        reference_out = _block_until_ready(reference.fn(*reference.args))
        for case in group_cases:
            out = _block_until_ready(case.fn(*case.args))
            stats = _comparison_stats(out, reference_out)
            rows.append(
                _validation_row(
                    case=case,
                    reference_variant=reference.variant,
                    args=args,
                    device_name=device_name,
                    group_metadata=group_metadata,
                    stats=stats,
                    max_residual_norm=None,
                    median_residual_norm=None,
                )
            )
    return rows


def _select_validation_reference_case(
    cases: Sequence[StageCase],
    *,
    preferred: Sequence[str],
    context: str,
) -> StageCase:
    by_variant = {case.variant: case for case in cases}
    for name in preferred:
        case = by_variant.get(name)
        if case is not None:
            return case
    for case in cases:
        if "jax_triton" not in case.variant:
            return case
    available = ", ".join(case.variant for case in cases)
    raise RuntimeError(
        f"--validate-solvers needs a non-Triton reference for {context}; "
        f"available variants: {available}"
    )


def _one_step_validation_group(variant: str) -> str:
    if variant.endswith("_real_precomputed_static"):
        return "real_precomputed_static"
    if variant.endswith("_real"):
        return "real"
    return "unknown"


def _validation_row(
    *,
    case: StageCase,
    reference_variant: str,
    args: argparse.Namespace,
    device_name: str,
    group_metadata: dict[str, Any],
    stats: dict[str, Any],
    max_residual_norm: float | None,
    median_residual_norm: float | None,
) -> dict[str, Any]:
    atol = _validation_atol(args)
    rtol = _validation_rtol(args)
    residual_tolerance = _validation_residual_tolerance(args)
    passes_tolerance = _comparison_passes(stats, atol=atol, rtol=rtol)
    residual_ok = (
        max_residual_norm is None
        or (
            np.isfinite(max_residual_norm)
            and max_residual_norm <= residual_tolerance
        )
    )
    all_finite = bool(stats["all_finite"]) and (
        max_residual_norm is None or np.isfinite(max_residual_norm)
    )
    status = "pass" if all_finite and passes_tolerance and residual_ok else "fail"
    notes: list[str] = []
    if not all_finite:
        notes.append("non-finite output or residual")
    if not passes_tolerance:
        notes.append("output differs from reference beyond tolerance")
    if not residual_ok:
        notes.append("block residual exceeds tolerance")
    return {
        **_row_context(args=args, device_name=device_name, group_metadata=group_metadata),
        "stage": case.stage,
        "variant": case.variant,
        "reference_variant": reference_variant,
        "layout": case.layout,
        "block_b": case.block_b,
        "stage_group": _stage_group(case.stage),
        "max_abs_diff": stats["max_abs_diff"],
        "max_rel_diff": stats["max_rel_diff"],
        "reference_max_abs": stats["reference_max_abs"],
        "max_abs_vm_diff": stats["max_abs_vm_diff"],
        "max_rel_vm_diff": stats["max_rel_vm_diff"],
        "max_residual_norm": max_residual_norm,
        "median_residual_norm": median_residual_norm,
        "all_finite": all_finite,
        "passes_tolerance": passes_tolerance,
        "status": status,
        "atol": atol,
        "rtol": rtol,
        "residual_tolerance": residual_tolerance,
        "notes": "; ".join(notes),
    }


def _comparison_stats(actual: Any, reference: Any) -> dict[str, Any]:
    actual_arrays = list(_numeric_leaves(actual))
    reference_arrays = list(_numeric_leaves(reference))
    if len(actual_arrays) != len(reference_arrays):
        raise RuntimeError(
            "validation output structure differs: "
            f"{len(actual_arrays)} arrays versus {len(reference_arrays)}"
        )
    max_abs = 0.0
    max_rel = 0.0
    reference_max_abs = 0.0
    all_finite = True
    for actual_arr, reference_arr in zip(actual_arrays, reference_arrays, strict=True):
        if actual_arr.shape != reference_arr.shape:
            raise RuntimeError(
                "validation output shape differs: "
                f"{actual_arr.shape} versus {reference_arr.shape}"
            )
        diff = np.abs(actual_arr - reference_arr)
        denom = np.maximum(np.abs(reference_arr), 1e-12)
        max_abs = max(max_abs, _nanmax0(diff))
        max_rel = max(max_rel, _nanmax0(diff / denom))
        reference_max_abs = max(reference_max_abs, _nanmax0(np.abs(reference_arr)))
        all_finite = all_finite and bool(np.all(np.isfinite(actual_arr)))
        all_finite = all_finite and bool(np.all(np.isfinite(reference_arr)))
    vm_abs, vm_rel = _vm_comparison_stats(actual_arrays, reference_arrays)
    return {
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "reference_max_abs": reference_max_abs,
        "max_abs_vm_diff": vm_abs,
        "max_rel_vm_diff": vm_rel,
        "all_finite": all_finite,
    }


def _numeric_leaves(value: Any) -> list[np.ndarray]:
    if isinstance(value, (tuple, list)):
        leaves: list[np.ndarray] = []
        for item in value:
            leaves.extend(_numeric_leaves(item))
        return leaves
    return [np.asarray(value)]


def _vm_comparison_stats(
    actual_arrays: Sequence[np.ndarray],
    reference_arrays: Sequence[np.ndarray],
) -> tuple[float | None, float | None]:
    if len(actual_arrays) < 2 or len(reference_arrays) < 2:
        return None, None
    if actual_arrays[0].shape != actual_arrays[1].shape:
        return None, None
    if reference_arrays[0].shape != reference_arrays[1].shape:
        return None, None
    actual_vm = actual_arrays[0] - actual_arrays[1]
    reference_vm = reference_arrays[0] - reference_arrays[1]
    diff = np.abs(actual_vm - reference_vm)
    denom = np.maximum(np.abs(reference_vm), 1e-12)
    return _nanmax0(diff), _nanmax0(diff / denom)


def _comparison_passes(stats: dict[str, Any], *, atol: float, rtol: float) -> bool:
    threshold = atol + rtol * float(stats["reference_max_abs"])
    return bool(stats["max_abs_diff"] <= threshold)


def _nanmax0(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.nanmax(values))


def _residual_stats(
    inputs: tuple[Any, Any, Any, Any, Any, Any, Any, Any],
    output: tuple[Any, Any],
) -> tuple[float, float]:
    residuals = _block_until_ready(double_cable_block_residual_norm(*inputs, *output))
    return float(jnp.max(residuals)), float(jnp.median(residuals))


def _validation_atol(args: argparse.Namespace) -> float:
    if args.validation_atol is not None:
        return float(args.validation_atol)
    return 1e-8 if _validation_precision(args) == "fp64" else 1e-3


def _validation_rtol(args: argparse.Namespace) -> float:
    if args.validation_rtol is not None:
        return float(args.validation_rtol)
    return 1e-8 if _validation_precision(args) == "fp64" else 2e-4


def _validation_residual_tolerance(args: argparse.Namespace) -> float:
    if args.validation_residual_tolerance is not None:
        return float(args.validation_residual_tolerance)
    return 1e-8 if _validation_precision(args) == "fp64" else 1e-3


def _validation_precision(args: argparse.Namespace) -> str:
    return str(args.precision or PRESETS[args.preset].precision)


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
        "layout": case.layout,
        "block_b": case.block_b,
        "stage_group": _stage_group(case.stage),
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
        "layout": case.layout,
        "block_b": case.block_b,
        "stage_group": _stage_group(case.stage),
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
        "membrane_backend": group_metadata.get("membrane_backend"),
        "membrane_model": group_metadata.get("membrane_model"),
        "membrane_gates_max": group_metadata.get("membrane_gates_max"),
        "membrane_channels_max": group_metadata.get("membrane_channels_max"),
        "membrane_backend_branches": group_metadata.get("membrane_backend_branches"),
        "membrane_gated_compartments": group_metadata.get("membrane_gated_compartments"),
        "membrane_leak_compartments": group_metadata.get("membrane_leak_compartments"),
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


def _membrane_backend_metadata(backend: Any, gates: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "membrane_gates_max": int(getattr(backend, "n_gates_max", 0)),
        "membrane_channels_max": int(getattr(backend, "n_channels_max", 0)),
        "membrane_backend_branches": None,
        "membrane_gated_compartments": None,
        "membrane_leak_compartments": None,
        "membrane_gated_gate_count": getattr(backend, "gated_gate_count", None),
        "membrane_gated_channel_count": getattr(backend, "gated_channel_count", None),
    }
    rows = getattr(backend, "rows", None)
    if rows is not None:
        metadata["membrane_backend_branches"] = len(rows)
    elif _is_gated_leak_stack_backend(backend):
        metadata["membrane_backend_branches"] = 1
    groups = getattr(backend, "groups", None)
    if groups is not None:
        metadata["membrane_backend_branches"] = len(groups)
    if _is_gated_leak_stack_backend(backend):
        mask_col = int(backend._gated_mask_col)
        mask = np.asarray(gates[0, :, mask_col])
        gated_count = int(np.count_nonzero(mask > 0.5))
        metadata["membrane_gated_compartments"] = gated_count
        metadata["membrane_leak_compartments"] = int(mask.shape[0]) - gated_count
        metadata["membrane_gated_gate_count"] = int(backend.gated_gate_count)
        metadata["membrane_gated_channel_count"] = int(backend.gated_channel_count)
    return metadata


def _stage_group(stage: str) -> str:
    if stage.startswith("membrane_"):
        return "membrane"
    if stage.startswith("extracellular_"):
        return "forcing"
    if stage == "system_assembly":
        return "assembly"
    if stage == "block_solve":
        return "solver"
    if stage == "observer_write":
        return "observer"
    if stage == "one_step_without_solve":
        return "one_step_without_solve"
    if stage == "one_step_proxy":
        return "one_step"
    return "other"


def _write_report(
    path: Path,
    rows: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    validation_rows: Sequence[dict[str, Any]] = (),
) -> None:
    stage_rows = sorted(rows, key=lambda item: (item["stage"], item["variant"]))
    block_rows = [row for row in rows if row["stage"] == "block_solve"]
    fastest_block = min(block_rows, key=lambda row: float(row["mean_ms"])) if block_rows else None
    primary_solver = _primary_block_solve(rows, metadata)
    if primary_solver is None:
        primary_solver = fastest_block
    one_step = _primary_one_step(rows, metadata)
    no_solve = _primary_one_step_without_solve(rows)
    hot_rows = sorted(rows, key=lambda item: float(item["mean_ms"]), reverse=True)
    group_sums = _stage_group_sums(rows, metadata)
    membrane_ratios = _membrane_ratio_notes(rows)
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
        f"- Membrane model: `{metadata['group']['membrane_model']}`",
        f"- Membrane gates/channels max: `{metadata['group'].get('membrane_gates_max')}` / `{metadata['group'].get('membrane_channels_max')}`",
        f"- Membrane branches: `{metadata['group'].get('membrane_backend_branches')}`",
        f"- MRG gated/leak compartments: `{metadata['group'].get('membrane_gated_compartments')}` / `{metadata['group'].get('membrane_leak_compartments')}`",
        "",
    ]
    if one_step is not None:
        one_step_ms = float(one_step["mean_ms"])
        lines.extend(
            [
                "## Hot-Step Decomposition",
                "",
                (
                    "Primary one-step proxy: "
                    f"`{one_step['variant']}` at {one_step_ms:.3f} ms mean."
                ),
                "",
                "| group | representative mean ms | share of primary one-step |",
                "| --- | ---: | ---: |",
            ]
        )
        for group, value in group_sums:
            if group == "one_step":
                continue
            share = 100.0 * value / one_step_ms if one_step_ms else 0.0
            lines.append(f"| {group} | {value:.3f} | {share:.1f}% |")
        lines.extend(
            [
                "",
                "### Hottest Individual Stages",
                "",
                "| rank | stage | variant | mean ms | share of primary one-step |",
                "| ---: | --- | --- | ---: | ---: |",
            ]
        )
        for rank, row in enumerate(hot_rows[:8], start=1):
            share = 100.0 * float(row["mean_ms"]) / one_step_ms if one_step_ms else 0.0
            lines.append(
                f"| {rank} | {row['stage']} | {row['variant']} | "
                f"{float(row['mean_ms']):.3f} | {share:.1f}% |"
            )
        lines.append("")
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
    if primary_solver is not None and one_step is not None:
        solver_share = 100.0 * float(primary_solver["mean_ms"]) / float(one_step["mean_ms"])
        lines.extend(
            [
                "## Solver Share",
                "",
                (
                    f"Primary block solve `{primary_solver['variant']}` is "
                    f"{float(primary_solver['mean_ms']):.3f} ms mean, "
                    f"or {solver_share:.1f}% of `{one_step['variant']}`."
                ),
                "",
            ]
        )
    if no_solve is not None and one_step is not None:
        no_solve_share = 100.0 * float(no_solve["mean_ms"]) / float(one_step["mean_ms"])
        lines.extend(
            [
                "## One-Step Without Solve Proxy",
                "",
                (
                    f"`{no_solve['variant']}` is {float(no_solve['mean_ms']):.3f} ms mean, "
                    f"or {no_solve_share:.1f}% of `{one_step['variant']}`."
                ),
                "",
                (
                    "This benchmark-only proxy fuses gate update, conductance terms, "
                    "extracellular RHS, and system assembly, then materializes the "
                    "assembled system plus updated gates to prevent dead-code elimination."
                ),
                "",
            ]
        )
    if membrane_ratios:
        lines.extend(["## MRG Membrane Compiler Signals", ""])
        lines.extend(f"- {note}" for note in membrane_ratios)
        lines.append("")
    if validation_rows:
        failed = [row for row in validation_rows if row["status"] != "pass"]
        lines.extend(
            [
                "## Numerical Solver Validation",
                "",
                (
                    "Status: "
                    f"`{'pass' if not failed else 'fail'}` "
                    f"({len(validation_rows) - len(failed)}/{len(validation_rows)} rows passed)."
                ),
                "",
                "| stage | variant | layout | block_b | reference | max abs diff | max abs Vm diff | max residual | status |",
                "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in validation_rows:
            lines.append(
                "| {stage} | {variant} | {layout} | {block_b} | {reference_variant} | {max_abs:.3e} | {vm_abs} | {residual} | {status} |".format(
                    stage=row["stage"],
                    variant=row["variant"],
                    layout=row.get("layout") or "",
                    block_b=row.get("block_b") or "",
                    reference_variant=row["reference_variant"],
                    max_abs=float(row["max_abs_diff"]),
                    vm_abs=_format_optional_scientific(row["max_abs_vm_diff"]),
                    residual=_format_optional_scientific(row["max_residual_norm"]),
                    status=row["status"],
                )
            )
        lines.extend(
            [
                "",
                (
                    "Validation runs after timing measurements, so it does not hide "
                    "first-run/cold compilation cost."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Stage Means",
            "",
            "| stage | variant | layout | block_b | mean ms | first run ms | max ms | output KiB |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stage_rows:
        lines.append(
            "| {stage} | {variant} | {layout} | {block_b} | {mean_ms:.3f} | {first_run_ms:.3f} | {max_ms:.3f} | {output_kib:.1f} |".format(
                stage=row["stage"],
                variant=row["variant"],
                layout=row.get("layout") or "",
                block_b=row.get("block_b") or "",
                mean_ms=float(row["mean_ms"]),
                first_run_ms=float(row["first_run_ms"]),
                max_ms=float(row["max_ms"]),
                output_kib=float(row["output_bytes"]) / 1024.0,
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `membrane_gate_update` and `membrane_conductance_terms` use the real prepared membrane backend.",
            "- `*_gated_only` and `*_mask_mix` rows appear for MRG gated/leak-stack backends; they separate generated gated-model work from leak/mask blending.",
            "- `system_assembly` uses real prepared double-cable coefficients and real extracellular forcing for one step.",
            "- `system_assembly/precomputed_static` precomposes static diagonal terms and absolute currents as a benchmark-only diagnostic.",
            "- `block_solve` runs selected solver functions on the real assembled system.",
            "- `one_step_without_solve` fuses membrane and assembly work without the block solve; it materializes assembled coefficients/RHS, so use it as a non-solve upper-bound proxy rather than subtracting it from `one_step_proxy`.",
            "- `one_step_proxy` fuses gate update, conductance terms, assembly, and the selected benchmark one-step block solver for one step; `_precomputed_static` variants use the same precomposed assembly inputs.",
            "- `observer_write` is present only for observer-only VmRaster runs.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_optional_scientific(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.3e}"


def _row_for(rows: Sequence[dict[str, Any]], stage: str, variant: str) -> dict[str, Any] | None:
    for row in rows:
        if row["stage"] == stage and row["variant"] == variant:
            return row
    return None


def _primary_one_step(rows: Sequence[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any] | None:
    active_solver = metadata["group"]["active_solver"]
    preferred = f"{active_solver}_real"
    row = _row_for(rows, "one_step_proxy", preferred)
    if row is not None:
        return row
    all_one_step = [item for item in rows if item["stage"] == "one_step_proxy"]
    return min(all_one_step, key=lambda item: float(item["mean_ms"])) if all_one_step else None


def _primary_one_step_without_solve(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    row = _row_for(rows, "one_step_without_solve", "real_materialized")
    if row is not None:
        return row
    all_no_solve = [item for item in rows if item["stage"] == "one_step_without_solve"]
    return min(all_no_solve, key=lambda item: float(item["mean_ms"])) if all_no_solve else None


def _stage_group_sums(
    rows: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
) -> list[tuple[str, float]]:
    primary_solver = _primary_block_solve(rows, metadata)
    selected = {
        "membrane": (
            _sum_stage(rows, "membrane_gate_update")
            + _sum_stage(rows, "membrane_conductance_terms")
        ),
        "forcing": _sum_stage(rows, "extracellular_rhs_drive"),
        "assembly": _sum_stage(rows, "system_assembly", variant="real_double_cable"),
        "solver": 0.0 if primary_solver is None else float(primary_solver["mean_ms"]),
        "observer": _sum_stage(rows, "observer_write"),
    }
    return sorted(
        ((group, value) for group, value in selected.items() if value > 0.0),
        key=lambda item: item[1],
        reverse=True,
    )


def _sum_stage(rows: Sequence[dict[str, Any]], stage: str, *, variant: str | None = None) -> float:
    values = [
        float(row["mean_ms"])
        for row in rows
        if row["stage"] == stage and (variant is None or row["variant"] == variant)
    ]
    return sum(values)


def _primary_block_solve(
    rows: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    active_solver = str(metadata["group"]["active_solver"])
    candidate_names = [active_solver]
    if active_solver in {"pcr_soa", "pcr_adaptive"}:
        candidate_names.append("pcr_soa_batched")
    if active_solver == "pcr":
        candidate_names.append("pcr_matrix_vmap")
    if active_solver == "thomas":
        candidate_names.append("thomas_vmap")
    for name in candidate_names:
        row = _row_for(rows, "block_solve", name)
        if row is not None:
            return row
    block_rows = [row for row in rows if row["stage"] == "block_solve"]
    return min(block_rows, key=lambda row: float(row["mean_ms"])) if block_rows else None


def _membrane_ratio_notes(rows: Sequence[dict[str, Any]]) -> list[str]:
    notes: list[str] = []
    gate = _first_stage(rows, "membrane_gate_update")
    gated_gate = _first_stage(rows, "membrane_gate_update_gated_only")
    if gate is not None and gated_gate is not None:
        ratio = _safe_ratio(float(gate["mean_ms"]), float(gated_gate["mean_ms"]))
        notes.append(
            "Gate update full MRG stack is "
            f"{float(gate['mean_ms']):.3f} ms versus "
            f"{float(gated_gate['mean_ms']):.3f} ms for generated gated-model work "
            f"({ratio:.2f}x)."
        )
    conductance = _first_stage(rows, "membrane_conductance_terms")
    gated_conductance = _first_stage(rows, "membrane_conductance_terms_gated_only")
    mix = _first_stage(rows, "membrane_conductance_terms_mask_mix")
    if conductance is not None and gated_conductance is not None:
        ratio = _safe_ratio(float(conductance["mean_ms"]), float(gated_conductance["mean_ms"]))
        notes.append(
            "Conductance full MRG stack is "
            f"{float(conductance['mean_ms']):.3f} ms versus "
            f"{float(gated_conductance['mean_ms']):.3f} ms for generated gated-model work "
            f"({ratio:.2f}x)."
        )
    if conductance is not None and mix is not None:
        conductance_ms = float(conductance["mean_ms"])
        share = 100.0 * float(mix["mean_ms"]) / conductance_ms if conductance_ms else 0.0
        notes.append(
            "The isolated leak/mask conductance blend is "
            f"{float(mix['mean_ms']):.3f} ms, about {share:.1f}% of full conductance terms."
        )
        notes.append(
            "The gated-only and mask-mix rows are separate JIT kernels for diagnosis; "
            "they are not additive timings for the fused full conductance stage."
        )
    return notes


def _first_stage(rows: Sequence[dict[str, Any]], stage: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if row["stage"] == stage]
    return candidates[0] if candidates else None


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("inf")


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
