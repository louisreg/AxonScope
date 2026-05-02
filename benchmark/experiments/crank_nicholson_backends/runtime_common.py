from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from axonscope.axons.unmyelinated import HodgkinHuxley
from axonscope.simresult import SimResult
from axonscope.solvers.CrankNicholson import (
    CrankNicholson,
    CrankNicholsonImplicit,
    CrankNicholsonImplicitFast,
    CrankNicholsonImplicitFastMultiStep,
    CrankNicholsonQuasiNewtonFast,
    CrankNicholsonSemiImplicit,
    CrankNicholson_unoptimized,
)

import settings as s
import utils as u


ArrayPairRunner = Callable[[], tuple[np.ndarray, np.ndarray]]
RunnerBuilder = Callable[["RuntimeProblem"], ArrayPairRunner]


@dataclass(frozen=True)
class RuntimeProblem:
    axon: HodgkinHuxley
    Nx: int
    Nt: int
    tsim: float
    dt: float
    q10: float
    g_bar: np.ndarray
    e_rev: np.ndarray
    idx_inj: int
    t_start_inj: float
    t_stop_inj: float
    inj_uA_per_cm2: float


@dataclass(frozen=True)
class TraceReference:
    t: np.ndarray
    Vm: np.ndarray
    x: np.ndarray


def make_problem(Nx: int) -> RuntimeProblem:
    axon = HodgkinHuxley(
        L=s.L,
        d=s.d,
        Nx=Nx,
        celsius=6.3,
        Vinit=s.Vinit,
        include_passive_leak=True,
        g_pas=0.001,
        e_pas=-70.0,
    )
    axon.insert_I_Clamp(
        position=s.position,
        t_start=s.t_start,
        duration=s.duration,
        amplitude=s.amplitude,
    )
    Nt = int(np.ceil(s.tsim / s.dt))
    return RuntimeProblem(
        axon=axon,
        Nx=Nx,
        Nt=Nt,
        tsim=s.tsim,
        dt=s.dt,
        q10=float(axon.ion_channel.q10),
        g_bar=np.asarray(axon.ion_channel.g_bar, dtype=np.float64),
        e_rev=np.asarray(axon.ion_channel.E_rev, dtype=np.float64),
        idx_inj=int(axon.idx_inj),
        t_start_inj=float(axon.t_start_inj),
        t_stop_inj=float(axon.t_stop_inj),
        inj_uA_per_cm2=float(np.asarray(axon.inj_uA_per_cm2)[int(axon.idx_inj)]),
    )


def _to_numpy(x):
    if hasattr(x, "block_until_ready"):
        x = x.block_until_ready()
    if hasattr(x, "detach"):
        x = x.detach().cpu()
        if hasattr(x, "tolist"):
            x = x.tolist()
    return np.asarray(x)


def normalize_output(result: tuple[object, object]) -> tuple[np.ndarray, np.ndarray]:
    t_vec, Vm = result
    return _to_numpy(t_vec), _to_numpy(Vm)


def make_simresult(problem: RuntimeProblem, result: tuple[object, object]) -> SimResult:
    t_vec, Vm = normalize_output(result)
    return SimResult(problem.axon, Vm, t_vec)


def reference_result(problem: RuntimeProblem) -> SimResult:
    return CrankNicholson().solve(problem.axon, tsim=problem.tsim, dt=problem.dt)


def nrv_cache_path(Nx: int) -> Path:
    return Path(__file__).resolve().parent / f"nrv_reference_Nx{Nx}.npz"


def save_reference_cache(path: Path, result: SimResult) -> None:
    np.savez(
        path,
        t=np.asarray(result.t),
        Vm=np.asarray(result.Vm),
        x=np.asarray(result.axon.x),
    )


def load_nrv_reference(problem: RuntimeProblem) -> TraceReference | None:
    path = nrv_cache_path(problem.Nx)
    if not path.exists():
        return None
    data = np.load(path)
    t = np.asarray(data["t"])
    Vm = np.asarray(data["Vm"])
    x = np.asarray(data["x"])

    if Vm.ndim != 2:
        return None
    if Vm.shape[0] == t.shape[0] and Vm.shape[1] == x.shape[0]:
        Vm_norm = Vm
    elif Vm.shape[1] == t.shape[0] and Vm.shape[0] == x.shape[0]:
        Vm_norm = Vm.T
    else:
        return None
    return TraceReference(t=t, Vm=Vm_norm, x=x)


def estimate_velocity(Vm: np.ndarray, t: np.ndarray, x: np.ndarray, threshold: float = -10.0, min_distance: float = 1.0) -> float:
    dt = float(t[1] - t[0])
    min_distance_pts = int(min_distance / dt)
    tAP = []
    xAP = []
    for j in range(Vm.shape[1]):
        peaks, _ = find_peaks(Vm[:, j], height=threshold, distance=min_distance_pts)
        tAP.extend(t[peaks])
        xAP.extend([x[j]] * len(peaks))
    if not tAP:
        return 0.0
    t_flat = np.array(tAP) / 1e3
    x_flat = np.array(xAP) / 1e6
    sort_idx = np.argsort(t_flat)
    t_flat = t_flat[sort_idx]
    x_flat = x_flat[sort_idx]
    x0 = x_flat[0]
    velocities = []
    mask_forward = x_flat >= x0
    if np.sum(mask_forward) >= 2:
        coeff = np.polyfit(t_flat[mask_forward], x_flat[mask_forward], 1)
        velocities.append(coeff[0])
    mask_backward = x_flat <= x0
    if np.sum(mask_backward) >= 2:
        t_sel = t_flat[mask_backward]
        x_sel = x_flat[mask_backward][::-1]
        coeff = np.polyfit(t_sel, x_sel, 1)
        velocities.append(coeff[0])
    return float(np.mean(velocities)) if velocities else 0.0


def probe_trace_errors(
    Vm: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    ref_Vm: np.ndarray,
    ref_t: np.ndarray,
    ref_x: np.ndarray,
) -> tuple[float, float]:
    positions = np.array([s.L / 4.0, s.L / 3.0, s.L / 2.0, 2.0 * s.L / 3.0, 3.0 * s.L / 4.0])
    errs = []
    for xp in positions:
        i = int(np.argmin(np.abs(x - xp)))
        j = int(np.argmin(np.abs(ref_x - xp)))
        ref_trace = np.interp(t, ref_t, ref_Vm[:, j])
        errs.append(Vm[:, i] - ref_trace)
    err = np.concatenate(errs)
    return float(np.max(np.abs(err))), float(np.sqrt(np.mean(err ** 2)))


def validate_against_reference(
    problem: RuntimeProblem,
    result: tuple[object, object],
    ref: SimResult,
    nrv_ref: TraceReference | None = None,
) -> tuple[SimResult, dict[str, float]]:
    sim = make_simresult(problem, result)
    Vm = np.asarray(sim.Vm)
    ref_Vm = np.asarray(ref.Vm)
    diff = Vm - ref_Vm

    vel = float(sim.average_velocity())
    ref_vel = float(ref.average_velocity())
    metrics = {
        "velocity_ms": vel,
        "axonscope_velocity_ms": ref_vel,
        "axonscope_velocity_abs_err_ms": abs(vel - ref_vel),
        "axonscope_max_abs_err_mV": float(np.max(np.abs(diff))),
        "axonscope_rms_err_mV": float(np.sqrt(np.mean(diff ** 2))),
        "has_nan": bool(np.isnan(Vm).any()),
        "reference": "AxonScope",
        "reference_label": "AxonScope",
    }
    if nrv_ref is not None:
        nrv_Vm = np.asarray(nrv_ref.Vm)
        nrv_vel = estimate_velocity(nrv_Vm, np.asarray(nrv_ref.t), np.asarray(nrv_ref.x))
        nrv_max, nrv_rms = probe_trace_errors(
            Vm,
            np.asarray(sim.t),
            np.asarray(sim.axon.x),
            nrv_Vm,
            np.asarray(nrv_ref.t),
            np.asarray(nrv_ref.x),
        )
        metrics.update(
            {
                "nrv_velocity_ms": nrv_vel,
                "nrv_velocity_abs_err_ms": abs(vel - nrv_vel),
                "nrv_max_abs_err_mV": nrv_max,
                "nrv_rms_err_mV": nrv_rms,
                "reference": "NRV",
                "reference_label": "NRV",
                "ref_velocity_ms": nrv_vel,
                "velocity_abs_err_ms": abs(vel - nrv_vel),
                "max_abs_err_mV": nrv_max,
                "rms_err_mV": nrv_rms,
            }
        )
    else:
        metrics.update(
            {
                "ref_velocity_ms": ref_vel,
                "velocity_abs_err_ms": abs(vel - ref_vel),
                "max_abs_err_mV": float(np.max(np.abs(diff))),
                "rms_err_mV": float(np.sqrt(np.mean(diff ** 2))),
            }
        )
    return sim, metrics


def benchmark_runner(run_once: ArrayPairRunner) -> tuple[tuple[np.ndarray, np.ndarray], float]:
    warmup = normalize_output(run_once())
    _ = warmup
    t0 = time.perf_counter()
    result = normalize_output(run_once())
    elapsed = time.perf_counter() - t0
    return result, elapsed


def save_validation(label: str, rows: list[dict[str, float]]) -> Path:
    out = Path(__file__).resolve().parent / f"validation_{label}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def save_plot(label: str, sim: SimResult, ref: SimResult, nrv_ref: TraceReference | None = None) -> Path:
    fig_dir = Path(__file__).resolve().parent / "figures"
    fig_dir.mkdir(exist_ok=True)
    out = fig_dir / f"{label}_coherence.png"

    x_positions = [s.L / 4.0, s.L / 3.0, s.L / 2.0, 2.0 * s.L / 3.0, 3.0 * s.L / 4.0]
    idx = [int(np.argmin(np.abs(np.asarray(sim.axon.x) - xp))) for xp in x_positions]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax_trace, ax_err, ax_raster = axes

    use_nrv_as_ref = nrv_ref is not None
    ref_t = np.asarray(nrv_ref.t) if use_nrv_as_ref else np.asarray(ref.t)
    ref_Vm = np.asarray(nrv_ref.Vm) if use_nrv_as_ref else np.asarray(ref.Vm)
    ref_x = np.asarray(nrv_ref.x) if use_nrv_as_ref else np.asarray(ref.axon.x)
    ref_name = "NRV" if use_nrv_as_ref else "AxonScope"

    for i, xp in zip(idx, x_positions):
        i_ref = int(np.argmin(np.abs(ref_x - xp)))
        ax_trace.plot(ref_t, ref_Vm[:, i_ref], lw=2.0, label=f"{ref_name} x={xp:.0f}um")
        ax_trace.plot(ref.t, ref.Vm[:, i], lw=1.1, ls=":", label=f"AxonScope x={xp:.0f}um")
        if nrv_ref is not None:
            i_nrv = int(np.argmin(np.abs(np.asarray(nrv_ref.x) - xp)))
            ax_trace.plot(nrv_ref.t, nrv_ref.Vm[:, i_nrv], lw=1.1, ls="-.", label=f"NRV x={xp:.0f}um")
        ax_trace.plot(sim.t, sim.Vm[:, i], ls="--", lw=1.4, label=f"test x={xp:.0f}um")
    ax_trace.set_title(label)
    ax_trace.set_xlabel("Time (ms)")
    ax_trace.set_ylabel("Vm (mV)")
    ax_trace.grid(True, alpha=0.3)
    ax_trace.legend(fontsize=7, ncol=2)

    diff_max_t = []
    for ti, t_val in enumerate(np.asarray(sim.t)):
        ref_idx = int(np.argmin(np.abs(ref_t - t_val)))
        row_err = []
        for xp in x_positions:
            i_sim = int(np.argmin(np.abs(np.asarray(sim.axon.x) - xp)))
            i_ref = int(np.argmin(np.abs(ref_x - xp)))
            row_err.append(abs(float(sim.Vm[ti, i_sim]) - float(ref_Vm[ref_idx, i_ref])))
        diff_max_t.append(max(row_err))
    ax_err.plot(np.asarray(sim.t), diff_max_t, color="tab:red")
    ax_err.set_title(f"Max |Vm - {ref_name}|")
    ax_err.set_xlabel("Time (ms)")
    ax_err.set_ylabel("Error (mV)")
    ax_err.grid(True, alpha=0.3)

    ref.rasterplot(ax_raster)
    ax_raster.set_title("AxonScope Raster")
    ax_raster.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def _status(metrics: dict[str, float], vel_tol: float, err_tol: float) -> str:
    if metrics["has_nan"]:
        return "nan"
    if metrics["velocity_abs_err_ms"] <= vel_tol and metrics["max_abs_err_mV"] <= err_tol:
        return "ok"
    return "warn"


def run_backend_script(
    *,
    label: str,
    build_runner: RunnerBuilder,
    vel_tol: float = 0.01,
    err_tol: float = 0.25,
    Nx_values: list[int] | None = None,
) -> None:
    Nx_values = s.Nx_v if Nx_values is None else Nx_values
    timing_rows = []
    validation_rows = []
    last_sim = None
    last_ref = None
    last_nrv = None

    for Nx in Nx_values:
        problem = make_problem(Nx)
        ref = reference_result(problem)
        nrv_ref = load_nrv_reference(problem)
        result, elapsed = benchmark_runner(build_runner(problem))
        sim, metrics = validate_against_reference(problem, result, ref, nrv_ref=nrv_ref)
        status = _status(metrics, vel_tol=vel_tol, err_tol=err_tol)

        timing_rows.append(elapsed)
        validation_rows.append(
            {
                "label": label,
                "Nx": Nx,
                "runtime_s": elapsed,
                "status": status,
                **metrics,
            }
        )
        print(
            f"Nx={Nx:<4d}  time={elapsed:.4f}s  "
            f"vel={metrics['velocity_ms']:.4f} m/s  "
            f"ref[{metrics['reference_label']}]={metrics['ref_velocity_ms']:.4f} m/s  "
            f"max_err={metrics['max_abs_err_mV']:.3e} mV  "
            f"status={status}",
            flush=True,
        )
        if nrv_ref is not None:
            print(
                f"           AxonScope vel={metrics['axonscope_velocity_ms']:.4f} m/s  "
                f"AxonScope max_err={metrics['axonscope_max_abs_err_mV']:.3e} mV",
                flush=True,
            )

        last_sim = sim
        last_ref = ref
        last_nrv = nrv_ref

    u.append_to_csv(u.res_to_df(Nx_values, timing_rows, label=label))
    validation_path = save_validation(label, validation_rows)
    if last_sim is not None and last_ref is not None:
        fig_path = save_plot(label, last_sim, last_ref, nrv_ref=last_nrv)
        print(f"Validation figure: {fig_path}", flush=True)
    print(f"Validation table: {validation_path}", flush=True)


def run_solver_family_script(label: str = "AxonScope", Nx: int | None = None) -> None:
    Nx = s.Nx_v[-1] if Nx is None else Nx
    solver_specs = [
        ("CrankNicholson_unoptimized", CrankNicholson_unoptimized()),
        ("CrankNicholson", CrankNicholson()),
        ("CrankNicholsonSemiImplicit", CrankNicholsonSemiImplicit()),
        ("CrankNicholsonImplicit", CrankNicholsonImplicit(n_newton=3)),
        ("CrankNicholsonImplicitFast", CrankNicholsonImplicitFast()),
        ("CrankNicholsonImplicitFastMultiStep", CrankNicholsonImplicitFastMultiStep()),
        ("CrankNicholsonQuasiNewtonFast", CrankNicholsonQuasiNewtonFast()),
    ]

    rows = []
    fig_dir = Path(__file__).resolve().parent / "figures"
    fig_dir.mkdir(exist_ok=True)
    final_problem = make_problem(Nx)
    final_ref = reference_result(final_problem)
    final_nrv = load_nrv_reference(final_problem)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_trace, ax_runtime, ax_vel, ax_err = axes.ravel()
    idx_plot = int(round((final_problem.Nx - 1) * 0.75))

    ax_trace.plot(final_ref.t, final_ref.Vm[:, idx_plot], lw=1.8, ls=":", label="AxonScope CN")
    if final_nrv is not None:
        idx_nrv = int(np.argmin(np.abs(np.asarray(final_nrv.x) - np.asarray(final_problem.axon.x)[idx_plot])))
        ax_trace.plot(final_nrv.t, final_nrv.Vm[:, idx_nrv], lw=2.4, label="NRV")

    for name, solver in solver_specs:
        problem = make_problem(Nx)
        ref = reference_result(problem)
        nrv_ref = load_nrv_reference(problem)
        def run_once(solver=solver, problem=problem):
            res = solver.solve(problem.axon, tsim=problem.tsim, dt=problem.dt)
            return np.asarray(res.t), np.asarray(res.Vm)

        result, elapsed = benchmark_runner(run_once)
        sim, metrics = validate_against_reference(problem, result, ref, nrv_ref=nrv_ref)
        rows.append({"label": label, "solver": name, "runtime_s": elapsed, **metrics})
        ax_trace.plot(sim.t, sim.Vm[:, idx_plot], lw=1.2, label=name)

    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent / "validation_AxonScope_solver_family.csv"
    df.to_csv(out, index=False)

    ax_runtime.bar(df["solver"], df["runtime_s"])
    ax_runtime.set_title("Runtime")
    ax_runtime.tick_params(axis="x", rotation=45)
    ax_runtime.grid(True, alpha=0.3, axis="y")

    ax_vel.bar(df["solver"], df["velocity_ms"])
    ax_vel.set_title("Velocity")
    ax_vel.tick_params(axis="x", rotation=45)
    ax_vel.grid(True, alpha=0.3, axis="y")

    ax_err.bar(df["solver"], df["max_abs_err_mV"])
    ref_title = "NRV" if final_nrv is not None else "AxonScope CN"
    ax_err.set_title(f"Max abs err vs {ref_title}")
    ax_err.tick_params(axis="x", rotation=45)
    ax_err.grid(True, alpha=0.3, axis="y")

    ax_trace.set_title(f"{label} CN Family")
    ax_trace.set_xlabel("Time (ms)")
    ax_trace.set_ylabel("Vm (mV)")
    ax_trace.grid(True, alpha=0.3)
    ax_trace.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig_path = fig_dir / "AxonScope_solver_family.png"
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Solver family figure: {fig_path}", flush=True)
    print(f"Solver family table: {out}", flush=True)


def run_solver_probe() -> None:
    try:
        import mlx.core as mx  # type: ignore

        status = f"available ({mx.__file__})"
    except Exception as exc:
        status = f"unavailable: {type(exc).__name__}: {exc}"

    out = Path(__file__).resolve().parent / "validation_mlx_probe.csv"
    pd.DataFrame([{"label": "mlx_probe", "status": status}]).to_csv(out, index=False)
    print(status)
    print(f"Validation table: {out}")
