"""Plot helpers for experimental pseudo-double validation outputs."""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _pyplot():
    cache_dir = Path(tempfile.gettempdir()) / "axonscope-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _finite_or_nan(value: Any) -> float:
    if value is None:
        return math.nan
    parsed = float(value)
    return parsed if math.isfinite(parsed) else math.nan


def _amplitudes(summaries: Sequence[dict[str, object]]) -> np.ndarray:
    return np.asarray([float(item["amplitude_uA"]) for item in summaries], dtype=float)


def write_validation_plots(result: dict[str, object], out_dir: Path) -> tuple[Path, ...]:
    """Write standard validation plots and return their paths."""

    plot_dir = Path(out_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        _plot_activation_summary(result, plot_dir / "activation_summary.png"),
        _plot_error_summary(result, plot_dir / "error_summary.png"),
        _plot_thresholds(result, plot_dir / "thresholds.png"),
        _plot_timings(result, plot_dir / "timings.png"),
    ]
    paths.extend(_plot_trace_samples(result, plot_dir))
    return tuple(path for path in paths if path is not None)


def _mode_label(result: dict[str, object]) -> str:
    return str(result.get("candidate_mode", "candidate"))


def _plot_activation_summary(result: dict[str, object], path: Path) -> Path | None:
    summaries = result.get("amplitude_summaries")
    if not isinstance(summaries, list) or not summaries:
        return None
    plt = _pyplot()
    x = _amplitudes(summaries)
    exact = np.asarray([int(item["exact_active_count"]) for item in summaries], dtype=float)
    candidate = np.asarray([int(item["candidate_active_count"]) for item in summaries], dtype=float)
    agreement = np.asarray([float(item["activation_agreement"]) for item in summaries], dtype=float)
    false_negative = np.asarray([int(item["false_negative_count"]) for item in summaries], dtype=float)
    false_positive = np.asarray([int(item["false_positive_count"]) for item in summaries], dtype=float)

    fig, (ax_count, ax_quality) = plt.subplots(
        2,
        1,
        figsize=(8.0, 6.0),
        sharex=True,
        constrained_layout=True,
    )
    ax_count.plot(x, exact, marker="o", linewidth=2.0, label="exact active")
    ax_count.plot(x, candidate, marker="s", linewidth=2.0, label="candidate active")
    ax_count.set_ylabel("active rows")
    ax_count.set_title(f"Activation summary: {_mode_label(result)}")
    ax_count.grid(True, alpha=0.3)
    ax_count.legend()

    ax_quality.plot(x, agreement, marker="o", linewidth=2.0, label="agreement")
    ax_quality.bar(x, false_negative, width=_bar_width(x), alpha=0.35, label="false negatives")
    ax_quality.bar(x, false_positive, width=_bar_width(x), alpha=0.35, label="false positives")
    ax_quality.set_xlabel("Amplitude [uA]")
    ax_quality.set_ylabel("agreement / count")
    ax_quality.set_ylim(bottom=0.0)
    ax_quality.grid(True, alpha=0.3)
    ax_quality.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_error_summary(result: dict[str, object], path: Path) -> Path | None:
    summaries = result.get("amplitude_summaries")
    if not isinstance(summaries, list) or not summaries:
        return None
    plt = _pyplot()
    x = _amplitudes(summaries)
    peak = np.asarray([_finite_or_nan(item.get("peak_abs_error_mean_mV")) for item in summaries])
    center = np.asarray(
        [_finite_or_nan(item.get("center_peak_abs_error_mean_mV")) for item in summaries]
    )
    rms = np.asarray([_finite_or_nan(item.get("rms_vm_error_mean_mV")) for item in summaries])
    time_error = np.asarray(
        [_finite_or_nan(item.get("activation_time_abs_error_mean_ms")) for item in summaries]
    )

    fig, (ax_voltage, ax_time) = plt.subplots(
        2,
        1,
        figsize=(8.0, 6.0),
        sharex=True,
        constrained_layout=True,
    )
    ax_voltage.plot(x, peak, marker="o", linewidth=2.0, label="peak abs error")
    ax_voltage.plot(x, center, marker="s", linewidth=2.0, label="center peak abs error")
    ax_voltage.plot(x, rms, marker="^", linewidth=2.0, label="RMS Vm error")
    ax_voltage.set_ylabel("error [mV]")
    ax_voltage.set_title(f"Physiology errors: {_mode_label(result)}")
    ax_voltage.grid(True, alpha=0.3)
    ax_voltage.legend()

    ax_time.plot(x, time_error, marker="o", linewidth=2.0, label="activation time error")
    ax_time.set_xlabel("Amplitude [uA]")
    ax_time.set_ylabel("error [ms]")
    ax_time.grid(True, alpha=0.3)
    ax_time.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_thresholds(result: dict[str, object], path: Path) -> Path | None:
    threshold = result.get("threshold_summary")
    if not isinstance(threshold, dict):
        return None
    reference = threshold.get("reference_thresholds_uA")
    candidate = threshold.get("candidate_thresholds_uA")
    if not isinstance(reference, list) or not isinstance(candidate, list):
        return None
    if not reference:
        return None
    plt = _pyplot()
    rows = np.arange(len(reference), dtype=float)
    ref_values = np.asarray([_finite_or_nan(value) for value in reference], dtype=float)
    cand_values = np.asarray([_finite_or_nan(value) for value in candidate], dtype=float)
    rel_error = threshold.get("threshold_rel_error_mean")
    title_suffix = (
        ""
        if rel_error is None
        else f" mean rel error={float(rel_error):.3g}"
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    ax.plot(rows, ref_values, marker="o", linewidth=0.0, markersize=7.0, label="exact")
    ax.plot(rows, cand_values, marker="s", linewidth=0.0, markersize=7.0, label="candidate")
    ax.set_xlabel("row")
    ax.set_ylabel("threshold [uA]")
    ax.set_title(f"Threshold estimates: {_mode_label(result)}{title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_timings(result: dict[str, object], path: Path) -> Path | None:
    timings = result.get("timings")
    if not isinstance(timings, list) or not timings:
        return None
    plt = _pyplot()
    x = np.asarray([float(item["amplitude_uA"]) for item in timings], dtype=float)
    reference = np.asarray([float(item["reference_seconds"]) for item in timings], dtype=float)
    candidate = np.asarray([float(item["candidate_seconds"]) for item in timings], dtype=float)
    speedup = np.asarray(
        [_finite_or_nan(item.get("candidate_speedup_vs_reference")) for item in timings],
        dtype=float,
    )

    fig, (ax_time, ax_speedup) = plt.subplots(
        2,
        1,
        figsize=(8.0, 6.0),
        sharex=True,
        constrained_layout=True,
    )
    ax_time.plot(x, reference * 1e3, marker="o", linewidth=2.0, label="exact")
    ax_time.plot(x, candidate * 1e3, marker="s", linewidth=2.0, label="candidate")
    ax_time.set_ylabel("elapsed [ms]")
    ax_time.set_title(f"Timing summary: {_mode_label(result)}")
    ax_time.grid(True, alpha=0.3)
    ax_time.legend()

    ax_speedup.axhline(1.0, color="0.35", linewidth=1.0, linestyle=":")
    ax_speedup.plot(x, speedup, marker="o", linewidth=2.0)
    ax_speedup.set_xlabel("Amplitude [uA]")
    ax_speedup.set_ylabel("exact / candidate")
    ax_speedup.grid(True, alpha=0.3)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_trace_samples(result: dict[str, object], plot_dir: Path) -> list[Path]:
    samples = result.get("trace_samples")
    if not isinstance(samples, list) or not samples:
        return []
    plt = _pyplot()
    threshold = _finite_or_nan(
        (result.get("parameters") or {}).get("activation_threshold_mV")
        if isinstance(result.get("parameters"), dict)
        else None
    )
    paths: list[Path] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        path = plot_dir / (
            "trace_"
            f"amp_{float(sample['amplitude_uA']):g}_"
            f"row_{int(sample['row'])}.png"
        )
        t = np.asarray(sample["t_ms"], dtype=float)
        positions = np.asarray(sample["positions_um"], dtype=float)
        center_col = int(sample["center_column"])
        peak_col = int(sample["reference_peak_column"])
        panels = [("center", center_col)]
        if peak_col != center_col:
            panels.append(("reference peak", peak_col))
        fig, axes = plt.subplots(
            len(panels),
            1,
            figsize=(8.5, 3.4 * len(panels)),
            sharex=True,
            constrained_layout=True,
        )
        axes_arr = np.atleast_1d(axes)
        ref_vm = np.asarray(sample["reference_vm_mV"], dtype=float)
        cand_vm = np.asarray(sample["candidate_vm_mV"], dtype=float)
        for ax, (label, col) in zip(axes_arr, panels, strict=False):
            ax.plot(t, ref_vm[:, col], color="black", linewidth=2.0, label="exact")
            ax.plot(t, cand_vm[:, col], color="C1", linewidth=2.0, label=_mode_label(result))
            if math.isfinite(threshold):
                ax.axhline(threshold, color="0.45", linestyle=":", linewidth=1.0)
            ax.set_ylabel("Vm [mV]")
            ax.set_title(f"{label} x={positions[col]:.1f} um")
            ax.grid(True, alpha=0.3)
            ax.legend()
        axes_arr[-1].set_xlabel("Time [ms]")
        fig.suptitle(
            f"Trace comparison: amplitude={float(sample['amplitude_uA']):g} uA, "
            f"row={int(sample['row'])}"
        )
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def _bar_width(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.6
    diffs = np.diff(np.sort(np.unique(x)))
    if diffs.size == 0:
        return 0.6
    return float(0.35 * np.min(diffs))


__all__ = ["write_validation_plots"]
