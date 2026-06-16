"""Validate experimental pseudo-double-cable candidates against exact double.

This module is intentionally benchmark/validation plumbing, not a public solver
API. Exact double-cable remains the reference model; pseudo modes stay opt-in
and experimental until their physiology metrics are acceptable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import jax
import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))

import axonscope as axs
from axonscope.analysis import ActivationCriterion
from axonscope.axons import CableFormulation
from axonscope.results import AxonSimulationResult
from axonscope.solvers import BatchOptions

from benchmark.pseudo_double.schur_runner import (
    PseudoDoubleSchurLocalConfig,
    run_schur_local_population,
)
from benchmark.pseudo_double.single_chain import (
    PseudoDoubleSingleChainConfig,
    SegmentScaledAnalyticalExtracellularContext,
    build_pseudo_double_single_chain_mrg,
    single_chain_vext_alpha,
)
from benchmark.pseudo_double.series_runner import (
    PseudoDoubleSeriesConfig,
    run_series_population,
)


VALIDATION_MODES = (
    "exact_double",
    "mrg_single_cable_surrogate",
    "pseudo_double_effective",
    "pseudo_double_single_myelinated_chain",
    "pseudo_double_series",
    "pseudo_double_split",
    "pseudo_double_schur_local",
    "pseudo_double_modal",
)
IMPLEMENTED_VALIDATION_MODES = frozenset(
    {
        "exact_double",
        "mrg_single_cable_surrogate",
        "pseudo_double_effective",
        "pseudo_double_single_myelinated_chain",
        "pseudo_double_series",
        "pseudo_double_split",
        "pseudo_double_schur_local",
    }
)
PSEUDO_DOUBLE_EXPERIMENTAL_MODES = frozenset(
    {
        "mrg_single_cable_surrogate",
        "pseudo_double_effective",
        "pseudo_double_single_myelinated_chain",
        "pseudo_double_series",
        "pseudo_double_split",
        "pseudo_double_schur_local",
        "pseudo_double_modal",
    }
)


@dataclass(frozen=True)
class RunConfig:
    size: int
    nodes: int
    diameter_um: float
    duration_ms: float
    dt_ms: float
    pulse_start_ms: float
    pulse_duration_ms: float
    electrode_z_um: float
    offset_span_um: float
    activation_threshold_mV: float
    activation_blanking_ms: float
    double_cable_block_solver: str
    recording: str
    probe_count: int


@dataclass(frozen=True)
class PseudoDoubleEffectiveConfig:
    """Experimental scalar effective pseudo-double parameters.

    V0 only calibrates the imposed extracellular coupling. It deliberately
    leaves axial/capacitance/leak scaling for a later solver-kernel
    implementation so this phase can validate physiology before broad plumbing.
    """

    vext_scale: float = 1.0

    def as_dict(self) -> dict[str, float]:
        return {"vext_scale": float(self.vext_scale)}


@dataclass(frozen=True)
class PseudoDoubleSplitConfig:
    """Experimental local-auxiliary split pseudo-double parameters.

    V0 keeps the solve on the existing scalar cable path and approximates the
    auxiliary periaxonal/myelin response by an implicit pointwise low-pass state
    driven by the extracellular waveform. It is intentionally validation-only:
    no public solver mode and no spatial periaxonal coupling yet.
    """

    vext_scale: float = 1.0
    direct_scale: float = 1.0
    aux_scale: float = 1.0
    aux_alpha: float = 1.0
    aux_tau_ms: float = 0.05

    def as_dict(self) -> dict[str, float]:
        return {
            "vext_scale": float(self.vext_scale),
            "direct_scale": float(self.direct_scale),
            "aux_scale": float(self.aux_scale),
            "aux_alpha": float(self.aux_alpha),
            "aux_tau_ms": float(self.aux_tau_ms),
        }


@dataclass(frozen=True)
class RowMetrics:
    activated: bool
    first_time_ms: float | None
    first_index: int | None
    peak_mV: float | None
    peak_time_ms: float | None
    peak_index: int | None
    peak_position_um: float | None
    center_peak_mV: float


def normalize_validation_mode(value: str) -> str:
    """Return a normalized validation mode or raise a clear error."""

    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "double": "exact_double",
        "exact": "exact_double",
        "single": "mrg_single_cable_surrogate",
        "single_cable": "mrg_single_cable_surrogate",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in VALIDATION_MODES:
        choices = ", ".join(VALIDATION_MODES)
        raise ValueError(f"unknown pseudo-double validation mode {value!r}; choices: {choices}")
    return normalized


def mode_metadata(mode: str) -> dict[str, object]:
    """Return JSON-friendly metadata describing one validation mode."""

    normalized = normalize_validation_mode(mode)
    return {
        "mode": normalized,
        "implemented": normalized in IMPLEMENTED_VALIDATION_MODES,
        "experimental": normalized in PSEUDO_DOUBLE_EXPERIMENTAL_MODES,
        "reference": normalized == "exact_double",
        "description": {
            "exact_double": "Exact MRG double-cable reference.",
            "mrg_single_cable_surrogate": (
                "MRG morphology and membranes forced through the existing "
                "single-cable extracellular path; baseline surrogate only."
            ),
            "pseudo_double_effective": (
                "Experimental scalar effective pseudo-double v0: MRG single-cable "
                "surrogate with calibrated extracellular coupling."
            ),
            "pseudo_double_single_myelinated_chain": (
                "Experimental one-voltage NODE/MYSA/FLUT/STIN myelinated chain: "
                "single-cable AxonScope model with segment-specific effective "
                "capacitance/leak and extracellular alpha coupling."
            ),
            "pseudo_double_series": (
                "Experimental coefficient-derived RC-series pseudo-double v1: "
                "local axolemma/myelin series reduction solved as one scalar "
                "tridiagonal system per step."
            ),
            "pseudo_double_split": (
                "Experimental split pseudo-double v0: MRG single-cable surrogate "
                "with a local implicit auxiliary extracellular-response state."
            ),
            "pseudo_double_schur_local": (
                "Experimental coefficient-derived Schur-local v1: diagonal-App "
                "local elimination of the double-cable periaxonal/myelin block."
            ),
            "pseudo_double_modal": "Planned modal two-scalar-solve pseudo-double mode.",
        }[normalized],
    }


def _formulation_for_mode(mode: str) -> CableFormulation:
    normalized = normalize_validation_mode(mode)
    if normalized in {
        "exact_double",
        "pseudo_double_series",
        "pseudo_double_schur_local",
    }:
        return CableFormulation.DOUBLE_CABLE
    if normalized in {
        "mrg_single_cable_surrogate",
        "pseudo_double_effective",
        "pseudo_double_single_myelinated_chain",
        "pseudo_double_split",
    }:
        return CableFormulation.SINGLE_CABLE
    raise NotImplementedError(
        f"{normalized!r} is registered as an experimental pseudo-double mode, "
        "but its solver kernel is not implemented yet."
    )


def _build_mrg_axon_for_mode(
    mode: str,
    *,
    nodes: int,
    diameter_um: float,
    single_chain_config: PseudoDoubleSingleChainConfig | None = None,
) -> axs.axons.Myelinated:
    normalized_mode = normalize_validation_mode(mode)
    if normalized_mode == "pseudo_double_single_myelinated_chain":
        return build_pseudo_double_single_chain_mrg(
            diameter_um=diameter_um,
            nodes=nodes,
            config=single_chain_config or PseudoDoubleSingleChainConfig(),
        )
    return axs.axons.MRG(
        diameter=diameter_um * axs.um,
        nodes=nodes,
        formulation=_formulation_for_mode(normalized_mode),
    )


def build_validation_population(
    mode: str,
    *,
    size: int,
    nodes: int,
    diameter_um: float,
    duration_ms: float | None = None,
    dt_ms: float | None = None,
    amplitude_uA: float,
    pulse_start_ms: float,
    pulse_duration_ms: float,
    electrode_z_um: float,
    offset_span_um: float,
    effective_config: PseudoDoubleEffectiveConfig | None = None,
    single_chain_config: PseudoDoubleSingleChainConfig | None = None,
    split_config: PseudoDoubleSplitConfig | None = None,
) -> list[axs.AxonInstance]:
    """Build one deterministic MRG validation population for a mode."""

    normalized_mode = normalize_validation_mode(mode)
    effective_config = effective_config or PseudoDoubleEffectiveConfig()
    single_chain_config = single_chain_config or PseudoDoubleSingleChainConfig()
    split_config = split_config or PseudoDoubleSplitConfig()
    axon = _build_mrg_axon_for_mode(
        normalized_mode,
        nodes=nodes,
        diameter_um=diameter_um,
        single_chain_config=single_chain_config,
    )
    center_x_um = 0.5 * float(axon.length)
    stimulus = _validation_stimulus_for_mode(
        normalized_mode,
        amplitude_uA=amplitude_uA,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        pulse_start_ms=pulse_start_ms,
        pulse_duration_ms=pulse_duration_ms,
        effective_config=effective_config,
        split_config=split_config,
    )
    electrode = axs.PointSourceElectrode(
        x=center_x_um * axs.um,
        z=electrode_z_um * axs.um,
        stimulus=stimulus,
    )
    if normalized_mode == "pseudo_double_single_myelinated_chain":
        context = SegmentScaledAnalyticalExtracellularContext(
            electrodes=[electrode],
            sigma=0.3 * axs.S_per_m,
            positions_um=axon.layout.position_values(unit="micrometer"),
            alpha=single_chain_vext_alpha(axon, single_chain_config),
        )
    else:
        context = axs.AnalyticalExtracellularContext(
            electrodes=[electrode],
            sigma=0.3 * axs.S_per_m,
        )

    if size <= 0:
        raise ValueError("size must be positive.")
    offsets = (
        np.asarray([0.0])
        if size == 1
        else np.linspace(-offset_span_um, offset_span_um, int(size), dtype=float)
    )
    instances: list[axs.AxonInstance] = []
    for offset_um in offsets:
        instance = axs.AxonInstance(axon, y=float(offset_um) * axs.um)
        instance.add_extracellular_context(context=context)
        instances.append(instance)
    return instances


def _validation_stimulus_for_mode(
    mode: str,
    *,
    amplitude_uA: float,
    duration_ms: float | None,
    dt_ms: float | None,
    pulse_start_ms: float,
    pulse_duration_ms: float,
    effective_config: PseudoDoubleEffectiveConfig,
    split_config: PseudoDoubleSplitConfig,
) -> axs.Stimulus:
    if mode == "pseudo_double_effective":
        return axs.Stimulus.pulse(
            start=pulse_start_ms * axs.ms,
            duration=pulse_duration_ms * axs.ms,
            amplitude=(amplitude_uA * effective_config.vext_scale) * axs.uA,
        )
    if mode == "pseudo_double_split":
        t_ms, current_uA = _split_auxiliary_current_samples(
            amplitude_uA=amplitude_uA,
            duration_ms=duration_ms,
            dt_ms=dt_ms,
            pulse_start_ms=pulse_start_ms,
            pulse_duration_ms=pulse_duration_ms,
            config=split_config,
        )
        return axs.Stimulus.from_samples(
            t_ms * axs.ms,
            current_uA,
            mode="hold",
            unit=axs.uA,
        )
    return axs.Stimulus.pulse(
        start=pulse_start_ms * axs.ms,
        duration=pulse_duration_ms * axs.ms,
        amplitude=amplitude_uA * axs.uA,
    )


def _split_auxiliary_current_samples(
    *,
    amplitude_uA: float,
    duration_ms: float | None,
    dt_ms: float | None,
    pulse_start_ms: float,
    pulse_duration_ms: float,
    config: PseudoDoubleSplitConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return current samples for the split v0 local auxiliary state.

    The auxiliary state follows an implicit pointwise first-order filter:

        Z[n] = (Z[n-1] + r * aux_alpha * I[n]) / (1 + r)

    with r = dt / tau. The scalar cable then sees

        vext_scale * (direct_scale * I[n] + aux_scale * Z[n]).
    """

    if config.aux_tau_ms <= 0.0:
        raise ValueError("split aux_tau_ms must be positive.")
    step_ms = float(dt_ms) if dt_ms is not None else min(config.aux_tau_ms / 5.0, 0.01)
    if step_ms <= 0.0:
        raise ValueError("split dt_ms must be positive.")
    stop_ms = (
        float(duration_ms)
        if duration_ms is not None
        else pulse_start_ms + pulse_duration_ms + 5.0 * config.aux_tau_ms
    )
    if stop_ms <= 0.0:
        raise ValueError("split duration_ms must be positive.")
    sample_count = max(2, int(math.ceil(stop_ms / step_ms)) + 1)
    t_ms = np.linspace(0.0, step_ms * (sample_count - 1), sample_count)
    pulse_stop_ms = pulse_start_ms + pulse_duration_ms
    raw = np.where(
        (t_ms >= pulse_start_ms) & (t_ms < pulse_stop_ms),
        float(amplitude_uA),
        0.0,
    )
    aux = np.zeros_like(raw)
    ratio = step_ms / float(config.aux_tau_ms)
    for index in range(1, raw.shape[0]):
        target = float(config.aux_alpha) * raw[index]
        aux[index] = (aux[index - 1] + ratio * target) / (1.0 + ratio)
    current = float(config.vext_scale) * (
        float(config.direct_scale) * raw + float(config.aux_scale) * aux
    )
    return t_ms, current


def _recording_policy(name: str, *, probe_count: int) -> axs.Recording:
    normalized = str(name).strip().lower()
    if normalized == "full":
        return axs.Recording.voltage()
    if normalized == "center":
        return axs.Recording.center(axs.signals.Vm)
    if normalized == "probes":
        return axs.Recording.probes(axs.signals.Vm, count=probe_count)
    raise ValueError("recording must be 'full', 'center', or 'probes'.")


def _run_mode(
    mode: str,
    *,
    amplitude_uA: float,
    config: RunConfig,
    effective_config: PseudoDoubleEffectiveConfig | None = None,
    single_chain_config: PseudoDoubleSingleChainConfig | None = None,
    series_config: PseudoDoubleSeriesConfig | None = None,
    split_config: PseudoDoubleSplitConfig | None = None,
    schur_config: PseudoDoubleSchurLocalConfig | None = None,
) -> tuple[AxonSimulationResult, float]:
    normalized_mode = normalize_validation_mode(mode)
    instances = build_validation_population(
        normalized_mode,
        size=config.size,
        nodes=config.nodes,
        diameter_um=config.diameter_um,
        duration_ms=config.duration_ms,
        dt_ms=config.dt_ms,
        amplitude_uA=amplitude_uA,
        pulse_start_ms=config.pulse_start_ms,
        pulse_duration_ms=config.pulse_duration_ms,
        electrode_z_um=config.electrode_z_um,
        offset_span_um=config.offset_span_um,
        effective_config=effective_config,
        single_chain_config=single_chain_config,
        split_config=split_config,
    )
    if normalized_mode == "pseudo_double_series":
        start = time.perf_counter()
        result = run_series_population(
            instances,
            duration_ms=config.duration_ms,
            dt_ms=config.dt_ms,
            recording=config.recording,
            probe_count=config.probe_count,
            config=series_config or PseudoDoubleSeriesConfig(),
        )
        elapsed = time.perf_counter() - start
        _materialize_vm(result)
        return result, elapsed
    if normalized_mode == "pseudo_double_schur_local":
        start = time.perf_counter()
        result = run_schur_local_population(
            instances,
            duration_ms=config.duration_ms,
            dt_ms=config.dt_ms,
            recording=config.recording,
            probe_count=config.probe_count,
            config=schur_config or PseudoDoubleSchurLocalConfig(),
        )
        elapsed = time.perf_counter() - start
        _materialize_vm(result)
        return result, elapsed
    simulation = axs.AxonSimulation(
        axs.AxonPopulation(instances),
        duration=config.duration_ms * axs.ms,
        dt=config.dt_ms * axs.ms,
        recording=_recording_policy(config.recording, probe_count=config.probe_count),
        batch_options=BatchOptions.full(
            double_cable_block_solver=config.double_cable_block_solver,
        ),
    )
    start = time.perf_counter()
    result = simulation.run()
    elapsed = time.perf_counter() - start
    if not isinstance(result, AxonSimulationResult):
        raise TypeError("pseudo-double validation expects a population result.")
    _materialize_vm(result)
    return result, elapsed


ReferenceRunMap = dict[float, tuple[AxonSimulationResult, float]]


def _run_reference_sweep(
    *,
    reference_mode: str,
    amplitudes_uA: Sequence[float],
    config: RunConfig,
) -> ReferenceRunMap:
    return {
        float(amplitude): _run_mode(
            reference_mode,
            amplitude_uA=float(amplitude),
            config=config,
        )
        for amplitude in amplitudes_uA
    }


def _materialize_vm(result: AxonSimulationResult) -> None:
    for row in result:
        np.asarray(row.Vm)


def _result_output_metadata(result: AxonSimulationResult) -> dict[str, object]:
    vm_arrays = []
    total_bytes = 0
    for cohort in result.cohorts:
        if cohort.Vm is None:
            continue
        vm = np.asarray(cohort.Vm)
        total_bytes += int(vm.nbytes)
        vm_arrays.append(
            {
                "shape": list(vm.shape),
                "dtype": str(vm.dtype),
                "bytes": int(vm.nbytes),
            }
        )
    dtypes = sorted({str(item["dtype"]) for item in vm_arrays})
    return {
        "vm_array_count": len(vm_arrays),
        "vm_arrays": vm_arrays,
        "vm_total_bytes": total_bytes,
        "vm_total_mib": total_bytes / float(1024**2),
        "vm_dtypes": dtypes,
    }


def _row_metrics(
    row: Any,
    *,
    threshold_mV: float,
    blanking_ms: float,
) -> RowMetrics:
    sim_result = row.to_sim_result() if hasattr(row, "to_sim_result") else row
    event = ActivationCriterion(
        threshold=threshold_mV * axs.mV,
        blanking=blanking_ms * axs.ms,
    ).evaluate(sim_result)
    vm = np.asarray(sim_result.voltage_values(unit="millivolt"), dtype=float)
    positions_um = np.asarray(sim_result.position_values(unit="micrometer"), dtype=float)
    center_col = int(vm.shape[1] // 2)
    peak_position_um = None
    if event.peak_index is not None:
        if sim_result.record_indices is None:
            peak_col = int(event.peak_index)
        else:
            record_indices = tuple(int(index) for index in sim_result.record_indices)
            peak_col = record_indices.index(int(event.peak_index))
        if 0 <= peak_col < positions_um.shape[0]:
            peak_position_um = float(positions_um[peak_col])
    return RowMetrics(
        activated=bool(event.activated),
        first_time_ms=None if event.first_time_ms is None else float(event.first_time_ms),
        first_index=None if event.first_index is None else int(event.first_index),
        peak_mV=None if event.peak_mV is None else float(event.peak_mV),
        peak_time_ms=None if event.peak_time_ms is None else float(event.peak_time_ms),
        peak_index=None if event.peak_index is None else int(event.peak_index),
        peak_position_um=peak_position_um,
        center_peak_mV=float(np.max(vm[:, center_col])),
    )


def _safe_abs_error(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return abs(float(a) - float(b))


def _rms_vm_error(reference_row: Any, candidate_row: Any) -> float | None:
    ref_vm = np.asarray(reference_row.voltage_values(unit="millivolt"), dtype=float)
    cand_vm = np.asarray(candidate_row.voltage_values(unit="millivolt"), dtype=float)
    if ref_vm.shape != cand_vm.shape:
        return None
    return float(np.sqrt(np.mean((cand_vm - ref_vm) ** 2)))


def _trace_sample(
    reference: AxonSimulationResult,
    candidate: AxonSimulationResult,
    *,
    amplitude_uA: float,
    row: int,
) -> dict[str, object] | None:
    if row < 0 or row >= len(reference) or row >= len(candidate):
        return None
    ref_result = reference[row].to_sim_result()
    cand_result = candidate[row].to_sim_result()
    ref_vm = np.asarray(ref_result.voltage_values(unit="millivolt"), dtype=float)
    cand_vm = np.asarray(cand_result.voltage_values(unit="millivolt"), dtype=float)
    if ref_vm.shape != cand_vm.shape or ref_vm.ndim != 2:
        return None
    positions_um = np.asarray(ref_result.position_values(unit="micrometer"), dtype=float)
    if positions_um.shape[0] != ref_vm.shape[1]:
        return None
    peak_flat = int(np.nanargmax(ref_vm))
    _, peak_col = np.unravel_index(peak_flat, ref_vm.shape)
    return {
        "amplitude_uA": float(amplitude_uA),
        "row": int(row),
        "t_ms": np.asarray(ref_result.time_values(unit="millisecond"), dtype=float).tolist(),
        "positions_um": positions_um.tolist(),
        "center_column": int(ref_vm.shape[1] // 2),
        "reference_peak_column": int(peak_col),
        "reference_vm_mV": ref_vm.tolist(),
        "candidate_vm_mV": cand_vm.tolist(),
    }


def _p95(values: Sequence[float]) -> float | None:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if finite.size == 0:
        return None
    return float(np.percentile(finite, 95))


def _mean(values: Sequence[float]) -> float | None:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def compare_mode_results(
    reference: AxonSimulationResult,
    candidate: AxonSimulationResult,
    *,
    amplitude_uA: float,
    threshold_mV: float,
    blanking_ms: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Compare candidate rows against exact double-cable reference rows."""

    if len(reference) != len(candidate):
        raise ValueError("reference and candidate results must have the same size.")
    rows: list[dict[str, object]] = []
    for index, (ref_row, cand_row) in enumerate(zip(reference, candidate)):
        ref_metrics = _row_metrics(
            ref_row,
            threshold_mV=threshold_mV,
            blanking_ms=blanking_ms,
        )
        cand_metrics = _row_metrics(
            cand_row,
            threshold_mV=threshold_mV,
            blanking_ms=blanking_ms,
        )
        rows.append(
            {
                "amplitude_uA": float(amplitude_uA),
                "row": int(index),
                "reference_activated": ref_metrics.activated,
                "candidate_activated": cand_metrics.activated,
                "activation_agrees": ref_metrics.activated == cand_metrics.activated,
                "false_negative": ref_metrics.activated and not cand_metrics.activated,
                "false_positive": cand_metrics.activated and not ref_metrics.activated,
                "reference_first_time_ms": ref_metrics.first_time_ms,
                "candidate_first_time_ms": cand_metrics.first_time_ms,
                "activation_time_abs_error_ms": _safe_abs_error(
                    ref_metrics.first_time_ms,
                    cand_metrics.first_time_ms,
                ),
                "reference_first_index": ref_metrics.first_index,
                "candidate_first_index": cand_metrics.first_index,
                "activation_index_agrees": (
                    ref_metrics.first_index is not None
                    and ref_metrics.first_index == cand_metrics.first_index
                ),
                "reference_peak_mV": ref_metrics.peak_mV,
                "candidate_peak_mV": cand_metrics.peak_mV,
                "peak_abs_error_mV": _safe_abs_error(
                    ref_metrics.peak_mV,
                    cand_metrics.peak_mV,
                ),
                "reference_peak_time_ms": ref_metrics.peak_time_ms,
                "candidate_peak_time_ms": cand_metrics.peak_time_ms,
                "peak_time_abs_error_ms": _safe_abs_error(
                    ref_metrics.peak_time_ms,
                    cand_metrics.peak_time_ms,
                ),
                "reference_peak_index": ref_metrics.peak_index,
                "candidate_peak_index": cand_metrics.peak_index,
                "reference_peak_position_um": ref_metrics.peak_position_um,
                "candidate_peak_position_um": cand_metrics.peak_position_um,
                "reference_center_peak_mV": ref_metrics.center_peak_mV,
                "candidate_center_peak_mV": cand_metrics.center_peak_mV,
                "center_peak_abs_error_mV": abs(
                    cand_metrics.center_peak_mV - ref_metrics.center_peak_mV
                ),
                "rms_vm_error_mV": _rms_vm_error(ref_row, cand_row),
            }
        )
    return rows, summarize_rows(rows)


def summarize_rows(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    """Return aggregate physiology metrics for one amplitude."""

    count = len(rows)
    if count == 0:
        raise ValueError("at least one row is required.")
    agreements = [bool(row["activation_agrees"]) for row in rows]
    false_negatives = [bool(row["false_negative"]) for row in rows]
    false_positives = [bool(row["false_positive"]) for row in rows]
    exact_active = [bool(row["reference_activated"]) for row in rows]
    candidate_active = [bool(row["candidate_activated"]) for row in rows]
    true_positives = [ref and cand for ref, cand in zip(exact_active, candidate_active)]
    peak_errors = [
        float(row["peak_abs_error_mV"])
        for row in rows
        if row["peak_abs_error_mV"] is not None
    ]
    center_peak_errors = [float(row["center_peak_abs_error_mV"]) for row in rows]
    time_errors = [
        float(row["activation_time_abs_error_ms"])
        for row in rows
        if row["activation_time_abs_error_ms"] is not None
    ]
    rms_errors = [
        float(row["rms_vm_error_mV"])
        for row in rows
        if row["rms_vm_error_mV"] is not None
    ]
    exact_active_count = int(sum(exact_active))
    candidate_active_count = int(sum(candidate_active))
    true_positive_count = int(sum(true_positives))
    return {
        "row_count": count,
        "activation_agreement": float(np.mean(agreements)),
        "exact_active_count": exact_active_count,
        "candidate_active_count": candidate_active_count,
        "false_negative_count": int(sum(false_negatives)),
        "false_positive_count": int(sum(false_positives)),
        "activation_recall": (
            None
            if exact_active_count == 0
            else true_positive_count / float(exact_active_count)
        ),
        "activation_precision": (
            None
            if candidate_active_count == 0
            else true_positive_count / float(candidate_active_count)
        ),
        "peak_abs_error_mean_mV": _mean(peak_errors),
        "peak_abs_error_p95_mV": _p95(peak_errors),
        "center_peak_abs_error_mean_mV": _mean(center_peak_errors),
        "activation_time_abs_error_mean_ms": _mean(time_errors),
        "activation_time_abs_error_p95_ms": _p95(time_errors),
        "rms_vm_error_mean_mV": _mean(rms_errors),
        "rms_vm_error_p95_mV": _p95(rms_errors),
    }


def _threshold_estimates(
    rows: Sequence[dict[str, object]],
    *,
    size: int,
    mode_prefix: str,
) -> list[float | None]:
    estimates: list[float | None] = []
    by_row = {index: [] for index in range(size)}
    for row in rows:
        by_row[int(row["row"])].append(row)
    for index in range(size):
        ordered = sorted(by_row[index], key=lambda item: float(item["amplitude_uA"]))
        threshold = None
        for row in ordered:
            if bool(row[f"{mode_prefix}_activated"]):
                threshold = float(row["amplitude_uA"])
                break
        estimates.append(threshold)
    return estimates


def summarize_thresholds(
    rows: Sequence[dict[str, object]],
    *,
    size: int,
) -> dict[str, object]:
    """Return coarse threshold estimates from an amplitude sweep."""

    reference = _threshold_estimates(rows, size=size, mode_prefix="reference")
    candidate = _threshold_estimates(rows, size=size, mode_prefix="candidate")
    comparable = [
        (ref, cand)
        for ref, cand in zip(reference, candidate)
        if ref is not None and cand is not None and ref != 0.0
    ]
    rel_errors = [abs(cand - ref) / abs(ref) for ref, cand in comparable]
    exact_activated = sum(value is not None for value in reference)
    candidate_activated = sum(value is not None for value in candidate)
    missed = sum(ref is not None and cand is None for ref, cand in zip(reference, candidate))
    extra = sum(ref is None and cand is not None for ref, cand in zip(reference, candidate))
    return {
        "reference_thresholds_uA": reference,
        "candidate_thresholds_uA": candidate,
        "comparable_count": len(comparable),
        "reference_activated_count": int(exact_activated),
        "candidate_activated_count": int(candidate_activated),
        "missed_activation_count": int(missed),
        "extra_activation_count": int(extra),
        "threshold_rel_error_mean": _mean(rel_errors),
        "threshold_rel_error_p95": _p95(rel_errors),
    }


def run_validation(
    *,
    reference_mode: str,
    candidate_mode: str,
    amplitudes_uA: Sequence[float],
    config: RunConfig,
    effective_config: PseudoDoubleEffectiveConfig | None = None,
    single_chain_config: PseudoDoubleSingleChainConfig | None = None,
    series_config: PseudoDoubleSeriesConfig | None = None,
    split_config: PseudoDoubleSplitConfig | None = None,
    schur_config: PseudoDoubleSchurLocalConfig | None = None,
    reference_runs: ReferenceRunMap | None = None,
    trace_sample_rows: Sequence[int] | None = None,
    trace_sample_amplitudes_uA: Sequence[float] | None = None,
) -> dict[str, object]:
    reference_mode = normalize_validation_mode(reference_mode)
    candidate_mode = normalize_validation_mode(candidate_mode)
    effective_config = effective_config or PseudoDoubleEffectiveConfig()
    single_chain_config = single_chain_config or PseudoDoubleSingleChainConfig()
    series_config = series_config or PseudoDoubleSeriesConfig()
    split_config = split_config or PseudoDoubleSplitConfig()
    schur_config = schur_config or PseudoDoubleSchurLocalConfig()
    if reference_mode != "exact_double":
        raise ValueError("reference_mode must be exact_double for Phase 7.6.4 validation.")
    if reference_runs is None:
        reference_runs = _run_reference_sweep(
            reference_mode=reference_mode,
            amplitudes_uA=amplitudes_uA,
            config=config,
        )
    rows: list[dict[str, object]] = []
    amplitude_summaries: list[dict[str, object]] = []
    timings: list[dict[str, object]] = []
    trace_samples: list[dict[str, object]] = []
    trace_rows = tuple(int(row) for row in (trace_sample_rows or ()))
    trace_amplitudes = (
        frozenset(float(value) for value in trace_sample_amplitudes_uA)
        if trace_sample_amplitudes_uA is not None
        else frozenset()
    )
    for amplitude in amplitudes_uA:
        reference, reference_seconds = reference_runs[float(amplitude)]
        candidate, candidate_seconds = _run_mode(
            candidate_mode,
            amplitude_uA=float(amplitude),
            config=config,
            effective_config=effective_config,
            single_chain_config=single_chain_config,
            series_config=series_config,
            split_config=split_config,
            schur_config=schur_config,
        )
        amplitude_rows, summary = compare_mode_results(
            reference,
            candidate,
            amplitude_uA=float(amplitude),
            threshold_mV=config.activation_threshold_mV,
            blanking_ms=config.activation_blanking_ms,
        )
        rows.extend(amplitude_rows)
        amplitude_summaries.append({"amplitude_uA": float(amplitude), **summary})
        if float(amplitude) in trace_amplitudes:
            for row in trace_rows:
                sample = _trace_sample(
                    reference,
                    candidate,
                    amplitude_uA=float(amplitude),
                    row=row,
                )
                if sample is not None:
                    trace_samples.append(sample)
        timings.append(
            {
                "amplitude_uA": float(amplitude),
                "reference_seconds": reference_seconds,
                "candidate_seconds": candidate_seconds,
                "candidate_speedup_vs_reference": (
                    None
                    if candidate_seconds == 0.0
                    else reference_seconds / candidate_seconds
                ),
                "reference_output": _result_output_metadata(reference),
                "candidate_output": _result_output_metadata(candidate),
            }
        )
    return {
        "schema_version": 1,
        "phase": "7.6.4",
        "reference_mode": reference_mode,
        "candidate_mode": candidate_mode,
        "reference_mode_metadata": mode_metadata(reference_mode),
        "candidate_mode_metadata": mode_metadata(candidate_mode),
        "candidate_effective_config": (
            effective_config.as_dict()
            if candidate_mode == "pseudo_double_effective"
            else None
        ),
        "candidate_single_chain_config": (
            single_chain_config.as_dict()
            if candidate_mode == "pseudo_double_single_myelinated_chain"
            else None
        ),
        "candidate_series_config": (
            series_config.as_dict()
            if candidate_mode == "pseudo_double_series"
            else None
        ),
        "candidate_split_config": (
            split_config.as_dict()
            if candidate_mode == "pseudo_double_split"
            else None
        ),
        "candidate_schur_local_config": (
            schur_config.as_dict()
            if candidate_mode == "pseudo_double_schur_local"
            else None
        ),
        "parameters": {
            "workload": "mrg_point_source_extracellular_validation",
            "backend": jax.default_backend(),
            "size": config.size,
            "nodes": config.nodes,
            "nt": int(round(config.duration_ms / config.dt_ms)),
            "diameter_um": config.diameter_um,
            "duration_ms": config.duration_ms,
            "dt_ms": config.dt_ms,
            "amplitudes_uA": [float(value) for value in amplitudes_uA],
            "pulse_start_ms": config.pulse_start_ms,
            "pulse_duration_ms": config.pulse_duration_ms,
            "electrode_z_um": config.electrode_z_um,
            "offset_span_um": config.offset_span_um,
            "activation_threshold_mV": config.activation_threshold_mV,
            "activation_blanking_ms": config.activation_blanking_ms,
            "double_cable_block_solver": config.double_cable_block_solver,
            "recording": config.recording,
            "probe_count": config.probe_count,
        },
        "amplitude_summaries": amplitude_summaries,
        "threshold_summary": summarize_thresholds(rows, size=config.size),
        "timings": timings,
        "rows": rows,
        "trace_samples": trace_samples,
    }


def score_validation_result(result: dict[str, object]) -> float:
    """Return a recall-first scalar score for pseudo-effective calibration."""

    summaries = result["amplitude_summaries"]
    threshold = result["threshold_summary"]
    if not isinstance(summaries, list) or not isinstance(threshold, dict):
        raise TypeError("validation result has an invalid summary structure.")
    false_negatives = sum(int(summary["false_negative_count"]) for summary in summaries)
    false_positives = sum(int(summary["false_positive_count"]) for summary in summaries)
    agreement_penalty = sum(
        1.0 - float(summary["activation_agreement"]) for summary in summaries
    )
    peak_errors = [
        float(summary["peak_abs_error_p95_mV"])
        for summary in summaries
        if summary["peak_abs_error_p95_mV"] is not None
    ]
    threshold_p95 = threshold.get("threshold_rel_error_p95")
    threshold_penalty = 0.0 if threshold_p95 is None else float(threshold_p95)
    missed_thresholds = int(threshold.get("missed_activation_count", 0))
    extra_thresholds = int(threshold.get("extra_activation_count", 0))
    peak_penalty = 0.0 if not peak_errors else float(np.mean(peak_errors)) / 100.0
    return (
        1000.0 * false_negatives
        + 300.0 * missed_thresholds
        + 100.0 * false_positives
        + 30.0 * extra_thresholds
        + 25.0 * agreement_penalty
        + 10.0 * threshold_penalty
        + peak_penalty
    )


def calibrate_pseudo_double_effective(
    *,
    amplitudes_uA: Sequence[float],
    config: RunConfig,
    scales: Sequence[float],
    reference_runs: ReferenceRunMap | None = None,
) -> tuple[PseudoDoubleEffectiveConfig, dict[str, object], ReferenceRunMap]:
    """Sweep vext_scale and choose the best recall-first effective config."""

    if not scales:
        raise ValueError("at least one calibration scale is required.")
    normalized_scales = tuple(float(scale) for scale in scales)
    if any(scale <= 0.0 for scale in normalized_scales):
        raise ValueError("calibration scales must be positive.")
    if reference_runs is None:
        reference_runs = _run_reference_sweep(
            reference_mode="exact_double",
            amplitudes_uA=amplitudes_uA,
            config=config,
        )

    trials = []
    for scale in normalized_scales:
        candidate_config = PseudoDoubleEffectiveConfig(vext_scale=scale)
        trial_result = run_validation(
            reference_mode="exact_double",
            candidate_mode="pseudo_double_effective",
            amplitudes_uA=amplitudes_uA,
            config=config,
            effective_config=candidate_config,
            reference_runs=reference_runs,
        )
        score = score_validation_result(trial_result)
        trials.append(
            {
                "vext_scale": scale,
                "score": score,
                "amplitude_summaries": trial_result["amplitude_summaries"],
                "threshold_summary": trial_result["threshold_summary"],
            }
        )
    best = min(trials, key=lambda item: (float(item["score"]), float(item["vext_scale"])))
    selected = PseudoDoubleEffectiveConfig(vext_scale=float(best["vext_scale"]))
    return (
        selected,
        {
            "method": "grid_search_vext_scale_v0",
            "objective": "recall_first_activation_threshold_peak_error",
            "selected": selected.as_dict(),
            "trials": trials,
        },
        reference_runs,
    )


def calibrate_pseudo_double_single_chain(
    *,
    amplitudes_uA: Sequence[float],
    config: RunConfig,
    scales: Sequence[float],
    single_chain_config: PseudoDoubleSingleChainConfig | None = None,
    reference_runs: ReferenceRunMap | None = None,
) -> tuple[PseudoDoubleSingleChainConfig, dict[str, object], ReferenceRunMap]:
    """Sweep single-chain vext_scale and choose the best recall-first config."""

    if not scales:
        raise ValueError("at least one calibration scale is required.")
    normalized_scales = tuple(float(scale) for scale in scales)
    if any(scale <= 0.0 for scale in normalized_scales):
        raise ValueError("calibration scales must be positive.")
    single_chain_config = single_chain_config or PseudoDoubleSingleChainConfig()
    if reference_runs is None:
        reference_runs = _run_reference_sweep(
            reference_mode="exact_double",
            amplitudes_uA=amplitudes_uA,
            config=config,
        )

    trials = []
    for scale in normalized_scales:
        candidate_config = replace(single_chain_config, vext_scale=scale)
        trial_result = run_validation(
            reference_mode="exact_double",
            candidate_mode="pseudo_double_single_myelinated_chain",
            amplitudes_uA=amplitudes_uA,
            config=config,
            single_chain_config=candidate_config,
            reference_runs=reference_runs,
        )
        score = score_validation_result(trial_result)
        trials.append(
            {
                "vext_scale": scale,
                "score": score,
                "single_chain_config": candidate_config.as_dict(),
                "amplitude_summaries": trial_result["amplitude_summaries"],
                "threshold_summary": trial_result["threshold_summary"],
            }
        )
    best = min(trials, key=lambda item: (float(item["score"]), float(item["vext_scale"])))
    selected = replace(single_chain_config, vext_scale=float(best["vext_scale"]))
    return (
        selected,
        {
            "method": "grid_search_single_chain_vext_scale_v0",
            "objective": "recall_first_activation_threshold_peak_error",
            "selected": selected.as_dict(),
            "trials": trials,
        },
        reference_runs,
    )


def calibrate_pseudo_double_series(
    *,
    amplitudes_uA: Sequence[float],
    config: RunConfig,
    scales: Sequence[float],
    series_config: PseudoDoubleSeriesConfig | None = None,
    reference_runs: ReferenceRunMap | None = None,
) -> tuple[PseudoDoubleSeriesConfig, dict[str, object], ReferenceRunMap]:
    """Sweep RC-series vext_scale and choose the best recall-first config."""

    if not scales:
        raise ValueError("at least one calibration scale is required.")
    normalized_scales = tuple(float(scale) for scale in scales)
    if any(scale <= 0.0 for scale in normalized_scales):
        raise ValueError("calibration scales must be positive.")
    series_config = series_config or PseudoDoubleSeriesConfig()
    if reference_runs is None:
        reference_runs = _run_reference_sweep(
            reference_mode="exact_double",
            amplitudes_uA=amplitudes_uA,
            config=config,
        )

    trials = []
    for scale in normalized_scales:
        candidate_config = replace(series_config, vext_scale=scale)
        trial_result = run_validation(
            reference_mode="exact_double",
            candidate_mode="pseudo_double_series",
            amplitudes_uA=amplitudes_uA,
            config=config,
            series_config=candidate_config,
            reference_runs=reference_runs,
        )
        score = score_validation_result(trial_result)
        trials.append(
            {
                "vext_scale": scale,
                "score": score,
                "series_config": candidate_config.as_dict(),
                "amplitude_summaries": trial_result["amplitude_summaries"],
                "threshold_summary": trial_result["threshold_summary"],
            }
        )
    best = min(trials, key=lambda item: (float(item["score"]), float(item["vext_scale"])))
    selected = replace(series_config, vext_scale=float(best["vext_scale"]))
    return (
        selected,
        {
            "method": "grid_search_series_vext_scale_v1",
            "objective": "recall_first_activation_threshold_peak_error",
            "selected": selected.as_dict(),
            "trials": trials,
        },
        reference_runs,
    )


def calibrate_pseudo_double_split(
    *,
    amplitudes_uA: Sequence[float],
    config: RunConfig,
    scales: Sequence[float],
    split_config: PseudoDoubleSplitConfig | None = None,
    reference_runs: ReferenceRunMap | None = None,
) -> tuple[PseudoDoubleSplitConfig, dict[str, object], ReferenceRunMap]:
    """Sweep split vext_scale and choose the best recall-first config."""

    if not scales:
        raise ValueError("at least one calibration scale is required.")
    normalized_scales = tuple(float(scale) for scale in scales)
    if any(scale <= 0.0 for scale in normalized_scales):
        raise ValueError("calibration scales must be positive.")
    split_config = split_config or PseudoDoubleSplitConfig()
    if reference_runs is None:
        reference_runs = _run_reference_sweep(
            reference_mode="exact_double",
            amplitudes_uA=amplitudes_uA,
            config=config,
        )

    trials = []
    for scale in normalized_scales:
        candidate_config = replace(split_config, vext_scale=scale)
        trial_result = run_validation(
            reference_mode="exact_double",
            candidate_mode="pseudo_double_split",
            amplitudes_uA=amplitudes_uA,
            config=config,
            split_config=candidate_config,
            reference_runs=reference_runs,
        )
        score = score_validation_result(trial_result)
        trials.append(
            {
                "vext_scale": scale,
                "score": score,
                "split_config": candidate_config.as_dict(),
                "amplitude_summaries": trial_result["amplitude_summaries"],
                "threshold_summary": trial_result["threshold_summary"],
            }
        )
    best = min(trials, key=lambda item: (float(item["score"]), float(item["vext_scale"])))
    selected = replace(split_config, vext_scale=float(best["vext_scale"]))
    return (
        selected,
        {
            "method": "grid_search_split_vext_scale_v0",
            "objective": "recall_first_activation_threshold_peak_error",
            "selected": selected.as_dict(),
            "trials": trials,
        },
        reference_runs,
    )


def calibrate_pseudo_double_schur_local(
    *,
    amplitudes_uA: Sequence[float],
    config: RunConfig,
    scales: Sequence[float],
    schur_config: PseudoDoubleSchurLocalConfig | None = None,
    reference_runs: ReferenceRunMap | None = None,
) -> tuple[PseudoDoubleSchurLocalConfig, dict[str, object], ReferenceRunMap]:
    """Sweep Schur-local vext_scale and choose the best recall-first config."""

    if not scales:
        raise ValueError("at least one calibration scale is required.")
    normalized_scales = tuple(float(scale) for scale in scales)
    if any(scale <= 0.0 for scale in normalized_scales):
        raise ValueError("calibration scales must be positive.")
    schur_config = schur_config or PseudoDoubleSchurLocalConfig()
    if reference_runs is None:
        reference_runs = _run_reference_sweep(
            reference_mode="exact_double",
            amplitudes_uA=amplitudes_uA,
            config=config,
        )

    trials = []
    for scale in normalized_scales:
        candidate_config = replace(schur_config, vext_scale=scale)
        trial_result = run_validation(
            reference_mode="exact_double",
            candidate_mode="pseudo_double_schur_local",
            amplitudes_uA=amplitudes_uA,
            config=config,
            schur_config=candidate_config,
            reference_runs=reference_runs,
        )
        score = score_validation_result(trial_result)
        trials.append(
            {
                "vext_scale": scale,
                "score": score,
                "schur_local_config": candidate_config.as_dict(),
                "amplitude_summaries": trial_result["amplitude_summaries"],
                "threshold_summary": trial_result["threshold_summary"],
            }
        )
    best = min(trials, key=lambda item: (float(item["score"]), float(item["vext_scale"])))
    selected = replace(schur_config, vext_scale=float(best["vext_scale"]))
    return (
        selected,
        {
            "method": "grid_search_schur_local_vext_scale_v1",
            "objective": "recall_first_activation_threshold_peak_error",
            "selected": selected.as_dict(),
            "trials": trials,
        },
        reference_runs,
    )


def write_outputs(result: dict[str, object], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    csv_path = out_dir / "rows.csv"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    rows = result["rows"]
    if not isinstance(rows, list):
        raise TypeError("result rows must be a list.")
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive.")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative.")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive.")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative.")
    return parsed


def _default_plot_trace_amplitudes(amplitudes_uA: Sequence[float]) -> tuple[float, ...]:
    ordered = tuple(sorted({float(value) for value in amplitudes_uA}))
    if len(ordered) <= 3:
        return ordered
    return (ordered[0], ordered[len(ordered) // 2], ordered[-1])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="exact_double", choices=VALIDATION_MODES)
    parser.add_argument(
        "--candidate",
        default="mrg_single_cable_surrogate",
        choices=VALIDATION_MODES,
    )
    parser.add_argument("--size", type=_positive_int, default=2)
    parser.add_argument("--nodes", type=_positive_int, default=3)
    parser.add_argument("--diameter-um", type=_positive_float, default=5.7)
    parser.add_argument("--duration", type=_positive_float, default=0.3)
    parser.add_argument("--dt", type=_positive_float, default=0.05)
    parser.add_argument(
        "--amplitudes-uA",
        type=float,
        nargs="+",
        default=(20.0, 60.0, 100.0),
    )
    parser.add_argument("--pulse-start-ms", type=float, default=0.10)
    parser.add_argument("--pulse-duration-ms", type=_positive_float, default=0.10)
    parser.add_argument("--electrode-z-um", type=_positive_float, default=120.0)
    parser.add_argument("--offset-span-um", type=float, default=40.0)
    parser.add_argument("--activation-threshold-mV", type=float, default=-20.0)
    parser.add_argument("--activation-blanking-ms", type=float, default=0.0)
    parser.add_argument(
        "--double-cable-block-solver",
        choices=("auto", "thomas", "pcr", "pcr_soa", "pcr_adaptive"),
        default="auto",
    )
    parser.add_argument(
        "--recording",
        choices=("full", "center", "probes"),
        default="full",
    )
    parser.add_argument("--probe-count", type=_positive_int, default=8)
    parser.add_argument(
        "--pseudo-vext-scale",
        type=_positive_float,
        default=1.0,
        help=(
            "Scalar extracellular coupling multiplier for "
            "pseudo_double_effective v0."
        ),
    )
    parser.add_argument(
        "--calibrate-vext-scales",
        type=_positive_float,
        nargs="+",
        default=None,
        help=(
            "Grid-search pseudo_double_effective, pseudo_double_single_myelinated_chain, "
            "pseudo_double_series, pseudo_double_split, or "
            "pseudo_double_schur_local vext_scale values before the final "
            "validation run."
        ),
    )
    parser.add_argument(
        "--single-chain-vext-scale",
        type=_positive_float,
        default=1.0,
        help="Global extracellular alpha multiplier for pseudo_double_single_myelinated_chain.",
    )
    for segment in ("node", "mysa", "flut", "stin"):
        parser.add_argument(
            f"--single-chain-cm-scale-{segment}",
            type=_positive_float,
            default=1.0,
            help=f"Cm multiplier for pseudo-chain {segment.upper()} compartments.",
        )
        parser.add_argument(
            f"--single-chain-gleak-scale-{segment}",
            type=_positive_float,
            default=1.0,
            help=f"Leak multiplier for pseudo-chain {segment.upper()} compartments.",
        )
        parser.add_argument(
            f"--single-chain-alpha-{segment}",
            type=_nonnegative_float,
            default=1.0,
            help=f"Extracellular alpha for pseudo-chain {segment.upper()} compartments.",
        )
    parser.add_argument(
        "--single-chain-axial-resistance-scale",
        type=_positive_float,
        default=1.0,
        help="Axial resistance multiplier for pseudo_double_single_myelinated_chain.",
    )
    parser.add_argument(
        "--single-chain-series-capacitance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use axolemma/myelin series capacitance for pseudo-chain internodes.",
    )
    parser.add_argument(
        "--single-chain-series-leak",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use axolemma/myelin series leak for pseudo-chain internodes.",
    )
    parser.add_argument(
        "--series-vext-scale",
        type=_positive_float,
        default=1.0,
        help="Extracellular multiplier for pseudo_double_series v1.",
    )
    parser.add_argument(
        "--series-capacitance-floor-fraction",
        type=_nonnegative_float,
        default=0.02,
        help=(
            "Minimum Ceff/Cm fallback for non-node compartments in "
            "pseudo_double_series v1."
        ),
    )
    parser.add_argument(
        "--series-conductance-floor-fraction",
        type=_nonnegative_float,
        default=0.0,
        help=(
            "Minimum Geff/Gm fallback for non-node compartments in "
            "pseudo_double_series v1."
        ),
    )
    parser.add_argument(
        "--split-vext-scale",
        type=_positive_float,
        default=1.0,
        help="Overall extracellular multiplier for pseudo_double_split v0.",
    )
    parser.add_argument(
        "--split-direct-scale",
        type=float,
        default=1.0,
        help="Direct raw extracellular drive weight for pseudo_double_split v0.",
    )
    parser.add_argument(
        "--split-aux-scale",
        type=float,
        default=1.0,
        help="Auxiliary filtered drive weight for pseudo_double_split v0.",
    )
    parser.add_argument(
        "--split-aux-alpha",
        type=float,
        default=1.0,
        help="Auxiliary filter target multiplier for pseudo_double_split v0.",
    )
    parser.add_argument(
        "--split-aux-tau-ms",
        type=_positive_float,
        default=0.05,
        help="Auxiliary implicit low-pass time constant for pseudo_double_split v0.",
    )
    parser.add_argument(
        "--schur-vext-scale",
        type=_positive_float,
        default=1.0,
        help="Extracellular RHS multiplier for pseudo_double_schur_local v1.",
    )
    parser.add_argument(
        "--schur-app-inverse-scale",
        type=_positive_float,
        default=1.0,
        help="Multiplier applied to the local diagonal App inverse for Schur v1.",
    )
    parser.add_argument(
        "--include-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also compare exact double-cable against the baseline "
            "mrg_single_cable_surrogate when the main candidate differs."
        ),
    )
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write validation PNG plots under OUT_DIR/plots.",
    )
    parser.add_argument(
        "--plot-trace-rows",
        type=_nonnegative_int,
        nargs="+",
        default=(0,),
        help="Population rows to include in trace-comparison plots when --plots is set.",
    )
    parser.add_argument(
        "--plot-trace-amplitudes-uA",
        type=_positive_float,
        nargs="+",
        default=None,
        help=(
            "Amplitudes to include in trace-comparison plots. Default: first, "
            "middle, and last amplitude from --amplitudes-uA."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("benchmark/results/pseudo_double"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-summary", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    amplitudes = tuple(float(value) for value in args.amplitudes_uA)
    if any(value <= 0.0 for value in amplitudes):
        parser.error("--amplitudes-uA values must be positive.")
    config = RunConfig(
        size=int(args.size),
        nodes=int(args.nodes),
        diameter_um=float(args.diameter_um),
        duration_ms=float(args.duration),
        dt_ms=float(args.dt),
        pulse_start_ms=float(args.pulse_start_ms),
        pulse_duration_ms=float(args.pulse_duration_ms),
        electrode_z_um=float(args.electrode_z_um),
        offset_span_um=float(args.offset_span_um),
        activation_threshold_mV=float(args.activation_threshold_mV),
        activation_blanking_ms=float(args.activation_blanking_ms),
        double_cable_block_solver=str(args.double_cable_block_solver),
        recording=str(args.recording),
        probe_count=int(args.probe_count),
    )
    reference = normalize_validation_mode(args.reference)
    candidate = normalize_validation_mode(args.candidate)
    if args.calibrate_vext_scales is not None and candidate not in {
        "pseudo_double_effective",
        "pseudo_double_single_myelinated_chain",
        "pseudo_double_series",
        "pseudo_double_split",
        "pseudo_double_schur_local",
    }:
        parser.error(
            "--calibrate-vext-scales is only valid with pseudo_double_effective "
            "or pseudo_double_single_myelinated_chain or pseudo_double_series "
            "or pseudo_double_split "
            "or pseudo_double_schur_local."
        )
    effective_config = PseudoDoubleEffectiveConfig(
        vext_scale=float(args.pseudo_vext_scale)
    )
    single_chain_config = PseudoDoubleSingleChainConfig(
        vext_scale=float(args.single_chain_vext_scale),
        cm_scale_node=float(args.single_chain_cm_scale_node),
        cm_scale_mysa=float(args.single_chain_cm_scale_mysa),
        cm_scale_flut=float(args.single_chain_cm_scale_flut),
        cm_scale_stin=float(args.single_chain_cm_scale_stin),
        gleak_scale_node=float(args.single_chain_gleak_scale_node),
        gleak_scale_mysa=float(args.single_chain_gleak_scale_mysa),
        gleak_scale_flut=float(args.single_chain_gleak_scale_flut),
        gleak_scale_stin=float(args.single_chain_gleak_scale_stin),
        axial_resistance_scale=float(args.single_chain_axial_resistance_scale),
        vext_alpha_node=float(args.single_chain_alpha_node),
        vext_alpha_mysa=float(args.single_chain_alpha_mysa),
        vext_alpha_flut=float(args.single_chain_alpha_flut),
        vext_alpha_stin=float(args.single_chain_alpha_stin),
        use_series_capacitance=bool(args.single_chain_series_capacitance),
        use_series_leak=bool(args.single_chain_series_leak),
    )
    series_config = PseudoDoubleSeriesConfig(
        vext_scale=float(args.series_vext_scale),
        capacitance_floor_fraction=float(args.series_capacitance_floor_fraction),
        conductance_floor_fraction=float(args.series_conductance_floor_fraction),
    )
    split_config = PseudoDoubleSplitConfig(
        vext_scale=float(args.split_vext_scale),
        direct_scale=float(args.split_direct_scale),
        aux_scale=float(args.split_aux_scale),
        aux_alpha=float(args.split_aux_alpha),
        aux_tau_ms=float(args.split_aux_tau_ms),
    )
    schur_config = PseudoDoubleSchurLocalConfig(
        vext_scale=float(args.schur_vext_scale),
        app_inverse_scale=float(args.schur_app_inverse_scale),
    )
    if args.dry_run:
        extras = []
        if candidate == "pseudo_double_effective":
            extras.append(f"pseudo_vext_scale={effective_config.vext_scale}")
        if candidate == "pseudo_double_single_myelinated_chain":
            extras.append(f"single_chain_vext_scale={single_chain_config.vext_scale}")
            extras.append(
                "single_chain_alpha="
                f"{single_chain_config.vext_alpha_node},"
                f"{single_chain_config.vext_alpha_mysa},"
                f"{single_chain_config.vext_alpha_flut},"
                f"{single_chain_config.vext_alpha_stin}"
            )
        if candidate == "pseudo_double_series":
            extras.append(f"series_vext_scale={series_config.vext_scale}")
            extras.append(
                "series_capacitance_floor_fraction="
                f"{series_config.capacitance_floor_fraction}"
            )
        if candidate == "pseudo_double_split":
            extras.append(f"split_vext_scale={split_config.vext_scale}")
            extras.append(f"split_aux_tau_ms={split_config.aux_tau_ms}")
        if candidate == "pseudo_double_schur_local":
            extras.append(f"schur_vext_scale={schur_config.vext_scale}")
            extras.append(f"schur_app_inverse_scale={schur_config.app_inverse_scale}")
        if args.calibrate_vext_scales is not None:
            scales = ",".join(str(float(value)) for value in args.calibrate_vext_scales)
            extras.append(f"calibrate_vext_scales={scales}")
        if bool(args.include_baseline) and candidate != "mrg_single_cable_surrogate":
            extras.append("include_baseline=true")
        if bool(args.plots):
            extras.append("plots=true")
        suffix = "" if not extras else " " + " ".join(extras)
        print(
            "pseudo_double_validation "
            f"reference={reference} candidate={candidate} "
            f"size={config.size} nodes={config.nodes} "
            f"amplitudes_uA={','.join(str(value) for value in amplitudes)}"
            f"{suffix}"
        )
        return
    reference_runs: ReferenceRunMap | None = None
    calibration: dict[str, object] | None = None
    if candidate == "pseudo_double_effective" and args.calibrate_vext_scales is not None:
        effective_config, calibration, reference_runs = calibrate_pseudo_double_effective(
            amplitudes_uA=amplitudes,
            config=config,
            scales=args.calibrate_vext_scales,
        )
    elif (
        candidate == "pseudo_double_single_myelinated_chain"
        and args.calibrate_vext_scales is not None
    ):
        single_chain_config, calibration, reference_runs = calibrate_pseudo_double_single_chain(
            amplitudes_uA=amplitudes,
            config=config,
            scales=args.calibrate_vext_scales,
            single_chain_config=single_chain_config,
        )
    elif candidate == "pseudo_double_series" and args.calibrate_vext_scales is not None:
        series_config, calibration, reference_runs = calibrate_pseudo_double_series(
            amplitudes_uA=amplitudes,
            config=config,
            scales=args.calibrate_vext_scales,
            series_config=series_config,
        )
    elif candidate == "pseudo_double_split" and args.calibrate_vext_scales is not None:
        split_config, calibration, reference_runs = calibrate_pseudo_double_split(
            amplitudes_uA=amplitudes,
            config=config,
            scales=args.calibrate_vext_scales,
            split_config=split_config,
        )
    elif candidate == "pseudo_double_schur_local" and args.calibrate_vext_scales is not None:
        schur_config, calibration, reference_runs = calibrate_pseudo_double_schur_local(
            amplitudes_uA=amplitudes,
            config=config,
            scales=args.calibrate_vext_scales,
            schur_config=schur_config,
        )
    elif bool(args.include_baseline) and candidate != "mrg_single_cable_surrogate":
        reference_runs = _run_reference_sweep(
            reference_mode=reference,
            amplitudes_uA=amplitudes,
            config=config,
        )
    trace_sample_rows: tuple[int, ...] = ()
    trace_sample_amplitudes: tuple[float, ...] = ()
    if bool(args.plots):
        trace_sample_rows = tuple(int(row) for row in args.plot_trace_rows)
        if any(row >= config.size for row in trace_sample_rows):
            parser.error("--plot-trace-rows values must be smaller than --size.")
        trace_sample_amplitudes = (
            tuple(float(value) for value in args.plot_trace_amplitudes_uA)
            if args.plot_trace_amplitudes_uA is not None
            else _default_plot_trace_amplitudes(amplitudes)
        )
    result = run_validation(
        reference_mode=reference,
        candidate_mode=candidate,
        amplitudes_uA=amplitudes,
        config=config,
        effective_config=effective_config,
        single_chain_config=single_chain_config,
        series_config=series_config,
        split_config=split_config,
        schur_config=schur_config,
        reference_runs=reference_runs,
        trace_sample_rows=trace_sample_rows,
        trace_sample_amplitudes_uA=trace_sample_amplitudes,
    )
    if calibration is not None:
        result["calibration"] = calibration
    if bool(args.include_baseline) and candidate != "mrg_single_cable_surrogate":
        result["baseline_comparison"] = run_validation(
            reference_mode=reference,
            candidate_mode="mrg_single_cable_surrogate",
            amplitudes_uA=amplitudes,
            config=config,
            reference_runs=reference_runs,
        )
    json_path, csv_path = write_outputs(result, args.out_dir)
    plot_paths: tuple[Path, ...] = ()
    if bool(args.plots):
        from benchmark.pseudo_double.plotting import write_validation_plots

        plot_paths = write_validation_plots(result, args.out_dir)
    if args.print_summary:
        threshold_summary = result["threshold_summary"]
        print(f"summary: {json_path}")
        print(f"rows: {csv_path}")
        if plot_paths:
            print(f"plots: {args.out_dir / 'plots'} ({len(plot_paths)} files)")
        candidate_effective_config = result.get("candidate_effective_config")
        if isinstance(candidate_effective_config, dict):
            print(
                "pseudo_effective_vext_scale="
                f"{candidate_effective_config.get('vext_scale')}"
            )
        candidate_single_chain_config = result.get("candidate_single_chain_config")
        if isinstance(candidate_single_chain_config, dict):
            print(
                "pseudo_single_chain_vext_scale="
                f"{candidate_single_chain_config.get('vext_scale')} "
                "alpha_node_mysa_flut_stin="
                f"{candidate_single_chain_config.get('vext_alpha_node')},"
                f"{candidate_single_chain_config.get('vext_alpha_mysa')},"
                f"{candidate_single_chain_config.get('vext_alpha_flut')},"
                f"{candidate_single_chain_config.get('vext_alpha_stin')}"
            )
        candidate_series_config = result.get("candidate_series_config")
        if isinstance(candidate_series_config, dict):
            print(
                "pseudo_series_vext_scale="
                f"{candidate_series_config.get('vext_scale')} "
                "capacitance_floor_fraction="
                f"{candidate_series_config.get('capacitance_floor_fraction')}"
            )
        candidate_split_config = result.get("candidate_split_config")
        if isinstance(candidate_split_config, dict):
            print(
                "pseudo_split_vext_scale="
                f"{candidate_split_config.get('vext_scale')} "
                f"aux_tau_ms={candidate_split_config.get('aux_tau_ms')}"
            )
        candidate_schur_config = result.get("candidate_schur_local_config")
        if isinstance(candidate_schur_config, dict):
            print(
                "pseudo_schur_vext_scale="
                f"{candidate_schur_config.get('vext_scale')} "
                f"app_inverse_scale={candidate_schur_config.get('app_inverse_scale')}"
            )
        calibration_summary = result.get("calibration")
        if isinstance(calibration_summary, dict):
            selected = calibration_summary.get("selected")
            trials = calibration_summary.get("trials")
            if isinstance(selected, dict) and isinstance(trials, list):
                print(
                    "calibration_selected_vext_scale="
                    f"{selected.get('vext_scale')} trials={len(trials)}"
                )
        print(
            "threshold_rel_error_mean="
            f"{_maybe_float(threshold_summary.get('threshold_rel_error_mean'))}"
        )
        baseline_comparison = result.get("baseline_comparison")
        if isinstance(baseline_comparison, dict):
            baseline_threshold = baseline_comparison.get("threshold_summary")
            if isinstance(baseline_threshold, dict):
                print(
                    "baseline_threshold_rel_error_mean="
                    f"{_maybe_float(baseline_threshold.get('threshold_rel_error_mean'))}"
                )
        for summary in result["amplitude_summaries"]:
            print(
                f"amplitude {summary['amplitude_uA']:g} uA: "
                f"agreement={summary['activation_agreement']:.3f} "
                f"false_negatives={summary['false_negative_count']} "
                f"peak_error_mean_mV={summary['peak_abs_error_mean_mV']}"
            )


if __name__ == "__main__":
    main()
