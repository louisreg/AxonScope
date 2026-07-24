"""Validate canonical full membrane recordings across JAX CPU and GPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import axonfleet as axs


SIGNALS = (
    axs.signals.Vm,
    axs.signals.GATES,
    axs.signals.CURRENTS,
    axs.signals.CONDUCTANCES,
    axs.signals.STATE_VARIABLES,
    axs.signals.MARKOV_OCCUPANCIES,
)
TOLERANCES = {
    "Vm": (2e-4, 2e-3),
    "gates": (2e-4, 2e-5),
    "currents": (5e-4, 5e-3),
    "conductances": (5e-4, 5e-4),
    "states": (2e-4, 2e-5),
    "occupancies": (2e-4, 2e-5),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="gpu_smoke")
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--case-filter")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--memory-trace", default="off")
    args, _ = parser.parse_known_args(argv)

    cases = _cases(args.case_filter)
    if args.dry_run:
        print("\n".join(cases))
        return 0
    if args.platform != "gpu":
        raise SystemExit("membrane_recording_validation requires a GPU platform.")

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    failed = False
    for name, factory in cases.items():
        print(f"Validating {name} full recording on CPU/GPU...")
        cpu_result, cpu_s = _run(factory(), axs.Device.cpu())
        gpu_result, gpu_s = _run(factory(), axs.Device.gpu(0))
        row, arrays = _compare(name, cpu_result, gpu_result)
        row["cpu_s"] = cpu_s
        row["gpu_s"] = gpu_s
        row["gpu_speedup"] = cpu_s / gpu_s if gpu_s > 0.0 else None
        rows.append(row)
        np.savez_compressed(args.output / f"{name}.npz", **arrays)
        failed |= row["status"] != "pass"

    payload = {
        "status": "fail" if failed else "pass",
        "platform": args.platform,
        "precision": "float32",
        "rows": rows,
    }
    (args.output / "validation.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    print(f"AxonFleet membrane recording validation: {args.output}")
    return 1 if failed else 0


def _run(simulation: axs.AxonSimulation, device: axs.Device):
    policy = axs.ExecutionPolicy(
        runtime=axs.runtime.jax,
        device=device,
        precision=axs.PrecisionPolicy.float32(),
    )
    start = perf_counter()
    simulation.execution_policy = policy
    result = simulation.run().single
    elapsed = perf_counter() - start
    return result, elapsed


def _compare(name: str, cpu, gpu):
    comparisons = []
    arrays: dict[str, np.ndarray] = {}
    failed = False
    for signal in SIGNALS:
        try:
            cpu_values = cpu.signal(signal)
        except KeyError:
            cpu_values = None
        try:
            gpu_values = gpu.signal(signal)
        except KeyError:
            gpu_values = None
        if cpu_values is None and gpu_values is None:
            continue
        if cpu_values is None or gpu_values is None:
            comparisons.append(
                {
                    "signal": signal.result_key,
                    "status": "fail",
                    "reason": "signal availability differs",
                }
            )
            failed = True
            continue
        if isinstance(cpu_values, dict):
            if set(cpu_values) != set(gpu_values):
                comparisons.append(
                    {
                        "signal": signal.result_key,
                        "status": "fail",
                        "reason": "recorded names differ",
                        "cpu_names": sorted(cpu_values),
                        "gpu_names": sorted(gpu_values),
                    }
                )
                failed = True
                continue
            items = ((key, cpu_values[key], gpu_values[key]) for key in sorted(cpu_values))
        else:
            items = ((signal.result_key, cpu_values, gpu_values),)
        rtol, atol = TOLERANCES[signal.result_key]
        for key, cpu_array, gpu_array in items:
            cpu_array = np.asarray(cpu_array)
            gpu_array = np.asarray(gpu_array)
            token = f"{signal.result_key}.{key}".replace("/", "_")
            arrays[f"cpu.{token}"] = cpu_array
            arrays[f"gpu.{token}"] = gpu_array
            same_shape = cpu_array.shape == gpu_array.shape
            difference = np.abs(cpu_array - gpu_array) if same_shape else np.asarray([np.inf])
            max_abs = float(np.max(difference)) if difference.size else 0.0
            rmse = float(np.sqrt(np.mean(difference**2))) if difference.size else 0.0
            passed = bool(
                same_shape
                and np.allclose(cpu_array, gpu_array, rtol=rtol, atol=atol, equal_nan=True)
            )
            comparisons.append(
                {
                    "signal": signal.result_key,
                    "name": key,
                    "shape": list(cpu_array.shape),
                    "rmse": rmse,
                    "max_abs": max_abs,
                    "rtol": rtol,
                    "atol": atol,
                    "status": "pass" if passed else "fail",
                }
            )
            failed |= not passed
    return {
        "case": name,
        "status": "fail" if failed else "pass",
        "comparisons": comparisons,
    }, arrays


def _cases(case_filter: str | None):
    cases = {
        "hh": _hh,
        "tigerholm": _tigerholm,
        "schild94": _schild94,
        "schild97": _schild97,
        "mrg_markov": _mrg_markov,
    }
    if case_filter:
        cases = {name: factory for name, factory in cases.items() if case_filter in name}
    if not cases:
        raise SystemExit(f"No recording validation case matches {case_filter!r}.")
    return cases


def _single_cable(axon, *, duration_ms: float, amplitude_nA: float):
    instance = axs.AxonInstance(axon)
    positions_um = axon.layout.position_values(unit="micrometer")
    instance.add_current_clamp(
        position=float(positions_um[len(positions_um) // 2]) * axs.um,
        current=axs.Stimulus.pulse(
            start=0.5 * axs.ms,
            duration=0.2 * axs.ms,
            amplitude=amplitude_nA * axs.nA,
        ),
    )
    return axs.AxonSimulation(
        instance,
        duration=duration_ms * axs.ms,
        dt=0.005 * axs.ms,
        recording=axs.Recording.full(),
        progress=False,
    )


def _hh():
    return _single_cable(
        axs.axons.HodgkinHuxley(
            length=1000 * axs.um,
            diameter=1 * axs.um,
            compartments=31,
            celsius=6.3 * axs.degC,
        ),
        duration_ms=2.0,
        amplitude_nA=2.0,
    )


def _tigerholm():
    return _single_cable(
        axs.axons.Tigerholm(
            length=1000 * axs.um,
            diameter=1 * axs.um,
            compartments=31,
        ),
        duration_ms=2.0,
        amplitude_nA=2.0,
    )


def _schild94():
    return _single_cable(
        axs.axons.Schild94(
            length=1000 * axs.um,
            diameter=0.8 * axs.um,
            compartments=31,
        ),
        duration_ms=2.0,
        amplitude_nA=1.0,
    )


def _schild97():
    return _single_cable(
        axs.axons.Schild97(
            length=1000 * axs.um,
            diameter=0.8 * axs.um,
            compartments=31,
        ),
        duration_ms=2.0,
        amplitude_nA=1.0,
    )


def _mrg_markov():
    template = axs.axons.MRGLikeDoubleCableTemplate(
        diameter=10.0 * axs.um,
        nodes=5,
    )
    defaults = template.default_membranes()
    node = axs.membranes.Composite(
        {
            "mrg_k_leak": axs.membranes.AxNode(
                gnapbar=0.0 * axs.mS_per_cm2,
                gnabar=0.0 * axs.mS_per_cm2,
            ),
            "nav11": axs.membranes.Nav11(
                gbar=11_900.0 * axs.mS_per_cm2,
                ena=50.0 * axs.mV,
                celsius=37.0 * axs.degC,
            ),
            "nav16": axs.membranes.Nav16(
                gbar=10.0 * axs.mS_per_cm2,
                ena=50.0 * axs.mV,
                celsius=37.0 * axs.degC,
            ),
        }
    )
    axon = axs.axons.MRG(
        diameter=10.0 * axs.um,
        nodes=5,
        membranes=axs.membranes.SectionLayout(
            node=node,
            mysa=defaults.membrane_for("MYSA"),
            flut=defaults.membrane_for("FLUT"),
            stin=defaults.membrane_for("STIN"),
        ),
    )
    instance = axs.AxonInstance(axon)
    center_index = int(axon.node_indices[len(axon.node_indices) // 2])
    instance.add_current_clamp(
        position=(
            float(axon.layout.position_values(unit="micrometer")[center_index])
            * axs.um
        ),
        current=axs.Stimulus.pulse(
            start=0.5 * axs.ms,
            duration=0.1 * axs.ms,
            amplitude=5.0 * axs.nA,
        ),
    )
    return axs.AxonSimulation(
        instance,
        duration=1.0 * axs.ms,
        dt=0.005 * axs.ms,
        recording=axs.Recording.full(),
        progress=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
