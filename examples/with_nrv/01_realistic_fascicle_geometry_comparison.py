"""Run AxonScope on a simple NRV-generated four-fascicle nerve.

Run:
    python examples/with_nrv/01_realistic_fascicle_geometry_comparison.py

NRV owns the external geometry, fiber placement, and LIFE/FEM footprint
sampling. AxonScope receives intrinsic axon layouts plus sampled
`ExtracellularFootprint` objects, then runs the recruitment sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console

import axonscope as axs
from axonscope.integrations import nrv as axs_nrv


@dataclass(frozen=True)
class ExampleConfig:
    """Editable constants for the NRV-to-AxonScope recruitment example."""

    nerve_diameter_um: float = 1_000.0
    nerve_length_um: float = 10_000.0
    axons_per_fascicle: int = 20
    percent_unmyelinated: float = 0.7
    delta_trace_um: float = 10.0
    fascicle_diameter_um: float = 250.0
    fascicle_offset_um: float = 250.0
    include_unmyelinated: bool = True
    duration_ms: float = 3.0
    dt_ms: float = 0.001
    stimulus_start_ms: float = 0.1
    pulse_duration_ms: float = 0.1
    observer_time_chunk_steps: int | None = 1000
    solver_progress: bool | str = False
    recruitment_amplitudes_uA: tuple[float, ...] = tuple(
        float(value) for value in np.linspace(0.0, 300.0, 21)
    )
    activation_threshold_mV: float = 0.0
    unmyelinated_compartments: int = 0
    life_diameter_um: float = 25.0
    life_length_um: float = 1_000.0
    life_fascicle_id: str = "0"
    fem_n_proc: int | None = None
    gmsh_n_core: int | None = 1


def main(config: ExampleConfig | None = None) -> None:
    if config is None:
        config = ExampleConfig()
    console = Console(width=110)

    import nrv

    console.print("[bold]1. Build NRV geometry[/bold]")
    middle_amplitude_uA = config.recruitment_amplitudes_uA[
        len(config.recruitment_amplitudes_uA) // 2
    ]
    nerve = axs_nrv.build_synthetic_4_fascicle_nerve(
        nrv,
        nerve_length_um=config.nerve_length_um,
        nerve_diameter_um=config.nerve_diameter_um,
        fascicle_diameter_um=config.fascicle_diameter_um,
        fascicle_offset_um=config.fascicle_offset_um,
        axons_per_fascicle=config.axons_per_fascicle,
        percent_unmyelinated=config.percent_unmyelinated,
        delta_trace_um=config.delta_trace_um,
    )
    life_setup = axs_nrv.attach_life_fem_electrode(
        nrv,
        nerve,
        nerve_length_um=config.nerve_length_um,
        life_fascicle_id=config.life_fascicle_id,
        life_diameter_um=config.life_diameter_um,
        life_length_um=config.life_length_um,
        stimulus_start_ms=config.stimulus_start_ms,
        pulse_duration_ms=config.pulse_duration_ms,
        validation_current_uA=float(middle_amplitude_uA),
        fem_n_proc=config.fem_n_proc,
        gmsh_n_core=config.gmsh_n_core,
    )

    rows = axs_nrv.extract_fiber_rows(
        nerve,
        include_unmyelinated=config.include_unmyelinated,
    )
    console.print(
        f"NRV generated {len(nerve.fascicles)} fascicles and {len(rows)} simulated fibers."
    )

    console.print("[bold]2. Sample NRV LIFE/FEM footprints on AxonScope axons[/bold]")
    contexts = [
        axs_nrv.life_context_from_fiber_row(
            row,
            life_setup=life_setup,
            nerve_length_um=config.nerve_length_um,
            unmyelinated_compartments=config.unmyelinated_compartments,
        )
        for row in rows
    ]
    pool = tuple(
        axs_nrv.life_simulation_from_context(
            context,
            current_uA=0.0,
            start_ms=config.stimulus_start_ms,
            pulse_duration_ms=config.pulse_duration_ms,
        )
        for context in contexts
    )

    def update_life_current(simulation: axs.AxonInstance, current: object) -> None:
        axs_nrv.replace_life_current(
            simulation,
            current,
            start_ms=config.stimulus_start_ms,
            pulse_duration_ms=config.pulse_duration_ms,
        )

    console.print("[bold]3. Run AxonScope recruitment sweep[/bold]")
    activation = axs.analysis.ActivationCriterion(
        threshold=config.activation_threshold_mV * axs.mV,
        blanking=config.stimulus_start_ms * axs.ms,
        target=axs.positions.ALL,
    )
    curve = axs.protocols.recruitment_sweep(
        pool,
        update=update_life_current,
        amplitudes=np.asarray(config.recruitment_amplitudes_uA, dtype=float) * axs.uA,
        duration=config.duration_ms * axs.ms,
        dt=config.dt_ms * axs.ms,
        criterion=activation,
        recording=axs.Recording.none(),
        batch_options=axs.BatchOptions.none(
            time_chunk_steps=config.observer_time_chunk_steps
        ),
        progress=True,
        solver_progress=config.solver_progress,
    )

    console.print("[bold]4. Plot AxonScope recruitment by fascicle[/bold]")
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    row_fascicles = np.asarray([context.row.fascicle_id for context in contexts], dtype=object)
    activated = np.asarray(curve.activated, dtype=bool)
    for fascicle_id in sorted(set(row_fascicles.tolist()), key=_fascicle_sort_key):
        mask = row_fascicles == fascicle_id
        recruitment = np.sum(activated[:, mask], axis=1) / float(np.sum(mask))
        ax.plot(curve.amplitudes_uA, recruitment, marker="o", label=f"fasc {fascicle_id}")
    ax.set_xlabel("LIFE current amplitude [uA]")
    ax.set_ylabel("Recruitment fraction")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("AxonScope recruitment on NRV-generated fibers")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    plt.show()


def _fascicle_sort_key(fascicle_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(fascicle_id))
    except ValueError:
        return (1, str(fascicle_id))


if __name__ == "__main__":
    main()
