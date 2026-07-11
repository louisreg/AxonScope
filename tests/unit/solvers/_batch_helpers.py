from __future__ import annotations

import numpy as np

import axonscope as axs
from axonscope import AxonInstance
from axonscope.analytical import PointSourceElectrode
from axonscope.axons import HodgkinHuxley
from axonscope.runtime.jax.observer_runtime import finalize_vm_raster_state
from axonscope.runtime.jax.solver_engines.types import JaxSolverEngine
from axonscope.stimulation import Stimulus


DIAGNOSTIC_DOUBLE_CABLE_BLOCK_SOLVERS = ("pcr", "pcr_soa", "pcr_adaptive")


def diagnostic_double_cable_solver_engine(
    solver: str,
    *,
    platform: str = "cpu",
    allow_internal: bool = False,
    block_b: int | None = None,
) -> JaxSolverEngine:
    return JaxSolverEngine(
        name=f"diagnostic_{solver}",
        platform=platform,
        single_cable_solver="jax_tridiagonal",
        double_cable_block_solver=solver,
        allow_internal_double_cable_block_solver=allow_internal,
        tiled_thomas_block_b=block_b,
    )


def kernel_observations(out):
    if out.observations is not None:
        return out.observations
    pending = out.pending_observation
    assert pending is not None
    return finalize_vm_raster_state(
        pending.plan,
        pending.state,
        nt=pending.nt,
        dt_ms=pending.dt_ms,
        synchronize=True,
    )


def hh_extracellular_axon(*, current_clamp: bool = True) -> AxonInstance:
    axon = AxonInstance(
        HodgkinHuxley(
            length=400.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=41,
            celsius=6.3 * axs.degC,
        )
    )
    if current_clamp:
        axon.add_current_clamp(
            position=200.0 * axs.um,
            current=Stimulus.pulse(start=0.4 * axs.ms, duration=0.05 * axs.ms, amplitude=0.8),
        )
    electrode = PointSourceElectrode(
        x=200e-6 * axs.m,
        y=100e-6 * axs.m,
        z=100e-6 * axs.m,
    )
    stim = Stimulus.pulse(start=0.3 * axs.ms, amplitude=20e-6, duration=0.1 * axs.ms, baseline=0.0)
    axon.add_extracellular_stimulation(
        stimulation=axs.analytical.point_source_stimulation(
            electrode,
            axon.layout.position_values(unit=axs.um) * axs.um,
            stimulus=stim,
            sigma=0.3 * axs.S_per_m,
        ),
        replace=True,
    )
    return axon


def drive_footprint_for_positions(drive, x_positions_m) -> np.ndarray:
    footprint = drive.footprint
    values = np.asarray(footprint.values_for_axon(), dtype=float)
    x_um = np.asarray(x_positions_m, dtype=float) * 1e6
    support_um = np.asarray(footprint.positions_um, dtype=float)
    if x_um.shape == support_um.shape and np.allclose(x_um, support_um):
        return values
    return np.interp(x_um, support_um, values)
