"""Solver-side axon arrays derived from descriptive layouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from axonscope.axons.flattened import FlattenedLayout, flatten_layout
from axonscope.axons.formulation import Formulation, infer_formulation, resolve_formulation
from axonscope.membranes import MembraneModel

if TYPE_CHECKING:
    from axonscope.axon_simulation import AxonSimulation
    from axonscope.axons.axon import Axon


def _all_same(values: np.ndarray) -> bool:
    return bool(np.allclose(values, values[0])) if values.shape[0] else True


def _uniform_periaxonal(
    *,
    Nx: int,
    dtype: np.dtype,
    xraxial_MOhm_per_cm: float,
    xg_S_cm2: float,
    xc_uF_cm2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.full((Nx,), float(xraxial_MOhm_per_cm), dtype=dtype),
        np.full((Nx,), float(xg_S_cm2), dtype=dtype),
        np.full((Nx,), float(xc_uF_cm2), dtype=dtype),
    )


@dataclass(frozen=True)
class SolverAxon:
    """Immutable NumPy arrays consumed by solver preparation.

    This is not part of the descriptive axon model. It is the numerical runtime
    representation built at the solver boundary from ``axon.layout`` plus any
    simulation-level extracellular overrides.
    """

    formulation: Formulation
    dtype: np.dtype
    x_um: np.ndarray
    n_compartments: int
    length_um: float
    compartment_lengths_um: np.ndarray
    dx_cm: np.ndarray
    h_um: np.ndarray
    h_cm: np.ndarray
    diam_um: np.ndarray
    Ra_ohm_cm: np.ndarray
    Cm_uF_cm2: np.ndarray
    membrane_models: tuple[MembraneModel, ...]
    section_names: tuple[str, ...]
    section_indices: np.ndarray
    section_tags: tuple[tuple[str, ...], ...]
    xraxial_MOhm_per_cm: np.ndarray
    xg_S_cm2: np.ndarray
    xc_uF_cm2: np.ndarray
    has_heterogeneous_cable_properties: bool

    @property
    def is_double_cable(self) -> bool:
        """Whether this solver axon uses the double-cable formulation."""

        return self.formulation == "double-cable"


def _overrides_complete(axon: "Axon | AxonSimulation") -> bool:
    return (
        getattr(axon, "_xraxial_override", None) is not None
        and getattr(axon, "_xg_override", None) is not None
        and getattr(axon, "_xc_override", None) is not None
    )


def _periaxonal_arrays(
    axon: "Axon | AxonSimulation",
    flat: FlattenedLayout,
    formulation: Formulation,
    *,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Nx = int(flat.Nx)
    if formulation == "double-cable":
        if any(layer is None for layer in flat.periaxonal_layers) and not _overrides_complete(axon):
            raise ValueError("Double-cable axons require periaxonal data on every section.")
        if all(layer is not None for layer in flat.periaxonal_layers):
            xraxial = np.asarray(
                [
                    layer.axial_resistance_MOhm_per_cm
                    for layer in flat.periaxonal_layers
                    if layer is not None
                ],
                dtype=dtype,
            )
            xg = np.asarray(
                [
                    layer.radial_conductance_S_cm2
                    for layer in flat.periaxonal_layers
                    if layer is not None
                ],
                dtype=dtype,
            )
            xc = np.asarray(
                [
                    layer.radial_capacitance_uF_cm2
                    for layer in flat.periaxonal_layers
                    if layer is not None
                ],
                dtype=dtype,
            )
        else:
            xraxial, xg, xc = _uniform_periaxonal(
                Nx=Nx,
                dtype=dtype,
                xraxial_MOhm_per_cm=1e9,
                xg_S_cm2=1e-6,
                xc_uF_cm2=0.0,
            )
    else:
        # Harmless defaults for single-cable imposed-field paths.
        xraxial, xg, xc = _uniform_periaxonal(
            Nx=Nx,
            dtype=dtype,
            xraxial_MOhm_per_cm=1e9,
            xg_S_cm2=1e10,
            xc_uF_cm2=0.0,
        )

    xraxial_override = getattr(axon, "_xraxial_override", None)
    xg_override = getattr(axon, "_xg_override", None)
    xc_override = getattr(axon, "_xc_override", None)
    if xraxial_override is not None:
        xraxial = xraxial_override.astype(dtype, copy=False)
    if xg_override is not None:
        xg = xg_override.astype(dtype, copy=False)
    if xc_override is not None:
        xc = xc_override.astype(dtype, copy=False)
    return xraxial, xg, xc


def build_solver_axon(axon: "Axon | AxonSimulation") -> SolverAxon:
    """Build solver-side arrays from an axon or axon simulation."""

    flat = flatten_layout(axon.layout)
    if flat.Nx < 2:
        raise ValueError(f"Axon layout requires at least 2 numerical compartments, got {flat.Nx}.")
    formulation = resolve_formulation(flat, getattr(axon, "formulation", None))
    dtype = np.dtype(flat.membrane_models[0].dtype)

    x_um = np.asarray(flat.x_um, dtype=dtype)
    compartment_lengths_um = np.asarray(flat.lengths_um, dtype=dtype)
    dx_cm = compartment_lengths_um * dtype.type(1e-4)
    h_um = np.diff(x_um)
    h_cm = h_um * dtype.type(1e-4)

    diam_um = np.asarray(flat.diam_um, dtype=dtype)
    Ra_ohm_cm = np.asarray(flat.Ra_ohm_cm, dtype=dtype)
    Cm_uF_cm2 = np.asarray(flat.Cm_uF_cm2, dtype=dtype)
    xraxial, xg, xc = _periaxonal_arrays(axon, flat, formulation, dtype=dtype)

    has_heterogeneous = (
        formulation == "double-cable"
        or not _all_same(diam_um)
        or not _all_same(Ra_ohm_cm)
        or not _all_same(Cm_uF_cm2)
    )
    return SolverAxon(
        formulation=formulation,
        dtype=dtype,
        x_um=x_um,
        n_compartments=int(flat.Nx),
        length_um=float(flat.length_um),
        compartment_lengths_um=compartment_lengths_um,
        dx_cm=dx_cm,
        h_um=h_um,
        h_cm=h_cm,
        diam_um=diam_um,
        Ra_ohm_cm=Ra_ohm_cm,
        Cm_uF_cm2=Cm_uF_cm2,
        membrane_models=tuple(flat.membrane_models),
        section_names=tuple(flat.section_names),
        section_indices=np.asarray(flat.section_indices, dtype=np.int32),
        section_tags=tuple(flat.section_tags),
        xraxial_MOhm_per_cm=xraxial,
        xg_S_cm2=xg,
        xc_uF_cm2=xc,
        has_heterogeneous_cable_properties=has_heterogeneous,
    )


__all__ = [
    "Formulation",
    "SolverAxon",
    "build_solver_axon",
    "infer_formulation",
    "resolve_formulation",
]
