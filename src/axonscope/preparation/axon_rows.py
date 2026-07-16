"""Backend-neutral NumPy materialization of solver-facing axon rows."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Sequence

import numpy as np

from axonscope.runtime.solver_axon import SolverAxon


@dataclass(frozen=True)
class MaterializedAxonRows:
    """Compact structure-of-arrays table for one collection of solver axons.

    Python axon descriptions remain the scientific authoring surface. This
    object is their backend-neutral numerical materialization: repeated
    ``SolverAxon`` objects occupy one template row and population rows select
    those templates through ``row_template_indices``.
    """

    row_template_indices: np.ndarray
    template_nx: np.ndarray
    has_heterogeneous_cable_properties: np.ndarray
    valid_mask: np.ndarray
    x_um: np.ndarray
    compartment_lengths_um: np.ndarray
    dx_cm: np.ndarray
    h_um: np.ndarray
    h_cm: np.ndarray
    diam_um: np.ndarray
    Ra_ohm_cm: np.ndarray
    Cm_uF_cm2: np.ndarray
    xraxial_MOhm_per_cm: np.ndarray
    xg_S_cm2: np.ndarray
    xc_uF_cm2: np.ndarray
    formulations: tuple[str, ...]
    membrane_models: tuple[tuple[Any, ...], ...]
    section_names: tuple[tuple[str, ...], ...]
    section_tags: tuple[tuple[tuple[str, ...], ...], ...]

    def __post_init__(self) -> None:
        arrays = (
            self.row_template_indices,
            self.template_nx,
            self.has_heterogeneous_cable_properties,
            self.valid_mask,
            self.x_um,
            self.compartment_lengths_um,
            self.dx_cm,
            self.h_um,
            self.h_cm,
            self.diam_um,
            self.Ra_ohm_cm,
            self.Cm_uF_cm2,
            self.xraxial_MOhm_per_cm,
            self.xg_S_cm2,
            self.xc_uF_cm2,
        )
        for array in arrays:
            np.asarray(array).setflags(write=False)

    @classmethod
    def from_solver_axons(
        cls,
        solver_axons: Sequence[SolverAxon],
        *,
        target_nx: int | None = None,
    ) -> "MaterializedAxonRows":
        """Materialize unique solver rows and population-to-template indices."""

        rows = tuple(solver_axons)
        if not rows:
            raise ValueError("solver_axons cannot be empty.")
        if any(not isinstance(row, SolverAxon) for row in rows):
            raise TypeError("solver_axons must contain only SolverAxon objects.")

        templates: list[SolverAxon] = []
        template_by_identity: dict[int, int] = {}
        row_indices = np.empty((len(rows),), dtype=np.int32)
        for row_index, row in enumerate(rows):
            identity = id(row)
            template_index = template_by_identity.get(identity)
            if template_index is None:
                template_index = len(templates)
                templates.append(row)
                template_by_identity[identity] = template_index
            row_indices[row_index] = template_index

        max_nx = max(int(row.n_compartments) for row in templates)
        resolved_nx = max_nx if target_nx is None else int(target_nx)
        if resolved_nx < max_nx:
            raise ValueError(
                f"target_nx must be >= the largest row ({max_nx}), got {resolved_nx}."
            )
        dtype = np.result_type(*(row.dtype for row in templates))
        template_nx = np.asarray(
            [int(row.n_compartments) for row in templates],
            dtype=np.int32,
        )
        valid_mask = np.arange(resolved_nx)[None, :] < template_nx[:, None]

        return cls(
            row_template_indices=row_indices,
            template_nx=template_nx,
            has_heterogeneous_cable_properties=np.asarray(
                [row.has_heterogeneous_cable_properties for row in templates],
                dtype=bool,
            ),
            valid_mask=valid_mask,
            x_um=_stack_space(templates, "x_um", resolved_nx, dtype=dtype, pad="edge"),
            compartment_lengths_um=_stack_space(
                templates,
                "compartment_lengths_um",
                resolved_nx,
                dtype=dtype,
            ),
            dx_cm=_stack_space(templates, "dx_cm", resolved_nx, dtype=dtype),
            h_um=_stack_edges(templates, "h_um", resolved_nx, dtype=dtype),
            h_cm=_stack_edges(templates, "h_cm", resolved_nx, dtype=dtype),
            diam_um=_stack_space(templates, "diam_um", resolved_nx, dtype=dtype),
            Ra_ohm_cm=_stack_space(templates, "Ra_ohm_cm", resolved_nx, dtype=dtype),
            Cm_uF_cm2=_stack_space(templates, "Cm_uF_cm2", resolved_nx, dtype=dtype),
            xraxial_MOhm_per_cm=_stack_space(
                templates,
                "xraxial_MOhm_per_cm",
                resolved_nx,
                dtype=dtype,
            ),
            xg_S_cm2=_stack_space(templates, "xg_S_cm2", resolved_nx, dtype=dtype),
            xc_uF_cm2=_stack_space(templates, "xc_uF_cm2", resolved_nx, dtype=dtype),
            formulations=tuple(str(row.formulation) for row in templates),
            membrane_models=tuple(tuple(row.membrane_models) for row in templates),
            section_names=tuple(tuple(row.section_names) for row in templates),
            section_tags=tuple(tuple(row.section_tags) for row in templates),
        )

    @property
    def size(self) -> int:
        """Number of population rows represented by the table."""

        return int(self.row_template_indices.shape[0])

    @property
    def template_count(self) -> int:
        """Number of unique numerical templates stored by the table."""

        return int(self.template_nx.shape[0])

    @property
    def nx(self) -> int:
        """Padded spatial width of materialized template rows."""

        return int(self.valid_mask.shape[1])

    @property
    def nbytes(self) -> int:
        """Host bytes owned by the numerical structure-of-arrays table."""

        return sum(
            int(array.nbytes)
            for array in (
                self.row_template_indices,
                self.template_nx,
                self.has_heterogeneous_cable_properties,
                self.valid_mask,
                self.x_um,
                self.compartment_lengths_um,
                self.dx_cm,
                self.h_um,
                self.h_cm,
                self.diam_um,
                self.Ra_ohm_cm,
                self.Cm_uF_cm2,
                self.xraxial_MOhm_per_cm,
                self.xg_S_cm2,
                self.xc_uF_cm2,
            )
        )

    def gather_space(self, values: np.ndarray) -> np.ndarray:
        """Gather a template-major spatial table into population row order."""

        array = np.asarray(values)
        if array.ndim != 2 or array.shape[0] != self.template_count:
            raise ValueError(
                "values must have shape (template_count, width); "
                f"got {array.shape}."
            )
        gathered = np.asarray(array[self.row_template_indices])
        gathered.setflags(write=False)
        return gathered

    @cached_property
    def x_positions_m(self) -> np.ndarray:
        """Return population-major padded positions in meters."""

        positions = self.gather_space(self.x_um).astype(float, copy=True)
        positions *= 1e-6
        positions.setflags(write=False)
        return positions


def _stack_space(
    rows: Sequence[SolverAxon],
    field: str,
    target_nx: int,
    *,
    dtype: np.dtype,
    pad: str = "zero",
) -> np.ndarray:
    output = np.zeros((len(rows), target_nx), dtype=dtype)
    for index, row in enumerate(rows):
        values = np.asarray(getattr(row, field), dtype=dtype)
        count = int(values.shape[0])
        output[index, :count] = values
        if pad == "edge" and count < target_nx:
            output[index, count:] = values[-1]
    output.setflags(write=False)
    return output


def _stack_edges(
    rows: Sequence[SolverAxon],
    field: str,
    target_nx: int,
    *,
    dtype: np.dtype,
) -> np.ndarray:
    target_edges = max(int(target_nx) - 1, 0)
    output = np.zeros((len(rows), target_edges), dtype=dtype)
    for index, row in enumerate(rows):
        values = np.asarray(getattr(row, field), dtype=dtype)
        output[index, : values.shape[0]] = values
    output.setflags(write=False)
    return output


__all__ = ["MaterializedAxonRows"]
