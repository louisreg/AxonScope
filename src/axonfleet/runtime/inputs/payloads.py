"""Runtime-neutral input payload contracts for compact batch inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

Array = Any


@dataclass(frozen=True)
class SparseIntracellularCurrentDensityBatch:
    """Point-clamp current density represented without the compartment axis.

    ``density_mid`` stores one sampled current-density waveform per sparse
    clamp slot with shape ``(B, Nt, K)``. ``indices`` and ``mask`` describe
    where each slot is injected in compartment space.
    """

    density_mid: Array
    indices: Array
    mask: Array
    target_nx: int

    def __post_init__(self) -> None:
        density_shape = tuple(int(dim) for dim in getattr(self.density_mid, "shape", ()))
        indices_shape = tuple(int(dim) for dim in getattr(self.indices, "shape", ()))
        mask_shape = tuple(int(dim) for dim in getattr(self.mask, "shape", ()))
        if len(density_shape) != 3:
            raise ValueError("density_mid must have shape (B, Nt, K).")
        expected_sparse_shape = (density_shape[0], density_shape[2])
        if indices_shape != expected_sparse_shape:
            raise ValueError(
                "indices must have shape (B, K) matching density_mid, "
                f"got {indices_shape} and {density_shape}."
            )
        if mask_shape != expected_sparse_shape:
            raise ValueError(
                "mask must have shape (B, K) matching density_mid, "
                f"got {mask_shape} and {density_shape}."
            )
        if int(self.target_nx) < 1:
            raise ValueError("target_nx must be >= 1.")
        object.__setattr__(self, "target_nx", int(self.target_nx))

    @property
    def batch_size(self) -> int:
        """Number of independent rows."""

        return int(self.density_mid.shape[0])

    @property
    def step_count(self) -> int:
        """Number of midpoint time samples."""

        return int(self.density_mid.shape[1])

    @property
    def max_sparse_entries(self) -> int:
        """Maximum number of clamp slots per row."""

        return int(self.density_mid.shape[2])


@dataclass(frozen=True)
class FactorizedExtracellularPotentialBatch:
    """Static-footprint extracellular potential without the dense time-space tensor.

    ``current_mid_A`` stores the dynamic stimulus samples with shape ``(Nt,)``
    for a shared single-drive waveform, ``(S, Nt)`` for shared multi-drive
    waveforms, ``(B, Nt)`` for row-specific single-drive waveforms, or
    ``(B, S, Nt)`` for row-specific multi-drive waveforms.
    ``current_initial_previous_A`` optionally stores the ``t=-dt/2`` sample
    used by double-cable batches. ``footprint_mV_per_A`` stores the static
    spatial footprint with shape ``(B, Nx)`` or ``(B, S, Nx)``. The dense
    midpoint potential is their product, summed over the optional drive axis:
    ``Vstim[B, Nt, Nx] = sum_S current_mid_A * footprint_mV_per_A``.
    ``current_row_scales`` stores the row amplitude payload for scaled shared
    waveforms as ``(B,)`` or ``(B, S)``.
    ``current_row_indices`` can compress repeated row-specific currents as
    ``current_mid_A[U, Nt]`` or ``current_mid_A[U, S, Nt]`` plus row indices
    ``[B]``.
    """

    current_mid_A: Array
    footprint_mV_per_A: Array
    target_nx: int
    current_initial_previous_A: Array | None = None
    single_cable_forcing_footprint_mV_per_A: Array | None = None
    current_row_indices: Array | None = None
    current_row_scales: Array | None = None

    def __post_init__(self) -> None:
        current_shape = tuple(int(dim) for dim in getattr(self.current_mid_A, "shape", ()))
        previous_shape = tuple(
            int(dim) for dim in getattr(self.current_initial_previous_A, "shape", ())
        )
        footprint_shape = tuple(
            int(dim) for dim in getattr(self.footprint_mV_per_A, "shape", ())
        )
        indices_shape = tuple(
            int(dim) for dim in getattr(self.current_row_indices, "shape", ())
        )
        scales_shape = tuple(
            int(dim) for dim in getattr(self.current_row_scales, "shape", ())
        )
        forcing_shape = tuple(
            int(dim)
            for dim in getattr(self.single_cable_forcing_footprint_mV_per_A, "shape", ())
        )
        if len(footprint_shape) not in {2, 3}:
            raise ValueError("footprint_mV_per_A must have shape (B, Nx) or (B, K, Nx).")
        batch_size = footprint_shape[0]
        drive_count = 1 if len(footprint_shape) == 2 else footprint_shape[1]
        if len(current_shape) not in {1, 2, 3}:
            raise ValueError(
                "current_mid_A must have shape (Nt,), (S, Nt), (B, Nt), "
                "or (B, S, Nt)."
            )
        if len(footprint_shape) == 2 and len(current_shape) == 3:
            raise ValueError(
                "multi-drive current_mid_A requires footprint_mV_per_A shape (B, S, Nx)."
            )
        if self.current_row_indices is not None and self.current_row_scales is not None:
            raise ValueError("current_row_indices and current_row_scales are mutually exclusive.")
        if self.current_row_scales is not None:
            if len(footprint_shape) == 2:
                if len(current_shape) != 1:
                    raise ValueError(
                        "rank-1 current_row_scales require current_mid_A shape (Nt,)."
                    )
                if scales_shape not in {(batch_size,), (batch_size, 1)}:
                    raise ValueError(
                        "rank-1 current_row_scales must have shape (B,) or (B, 1), "
                        f"got {scales_shape} for footprint shape {footprint_shape}."
                    )
            else:
                if len(current_shape) != 2 or current_shape[0] != drive_count:
                    raise ValueError(
                        "multi-drive current_row_scales require current_mid_A shape "
                        f"(S, Nt)=({drive_count}, Nt), got {current_shape}."
                    )
                if scales_shape != (batch_size, drive_count):
                    raise ValueError(
                        "multi-drive current_row_scales must have shape (B, S), "
                        f"got {scales_shape} for footprint shape {footprint_shape}."
                    )
        if self.current_row_indices is not None:
            valid_indexed_layout = (
                len(footprint_shape) == 2
                and len(current_shape) == 2
            ) or (
                len(footprint_shape) == 3
                and len(current_shape) == 3
                and current_shape[1] == drive_count
            )
            if not valid_indexed_layout:
                raise ValueError(
                    "current_row_indices require current_mid_A shape (U, Nt) "
                    "or (U, S, Nt) matching rank-1 or rank-S footprints."
                )
            if current_shape[0] < 1:
                raise ValueError("compressed current_mid_A must contain at least one pattern.")
            if indices_shape != (batch_size,):
                raise ValueError(
                    "current_row_indices must have shape (B,) matching footprint_mV_per_A, "
                    f"got {indices_shape} and {footprint_shape}."
                )
        elif len(current_shape) == 2:
            if len(footprint_shape) == 2 and current_shape[0] != batch_size:
                raise ValueError(
                    "current_mid_A batch size must match footprint_mV_per_A; "
                    f"got {current_shape} and {footprint_shape}."
                )
            if len(footprint_shape) == 3 and current_shape[0] != drive_count:
                raise ValueError(
                    "shared multi-drive current_mid_A must have shape (S, Nt); "
                    f"got {current_shape} and {footprint_shape}."
                )
        if (
            len(current_shape) == 3
            and self.current_row_indices is None
            and current_shape[:2] != (batch_size, drive_count)
        ):
            raise ValueError(
                "current_mid_A batch/drive axes must match footprint_mV_per_A, "
                f"got {current_shape} and {footprint_shape}."
            )
        if self.current_initial_previous_A is not None:
            if self.current_row_scales is not None:
                valid_previous_shapes = {()} if len(footprint_shape) == 2 else {(drive_count,)}
            elif self.current_row_indices is not None:
                valid_previous_shapes = (
                    {(current_shape[0],)}
                    if len(footprint_shape) == 2
                    else {(current_shape[0], drive_count)}
                )
            elif len(footprint_shape) == 3:
                if len(current_shape) == 2:
                    valid_previous_shapes = {(drive_count,)}
                else:
                    valid_previous_shapes = {(batch_size, drive_count)}
            else:
                valid_previous_shapes = {(), (batch_size,)}
            if previous_shape not in valid_previous_shapes:
                raise ValueError(
                    "current_initial_previous_A must match the selected current "
                    "layout: scalar/(B,) for rank-1, (S,) for shared/scaled "
                    "multi-drive, or (U, S)/(B, S) for indexed/row-specific "
                    "multi-drive; "
                    f"got {previous_shape} for footprint shape {footprint_shape}."
                )
        footprint_width = footprint_shape[-1]
        if int(self.target_nx) != footprint_width:
            raise ValueError(
                "target_nx must match footprint_mV_per_A width, "
                f"got target_nx={self.target_nx} and shape {footprint_shape}."
            )
        if int(self.target_nx) < 1:
            raise ValueError("target_nx must be >= 1.")
        if self.single_cable_forcing_footprint_mV_per_A is not None:
            if forcing_shape != footprint_shape:
                raise ValueError(
                    "single_cable_forcing_footprint_mV_per_A must match "
                    f"footprint_mV_per_A shape {footprint_shape}, got {forcing_shape}."
                )
        object.__setattr__(self, "target_nx", int(self.target_nx))

    @property
    def batch_size(self) -> int:
        """Number of independent rows."""

        return int(self.footprint_mV_per_A.shape[0])

    @property
    def drive_count(self) -> int:
        """Maximum number of factorized drives per row."""

        shape = getattr(self.footprint_mV_per_A, "shape", ())
        return 1 if len(shape) == 2 else int(shape[1])

    @property
    def step_count(self) -> int:
        """Number of midpoint time samples."""

        current = self.current_mid_A
        if len(current.shape) == 1:
            return int(current.shape[0])
        if len(current.shape) == 2:
            return int(current.shape[1])
        return int(current.shape[2])

    @property
    def shared_current(self) -> bool:
        """Whether all rows share the same temporal waveform."""

        if self.current_row_indices is not None or self.current_row_scales is not None:
            return False
        current_shape = tuple(int(dim) for dim in getattr(self.current_mid_A, "shape", ()))
        footprint_shape = tuple(
            int(dim) for dim in getattr(self.footprint_mV_per_A, "shape", ())
        )
        if len(current_shape) == 1:
            return True
        return (
            len(footprint_shape) == 3
            and len(current_shape) == 2
            and current_shape[0] == footprint_shape[1]
        )

    @property
    def scaled_shared_waveform(self) -> bool:
        """Whether rows scale shared temporal waveform shapes."""

        return self.current_row_scales is not None
