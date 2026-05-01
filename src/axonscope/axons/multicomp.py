from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import jax.numpy as jnp

from axonscope.axons.base import AxonBase
from axonscope.channel_models.base_channel_model import IonChannelModelBase
from axonscope.channel_models.passive import PassiveICM
from axonscope.icm import HeterogeneousICMBackend, ICMBackend


@dataclass(frozen=True)
class MultiCompGeometry:
    """Per-compartment cable geometry and material description."""

    diam_um: jnp.ndarray
    Ra_ohm_cm: jnp.ndarray
    Cm_uF_cm2: jnp.ndarray


class DoubleCableAxon(AxonBase):
    """Base class for one-dimensional double-cable style axons.

    This class encapsulates the shared geometry and extracellular-layer plumbing
    used by MRG-like myelinated models and future derivatives.
    """

    def __init__(
        self,
        *,
        ion_channel: IonChannelModelBase,
        lengths_um: Sequence[float],
        diam_um: Sequence[float],
        Ra_ohm_cm: Sequence[float],
        Cm_uF_cm2: Sequence[float],
        Vinit: float,
        Temp: float,
        fiber_d_um: float,
        kind_vec: Optional[Sequence[str]] = None,
        is_node: Optional[Sequence[bool]] = None,
        xraxial_MOhm_cm: Optional[Sequence[float]] = None,
        xg_S_cm2: Optional[Sequence[float]] = None,
        xc_uF_cm2: Optional[Sequence[float]] = None,
        Veinit: float = 0.0,
        enable_extracellular: bool = True,
    ) -> None:
        lengths = jnp.asarray(lengths_um, dtype=ion_channel.dtype)
        diam = jnp.asarray(diam_um, dtype=ion_channel.dtype)
        Ra_vec = jnp.asarray(Ra_ohm_cm, dtype=ion_channel.dtype)
        Cm_vec = jnp.asarray(Cm_uF_cm2, dtype=ion_channel.dtype)

        Nx = int(lengths.shape[0])
        if Nx < 2:
            raise ValueError("DoubleCableAxon requires at least 2 compartments.")
        if diam.shape != (Nx,) or Ra_vec.shape != (Nx,) or Cm_vec.shape != (Nx,):
            raise ValueError("All per-compartment vectors must have shape (Nx,).")

        edges = jnp.concatenate([jnp.array([0.0], dtype=ion_channel.dtype), jnp.cumsum(lengths)])
        x_centers = 0.5 * (edges[:-1] + edges[1:])

        super().__init__(
            ion_channel=ion_channel,
            L=None,
            d=float(fiber_d_um),
            Nx=Nx,
            x_vec=x_centers,
            Ra=float(jnp.mean(Ra_vec)),
            Cm=float(jnp.mean(Cm_vec)),
            Vinit=Vinit,
            Temp=Temp,
        )

        self.L = float(jnp.sum(lengths))
        self.diam_vec = diam
        self.Ra_vec = Ra_vec
        self.Cm_vec = Cm_vec
        self.compartment_lengths_um = lengths
        self.has_heterogeneous_cable_properties = True

        self.dx_cm = lengths * 1e-4
        if self.Nx > 1:
            self.h_um = jnp.diff(x_centers)
            self.h_cm = self.h_um * 1e-4

        if kind_vec is not None:
            if len(kind_vec) != self.Nx:
                raise ValueError("kind_vec must have length Nx.")
            self.section_kinds = tuple(kind_vec)
            self.kind_vec = self.section_kinds

        if is_node is not None:
            if len(is_node) != self.Nx:
                raise ValueError("is_node must have length Nx.")
            self.node_mask = jnp.asarray(is_node, dtype=bool)
            node_idx = jnp.where(self.node_mask)[0]
            self.node_indices = node_idx.astype(jnp.int32)

        if xraxial_MOhm_cm is not None and xg_S_cm2 is not None and xc_uF_cm2 is not None:
            self.set_extracellular_layer(
                xraxial_MOhm_per_cm=jnp.asarray(xraxial_MOhm_cm, dtype=ion_channel.dtype),
                xg_S_per_cm2=jnp.asarray(xg_S_cm2, dtype=ion_channel.dtype),
                xc_uF_per_cm2=jnp.asarray(xc_uF_cm2, dtype=ion_channel.dtype),
                use_extracellular=enable_extracellular,
                Veinit=Veinit,
            )

    def build_icm_backend(self) -> ICMBackend:
        """Return the compute backend associated with this double-cable axon."""
        build_backend = getattr(self.ion_channel, "build_backend", None)
        if callable(build_backend):
            return build_backend()
        return super().build_icm_backend()


class AxonMultiCompBase(AxonBase):
    """Container for heterogeneous multi-compartment axons.

    This class is solver-agnostic and focuses on geometry vectors, per-segment
    ion-channel instances, and a bridge to the heterogeneous ICM backend.
    """

    def __init__(
        self,
        *,
        L: Optional[float] = None,
        Nx: Optional[int] = None,
        x_vec: Optional[jnp.ndarray] = None,
        geometry: MultiCompGeometry,
        icm_vec: Sequence[IonChannelModelBase],
        Vinit: float = -70.0,
        Temp: float = 37.0,
    ) -> None:
        if len(icm_vec) == 0:
            raise ValueError("icm_vec cannot be empty.")

        dtype = icm_vec[0].dtype
        diam_vec = jnp.asarray(geometry.diam_um, dtype=dtype)
        Ra_vec = jnp.asarray(geometry.Ra_ohm_cm, dtype=dtype)
        Cm_vec = jnp.asarray(geometry.Cm_uF_cm2, dtype=dtype)

        if x_vec is not None:
            x = jnp.asarray(x_vec, dtype=dtype)
            Nx_effective = int(x.shape[0])
        else:
            if L is None or Nx is None:
                raise ValueError("Provide either x_vec, or both L and Nx.")
            x = None
            Nx_effective = int(Nx)

        if Nx_effective < 2:
            raise ValueError("Multi-comp axon requires Nx >= 2.")

        if len(icm_vec) != Nx_effective:
            raise ValueError(f"icm_vec size must be Nx={Nx_effective}, got {len(icm_vec)}.")

        if (
            diam_vec.shape != (Nx_effective,)
            or Ra_vec.shape != (Nx_effective,)
            or Cm_vec.shape != (Nx_effective,)
        ):
            raise ValueError("Geometry vectors must all have shape (Nx,).")

        super().__init__(
            ion_channel=icm_vec[0],
            d=float(jnp.mean(diam_vec)),
            Nx=Nx_effective if x is None else None,
            L=float(L) if x is None else None,
            x_vec=x,
            Ra=float(jnp.mean(Ra_vec)),
            Cm=float(jnp.mean(Cm_vec)),
            Vinit=Vinit,
            Temp=Temp,
        )
        if x is not None and L is not None:
            self.L = float(L)

        self.dtype = dtype
        self.has_heterogeneous_cable_properties = True

        self.diam_vec = diam_vec
        self.Ra_vec = Ra_vec
        self.Cm_vec = Cm_vec
        self.channel_models = tuple(icm_vec)
        self.icm_vec = self.channel_models

    def build_icm_backend(self) -> ICMBackend:
        """Return the compute backend associated with this axon description."""
        return HeterogeneousICMBackend.from_icm_vec(self.channel_models)


class GenericMultiCompAxon(AxonMultiCompBase):
    """Simple heterogeneous axon scaffold used to build/test multicomp backends."""

    def __init__(
        self,
        *,
        L: float = 1000.0,
        Nx: int = 101,
        diam_um: float = 1.0,
        Ra_ohm_cm: float = 100.0,
        Cm_uF_cm2: float = 1.0,
        icm_vec: Optional[Sequence[IonChannelModelBase]] = None,
        Vinit: float = -70.0,
        Temp: float = 37.0,
    ) -> None:
        if icm_vec is None:
            icm_vec = [PassiveICM(Rm=1e4, EL=-70.0) for _ in range(Nx)]

        geometry = MultiCompGeometry(
            diam_um=jnp.full((Nx,), diam_um),
            Ra_ohm_cm=jnp.full((Nx,), Ra_ohm_cm),
            Cm_uF_cm2=jnp.full((Nx,), Cm_uF_cm2),
        )
        super().__init__(
            L=L,
            Nx=Nx,
            geometry=geometry,
            icm_vec=icm_vec,
            Vinit=Vinit,
            Temp=Temp,
        )
