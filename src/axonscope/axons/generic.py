from typing import Optional

import jax.numpy as jnp

from axonscope.axons.base import AxonBase
from axonscope.channel_models.base_channel_model import IonChannelModelBase
from axonscope.channel_models.passive import PassiveICM


class GenericAxon(AxonBase):
    """Generic axon for cable simulations with a given ion channel model."""

    def __init__(
        self,
        ion_channel: IonChannelModelBase,
        d: float,
        Nx: Optional[int],
        L: Optional[float] = None,
        x_vec: Optional[jnp.ndarray] = None,
        Ra: float = 100.0,
        Cm: float = 1.0,
        Vinit: float = -70.0,
        Temp: float = 37.0,
    ):
        super().__init__(
            ion_channel=ion_channel,
            L=L,
            d=d,
            Nx=Nx,
            x_vec=x_vec,
            Ra=Ra,
            Cm=Cm,
            Vinit=Vinit,
            Temp=Temp,
        )


class Passive(AxonBase):
    """Axon with passive membrane properties (leak conductance)."""

    def __init__(
        self,
        d: float,
        Nx: Optional[int],
        L: Optional[float] = None,
        x_vec: Optional[jnp.ndarray] = None,
        Rm: float = 1e4,
        Cm: float = 1.0,
        Ra: float = 100.0,
        EL: float = -70.0,
        Vinit: float = -70.0,
        Temp: float = 37.0,
    ):
        ion_channel = PassiveICM(Rm=Rm, EL=EL)
        super().__init__(
            ion_channel=ion_channel,
            L=L,
            d=d,
            Nx=Nx,
            x_vec=x_vec,
            Ra=Ra,
            Cm=Cm,
            Vinit=Vinit,
            Temp=Temp,
        )
