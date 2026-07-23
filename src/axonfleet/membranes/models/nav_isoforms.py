"""Public Balbi Nav1.1-Nav1.9 parameterizations of one shared source model."""

from __future__ import annotations

from axonfleet.membranes.models.nav_balbi import BalbiNav
from axonfleet.membranes.types import Rate, Voltage
from axonfleet.utils.units import mV, ms


class Nav11(BalbiNav):
    """Balbi Nav1.1 parameterization."""

    model_kind = "nav11"
    source_model = BalbiNav


class Nav12(BalbiNav):
    """Balbi Nav1.2 parameterization."""

    model_kind = "nav12"
    source_model = BalbiNav
    C1C2b2: Rate = 16 / ms
    C1C2v2: Voltage = -5 * mV
    C2C1v1: Voltage = -35 * mV
    C2C1b2: Rate = 16 / ms
    C2C1v2: Voltage = -5 * mV
    C2O1b2: Rate = 16 / ms
    C2O1v2: Voltage = -10 * mV
    O1C2v1: Voltage = -40 * mV
    O1C2b2: Rate = 16 / ms
    O1C2v2: Voltage = -10 * mV
    C2O2b2: Rate = 0.13 / ms
    C2O2v2: Voltage = -20 * mV
    O2C2v1: Voltage = -60 * mV
    O2C2k1: Voltage = 6 * mV
    O2C2b2: Rate = 0.7 / ms
    O2C2v2: Voltage = -10 * mV
    O2C2k2: Voltage = -15 * mV
    O1I1b1: Rate = 3 / ms
    O1I1v1: Voltage = -41 * mV
    O1I1k1: Voltage = 12 * mV
    O1I1b2: Rate = 16 / ms
    O1I1v2: Voltage = -11 * mV
    O1I1k2: Voltage = -12 * mV
    I1O1v1: Voltage = -42 * mV
    I1C1b1: Rate = 0.55 / ms
    I1C1v1: Voltage = -65 * mV
    C1I1b2: Rate = 0.55 / ms
    C1I1v2: Voltage = -65 * mV
    C1I1k2: Voltage = -11 * mV
    I1I2b2: Rate = 0.0022 / ms
    I2I1b1: Rate = 0.017 / ms


class Nav13(BalbiNav):
    """Balbi Nav1.3 parameterization."""

    model_kind = "nav13"
    source_model = BalbiNav
    C1C2b2: Rate = 8 / ms
    C1C2k2: Voltage = -9 * mV
    C2C1b1: Rate = 2 / ms
    C2C1k1: Voltage = 9 * mV
    C2C1b2: Rate = 8 / ms
    C2C1k2: Voltage = -9 * mV
    C2O1b2: Rate = 8 / ms
    C2O1v2: Voltage = -17 * mV
    C2O1k2: Voltage = -9 * mV
    O1C2b1: Rate = 2 / ms
    O1C2v1: Voltage = -47 * mV
    O1C2k1: Voltage = 9 * mV
    O1C2b2: Rate = 8 / ms
    O1C2v2: Voltage = -17 * mV
    O1C2k2: Voltage = -9 * mV
    C2O2b2: Rate = 0.13 / ms
    C2O2v2: Voltage = -15 * mV
    C2O2k2: Voltage = -5 * mV
    O2C2b1: Rate = 1 / ms
    O2C2v1: Voltage = -40 * mV
    O2C2k1: Voltage = 3 * mV
    O2C2k2: Voltage = -3 * mV
    O1I1b1: Rate = 2 / ms
    O1I1v1: Voltage = -52 * mV
    O1I1b2: Rate = 8 / ms
    O1I1v2: Voltage = -22 * mV
    O1I1k2: Voltage = -13 * mV
    I1O1v1: Voltage = -52 * mV
    I1C1b1: Rate = 0.062 / ms
    I1C1v1: Voltage = -70 * mV
    I1C1k1: Voltage = 10 * mV
    C1I1b2: Rate = 0.09 / ms
    C1I1v2: Voltage = -68 * mV
    C1I1k2: Voltage = -8 * mV
    I1I2b2: Rate = 0.0001 / ms
    I2I1b1: Rate = 0.0001 / ms


class Nav14(BalbiNav):
    """Balbi Nav1.4 parameterization."""

    model_kind = "nav14"
    source_model = BalbiNav
    C1C2b2: Rate = 16 / ms
    C1C2v2: Voltage = -3 * mV
    C1C2k2: Voltage = -9 * mV
    C2C1v1: Voltage = -33 * mV
    C2C1k1: Voltage = 9 * mV
    C2C1b2: Rate = 16 / ms
    C2C1v2: Voltage = -3 * mV
    C2C1k2: Voltage = -9 * mV
    C2O1b2: Rate = 16 / ms
    C2O1v2: Voltage = -8 * mV
    C2O1k2: Voltage = -9 * mV
    O1C2b1: Rate = 1 / ms
    O1C2v1: Voltage = -38 * mV
    O1C2k1: Voltage = 9 * mV
    O1C2b2: Rate = 16 / ms
    O1C2v2: Voltage = -8 * mV
    O1C2k2: Voltage = -9 * mV
    C2O2b2: Rate = 0.03 / ms
    C2O2v2: Voltage = -20 * mV
    C2O2k2: Voltage = -8 * mV
    O2C2b1: Rate = 3 / ms
    O2C2k1: Voltage = 8 * mV
    O2C2b2: Rate = 0.1 / ms
    O2C2k2: Voltage = -8 * mV
    O1I1b1: Rate = 0 / ms
    O1I1v1: Voltage = -10 * mV
    O1I1k1: Voltage = 10 * mV
    O1I1b2: Rate = 16 / ms
    O1I1v2: Voltage = -10 * mV
    O1I1k2: Voltage = -10 * mV
    I1O1v1: Voltage = -10 * mV
    I1C1b1: Rate = 0.35 / ms
    I1C1v1: Voltage = -70 * mV
    I1C1k1: Voltage = 10 * mV
    C1I1b2: Rate = 0.8 / ms
    C1I1v2: Voltage = -70 * mV
    C1I1k2: Voltage = -7 * mV
    I1I2v2: Voltage = -70 * mV
    I1I2k2: Voltage = -12 * mV
    I2I1b1: Rate = 0.007 / ms
    I2I1v1: Voltage = -70 * mV
    I2I1k1: Voltage = 12 * mV


class Nav15(BalbiNav):
    """Balbi Nav1.5 parameterization."""

    model_kind = "nav15"
    source_model = BalbiNav
    C1C2b2: Rate = 10 / ms
    C1C2v2: Voltage = -13 * mV
    C2C1b1: Rate = 1 / ms
    C2C1v1: Voltage = -43 * mV
    C2C1k1: Voltage = 8 * mV
    C2C1b2: Rate = 10 / ms
    C2C1v2: Voltage = -13 * mV
    C2O1b2: Rate = 10 / ms
    C2O1v2: Voltage = -23 * mV
    O1C2b1: Rate = 1 / ms
    O1C2v1: Voltage = -53 * mV
    O1C2k1: Voltage = 8 * mV
    O1C2b2: Rate = 10 / ms
    O1C2v2: Voltage = -23 * mV
    C2O2b2: Rate = 0.05 / ms
    C2O2k2: Voltage = -10 * mV
    O2C2k1: Voltage = 10 * mV
    O2C2b2: Rate = 0.08 / ms
    O1I1b1: Rate = 7 / ms
    O1I1v1: Voltage = -44 * mV
    O1I1b2: Rate = 10 / ms
    O1I1v2: Voltage = -19 * mV
    O1I1k2: Voltage = -13 * mV
    I1O1v1: Voltage = -20 * mV
    I1C1b1: Rate = 0.19 / ms
    I1C1v1: Voltage = -110 * mV
    C1I1b2: Rate = 0.016 / ms
    C1I1v2: Voltage = -92 * mV
    C1I1k2: Voltage = -6 * mV
    I1I2b2: Rate = 0.00022 / ms
    I1I2v2: Voltage = -50 * mV
    I2I1b1: Rate = 0.0018 / ms
    I2I1k1: Voltage = 30 * mV


class Nav16(BalbiNav):
    """Balbi Nav1.6 parameterization."""

    model_kind = "nav16"
    source_model = BalbiNav
    C1C2b2: Rate = 14 / ms
    C1C2v2: Voltage = -8 * mV
    C2C1b1: Rate = 2 / ms
    C2C1v1: Voltage = -38 * mV
    C2C1k1: Voltage = 9 * mV
    C2C1b2: Rate = 14 / ms
    C2C1v2: Voltage = -8 * mV
    C2O1b2: Rate = 14 / ms
    C2O1v2: Voltage = -18 * mV
    O1C2b1: Rate = 4 / ms
    O1C2v1: Voltage = -48 * mV
    O1C2k1: Voltage = 9 * mV
    O1C2b2: Rate = 14 / ms
    O1C2v2: Voltage = -18 * mV
    C2O2b2: Rate = 0.0001 / ms
    C2O2k2: Voltage = -8 * mV
    O2C2b1: Rate = 0.0001 / ms
    O2C2v1: Voltage = -55 * mV
    O2C2k1: Voltage = 10 * mV
    O2C2b2: Rate = 0.0001 / ms
    O2C2k2: Voltage = -5 * mV
    O1I1b1: Rate = 6 / ms
    O1I1v1: Voltage = -40 * mV
    O1I1b2: Rate = 10 / ms
    O1I1v2: Voltage = 15 * mV
    O1I1k2: Voltage = -18 * mV
    I1O1v1: Voltage = -40 * mV
    I1C1b1: Rate = 0.1 / ms
    I1C1v1: Voltage = -86 * mV
    I1C1k1: Voltage = 9 * mV
    C1I1b2: Rate = 0.08 / ms
    C1I1v2: Voltage = -55 * mV
    C1I1k2: Voltage = -12 * mV
    I1I2b2: Rate = 0.00022 / ms
    I1I2v2: Voltage = -50 * mV
    I2I1b1: Rate = 0.0018 / ms
    I2I1k1: Voltage = 30 * mV


class Nav17(BalbiNav):
    """Balbi Nav1.7 parameterization."""

    model_kind = "nav17"
    source_model = BalbiNav
    C1C2b2: Rate = 16 / ms
    C1C2v2: Voltage = -18 * mV
    C1C2k2: Voltage = -9 * mV
    C2C1b1: Rate = 6 / ms
    C2C1v1: Voltage = -48 * mV
    C2C1k1: Voltage = 9 * mV
    C2C1b2: Rate = 16 / ms
    C2C1v2: Voltage = -18 * mV
    C2C1k2: Voltage = -9 * mV
    C2O1b2: Rate = 16 / ms
    C2O1v2: Voltage = -23 * mV
    C2O1k2: Voltage = -9 * mV
    O1C2b1: Rate = 2 / ms
    O1C2v1: Voltage = -53 * mV
    O1C2k1: Voltage = 9 * mV
    O1C2b2: Rate = 16 / ms
    O1C2v2: Voltage = -23 * mV
    O1C2k2: Voltage = -9 * mV
    C2O2b2: Rate = 0.01 / ms
    C2O2v2: Voltage = -35 * mV
    C2O2k2: Voltage = -5 * mV
    O2C2b1: Rate = 3 / ms
    O2C2v1: Voltage = -75 * mV
    O2C2k1: Voltage = 5 * mV
    O2C2b2: Rate = 0.01 / ms
    O2C2v2: Voltage = -35 * mV
    O2C2k2: Voltage = -5 * mV
    O1I1b1: Rate = 4 / ms
    O1I1v1: Voltage = -52 * mV
    O1I1k1: Voltage = 12 * mV
    O1I1b2: Rate = 8 / ms
    O1I1v2: Voltage = -27 * mV
    O1I1k2: Voltage = -12 * mV
    I1O1v1: Voltage = -52 * mV
    I1C1b1: Rate = 0.085 / ms
    I1C1v1: Voltage = -110 * mV
    I1C1k1: Voltage = 5 * mV
    C1I1b2: Rate = 0.025 / ms
    C1I1v2: Voltage = -55 * mV
    C1I1k2: Voltage = -20 * mV
    I1I2b2: Rate = 0.00001 / ms
    I1I2v2: Voltage = -80 * mV
    I1I2k2: Voltage = -20 * mV
    I2I1b1: Rate = 0.00001 / ms
    I2I1v1: Voltage = -80 * mV
    I2I1k1: Voltage = 20 * mV


class Nav18(BalbiNav):
    """Balbi Nav1.8 parameterization."""

    model_kind = "nav18"
    source_model = BalbiNav
    C1C2b2: Rate = 5 / ms
    C1C2v2: Voltage = 17 * mV
    C1C2k2: Voltage = -8 * mV
    C2C1b1: Rate = 1 / ms
    C2C1v1: Voltage = -23 * mV
    C2C1k1: Voltage = 8 * mV
    C2C1b2: Rate = 5 / ms
    C2C1v2: Voltage = 17 * mV
    C2C1k2: Voltage = -8 * mV
    C2O1b2: Rate = 5 / ms
    C2O1v2: Voltage = 13 * mV
    C2O1k2: Voltage = -8 * mV
    O1C2b1: Rate = 1 / ms
    O1C2v1: Voltage = -27 * mV
    O1C2k1: Voltage = 8 * mV
    O1C2b2: Rate = 5 / ms
    O1C2v2: Voltage = 13 * mV
    O1C2k2: Voltage = -8 * mV
    C2O2b2: Rate = 0.02 / ms
    C2O2v2: Voltage = 15 * mV
    C2O2k2: Voltage = -8 * mV
    O2C2b1: Rate = 0.8 / ms
    O2C2v1: Voltage = -60 * mV
    O2C2k1: Voltage = 5 * mV
    O2C2b2: Rate = 0.002 / ms
    O2C2v2: Voltage = 10 * mV
    O2C2k2: Voltage = -6 * mV
    O1I1b1: Rate = 0.8 / ms
    O1I1v1: Voltage = -21 * mV
    O1I1k1: Voltage = 10 * mV
    O1I1b2: Rate = 1 / ms
    O1I1v2: Voltage = -1 * mV
    O1I1k2: Voltage = -7 * mV
    I1O1v1: Voltage = -21 * mV
    I1C1b1: Rate = 0.28 / ms
    I1C1k1: Voltage = 9.5 * mV
    C1I1b2: Rate = 0.02 / ms
    C1I1v2: Voltage = -10 * mV
    C1I1k2: Voltage = -20 * mV
    I1I2b2: Rate = 0.001 / ms
    I1I2v2: Voltage = -50 * mV
    I1I2k2: Voltage = -3 * mV
    I2I1b1: Rate = 0.0003 / ms
    I2I1v1: Voltage = -50 * mV
    I2I1k1: Voltage = 5 * mV


class Nav19(BalbiNav):
    """Balbi Nav1.9 parameterization."""

    model_kind = "nav19"
    source_model = BalbiNav
    C1C2b2: Rate = 0.8 / ms
    C1C2v2: Voltage = -21 * mV
    C1C2k2: Voltage = -9 * mV
    C2C1b1: Rate = 0.05 / ms
    C2C1v1: Voltage = -56 * mV
    C2C1b2: Rate = 0.8 / ms
    C2C1v2: Voltage = -21 * mV
    C2C1k2: Voltage = -9 * mV
    C2O1b2: Rate = 0.8 / ms
    C2O1v2: Voltage = -61 * mV
    C2O1k2: Voltage = -9 * mV
    O1C2b1: Rate = 0.5 / ms
    O1C2v1: Voltage = -96 * mV
    O1C2b2: Rate = 0.8 / ms
    O1C2v2: Voltage = -61 * mV
    O1C2k2: Voltage = -9 * mV
    C2O2b2: Rate = 0.0001 / ms
    C2O2v2: Voltage = -5 * mV
    C2O2k2: Voltage = -8 * mV
    O2C2b1: Rate = 0.0001 / ms
    O2C2v1: Voltage = -65 * mV
    O2C2b2: Rate = 0.0001 / ms
    O2C2v2: Voltage = -15 * mV
    O2C2k2: Voltage = -12 * mV
    O1I1b1: Rate = 0.04 / ms
    O1I1v1: Voltage = -59 * mV
    O1I1k1: Voltage = 8 * mV
    O1I1b2: Rate = 0.8 / ms
    O1I1v2: Voltage = 1 * mV
    O1I1k2: Voltage = -10 * mV
    I1O1b1: Rate = 0.0001 / ms
    I1O1v1: Voltage = -60 * mV
    I1O1k1: Voltage = 8 * mV
    I1C1b1: Rate = 0.06 / ms
    I1C1v1: Voltage = -59 * mV
    I1C1k1: Voltage = 8 * mV
    C1I1b2: Rate = 0.04 / ms
    C1I1v2: Voltage = -59 * mV
    C1I1k2: Voltage = -8 * mV
    I1I2b2: Rate = 0.0016 / ms
    I1I2v2: Voltage = -60 * mV
    I1I2k2: Voltage = -20 * mV
    I2I1b1: Rate = 0.0115 / ms
    I2I1v1: Voltage = -100 * mV
    I2I1k1: Voltage = 8 * mV


__all__ = [
    "Nav11",
    "Nav12",
    "Nav13",
    "Nav14",
    "Nav15",
    "Nav16",
    "Nav17",
    "Nav18",
    "Nav19",
]
