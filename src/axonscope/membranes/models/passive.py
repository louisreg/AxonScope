"""Passive leak membrane equations written as plain Python source."""

from __future__ import annotations

from axonscope.membranes.model import Model, section
from axonscope.membranes.types import ConductanceDensity, CurrentDensity, ResistanceArea, Voltage
from axonscope.utils.units import cm2, mV, ohm


class Passive(Model):
    """Passive leak membrane equations in canonical units."""

    model_kind = "passive"
    metadata = {
        "display_name": "Passive leak",
        "family": "passive",
        "source_reference": "generic passive membrane",
        "current_sign_convention": "outward_positive",
        "notes": (
            "Rm is a membrane resistance-area in ohm*cm2; "
            "the compiler converts g_l = 1 / Rm to mS/cm2."
        ),
    }

    Rm: ResistanceArea = 1.0e4 * ohm * cm2
    EL: Voltage = -70.0 * mV

    @section("leak")
    def leak(self, Vm: Voltage):
        """Passive leak equations in canonical units.

        Rm: membrane resistance-area
        EL, Vm: membrane voltage
        """

        g_l: ConductanceDensity = 1.0 / self.Rm
        I_l: CurrentDensity = g_l * (Vm - self.EL)
        return I_l, g_l
