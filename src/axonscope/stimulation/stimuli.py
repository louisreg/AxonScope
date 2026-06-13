from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


ArrayLike = np.ndarray | list[float] | tuple[float, ...]


@dataclass(frozen=True)
class Stimulus:
    """
    Piecewise stimulus waveform.

    Times are in ms.
    Amplitudes are user-defined:
      - nA for intracellular current clamp
      - uA or mA depending on extracellular convention
      - arbitrary scalar for normalized protocols
    """

    t: np.ndarray
    y: np.ndarray
    mode: Literal["hold", "linear"] = "hold"

    def __post_init__(self):
        t = np.asarray(self.t, dtype=float)
        y = np.asarray(self.y, dtype=float)

        if t.ndim != 1 or y.ndim != 1:
            raise ValueError("Stimulus.t and Stimulus.y must be 1D arrays.")
        if len(t) != len(y):
            raise ValueError("Stimulus.t and Stimulus.y must have the same length.")
        if len(t) == 0:
            raise ValueError("Stimulus cannot be empty.")
        if np.any(np.diff(t) < 0):
            order = np.argsort(t)
            t = t[order]
            y = y[order]

        # Remove duplicated times by keeping the last value.
        unique_t, last_indices = np.unique(t, return_index=False, return_inverse=False), None
        if len(unique_t) != len(t):
            new_t = []
            new_y = []
            for ti in unique_t:
                idx = np.where(t == ti)[0][-1]
                new_t.append(ti)
                new_y.append(y[idx])
            t = np.asarray(new_t, dtype=float)
            y = np.asarray(new_y, dtype=float)

        object.__setattr__(self, "t", t)
        object.__setattr__(self, "y", y)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def constant(cls, value: float, start: float = 0.0) -> "Stimulus":
        return cls(t=np.asarray([start]), y=np.asarray([value]))

    @classmethod
    def pulse(
        cls,
        start: float,
        amplitude: float,
        duration: float,
        baseline: float = 0.0,
    ) -> "Stimulus":
        return cls(
            t=np.asarray([0.0, start, start + duration]),
            y=np.asarray([baseline, amplitude, baseline]),
            mode="hold",
        )

    @classmethod
    def biphasic(
        cls,
        start: float,
        cathodic_amplitude: float,
        cathodic_duration: float,
        anodic_amplitude: float | None = None,
        interphase: float = 0.0,
        anodic_first: bool = False,
        baseline: float = 0.0,
    ) -> "Stimulus":
        """
        Charge-balanced biphasic pulse by default.

        Convention:
        - cathodic phase is negative
        - anodic phase is positive
        """
        cath = -abs(cathodic_amplitude)

        if anodic_amplitude is None:
            anodic_amplitude = abs(cathodic_amplitude)

        anod = abs(anodic_amplitude)
        anodic_duration = abs(cath * cathodic_duration / anod) if anod != 0 else 0.0

        if anodic_first:
            a1, d1 = anod, anodic_duration
            a2, d2 = cath, cathodic_duration
        else:
            a1, d1 = cath, cathodic_duration
            a2, d2 = anod, anodic_duration

        t0 = start
        t1 = t0 + d1
        t2 = t1 + interphase
        t3 = t2 + d2

        return cls(
            t=np.asarray([0.0, t0, t1, t2, t3]),
            y=np.asarray([baseline, a1, baseline, a2, baseline]),
            mode="hold",
        )

    @classmethod
    def sinus(
        cls,
        start: float,
        duration: float,
        amplitude: float,
        frequency_khz: float,
        offset: float = 0.0,
        phase: float = 0.0,
        dt: float | None = None,
    ) -> "Stimulus":
        if dt is None:
            dt = 1.0 / (100.0 * frequency_khz)

        n = int(np.ceil(duration / dt)) + 1
        local_t = np.linspace(0.0, duration, n)
        y = offset + amplitude * np.sin(2.0 * np.pi * frequency_khz * local_t + phase)

        t = start + local_t
        return cls(t=t, y=y, mode="linear")

    @classmethod
    def ramp(
        cls,
        start: float,
        duration: float,
        start_value: float,
        stop_value: float,
        dt: float,
    ) -> "Stimulus":
        n = int(np.ceil(duration / dt)) + 1
        local_t = np.linspace(0.0, duration, n)
        y = np.linspace(start_value, stop_value, n)
        return cls(t=start + local_t, y=y, mode="linear")

    @classmethod
    def from_samples(
        cls,
        t: ArrayLike,
        y: ArrayLike,
        mode: Literal["hold", "linear"] = "hold",
    ) -> "Stimulus":
        return cls(t=np.asarray(t), y=np.asarray(y), mode=mode)

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def shifted(self, dt: float) -> "Stimulus":
        return Stimulus(self.t + dt, self.y, self.mode)

    def scaled(self, factor: float) -> "Stimulus":
        return Stimulus(self.t, factor * self.y, self.mode)

    def offset(self, value: float) -> "Stimulus":
        return Stimulus(self.t, self.y + value, self.mode)

    def insert_samples(self, t_new: ArrayLike) -> "Stimulus":
        """
        Return a new stimulus evaluated on the union of old and new samples.
        """
        from axonscope.stimulus_eval import evaluate_stimulus_numpy

        t_union = np.unique(np.concatenate([self.t, np.asarray(t_new, dtype=float)]))
        y_union = evaluate_stimulus_numpy(self, t_union)
        return Stimulus(t_union, y_union, self.mode)

    def synchronize(self, other: "Stimulus") -> tuple["Stimulus", "Stimulus"]:
        """
        Return two stimuli evaluated on the same time grid.
        """
        t_union = np.unique(np.concatenate([self.t, other.t]))
        return self.insert_samples(t_union), other.insert_samples(t_union)

    def __add__(self, other: float | "Stimulus") -> "Stimulus":
        if isinstance(other, Stimulus):
            a, b = self.synchronize(other)
            return Stimulus(a.t, a.y + b.y, self.mode)
        return Stimulus(self.t, self.y + float(other), self.mode)

    def __sub__(self, other: float | "Stimulus") -> "Stimulus":
        if isinstance(other, Stimulus):
            a, b = self.synchronize(other)
            return Stimulus(a.t, a.y - b.y, self.mode)
        return Stimulus(self.t, self.y - float(other), self.mode)

    def __mul__(self, other: float | "Stimulus") -> "Stimulus":
        if isinstance(other, Stimulus):
            a, b = self.synchronize(other)
            return Stimulus(a.t, a.y * b.y, self.mode)
        return Stimulus(self.t, self.y * float(other), self.mode)

    __rmul__ = __mul__
