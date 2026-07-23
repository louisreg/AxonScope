"""Backend-independent temporal stimulation waveforms.

This module only describes time courses. A `Stimulus` does not know whether it
will become an intracellular clamp current or an electrode current until a
physical object consumes it. Public constructor times must carry units and are
stored in milliseconds; amplitude units are preserved when Pint quantities are
provided and normalized later by the consuming clamp or electrode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from axonfleet.utils import units


_ArrayLike = Any
_UnitLike = Any


def _readonly_float_array(values: Any) -> np.ndarray:
    arr = np.array(values, dtype=float, copy=True, order="C")
    arr.setflags(write=False)
    return arr


def _is_zero_value(value: Any) -> bool:
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def _coerce_amplitudes(values: Any, unit: _UnitLike | None) -> tuple[np.ndarray, str | None]:
    """Return numeric amplitudes and their optional canonical unit label.

    If `unit` is provided, all values are converted to that unit. If no unit is
    provided but `values` contains Pint-like quantities, the first quantity unit
    is inferred and all quantity entries are converted to it.
    """

    unit_label = units.unit_label(unit)
    if unit_label is not None:
        return units.to_array(values, unit_label, dtype=float), unit_label
    if units.is_quantity_like(values):
        inferred = units.quantity_unit(values)
        if inferred is None:
            return np.asarray(values.magnitude, dtype=float), None
        inferred = units.unit_label(inferred)
        return units.to_array(values, inferred, dtype=float), inferred
    if isinstance(values, np.ndarray) and values.dtype != object:
        return np.asarray(values, dtype=float), None

    arr = np.asarray(values, dtype=object)
    flat = list(arr.reshape(-1))
    quantity_units = [units.quantity_unit(value) for value in flat if units.is_quantity_like(value)]
    if not quantity_units:
        return np.asarray(values, dtype=float), None
    inferred = units.unit_label(quantity_units[0])
    converted = [
        units.to_scalar(value, inferred) if units.is_quantity_like(value) else float(value)
        for value in flat
    ]
    return np.asarray(converted, dtype=float).reshape(arr.shape), inferred


@dataclass(frozen=True)
class Stimulus:
    """Piecewise temporal waveform.

    Times are stored in milliseconds. Amplitudes are scalar until a physical
    object interprets them: intracellular clamps normalize to nA, while
    extracellular electrodes normalize to A. Pint quantities can also be passed
    directly as waveform amplitudes; the unit label is preserved and converted
    by the consuming object.
    """

    t: np.ndarray
    y: np.ndarray
    mode: Literal["hold", "linear"] = "hold"
    y_unit: str | None = None
    _scale_shape: tuple[Any, ...] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self):
        """Validate, sort, deduplicate, and normalize the sample arrays."""

        t = units.to_ms_array(self.t, dtype=float)
        y, y_unit = _coerce_amplitudes(self.y, self.y_unit)

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

        t = _readonly_float_array(t)
        y = _readonly_float_array(y)

        object.__setattr__(self, "t", t)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "y_unit", y_unit)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def constant(
        cls,
        value: Any,
        start: Any | None = None,
        *,
        unit: _UnitLike | None = None,
    ) -> "Stimulus":
        """Build a constant waveform.

        Parameters
        ----------
        value:
            Amplitude after `start`. Plain numbers are unitless until the
            stimulus is attached to a physical object.
        start:
            Start time, with units. When omitted, the waveform starts at 0 ms.
        unit:
            Optional amplitude unit for plain numeric values.
        """

        start = 0.0 if start is None else units.require_time_ms(start, name="start")
        return cls(
            t=np.asarray([start]),
            y=np.asarray([value], dtype=object),
            y_unit=unit,
            _scale_shape=("constant",),
        )

    @classmethod
    def pulse(
        cls,
        start: Any,
        amplitude: Any,
        duration: Any,
        baseline: Any = 0.0,
        unit: _UnitLike | None = None,
    ) -> "Stimulus":
        """Build a rectangular monophasic pulse waveform.

        The waveform is baseline before `start`, equal to `amplitude` for
        `duration`, then returns to baseline. Times must carry units.
        """

        start = units.require_time_ms(start, name="start")
        duration = units.require_time_ms(duration, name="duration")
        scale_shape = ("pulse",) if _is_zero_value(baseline) else None
        return cls(
            t=np.asarray([0.0, start, start + duration]),
            y=np.asarray([baseline, amplitude, baseline], dtype=object),
            mode="hold",
            y_unit=unit,
            _scale_shape=scale_shape,
        )

    @classmethod
    def biphasic(
        cls,
        start: Any,
        cathodic_amplitude: Any,
        cathodic_duration: Any,
        anodic_amplitude: Any | None = None,
        interphase: Any | None = None,
        anodic_first: bool = False,
        baseline: Any = 0.0,
        unit: _UnitLike | None = None,
    ) -> "Stimulus":
        """Build a charge-balanced biphasic pulse by default.

        The cathodic phase is stored as a negative amplitude and the anodic
        phase as a positive amplitude. If `anodic_amplitude` is omitted, the
        anodic phase uses the opposite amplitude and the same duration.
        """
        start = units.require_time_ms(start, name="start")
        cathodic_duration = units.require_time_ms(cathodic_duration, name="cathodic_duration")
        interphase = (
            0.0
            if interphase is None
            else units.require_time_ms(interphase, name="interphase")
        )
        amplitudes, inferred_unit = _coerce_amplitudes(
            [cathodic_amplitude, anodic_amplitude if anodic_amplitude is not None else 0.0, baseline],
            unit,
        )
        cathodic_amplitude = amplitudes[0]
        if anodic_amplitude is not None:
            anodic_amplitude = amplitudes[1]
        baseline = amplitudes[2]
        cath = -abs(cathodic_amplitude)

        balanced = anodic_amplitude is None
        if balanced:
            anodic_amplitude = abs(cathodic_amplitude)

        anod = abs(anodic_amplitude)
        anodic_duration = (
            abs(cath * cathodic_duration / anod)
            if anod != 0
            else cathodic_duration
        )

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
            y_unit=inferred_unit,
            _scale_shape=("balanced_biphasic", bool(anodic_first))
            if balanced and float(baseline) == 0.0
            else None,
        )

    @classmethod
    def sinus(
        cls,
        start: Any,
        duration: Any,
        amplitude: Any,
        frequency_khz: Any,
        offset: Any = 0.0,
        phase: float = 0.0,
        dt: Any | None = None,
        unit: _UnitLike | None = None,
    ) -> "Stimulus":
        """Build a sampled sinusoidal waveform.

        `frequency_khz` accepts plain kilohertz values or Pint-like frequency
        quantities. If `dt` is omitted, a conservative sampling step is chosen
        from the frequency.
        """

        start = units.require_time_ms(start, name="start")
        duration = units.require_time_ms(duration, name="duration")
        frequency_khz = units.to_scalar(frequency_khz, "kilohertz")
        amplitudes, inferred_unit = _coerce_amplitudes([amplitude, offset], unit)
        amplitude = amplitudes[0]
        offset = amplitudes[1]
        if dt is None:
            dt = 1.0 / (100.0 * frequency_khz)
        else:
            dt = units.require_time_ms(dt, name="dt")

        n = int(np.ceil(duration / dt)) + 1
        local_t = np.linspace(0.0, duration, n)
        y = offset + amplitude * np.sin(2.0 * np.pi * frequency_khz * local_t + phase)

        t = start + local_t
        return cls(t=t, y=y, mode="linear", y_unit=inferred_unit)

    @classmethod
    def ramp(
        cls,
        start: Any,
        duration: Any,
        start_value: Any,
        stop_value: Any,
        dt: Any,
        unit: _UnitLike | None = None,
    ) -> "Stimulus":
        """Build a sampled linear ramp waveform."""

        start = units.require_time_ms(start, name="start")
        duration = units.require_time_ms(duration, name="duration")
        dt = units.require_time_ms(dt, name="dt")
        amplitudes, inferred_unit = _coerce_amplitudes(
            [start_value, stop_value],
            unit,
        )
        n = int(np.ceil(duration / dt)) + 1
        local_t = np.linspace(0.0, duration, n)
        return cls(
            t=start + local_t,
            y=np.linspace(amplitudes[0], amplitudes[1], n),
            mode="linear",
            y_unit=inferred_unit,
        )

    @classmethod
    def from_samples(
        cls,
        t: _ArrayLike,
        y: _ArrayLike,
        mode: Literal["hold", "linear"] = "hold",
        unit: _UnitLike | None = None,
    ) -> "Stimulus":
        """Build a waveform from explicit samples.

        Parameters
        ----------
        t:
            Sample times, with units.
        y:
            Sample amplitudes. Pint-like values preserve their unit metadata.
        mode:
            Interpolation mode, either sample-and-hold (`"hold"`) or
            piecewise-linear (`"linear"`).
        unit:
            Optional amplitude unit for plain numeric samples.
        """

        return cls(
            t=units.require_time_array_ms(t, name="t"),
            y=np.asarray(y, dtype=object),
            mode=mode,
            y_unit=unit,
        )

    def as_unit(self, unit: _UnitLike) -> "Stimulus":
        """Return this waveform with amplitudes expressed in `unit`.

        Plain unitless amplitudes are interpreted as already being in `unit`.
        The original stimulus is not modified.
        """

        unit_label = units.unit_label(unit)
        if unit_label is None:
            if self.y_unit is None:
                return self
            return Stimulus(
                self.t,
                self.y,
                self.mode,
                y_unit=None,
                _scale_shape=self._scale_shape,
            )
        if self.y_unit == unit_label:
            return self
        if self.y_unit is None:
            return Stimulus(
                self.t,
                self.y,
                self.mode,
                y_unit=unit_label,
                _scale_shape=self._scale_shape,
            )
        y = units.to_array(units.Q_(self.y, self.y_unit), unit_label)
        return Stimulus(
            self.t,
            y,
            self.mode,
            y_unit=unit_label,
            _scale_shape=self._scale_shape,
        )

    def evaluate(self, t: _ArrayLike, *, unit: _UnitLike | None = None) -> np.ndarray | float:
        """Evaluate the stimulus on a time grid.

        Plain numeric times are interpreted as milliseconds. Pint-like times
        are converted automatically. If `unit` is given, amplitudes are returned
        as numeric values expressed in that unit.
        """

        t_query = units.to_ms_array(t, dtype=float)
        scalar_input = np.asarray(t_query).ndim == 0
        tq = np.atleast_1d(np.asarray(t_query, dtype=float))

        if self.mode == "linear":
            values = np.interp(tq, self.t, self.y, left=self.y[0], right=self.y[-1])
        else:
            idx = np.searchsorted(self.t, tq, side="right") - 1
            idx = np.clip(idx, 0, len(self.y) - 1)
            values = self.y[idx]

        values = np.asarray(values, dtype=float)
        unit_label = units.unit_label(unit)
        if unit_label is not None and self.y_unit is not None and self.y_unit != unit_label:
            values = units.to_array(units.Q_(values, self.y_unit), unit_label, dtype=float)

        if scalar_input:
            return float(np.asarray(values, dtype=float)[0])
        return values

    def plot(
        self,
        t: _ArrayLike | None = None,
        ax: Any | None = None,
        *,
        time_unit: _UnitLike = "millisecond",
        amplitude_unit: _UnitLike | None = None,
        label: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        grid: bool = True,
        **plot_kwargs: Any,
    ) -> Any:
        """Plot this stimulus on a Matplotlib axis.

        If `t` is omitted, a dense grid spanning the stored samples is created.
        The method returns the axis so callers can keep customizing the figure.
        """

        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        if t is None:
            t_min = float(self.t[0])
            t_max = float(self.t[-1])
            if t_max <= t_min:
                t_max = t_min + 1.0
            t = np.linspace(t_min, t_max, 1000)

        time_unit_label = units.unit_label(time_unit) or "millisecond"
        amplitude_unit_label = units.unit_label(amplitude_unit) or self.y_unit
        t_ms = units.to_ms_array(t, dtype=float)
        x = units.to_array(units.Q_(t_ms, "millisecond"), time_unit_label, dtype=float)
        y = self.evaluate(t_ms, unit=amplitude_unit_label)

        if label is not None:
            plot_kwargs.setdefault("label", label)
        plot_kwargs.setdefault("linewidth", 2.0)
        ax.plot(x, y, **plot_kwargs)

        time_display = units.short_unit_label(time_unit_label) or time_unit_label
        ax.set_xlabel(xlabel or f"Time [{time_display}]")
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        elif amplitude_unit_label is not None:
            amplitude_display = units.short_unit_label(amplitude_unit_label) or amplitude_unit_label
            ax.set_ylabel(f"Amplitude [{amplitude_display}]")
        else:
            ax.set_ylabel("Amplitude")
        if grid:
            ax.grid(True, alpha=0.3)
        return ax

    def shifted(self, dt: Any) -> "Stimulus":
        """Return a waveform shifted by unit-bearing `dt`."""

        return Stimulus(
            self.t + units.require_time_ms(dt, name="dt"),
            self.y,
            self.mode,
            y_unit=self.y_unit,
            _scale_shape=self._scale_shape,
        )

    def scaled(self, factor: float) -> "Stimulus":
        """Return a waveform with amplitudes multiplied by `factor`."""

        return Stimulus(
            self.t,
            float(factor) * self.y,
            self.mode,
            y_unit=self.y_unit,
            _scale_shape=self._scale_shape,
        )

    def offset(self, value: float) -> "Stimulus":
        """Return a waveform with an additive amplitude offset."""

        return Stimulus(
            self.t,
            self.y + float(value),
            self.mode,
            y_unit=self.y_unit,
        )

    def insert_samples(self, t_new: _ArrayLike) -> "Stimulus":
        """Return a stimulus evaluated on the union of old and new samples."""

        t_new_ms = units.require_time_array_ms(t_new, name="t_new", dtype=float)
        return self._with_samples_ms(np.unique(np.concatenate([self.t, t_new_ms])))

    def _with_samples_ms(self, t_ms: np.ndarray) -> "Stimulus":
        """Return this waveform evaluated on canonical millisecond samples."""

        return Stimulus(
            t_ms,
            self.evaluate(t_ms),
            self.mode,
            y_unit=self.y_unit,
        )

    def synchronize(self, other: "Stimulus") -> tuple["Stimulus", "Stimulus"]:
        """Return two waveforms evaluated on the same time grid."""

        if not isinstance(other, Stimulus):
            raise TypeError("other must be a Stimulus.")
        t_ms = np.unique(np.concatenate([self.t, other.t]))
        return self._with_samples_ms(t_ms), other._with_samples_ms(t_ms)

    def __add__(self, other: float | "Stimulus") -> "Stimulus":
        """Return the pointwise sum with a scalar or another stimulus."""

        if not isinstance(other, Stimulus):
            return Stimulus(
                self.t,
                self.y + float(other),
                self.mode,
                y_unit=self.y_unit,
            )

        left, right = self.synchronize(other)
        if left.y_unit is None and right.y_unit is not None:
            left = left.as_unit(right.y_unit)
        if left.y_unit is not None:
            right = right.as_unit(left.y_unit)
        return Stimulus(
            left.t,
            left.y + right.y,
            self.mode,
            y_unit=left.y_unit,
        )

    def __sub__(self, other: float | "Stimulus") -> "Stimulus":
        """Return the pointwise difference with a scalar or stimulus."""

        if not isinstance(other, Stimulus):
            return self.offset(-float(other))
        left, right = self.synchronize(other)
        if left.y_unit is None and right.y_unit is not None:
            left = left.as_unit(right.y_unit)
        if left.y_unit is not None:
            right = right.as_unit(left.y_unit)
        return Stimulus(
            left.t,
            left.y - right.y,
            self.mode,
            y_unit=left.y_unit,
        )

    def __mul__(self, other: float | "Stimulus") -> "Stimulus":
        """Return a scaled waveform or the pointwise waveform product."""

        if not isinstance(other, Stimulus):
            return self.scaled(float(other))
        left, right = self.synchronize(other)
        return Stimulus(left.t, left.y * right.y, self.mode)

    __rmul__ = __mul__


__all__ = ["Stimulus"]
