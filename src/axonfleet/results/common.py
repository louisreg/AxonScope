"""Shared one-axon result facade."""

from __future__ import annotations

from typing import Any


class SingleAxonResultMixin:
    """Common analysis and view surface for one-axon result objects."""

    def time_values(self, *, unit: Any = "millisecond"):
        """Return result times as plain values in `unit`."""

        from axonfleet.results.views import time_values

        return time_values(self, unit=unit)

    def position_values(self, *, unit: Any = "micrometer"):
        """Return recorded axon positions as plain values in `unit`."""

        from axonfleet.results.views import position_values

        return position_values(self, unit=unit)

    def voltage_values(self, *, unit: Any = "millivolt"):
        """Return membrane voltages as plain values in `unit`."""

        from axonfleet.results.views import voltage_values

        return voltage_values(self, unit=unit)

    def peak_voltage_values(self, *, unit: Any = "millivolt"):
        """Return the peak membrane voltage for each recorded column."""

        from axonfleet.results.views import peak_voltage_values

        return peak_voltage_values(self, unit=unit)

    def analyze(self, *definitions: Any) -> Any:
        """Evaluate public analysis definitions on this result."""

        from axonfleet.analysis import analyze

        return analyze(self, *definitions)

    def report(self, *definitions: Any) -> Any:
        """Return an analysis report for one or more definitions."""

        from axonfleet.analysis import AnalysisReport, analyze

        analyzed = analyze(self, *definitions)
        if hasattr(analyzed, "analyses"):
            return analyzed
        return AnalysisReport(analyses=(analyzed,))

    def nearest_position_index(self, position: Any) -> int:
        """Return the recorded column nearest to `position`."""

        from axonfleet.results.views import nearest_position_index

        return nearest_position_index(self, position)

    def trace_values(
        self,
        *,
        position: Any | None = None,
        index: int | None = None,
        time_unit: Any = "millisecond",
        voltage_unit: Any = "millivolt",
    ):
        """Return one voltage trace as ``(time, Vm)`` arrays."""

        from axonfleet.results.views import trace_values

        return trace_values(
            self,
            position=position,
            index=index,
            time_unit=time_unit,
            voltage_unit=voltage_unit,
        )

    def plot_trace(
        self,
        ax: Any | None = None,
        *,
        position: Any | None = None,
        index: int | None = None,
        time_unit: Any = "millisecond",
        voltage_unit: Any = "millivolt",
        label: str | None = None,
        title: str | None = None,
        grid: bool = True,
        **plot_kwargs: Any,
    ) -> Any:
        """Plot one membrane-voltage trace."""

        from axonfleet.results.views import plot_trace

        return plot_trace(
            self,
            ax=ax,
            position=position,
            index=index,
            time_unit=time_unit,
            voltage_unit=voltage_unit,
            label=label,
            title=title,
            grid=grid,
            **plot_kwargs,
        )

    def plot_traces(
        self,
        ax: Any | None = None,
        *,
        positions: Any | None = None,
        indices: Any | None = None,
        labels: Any | None = None,
        time_unit: Any = "millisecond",
        voltage_unit: Any = "millivolt",
        title: str = "Vm traces",
        grid: bool = True,
        legend: bool = True,
        **plot_kwargs: Any,
    ) -> Any:
        """Plot several membrane-voltage traces."""

        from axonfleet.results.views import plot_traces

        return plot_traces(
            self,
            ax=ax,
            positions=positions,
            indices=indices,
            labels=labels,
            time_unit=time_unit,
            voltage_unit=voltage_unit,
            title=title,
            grid=grid,
            legend=legend,
            **plot_kwargs,
        )

    def plot_peak_voltage(
        self,
        ax: Any | None = None,
        *,
        position_unit: Any = "micrometer",
        voltage_unit: Any = "millivolt",
        title: str = "Peak Vm by recorded position",
        grid: bool = True,
        **plot_kwargs: Any,
    ) -> Any:
        """Plot peak membrane voltage over recorded positions."""

        from axonfleet.results.views import plot_peak_voltage

        return plot_peak_voltage(
            self,
            ax=ax,
            position_unit=position_unit,
            voltage_unit=voltage_unit,
            title=title,
            grid=grid,
            **plot_kwargs,
        )

    def plot_recorded_axis(
        self,
        ax: Any | None = None,
        *,
        selectors: Any | None = None,
        markers: Any | None = None,
        position_unit: Any = "micrometer",
        title: str = "Recorded Vm positions",
        grid: bool = True,
    ) -> Any:
        """Plot recorded Vm columns and optional position selectors."""

        from axonfleet.results.views import plot_recorded_axis

        return plot_recorded_axis(
            self,
            ax=ax,
            selectors=selectors,
            markers=markers,
            position_unit=position_unit,
            title=title,
            grid=grid,
        )

    def plot_recording_group(
        self,
        group: str,
        ax: Any | None = None,
        *,
        position: Any | None = None,
        index: int | None = None,
        labels: Any | None = None,
        time_unit: Any = "millisecond",
        title: str | None = None,
        ylabel: str | None = None,
        grid: bool = True,
        legend: bool = True,
        **plot_kwargs: Any,
    ) -> Any:
        """Plot all channels from one named recording group."""

        from axonfleet.results.views import plot_recording_group

        return plot_recording_group(
            self,
            group,
            ax=ax,
            position=position,
            index=index,
            labels=labels,
            time_unit=time_unit,
            title=title,
            ylabel=ylabel,
            grid=grid,
            legend=legend,
            **plot_kwargs,
        )

    def plot_map(
        self,
        ax: Any | None = None,
        *,
        time_unit: Any = "millisecond",
        position_unit: Any = "micrometer",
        voltage_unit: Any = "millivolt",
        cmap: str = "viridis",
        colorbar: bool = True,
        title: str = "Vm propagation",
        **imshow_kwargs: Any,
    ) -> Any:
        """Plot the membrane-voltage space-time map."""

        from axonfleet.results.views import plot_map

        return plot_map(
            self,
            ax=ax,
            time_unit=time_unit,
            position_unit=position_unit,
            voltage_unit=voltage_unit,
            cmap=cmap,
            colorbar=colorbar,
            title=title,
            **imshow_kwargs,
        )

__all__ = ["SingleAxonResultMixin"]
