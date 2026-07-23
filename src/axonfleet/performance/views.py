"""View helpers for simulation performance estimates."""

from __future__ import annotations

from typing import Any, TextIO

from axonfleet.views.summary import rows_to_dataframe


def simulation_estimate_rows(
    estimate: Any,
    *,
    section: str = "items",
) -> tuple[dict[str, Any], ...]:
    """Return table rows for one simulation estimate section."""

    if section == "items":
        return tuple(item.to_dict() for item in estimate.items)
    if section == "groups":
        return tuple(group.to_dict() for group in estimate.groups)
    raise ValueError("section must be 'items' or 'groups'.")


def simulation_estimate_to_dataframe(
    estimate: Any,
    *,
    section: str = "items",
) -> Any:
    """Return one simulation estimate section as a pandas DataFrame."""

    return rows_to_dataframe(simulation_estimate_rows(estimate, section=section))


def format_simulation_estimate(estimate: Any) -> str:
    """Format a readable table-like estimate report."""

    lines = [
        "AxonFleet simulation estimate",
        "summary:",
        f"  axons={estimate.axon_count}, groups={len(estimate.groups)}, Nt={estimate.step_count}",
        f"  duration={estimate.duration_ms:g} ms, dt={estimate.dt_ms:g} ms",
        (
            f"  runtime={estimate.runtime.value}, device={estimate.device.kind}, "
            f"precision={estimate.precision.solver_dtype}, max_Nx={estimate.max_compartments}"
        ),
        f"  total={estimate.total_mib:.3f} MiB, retained={estimate.retained_mib:.3f} MiB",
        "groups:",
        "  id kind                    rows Nx  pad  rec->kernel  observer                 retained total",
    ]
    for group in estimate.groups:
        lines.append(
            "  "
            f"{group.group_id:<2d} {group.batch_kind:<23s} "
            f"{group.size:<4d} {group.nx:<3d} {group.padded_compartments:<4d} "
            f"{group.recording_mode}->{group.kernel_recording_mode:<7s} "
            f"{group.observer_output:<24s} "
            f"{_bytes_text(group.retained_bytes):>8s} {_bytes_text(group.total_bytes):>8s}"
        )
    lines.extend(
        [
            "arrays:",
            "  name                                   shape            dtype      kept       MiB role",
        ]
    )
    for item in estimate.items:
        shape = "x".join(str(dim) for dim in item.shape) or "scalar"
        kept = "retained" if item.retained else "temporary"
        lines.append(
            f"  {item.name:38s} {shape:16s} {item.dtype:10s} "
            f"{kept:9s} {item.mib:8.3f} {item.role}"
        )
    if estimate.warnings:
        lines.append("warnings:")
        lines.extend(f"  - {warning}" for warning in estimate.warnings)
    if estimate.recommendations:
        lines.append("recommendations:")
        lines.extend(f"  - {recommendation}" for recommendation in estimate.recommendations)
    return "\n".join(lines)


def print_simulation_estimate(
    estimate: Any,
    file: TextIO | None = None,
    *,
    rich: bool | None = None,
) -> None:
    """Print the estimate report, using Rich tables for terminals."""

    if rich is False or file is not None:
        print(format_simulation_estimate(estimate), file=file)
        return

    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table

    console = Console(width=120)
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("axons", str(estimate.axon_count))
    summary.add_row("groups", str(len(estimate.groups)))
    summary.add_row("steps", str(estimate.step_count))
    summary.add_row("time", f"{estimate.duration_ms:g} ms @ {estimate.dt_ms:g} ms")
    summary.add_row("runtime", f"{estimate.runtime.value} / {estimate.device.kind}")
    summary.add_row("precision", estimate.precision.solver_dtype)
    summary.add_row("total", _bytes_text(estimate.total_bytes))
    summary.add_row("retained", _bytes_text(estimate.retained_bytes))

    groups = Table(title="Dispatch Groups", show_lines=False)
    for column in (
        "id",
        "kind",
        "rows",
        "Nx",
        "pad",
        "recording",
        "observer",
        "retained",
        "total",
    ):
        groups.add_column(column, overflow="fold")
    for group in estimate.groups:
        groups.add_row(
            str(group.group_id),
            group.batch_kind,
            str(group.size),
            str(group.nx),
            str(group.padded_compartments),
            f"{group.recording_mode}->{group.kernel_recording_mode}",
            group.observer_output,
            _bytes_text(group.retained_bytes),
            _bytes_text(group.total_bytes),
        )

    arrays = Table(title="Estimated Arrays", show_lines=False)
    for column in ("name", "shape", "dtype", "kept", "MiB", "role"):
        arrays.add_column(column, overflow="fold")
    for item in estimate.items:
        arrays.add_row(
            item.name,
            "x".join(str(dim) for dim in item.shape) or "scalar",
            item.dtype,
            "retained" if item.retained else "temporary",
            f"{item.mib:.3f}",
            item.role,
        )

    console.print(
        Panel(
            Group(summary, groups, arrays),
            title="AxonFleet simulation estimate",
            expand=False,
        )
    )


def _bytes_text(value: int) -> str:
    if value == 0:
        return "0"
    if value < 1024:
        return f"{value} B"
    if value < 1024**2:
        return f"{value / 1024:.2f} KiB"
    return f"{value / (1024**2):.3f} MiB"


__all__ = [
    "format_simulation_estimate",
    "print_simulation_estimate",
    "simulation_estimate_rows",
    "simulation_estimate_to_dataframe",
]
