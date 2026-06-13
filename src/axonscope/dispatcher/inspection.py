"""Human-readable inspection helpers for pool dispatch plans."""

from __future__ import annotations

from typing import Any, Sequence

from axonscope.axon_simulation import AxonSimulation
from axonscope.axons.axon import Axon
from axonscope.dispatcher.plan import DispatchGroup, DispatchPlan, build_dispatch_plan


PoolLike = DispatchPlan | Sequence[Axon | AxonSimulation]


def dispatch_method_for_group(group: DispatchGroup) -> str:
    """Return the execution method implied by one dispatch group."""

    if group.size < 2:
        return "scalar"
    prefix = "batch" if group.geometry_shared else "parameter-batch"
    return f"{prefix}-{group.mode}-cable"


def as_dispatch_plan(pool_or_plan: PoolLike) -> DispatchPlan:
    """Return a dispatch plan from either a plan or a public pool."""

    if isinstance(pool_or_plan, DispatchPlan):
        return pool_or_plan
    return build_dispatch_plan(pool_or_plan)


def describe_dispatch_plan(pool_or_plan: PoolLike) -> str:
    """Return a compact text table describing dispatcher groups."""

    plan = as_dispatch_plan(pool_or_plan)
    lines = [
        f"Dispatch plan: {len(plan.items)} rows, {len(plan.groups)} groups",
        "group  method                         size  Nx   geometry    pool indices",
        "-----  -----------------------------  ----  ---  ----------  ------------",
    ]
    for group in plan.groups:
        if group.has_padding:
            geometry = "padded"
        else:
            geometry = "shared" if group.geometry_shared else "batched"
        indices = ", ".join(str(index) for index in group.pool_indices)
        lines.append(
            f"{group.group_id:>5}  "
            f"{dispatch_method_for_group(group):<29}  "
            f"{group.size:>4}  "
            f"{group.nx:>3}  "
            f"{geometry:<10}  "
            f"[{indices}]"
        )
    return "\n".join(lines)


def print_dispatch_plan(pool_or_plan: PoolLike) -> None:
    """Print a compact dispatch-plan table."""

    print(describe_dispatch_plan(pool_or_plan))


def plot_dispatch_plan(
    pool_or_plan: PoolLike,
    ax: Any | None = None,
    *,
    show_indices: bool = True,
) -> Any:
    """Plot dispatch groups as rows over input pool indices."""

    plan = as_dispatch_plan(pool_or_plan)
    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=(7.0, max(2.4, 0.45 * len(plan.groups) + 1.2)))

    palette = {
        "scalar": "tab:gray",
        "batch-single-cable": "tab:blue",
        "batch-double-cable": "tab:green",
        "parameter-batch-single-cable": "tab:orange",
        "parameter-batch-double-cable": "tab:red",
    }
    for y, group in enumerate(plan.groups):
        method = dispatch_method_for_group(group)
        indices = group.pool_indices
        color = palette.get(method, "tab:purple")
        ax.scatter(indices, [y] * len(indices), s=90, color=color, label=method)
        if len(indices) > 1:
            ax.plot(indices, [y] * len(indices), color=color, alpha=0.45, linewidth=3.0)
        if show_indices:
            for index in indices:
                ax.text(index, y + 0.12, str(index), ha="center", va="bottom", fontsize=8)

    labels = [
        f"G{group.group_id}: {dispatch_method_for_group(group)} "
        f"(B={group.size}, Nx={group.nx}"
        f"{', padded' if group.has_padding else ''})"
        for group in plan.groups
    ]
    ax.set_yticks(range(len(plan.groups)), labels)
    ax.set_xlabel("Input pool index")
    ax.set_title("Dispatch plan")
    ax.grid(True, axis="x", alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    if unique:
        ax.legend(unique.values(), unique.keys(), loc="best", fontsize=8)
    return ax


__all__ = [
    "as_dispatch_plan",
    "describe_dispatch_plan",
    "dispatch_method_for_group",
    "plot_dispatch_plan",
    "print_dispatch_plan",
]
