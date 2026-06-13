from __future__ import annotations

import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from axonscope.benchmarking import collect_benchmark_metadata
from axonscope.stimulation import ExtracellularContext
from axonscope.dispatcher.runtime_batches import scale_extracellular_contexts


@dataclass(frozen=True)
class TimingStats:
    repeats: int
    mean_s: float
    median_s: float
    min_s: float
    max_s: float
    std_s: float

    @classmethod
    def from_samples(cls, samples_s: Sequence[float]) -> "TimingStats":
        samples = [float(value) for value in samples_s]
        return cls(
            repeats=len(samples),
            mean_s=float(statistics.fmean(samples)),
            median_s=float(statistics.median(samples)),
            min_s=float(min(samples)),
            max_s=float(max(samples)),
            std_s=float(statistics.pstdev(samples)) if len(samples) > 1 else 0.0,
        )


def scaled_context_batch(
    contexts: Sequence[ExtracellularContext],
    *,
    batch_size: int,
    start: float = 0.5,
    stop: float = 1.5,
) -> list[tuple[ExtracellularContext, ...]]:
    """Build a simple amplitude sweep over one shared context set."""

    scales = np.linspace(start, stop, batch_size)
    return [
        scale_extracellular_contexts(contexts, float(scale))
        for scale in scales
    ]


def time_call(fn: Callable[[], object]) -> tuple[float, object]:
    start = time.perf_counter()
    value = fn()
    block_until_ready(value)
    return time.perf_counter() - start, value


def block_until_ready(value: object) -> None:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            block_until_ready(item)


def write_rows(
    rows: Sequence[object],
    out_dir: Path,
    *,
    prefix: str,
    metadata: dict[str, object],
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{prefix}.json"
    csv_path = out_dir / f"{prefix}.csv"

    payload = {
        "schema_version": 1,
        "metadata": {**collect_benchmark_metadata(), **metadata},
        "results": [row_to_dict(row) for row in rows],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    flat_rows = [flatten_row(row_to_dict(row)) for row in rows]
    fieldnames = sorted({key for row in flat_rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)
    return json_path, csv_path


def row_to_dict(row: object) -> dict[str, object]:
    return asdict(row)


def flatten_row(row: dict[str, object]) -> dict[str, object]:
    flat = dict(row)
    for key in ("scalar_warm", "batch_warm", "warm"):
        if key not in flat:
            continue
        stats = flat.pop(key)
        if isinstance(stats, dict):
            for stat_name, value in stats.items():
                flat[f"{key}.{stat_name}"] = value
    return flat
