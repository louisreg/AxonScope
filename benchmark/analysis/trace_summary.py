from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize benchmark event and trace artifacts.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    events_path = args.run_dir / "events.jsonl"
    if not events_path.is_file():
        print(f"No events.jsonl found in {args.run_dir}")
        return 1

    totals: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)
    for event in _read_events(events_path):
        name = str(event.get("name", "unknown"))
        totals[name] += int(event.get("duration_ns", 0) or 0)
        counts[name] += 1

    print("stage,count,total_ms")
    for name in sorted(totals):
        print(f"{name},{counts[name]},{totals[name] / 1_000_000.0:.3f}")

    trace_files = sorted(args.run_dir.glob("**/*.trace.json.gz"))
    profile_files = sorted(args.run_dir.glob("**/*.prof"))
    if trace_files or profile_files:
        print("\nartifacts:")
        for path in (*trace_files, *profile_files):
            print(f"  {path}")
    return 0


def _read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


if __name__ == "__main__":
    raise SystemExit(main())

