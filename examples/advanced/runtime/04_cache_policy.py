"""Inspect and clean AxonFleet's deterministic persistent artifact cache."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import axonfleet as axs


def main() -> None:
    previous = os.environ.get("AXONFLEET_CACHE")
    try:
        with TemporaryDirectory() as temporary:
            os.environ["AXONFLEET_CACHE"] = str(Path(temporary) / "cache")

            axs.membranes.inspect_generated_code(
                axs.membranes.HodgkinHuxley(),
                files=("jax_model.py",),
            )
            snapshot = axs.cache.inspect()
            generated = next(
                section
                for section in snapshot.sections
                if section.name == "model_codegen"
            )

            print(f"cache: {snapshot.directory}")
            print(f"generated files: {generated.file_count}")
            print(f"total bytes: {snapshot.bytes}")

            cleaned = axs.cache.clean()
            print(f"files after clean: {cleaned.file_count}")
    finally:
        if previous is None:
            os.environ.pop("AXONFLEET_CACHE", None)
        else:
            os.environ["AXONFLEET_CACHE"] = previous


if __name__ == "__main__":
    main()
