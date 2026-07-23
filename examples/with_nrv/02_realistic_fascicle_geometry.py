"""Run AxonFleet on a realistic NRV histology-contour nerve.

Run:
    python examples/with_nrv/02_realistic_fascicle_geometry.py

NRV owns the external geometry, fiber placement, and LIFE/FEM footprint
sampling. AxonFleet receives intrinsic axon layouts plus sampled
`ExtracellularFootprint` objects, then runs the recruitment sweep.
"""

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _fascicle_recruitment_common import (  # noqa: E402
    ExampleConfig,
    build_realistic_histology_geometry,
    run_fascicle_recruitment_example,
)


def main(config: ExampleConfig | None = None):
    if config is None:
        config = ExampleConfig()
    return run_fascicle_recruitment_example(
        config=config,
        build_geometry=build_realistic_histology_geometry,
        geometry_label="histology-contour",
    )


if __name__ == "__main__":
    main()
