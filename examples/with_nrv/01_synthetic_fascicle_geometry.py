"""Run AxonFleet on the synthetic two-fascicle NRV tutorial geometry.

Run:
    python examples/with_nrv/01_synthetic_fascicle_geometry.py

The geometry follows NRV tutorial 4: a cylindrical nerve with one circular
fascicle and one custom CShape elliptical fascicle. The remaining workflow is
the same as the realistic fascicle example: NRV owns geometry/fiber placement
and LIFE/FEM footprint sampling, then AxonFleet runs the recruitment sweep.
"""

from __future__ import annotations

from dataclasses import replace
from math import cos, pi, sin
import sys
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _fascicle_recruitment_common import (  # noqa: E402
    ExampleConfig,
    NrvGeometry,
    run_fascicle_recruitment_example,
)


def main(config: ExampleConfig | None = None):
    if config is None:
        config = ExampleConfig(
            nerve_diameter_um=500.0,
            nerve_length_um=5_000.0,
            delta_trace_um=5.0,
            life_fascicle_id="2",
        )
    else:
        config = replace(
            config,
            nerve_diameter_um=500.0,
            nerve_length_um=5_000.0,
            delta_trace_um=5.0,
            life_fascicle_id="2",
        )
    return run_fascicle_recruitment_example(
        config=config,
        build_geometry=build_synthetic_tutorial_geometry,
        geometry_label="synthetic tutorial",
    )


def build_synthetic_tutorial_geometry(
    nrv_module: Any,
    config: ExampleConfig,
) -> NrvGeometry:
    outer_d_mm = 5
    nerve = nrv_module.nerve(
        length=int(config.nerve_length_um),
        diameter=int(config.nerve_diameter_um),
        Outer_D=outer_d_mm,
    )

    fascicle_1_diameter_um = 200.0
    fascicle_1_center = (-100.0, 0.0)
    fascicle_1 = nrv_module.fascicle(diameter=fascicle_1_diameter_um, ID=1)
    nerve.add_fascicle(
        fascicle=fascicle_1,
        y=fascicle_1_center[0],
        z=fascicle_1_center[1],
    )

    fascicle_2_diameter_um = (220.0, 110.0)
    fascicle_2_center = (100.0, 0.0)
    fascicle_2_geometry = nrv_module.create_cshape(
        center=fascicle_2_center,
        diameter=fascicle_2_diameter_um,
        rot=90,
        degree=True,
    )
    fascicle_2 = nrv_module.fascicle(ID=2)
    fascicle_2.set_geometry(geometry=fascicle_2_geometry)
    nerve.add_fascicle(fascicle=fascicle_2)

    nerve_contour = _circle_contour(
        center=(0.0, 0.0),
        radius=config.nerve_diameter_um / 2.0,
    )
    fascicle_contours = (
        _circle_contour(
            center=fascicle_1_center,
            radius=fascicle_1_diameter_um / 2.0,
        ),
        _ellipse_contour(
            center=fascicle_2_center,
            diameter=fascicle_2_diameter_um,
            rotation_deg=90.0,
        ),
    )
    return NrvGeometry(
        nerve=nerve,
        nerve_contour=nerve_contour,
        fascicle_contours=fascicle_contours,
        life_fascicle_id="2",
    )


def _circle_contour(
    *,
    center: tuple[float, float],
    radius: float,
    points: int = 96,
) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * pi, points, endpoint=False)
    return np.column_stack(
        [
            center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
        ]
    )


def _ellipse_contour(
    *,
    center: tuple[float, float],
    diameter: tuple[float, float],
    rotation_deg: float,
    points: int = 96,
) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * pi, points, endpoint=False)
    semi_a = diameter[0] / 2.0
    semi_b = diameter[1] / 2.0
    rotation = rotation_deg * pi / 180.0
    x = semi_a * np.cos(theta)
    y = semi_b * np.sin(theta)
    return np.column_stack(
        [
            center[0] + x * cos(rotation) - y * sin(rotation),
            center[1] + x * sin(rotation) + y * cos(rotation),
        ]
    )


if __name__ == "__main__":
    main()
