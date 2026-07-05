"""Single-cable myelinated-chain pseudo-double validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

import numpy as np

import axonscope as axs


class PseudoDoubleSegmentType(IntEnum):
    """Segment classes preserved by the pseudo-double single-chain model."""

    NODE = 0
    MYSA = 1
    FLUT = 2
    STIN = 3


_SEGMENT_KEYS = ("node", "mysa", "flut", "stin")
_SEGMENT_TYPE_BY_NAME = {
    "node": PseudoDoubleSegmentType.NODE,
    "mysa": PseudoDoubleSegmentType.MYSA,
    "flut": PseudoDoubleSegmentType.FLUT,
    "stin": PseudoDoubleSegmentType.STIN,
}
_DEFAULT_NODE_GL_S_CM2 = 0.007
_DEFAULT_NODE_EL_MV = -90.0
_DEFAULT_INTERNODE_EL_MV = -80.0


@dataclass(frozen=True)
class PseudoDoubleSingleChainConfig:
    """Experimental one-voltage NODE/MYSA/FLUT/STIN chain parameters."""

    vext_scale: float = 1.0
    cm_scale_node: float = 1.0
    cm_scale_mysa: float = 1.0
    cm_scale_flut: float = 1.0
    cm_scale_stin: float = 1.0
    gleak_scale_node: float = 1.0
    gleak_scale_mysa: float = 1.0
    gleak_scale_flut: float = 1.0
    gleak_scale_stin: float = 1.0
    axial_resistance_scale: float = 1.0
    vext_alpha_node: float = 1.0
    vext_alpha_mysa: float = 1.0
    vext_alpha_flut: float = 1.0
    vext_alpha_stin: float = 1.0
    use_series_capacitance: bool = True
    use_series_leak: bool = True
    active_nodes_only: bool = True

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "vext_scale": float(self.vext_scale),
            "cm_scale_node": float(self.cm_scale_node),
            "cm_scale_mysa": float(self.cm_scale_mysa),
            "cm_scale_flut": float(self.cm_scale_flut),
            "cm_scale_stin": float(self.cm_scale_stin),
            "gleak_scale_node": float(self.gleak_scale_node),
            "gleak_scale_mysa": float(self.gleak_scale_mysa),
            "gleak_scale_flut": float(self.gleak_scale_flut),
            "gleak_scale_stin": float(self.gleak_scale_stin),
            "axial_resistance_scale": float(self.axial_resistance_scale),
            "vext_alpha_node": float(self.vext_alpha_node),
            "vext_alpha_mysa": float(self.vext_alpha_mysa),
            "vext_alpha_flut": float(self.vext_alpha_flut),
            "vext_alpha_stin": float(self.vext_alpha_stin),
            "use_series_capacitance": bool(self.use_series_capacitance),
            "use_series_leak": bool(self.use_series_leak),
            "active_nodes_only": bool(self.active_nodes_only),
        }


def segment_scaled_point_source_stimulation(
    electrode: axs.analytical.PointSourceElectrode,
    *,
    positions_um: Sequence[float],
    alpha: Sequence[float],
    sigma: object,
    drive_id: str = "segment_scaled_point_source",
) -> axs.ExtracellularStimulation:
    """Build a pseudo-double point-source stimulation scaled by alpha(x).

    This belongs to the pseudo-double validation benchmark. It samples the
    public point-source helper, then applies the experimental segment coupling
    directly to the typed footprint consumed by solvers.
    """

    if not isinstance(electrode, axs.analytical.PointSourceElectrode):
        raise TypeError("electrode must be a PointSourceElectrode.")
    if electrode.stimulus is None:
        raise ValueError("electrode must carry an attached stimulus.")
    positions = np.asarray(positions_um, dtype=float)
    scale = np.asarray(alpha, dtype=float)
    if positions.ndim != 1 or scale.ndim != 1:
        raise ValueError("positions_um and alpha must be one-dimensional.")
    if positions.shape != scale.shape:
        raise ValueError("positions_um and alpha must have the same shape.")
    if positions.size < 2:
        raise ValueError("at least two positions are required.")
    if not np.all(np.diff(positions) > 0.0):
        raise ValueError("positions_um must be strictly increasing.")
    if not np.all(np.isfinite(scale)):
        raise ValueError("alpha must contain finite values.")

    base = axs.analytical.point_source_footprint(
        electrode,
        positions * axs.um,
        sigma=sigma,
    )
    footprint = axs.ExtracellularFootprint.shared(
        values=base.values_for_axon() * scale,
        positions=positions * axs.um,
        source_id=base.source_id,
        metadata={
            **dict(base.metadata),
            "builder": "benchmark.pseudo_double.segment_scaled_point_source_stimulation",
            "segment_scaled": True,
        },
    )
    drive = axs.ExtracellularDrive(
        id=axs.DriveId(drive_id),
        footprint=footprint,
        stimulus=electrode.stimulus,
        metadata={"source": "pseudo_double_segment_scaled_point_source"},
    )
    return axs.ExtracellularStimulation([drive])


def build_pseudo_double_single_chain_mrg(
    *,
    diameter_um: float,
    nodes: int,
    config: PseudoDoubleSingleChainConfig | None = None,
) -> axs.axons.Myelinated:
    """Build a single-cable MRG-like chain preserving NODE/MYSA/FLUT/STIN."""

    config = config or PseudoDoubleSingleChainConfig()
    _validate_single_chain_config(config)
    geometry = axs.axons.build_mrg_like_geometry(
        diameter=float(diameter_um) * axs.um,
        nodes=int(nodes),
    )
    elements = []
    for index, name in enumerate(geometry.section_names):
        key = _segment_key(name)
        section = axs.axons.Section(
            name,
            membrane=_membrane_for_segment(key, geometry, index, config),
            diameter=geometry.diam_um[index] * axs.um,
            Ra=(geometry.Ra_ohm_cm[index] * config.axial_resistance_scale) * axs.ohm_cm,
            Cm=_effective_cm_uF_cm2(geometry, index, key, config) * axs.uF_per_cm2,
            periaxonal=None,
            tags=("myelinated", "node") if key == "node" else ("myelinated", key),
        )
        elements.append(
            axs.axons.LayoutElement(
                section,
                length=geometry.lengths_um[index] * axs.um,
                compartments=1,
            )
        )
    axon = axs.axons.Myelinated(
        layout=axs.axons.Layout(elements),
        formulation=axs.axons.CableFormulation.SINGLE_CABLE,
        diameter=float(diameter_um) * axs.um,
    )
    return axon


def single_chain_segment_type(axon: axs.axons.Myelinated) -> np.ndarray:
    """Return per-compartment NODE/MYSA/FLUT/STIN integer segment codes."""

    values: list[int] = []
    for element in axon.layout.elements:
        key = _segment_key(element.section.name)
        values.extend([int(_SEGMENT_TYPE_BY_NAME[key])] * int(element.compartments))
    return np.asarray(values, dtype=np.int8)


def single_chain_vext_alpha(
    axon: axs.axons.Myelinated,
    config: PseudoDoubleSingleChainConfig | None = None,
) -> np.ndarray:
    """Return per-compartment extracellular coupling alpha values."""

    config = config or PseudoDoubleSingleChainConfig()
    scale = {
        "node": config.vext_alpha_node,
        "mysa": config.vext_alpha_mysa,
        "flut": config.vext_alpha_flut,
        "stin": config.vext_alpha_stin,
    }
    values: list[float] = []
    for element in axon.layout.elements:
        key = _segment_key(element.section.name)
        values.extend([float(config.vext_scale) * float(scale[key])] * int(element.compartments))
    return np.asarray(values, dtype=float)


def single_chain_segment_counts(axon: axs.axons.Myelinated) -> dict[str, int]:
    """Return segment counts for the flattened single-chain layout."""

    counts = {key: 0 for key in _SEGMENT_KEYS}
    for element in axon.layout.elements:
        key = _segment_key(element.section.name)
        counts[key] += int(element.compartments)
    return counts


def _validate_single_chain_config(config: PseudoDoubleSingleChainConfig) -> None:
    for name, value in config.as_dict().items():
        if isinstance(value, bool):
            continue
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite.")
        if name.endswith("_scale") or name.startswith("cm_scale") or name.startswith("gleak_scale"):
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if name.startswith("vext_alpha"):
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative.")
    if not config.active_nodes_only:
        raise NotImplementedError("pseudo-chain v0 supports active_nodes_only=True only.")


def _segment_key(name: str) -> str:
    key = str(name).strip().lower()
    if key not in _SEGMENT_TYPE_BY_NAME:
        raise ValueError(f"unsupported pseudo-chain segment {name!r}.")
    return key


def _scale_for_segment(config: PseudoDoubleSingleChainConfig, prefix: str, key: str) -> float:
    return float(getattr(config, f"{prefix}_{key}"))


def _series_equivalent_float(left: float, right: float, *, eps: float = 1e-12) -> float:
    if right <= eps:
        return float(left)
    return float((left * right) / max(left + right, eps))


def _effective_cm_uF_cm2(
    geometry,
    index: int,
    key: str,
    config: PseudoDoubleSingleChainConfig,
) -> float:
    axolemma = float(geometry.Cm_uF_cm2[index])
    myelin = float(geometry.periaxonal_layers[index].radial_capacitance_uF_cm2)
    value = (
        _series_equivalent_float(axolemma, myelin)
        if config.use_series_capacitance and key != "node"
        else axolemma
    )
    return max(value * _scale_for_segment(config, "cm_scale", key), 1e-12)


def _effective_leak_mS_cm2(
    geometry,
    index: int,
    key: str,
    config: PseudoDoubleSingleChainConfig,
) -> float:
    axolemma = float(geometry.leak_mS_cm2[index])
    myelin = float(geometry.periaxonal_layers[index].radial_conductance_S_cm2) * 1e3
    value = (
        _series_equivalent_float(axolemma, myelin)
        if config.use_series_leak and key != "node"
        else axolemma
    )
    return max(value * _scale_for_segment(config, "gleak_scale", key), 1e-12)


def _membrane_for_segment(
    key: str,
    geometry,
    index: int,
    config: PseudoDoubleSingleChainConfig,
):
    if key == "node":
        return axs.membranes.AxNode(
            gl=(
                _DEFAULT_NODE_GL_S_CM2
                * 1e3
                * config.gleak_scale_node
                * axs.mS_per_cm2
            ),
            el=_DEFAULT_NODE_EL_MV * axs.mV,
        )
    leak_mS_cm2 = _effective_leak_mS_cm2(geometry, index, key, config)
    return axs.membranes.Passive(
        Rm=1e3 / leak_mS_cm2,
        EL=_DEFAULT_INTERNODE_EL_MV,
    )


__all__ = [
    "PseudoDoubleSegmentType",
    "PseudoDoubleSingleChainConfig",
    "build_pseudo_double_single_chain_mrg",
    "segment_scaled_point_source_stimulation",
    "single_chain_segment_counts",
    "single_chain_segment_type",
    "single_chain_vext_alpha",
]
