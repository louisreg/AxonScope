from __future__ import annotations

import numpy as np

from benchmark.pseudo_double.single_chain import (
    PseudoDoubleSegmentType,
    PseudoDoubleSingleChainConfig,
    build_pseudo_double_single_chain_mrg,
    segment_scaled_point_source_stimulation,
    single_chain_segment_counts,
    single_chain_segment_type,
    single_chain_vext_alpha,
)


def test_single_chain_preserves_mrg_segment_taxonomy():
    axon = build_pseudo_double_single_chain_mrg(diameter_um=5.7, nodes=3)

    counts = single_chain_segment_counts(axon)
    segment_type = single_chain_segment_type(axon)

    assert axon.resolved_formulation == "single-cable"
    assert counts["node"] == 3
    assert counts["mysa"] > 0
    assert counts["flut"] > 0
    assert counts["stin"] > 0
    assert segment_type.shape == (axon.n_compartments,)
    assert int(PseudoDoubleSegmentType.NODE) in segment_type
    assert int(PseudoDoubleSegmentType.MYSA) in segment_type
    assert int(PseudoDoubleSegmentType.FLUT) in segment_type
    assert int(PseudoDoubleSegmentType.STIN) in segment_type


def test_single_chain_uses_single_cable_sections_without_periaxonal_layers():
    axon = build_pseudo_double_single_chain_mrg(diameter_um=5.7, nodes=3)

    assert all(element.section.periaxonal is None for element in axon.layout.elements)
    assert all(element.section.Cm_uF_cm2 > 0.0 for element in axon.layout.elements)
    assert all(element.section.Ra_ohm_cm > 0.0 for element in axon.layout.elements)


def test_single_chain_alpha_vector_is_segment_specific():
    config = PseudoDoubleSingleChainConfig(
        vext_scale=2.0,
        vext_alpha_node=1.0,
        vext_alpha_mysa=0.8,
        vext_alpha_flut=0.6,
        vext_alpha_stin=0.4,
    )
    axon = build_pseudo_double_single_chain_mrg(
        diameter_um=5.7,
        nodes=3,
        config=config,
    )

    alpha = single_chain_vext_alpha(axon, config)

    assert alpha.shape == (axon.n_compartments,)
    np.testing.assert_allclose(np.unique(alpha), np.asarray([0.8, 1.2, 1.6, 2.0]))


def test_segment_scaled_stimulation_multiplies_footprint():
    import axonscope as axs

    electrode = axs.analytical.PointSourceElectrode(
        x=0.0 * axs.um,
        z=100.0 * axs.um,
        stimulus=axs.Stimulus.constant(1.0 * axs.uA),
    )
    scaled = segment_scaled_point_source_stimulation(
        electrode,
        positions_um=(0.0, 10.0, 20.0),
        alpha=(1.0, 0.5, 0.25),
        sigma=0.3 * axs.S_per_m,
    )
    x_m = np.asarray([0.0, 10e-6, 20e-6])
    base = electrode.footprint_for_axon(x_m, sigma_S_m=0.3)

    np.testing.assert_allclose(
        scaled.drives[0].footprint.values_for_axon(),
        base * np.asarray([1.0, 0.5, 0.25]),
    )
