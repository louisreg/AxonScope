"""Structural contracts for the Gaines membrane families.

Numerical rates, currents, and section defaults are validated directly against
NRV in ``tests/nrv/numerics/test_gaines_membranes_vs_nrv.py``.
"""

import numpy as np

import axonfleet as axs


def _section_membrane(axon, section_name: str):
    return next(
        element.section.membrane
        for element in axon.layout.elements
        if element.section.name.lower() == section_name.lower()
    )


def test_gaines_families_share_node_and_internode_source_topologies():
    assert (
        axs.membranes.GainesMotorNode.source_model
        is axs.membranes.GainesSensoryNode.source_model
    )
    assert (
        axs.membranes.GainesMotorInternode.source_model
        is axs.membranes.GainesSensoryInternode.source_model
    )
    assert (
        axs.membranes.GainesMotorNode.source_model
        is not axs.membranes.GainesMotorInternode.source_model
    )


def test_gaines_axons_reuse_mrg_geometry_and_family_membranes():
    motor = axs.axons.GainesMotor(diameter=10.0 * axs.um, nodes=3)
    sensory = axs.axons.GainesSensory(diameter=10.0 * axs.um, nodes=3)
    mrg = axs.axons.MRG(diameter=10.0 * axs.um, nodes=3)

    assert motor.v_init == -85.9411
    assert sensory.v_init == -79.3565
    assert motor.formulation is axs.axons.CableFormulation.DOUBLE_CABLE
    assert sensory.formulation is axs.axons.CableFormulation.DOUBLE_CABLE
    np.testing.assert_allclose(
        motor.layout.position_values(unit="micrometer"),
        mrg.layout.position_values(unit="micrometer"),
    )

    for axon, node_model, internode_model in (
        (motor, axs.membranes.GainesMotorNode, axs.membranes.GainesMotorInternode),
        (
            sensory,
            axs.membranes.GainesSensoryNode,
            axs.membranes.GainesSensoryInternode,
        ),
    ):
        assert _section_membrane(axon, "node").kind == node_model.kind_name()
        for section_name in ("MYSA", "FLUT", "STIN"):
            assert (
                _section_membrane(axon, section_name).kind
                == internode_model.kind_name()
            )
