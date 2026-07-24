from __future__ import annotations

import axonfleet as axs
from axonfleet.membranes.explain import (
    MembraneComponentExplanation,
    MembraneMechanismExplanation,
    MembraneModelExplanation,
    MembraneRecordingOutputExplanation,
)


def test_membrane_model_explain_reports_source_units_cache_and_targets(tmp_path, monkeypatch):
    monkeypatch.setenv("AXONFLEET_CACHE", str(tmp_path / "codegen"))
    model = axs.membranes.Passive()

    report = model.explain()

    assert isinstance(report, MembraneModelExplanation)
    assert report.model_kind == "passive"
    assert report.components == (
        MembraneComponentExplanation(
            label="passive",
            model_kind="passive",
        ),
    )
    assert isinstance(
        report.recording_outputs,
        MembraneRecordingOutputExplanation,
    )
    assert report.recording_outputs.currents == ("I_l",)
    assert report.recording_outputs.conductances == ("g_l",)
    assert report.recording_outputs.observables == ("passive.g_l",)
    assert len(report.sources) == 1
    source = report.sources[0]
    assert source.model_name == "passive"
    assert source.cache_status in {"hit", "miss"}
    assert source.cache_reason in {
        "manifest_match",
        "manifest_missing",
        "generated_files_missing:jax_model.py,numpy_model.py",
    }
    assert len(source.graph_hash) == 40
    assert source.optimized_graph_hash == source.graph_hash
    assert source.function_names == ("leak",)
    assert source.generated_targets == ("jax", "numpy")
    assert [symbol.name for symbol in source.inputs] == ["Vm"]
    assert {symbol.name for symbol in source.parameters} == {"Rm", "EL"}
    assert {symbol.role for symbol in source.parameters} == {
        "resistance_area",
        "voltage",
    }
    assert source.currents == ("I_l",)
    assert source.observables == ("g_l",)
    assert source.source_outputs["all"] == ("I_l", "g_l")
    assert len(source.sections) == 1
    assert source.sections[0].name == "leak"
    assert source.sections[0].assignments == ("g_l", "I_l")
    assert source.targets[0].target == "jax"
    assert source.targets[0].arg_names == ("Vm", "Rm", "EL")
    assert source.targets[0].output_names == ("I_l", "g_l")

    text = report.format()
    assert "AxonFleet membrane model explanation" in text
    assert "model=passive" in text
    assert "components=(passive:passive)" in text
    assert "recording_outputs=" in text
    assert f"graph_hash={source.graph_hash}" in text
    assert f"optimized_graph_hash={source.optimized_graph_hash}" in text
    assert "generated model_step targets" in text
    assert "Rm(resistance_area" in text


def test_membrane_explain_reports_composite_labels_and_recording_outputs(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AXONFLEET_CACHE", str(tmp_path / "codegen"))
    model = axs.membranes.Composite(
        {
            "rattay": axs.membranes.RattayAberham(),
            "extra_leak": axs.membranes.Passive(Rm=12_000.0, EL=-68.0),
        }
    )

    report = model.explain()

    assert report.model_kind == "composite"
    assert report.components == (
        MembraneComponentExplanation(
            label="rattay",
            model_kind="rattay_aberham",
        ),
        MembraneComponentExplanation(
            label="extra_leak",
            model_kind="passive",
        ),
    )
    assert report.recording_outputs.gates == (
        "rattay.m",
        "rattay.h",
        "rattay.n",
    )
    assert report.recording_outputs.currents == ("I_na", "I_k", "I_l")
    assert report.recording_outputs.conductances == ("g_na", "g_k", "g_l")
    assert report.recording_outputs.current_aggregates == ("I_l",)
    assert report.recording_outputs.conductance_aggregates == ("g_l",)
    assert "extra_leak.g_l" in report.recording_outputs.observables

    text = report.format()
    assert "model=composite" in text
    assert "components=(rattay:rattay_aberham, extra_leak:passive)" in text
    assert "gates: (rattay.m, rattay.h, rattay.n)" in text
    assert "currents: (I_na, I_k, I_l) aggregates=(I_l)" in text
    assert "conductances: (g_na, g_k, g_l) aggregates=(g_l)" in text


def test_membrane_explain_reuses_generated_cache_and_reports_model_step_pruning(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AXONFLEET_CACHE", str(tmp_path / "codegen"))
    model = axs.membranes.HodgkinHuxley()
    model.explain()

    report = axs.membranes.explain(model)

    source = report.sources[0]
    assert source.cache_status == "hit"
    assert source.cache_reason == "manifest_match"
    assert source.function_names == ("rates", "currents")
    assert [section.name for section in source.sections] == ["rates", "currents"]
    assert {"m", "h", "n"}.issubset({state.name for state in source.states})
    assert source.gates == ("m", "h", "n")
    jax_target = next(target for target in source.targets if target.target == "jax")
    assert {"I_na", "I_k", "I_l", "g_na", "g_k", "g_l"}.issubset(
        set(jax_target.output_names)
    )
    assert "alpha_m" in jax_target.pruned_from_model_step
    assert "q10" in jax_target.pruned_from_model_step
    assert "g_na" in jax_target.retained_assignments
    assert len(jax_target.backend_lowering_key) == 40
    assert jax_target.cache_key == source.cache_key
    assert jax_target.compiler_version == source.metadata["source_compiler"]
    assert jax_target.contract_version == source.metadata["source_contract"]
    assert jax_target.parameter_specialization == "runtime_overridable_defaults"
    assert len(jax_target.helper_identity_hash) == 40
    assert len(jax_target.dependency_hash) == 40
    assert jax_target.static_shape_policy == "runtime_node_count"
    assert jax_target.recording_policy == "all_source_outputs"
    assert jax_target.precision_policy == "runtime_backend_dtype"
    assert jax_target.optimization_level == "identity"

    text = report.format()
    assert "rates (rates)" in text
    assert "currents (currents)" in text
    assert "backend_lowering_key" in text
    assert "parameters:runtime_overridable_defaults" in text
    assert "recording:all_source_outputs" in text
    assert "pruned_from_model_step" in text


def test_membrane_explain_reports_mechanism_boundaries(tmp_path, monkeypatch):
    monkeypatch.setenv("AXONFLEET_CACHE", str(tmp_path / "codegen"))
    model = axs.membranes.Sundt()

    report = model.explain()

    source = report.sources[0]
    assert [mechanism.name for mechanism in source.mechanisms] == [
        "na_hh",
        "borg_kdr",
    ]
    assert isinstance(source.mechanisms[0], MembraneMechanismExplanation)

    na_hh = source.mechanisms[0]
    assert na_hh.function_name == "na_hh_rates"
    assert "alpha_m" in na_hh.assignments
    assert "beta_h" in na_hh.assignments
    assert {"Vm", "celsius", "mshift", "hshift", "ishift"}.issubset(
        set(na_hh.external_dependencies)
    )

    borg = source.mechanisms[1]
    assert borg.function_name == "borg_kdr_rates"
    assert "alpha_n" in borg.assignments
    assert "beta_l" in borg.assignments
    assert {"Vm", "celsius", "vhalfn", "vhalfl"}.issubset(
        set(borg.external_dependencies)
    )

    metadata_mechanisms = source.metadata["source_mechanisms"]
    assert [entry["name"] for entry in metadata_mechanisms] == ["na_hh", "borg_kdr"]
    assert metadata_mechanisms[0]["function"] == "na_hh_rates"
    assert "alpha_m" in metadata_mechanisms[0]["assignments"]
    assert source.metadata["source_provenance"]["sections"][0]["name"] == "mechanism:na_hh"

    text = report.format()
    assert "mechanisms:" in text
    assert "na_hh (na_hh_rates)" in text
    assert "borg_kdr (borg_kdr_rates)" in text
    assert "external_dependencies=" in text


def test_membrane_explain_reports_stateful_step_program(tmp_path, monkeypatch):
    monkeypatch.setenv("AXONFLEET_CACHE", str(tmp_path / "codegen"))
    model = axs.membranes.Schild94(diameter=0.8 * axs.um)

    report = model.explain()

    source = report.sources[0]
    assert source.step is not None
    assert [update.state for update in source.step.state_initials] == [
        "cai",
        "Oc",
        "cao",
        "c_kca",
    ]
    assert [update.state for update in source.step.prepare_state_updates] == [
        "cai",
        "Oc",
        "cao",
    ]
    assert [update.state for update in source.step.finalize_state_updates] == [
        "c_kca",
        "cai",
        "Oc",
        "cao",
    ]
    assert source.step.total_outward_current is not None
    assert "I_ion" in source.step.total_outward_current
    assert source.step.explicit_outward_current is not None
    assert "background_current" in source.step.explicit_outward_current
    assert source.step.correction_current is not None
    assert "eca_static" in source.step.correction_current
    assert source.step.prepare_gate_source == "previous"
    assert source.step.linearization_gate_source == "previous"
    assert [diagnostic.state for diagnostic in source.step.diagnostics] == [
        "I_na",
        "I_k",
        "I_ca",
        "I_total_rhs_uAcm2",
    ]

    text = report.format()
    assert "step_program=initials=" in text
    assert "prepare=(cai<-" in text
    assert "Oc<-" in text
    assert "cao<-" in text
    assert "solver_terms=(total=" in text
