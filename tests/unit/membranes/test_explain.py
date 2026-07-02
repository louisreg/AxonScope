from __future__ import annotations

import axonscope as axs


def test_membrane_model_explain_reports_source_units_cache_and_targets(tmp_path, monkeypatch):
    monkeypatch.setenv("AXONSCOPE_MODEL_CODEGEN_CACHE", str(tmp_path / "codegen"))
    model = axs.membranes.Passive()

    report = model.explain()

    assert isinstance(report, axs.membranes.MembraneModelExplanation)
    assert report.model_kind == "passive"
    assert len(report.sources) == 1
    source = report.sources[0]
    assert source.model_name == "passive"
    assert source.cache_status in {"hit", "miss"}
    assert source.cache_reason in {"manifest_match", "manifest_missing"}
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
    assert "AxonScope membrane model explanation" in text
    assert "model=passive" in text
    assert "generated model_step targets" in text
    assert "Rm(resistance_area" in text


def test_membrane_explain_reuses_generated_cache_and_reports_model_step_pruning(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AXONSCOPE_MODEL_CODEGEN_CACHE", str(tmp_path / "codegen"))
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

    text = report.format()
    assert "rates (rates)" in text
    assert "currents (currents)" in text
    assert "pruned_from_model_step" in text


def test_membrane_explain_reports_stateful_step_program(tmp_path, monkeypatch):
    monkeypatch.setenv("AXONSCOPE_MODEL_CODEGEN_CACHE", str(tmp_path / "codegen"))
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
        "I_na_total_uAcm2",
        "I_k_total_uAcm2",
        "I_ca_total_uAcm2",
        "I_total_rhs_uAcm2",
    ]

    text = report.format()
    assert "step_program=initials=" in text
    assert "prepare=(cai<-" in text
    assert "Oc<-" in text
    assert "cao<-" in text
    assert "solver_terms=(total=" in text
