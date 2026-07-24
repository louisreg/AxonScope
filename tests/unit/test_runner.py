from dataclasses import FrozenInstanceError

import pytest

import axonfleet as axs
import axonfleet.runner as runner_module
from axonfleet.dispatcher.numeric_axis import ExtracellularWaveformAxisInput


def _simulation() -> axs.AxonSimulation:
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
    )
    return axs.AxonSimulation(
        axon,
        duration=0.1 * axs.ms,
        dt=0.05 * axs.ms,
    )


def _extracellular_simulation() -> axs.AxonSimulation:
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=5,
    )
    positions = axon.layout.position_values(unit=axs.um) * axs.um
    electrode = axs.analytical.PointSourceElectrode(
        x=50.0 * axs.um,
        y=10.0 * axs.um,
        z=0.0 * axs.um,
    )
    stimulation = axs.analytical.point_source_stimulation(
        electrode,
        positions,
        sigma=0.3 * axs.S_per_m,
        stimulus=axs.Stimulus.pulse(
            start=0.05 * axs.ms,
            duration=0.05 * axs.ms,
            amplitude=0.0 * axs.uA,
        ),
    )
    instance = axs.AxonInstance(axon)
    instance.add_extracellular_stimulation(stimulation=stimulation)
    return axs.AxonSimulation(
        instance,
        duration=0.1 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.center(axs.signals.Vm),
    )


def test_simulation_plan_is_immutable_and_contains_no_dispatch_state():
    plan = _simulation().plan()

    assert isinstance(plan, axs.RunnablePlan)
    assert isinstance(plan, axs.SimulationPlan)
    assert isinstance(plan.population, axs.PopulationPlan)
    assert plan.expected_rows == 1
    assert not hasattr(plan, "groups")
    assert not hasattr(plan, "runtime_context")
    with pytest.raises(FrozenInstanceError):
        plan.duration = 1.0 * axs.ms


def test_plan_creation_does_not_build_dispatch_plan(monkeypatch):
    def fail_build(_instances):
        raise AssertionError("plan creation must not materialize dispatch state")

    monkeypatch.setattr(runner_module, "build_dispatch_plan", fail_build)

    plan = _simulation().plan()

    assert plan.expected_rows == 1


def test_plan_creation_does_not_materialize_population(monkeypatch):
    def fail_population(_source):
        raise AssertionError("plan creation must not materialize population rows")

    monkeypatch.setattr(runner_module, "AxonPopulation", fail_population)

    plan = _simulation().plan()

    assert plan.expected_rows == 1
    assert isinstance(plan.population, axs.PopulationPlan)


def test_membrane_validation_is_deferred_to_runner_preparation():
    invalid = axs.membranes.Passive(Rm=1.0 * axs.ms)
    composite = axs.membranes.Composite(
        {"invalid_leak": invalid, "default_leak": axs.membranes.Passive()}
    )
    section_layout = axs.membranes.SectionLayout(axon=composite)
    section = axs.axons.Section(
        "axon",
        membrane=section_layout.membrane_for("axon"),
        diameter=0.5 * axs.um,
    )
    layout = axs.axons.Layout.single_uniform(
        section,
        length=100.0 * axs.um,
        compartments=3,
    )
    simulation = axs.AxonSimulation(
        axs.axons.Axon(layout=layout),
        duration=0.1 * axs.ms,
        dt=0.05 * axs.ms,
    )

    assert composite.components[0] is invalid
    assert section_layout.membrane_for("axon") is composite
    assert layout.position_values().shape == (3,)
    plan = simulation.plan()

    with pytest.raises(TypeError, match="Cannot convert"):
        axs.Runner().estimate(plan)


def test_runner_materializes_population_once_per_plan(monkeypatch):
    original = runner_module.AxonPopulation
    calls = []

    class TrackedPopulation(original):
        def __init__(self, source):
            calls.append(source)
            super().__init__(source)

    monkeypatch.setattr(runner_module, "AxonPopulation", TrackedPopulation)
    plan = _simulation().plan()
    runner = axs.Runner()

    first = runner._population(plan.population)
    second = runner._population(plan.population)

    assert first is second
    assert len(calls) == 1


def test_numeric_axis_plan_composes_source_without_expanding_python_rows():
    source = _simulation().plan()
    waveforms = tuple(
        axs.Stimulus.pulse(
            start=0.0 * axs.ms,
            duration=0.05 * axs.ms,
            amplitude=amplitude * axs.uA,
        )
        for amplitude in (1.0, 2.0, 3.0)
    )
    axis_input = ExtracellularWaveformAxisInput(
        waveforms=waveforms,
        source_drive_waveforms=((waveforms[0],),),
        selected_drive_indices=(0,),
    )

    plan = source.with_numeric_axis(axis_input)

    assert isinstance(plan, axs.NumericAxisPlan)
    assert plan.source is source
    assert plan.expected_rows == 3
    assert plan.source.population.source is source.population.source


def test_runner_estimates_numeric_axis_peak_as_one_compact_execution():
    source = _extracellular_simulation().plan()
    update = axs.protocols.ExtracellularWaveformUpdate(
        lambda value: axs.Stimulus.pulse(
            start=0.05 * axs.ms,
            duration=0.05 * axs.ms,
            amplitude=value,
        )
    )
    builder = update.prepare_numeric_axis((source.population.source,))
    plan = source.with_numeric_axis(
        builder.numeric_axis_input((1.0 * axs.uA, 2.0 * axs.uA, 3.0 * axs.uA))
    )

    estimate = axs.Runner().estimate(plan)
    inspection = axs.Runner().inspect(plan)

    assert isinstance(estimate, axs.PlanEstimate)
    assert estimate.simulation_executions_min == 1
    assert estimate.simulation_executions_max == 1
    assert estimate.peak.axon_count == 3
    assert estimate.peak.groups[0].size == 3
    assert isinstance(inspection, axs.PlanInspection)
    assert inspection.components[0].simulation.planning.axon_count == 3
    assert inspection.components[0].simulation.dispatch_groups[0].size == 3


def test_runner_estimates_sweep_peak_separately_from_repeated_work():
    simulation = _extracellular_simulation()
    plan = axs.protocols.pool_sweep_plan(
        simulation.population,
        update=axs.protocols.ExtracellularWaveformUpdate(
            lambda value: axs.Stimulus.pulse(
                start=0.05 * axs.ms,
                duration=0.05 * axs.ms,
                amplitude=value,
            )
        ),
        values=(1.0 * axs.uA, 2.0 * axs.uA, 3.0 * axs.uA),
        observe=lambda result: result,
        duration=simulation.duration,
        dt=simulation.dt,
        recording=simulation.recording,
        value_batch_size=2,
    )

    estimate = axs.Runner().estimate(plan)
    inspection = axs.Runner().inspect(plan)

    assert estimate.expected_rows == 3
    assert estimate.simulation_executions_min == 2
    assert estimate.simulation_executions_max == 2
    assert estimate.peak.axon_count == 2
    assert "not cumulative work" in estimate.format()
    assert inspection.simulation_executions_min == 2
    assert inspection.components[0].simulation.planning.axon_count == 2


def test_runner_composed_reports_bound_threshold_and_study_work():
    source = _simulation().plan()
    threshold = axs.ThresholdPlan(
        source=source,
        update=lambda row, value: row,
        decode=lambda result: (False,),
        bounds=(0.0 * axs.uA, 10.0 * axs.uA),
        row_labels=("axon",),
        max_iterations=20,
    )
    study = axs.StudyPlan(
        name="threshold_study",
        tasks=(
            axs.StudyTask("source", source),
            axs.StudyTask("threshold", threshold, depends_on=("source",)),
        ),
    )

    estimate = axs.Runner().estimate(study)
    inspection = axs.Runner().inspect(study)

    assert estimate.simulation_executions_min == 3
    assert estimate.simulation_executions_max == 23
    assert estimate.peak_bytes == max(
        component.peak.total_bytes for component in estimate.components
    )
    assert estimate.components[1].depends_on == ("source",)
    assert estimate.name == "threshold_study"
    assert inspection.simulation_executions_min == 3
    assert inspection.simulation_executions_max == 23
    assert inspection.components[1].depends_on == ("source",)
    assert inspection.name == "threshold_study"


def test_composed_estimate_and_inspection_do_not_execute_solver(monkeypatch):
    source = _simulation().plan()
    study = axs.StudyPlan(
        name="inspection_only",
        tasks=(axs.StudyTask("source", source),),
    )
    runner = axs.Runner()

    def fail_execution(*_args, **_kwargs):
        raise AssertionError("estimate/inspect must not execute solver work")

    monkeypatch.setattr(runner, "_run_simulation", fail_execution)

    assert runner.estimate(study).simulation_executions_min == 1
    assert runner.inspect(study).simulation_executions_min == 1


def test_runner_clear_drops_runner_owned_population_dispatch_and_preparation_state():
    runner = axs.Runner()
    plan = _simulation().plan()

    dispatch_plan = runner._dispatch_plan(plan)
    runner._prepared_cohorts.for_current_group(dispatch_plan.groups[0])
    assert len(runner._dispatch_plans) == 1
    assert len(runner._prepared_cohorts) == 1

    runner.clear()

    assert len(runner._dispatch_plans) == 0
    assert len(runner._prepared_cohorts) == 0


def test_prepared_cohort_reuse_is_isolated_per_runner():
    plan = _simulation().plan()
    first_runner = axs.Runner()
    second_runner = axs.Runner()
    group = first_runner._dispatch_plan(plan).groups[0]

    first = first_runner._prepared_cohorts.for_current_group(group)
    reused = first_runner._prepared_cohorts.for_current_group(group)
    independent = second_runner._prepared_cohorts.for_current_group(group)

    assert reused is first
    assert independent is not first


def test_pool_sweep_plan_defers_update_and_numeric_preparation():
    simulation = _simulation()
    calls = []

    def update(row, value):
        calls.append((row, value))
        return row

    plan = axs.protocols.pool_sweep_plan(
        simulation.population,
        update=update,
        values=(1.0, 2.0, 3.0),
        observe=lambda result: result,
        duration=simulation.duration,
        dt=simulation.dt,
        recording=axs.Recording.center(axs.signals.Vm),
        batch_options=None,
        execution_policy=None,
        value_batch_size=1,
    )

    assert isinstance(plan, axs.SweepPlan)
    assert plan.expected_rows == 3
    assert [values for _, values in plan.value_batches()] == [(1.0,), (2.0,), (3.0,)]
    assert calls == []


def test_threshold_plan_defers_bounds_and_updates():
    simulation = _simulation()
    calls = []

    def bounds(row):
        calls.append(("bounds", row))
        return 0.0 * axs.uA, 10.0 * axs.uA

    def update(row, value):
        calls.append(("update", row, value))
        return row

    plan = axs.protocols.find_threshold_plan(
        simulation.population,
        update=update,
        bounds=bounds,
        duration=simulation.duration,
        dt=simulation.dt,
        criterion=axs.analysis.Activation(),
        tolerance=0.5 * axs.uA,
        recording=axs.Recording.none(),
    )

    assert isinstance(plan, axs.ThresholdPlan)
    assert plan.expected_rows == 1
    assert plan.bounds is bounds
    assert calls == []


def test_study_plan_uses_stable_dependency_order(monkeypatch):
    plans = tuple(_simulation().plan() for _ in range(3))
    labels = {id(plan): label for plan, label in zip(plans, "abc", strict=True)}
    study = axs.StudyPlan(
        name="ordered",
        tasks=(
            axs.StudyTask("c", plans[2], depends_on=("b",)),
            axs.StudyTask("a", plans[0]),
            axs.StudyTask("b", plans[1], depends_on=("a",)),
        ),
    )
    runner = axs.Runner()
    executed = []

    def fake_run_leaf(plan, *, cancellation):
        del cancellation
        label = labels[id(plan)]
        executed.append(label)
        return label.upper()

    monkeypatch.setattr(runner, "_run_leaf", fake_run_leaf)

    result = runner.run(study)

    assert executed == ["a", "b", "c"]
    assert result.keys == ("a", "b", "c")
    assert result.values == ("A", "B", "C")
    assert result["b"] == "B"
    assert result.name == "ordered"


def test_study_plan_rejects_invalid_dependencies():
    plan = _simulation().plan()

    with pytest.raises(ValueError, match="unknown task"):
        axs.StudyPlan(
            name="invalid",
            tasks=(axs.StudyTask("a", plan, depends_on=("missing",)),),
        )
    with pytest.raises(ValueError, match="cycle"):
        axs.StudyPlan(
            name="cyclic",
            tasks=(
                axs.StudyTask("a", plan, depends_on=("b",)),
                axs.StudyTask("b", plan, depends_on=("a",)),
            ),
        )
    with pytest.raises(ValueError, match="repeats task key"):
        axs.StudyPlan(
            name="duplicate",
            tasks=(axs.StudyTask("a", plan), axs.StudyTask("a", plan)),
        )
    with pytest.raises(ValueError, match="study name"):
        axs.StudyPlan(name="", tasks=(axs.StudyTask("a", plan),))


def test_study_failure_retains_completed_results(monkeypatch):
    plans = tuple(_simulation().plan() for _ in range(3))
    study = axs.StudyPlan(
        name="failure",
        tasks=tuple(
            axs.StudyTask(key, plan)
            for key, plan in zip("abc", plans, strict=True)
        ),
    )
    indices = {id(plan): index for index, plan in enumerate(plans)}
    runner = axs.Runner()
    calls = []

    def fake_run_leaf(plan, *, cancellation):
        del cancellation
        index = indices[id(plan)]
        calls.append(index)
        if index == 1:
            raise ValueError("broken")
        return index

    monkeypatch.setattr(runner, "_run_leaf", fake_run_leaf)

    with pytest.raises(axs.StudyExecutionError) as caught:
        runner.run(study)

    assert calls == [0, 1]
    assert caught.value.task_key == "b"
    assert caught.value.completed.items() == (("a", 0),)
    assert caught.value.pending_keys == ("c",)
    assert isinstance(caught.value.cause, ValueError)


def test_study_cancellation_stops_before_next_task(monkeypatch):
    plans = tuple(_simulation().plan() for _ in range(3))
    study = axs.StudyPlan.from_plans(plans, name="cancelled")
    runner = axs.Runner()
    cancellation = axs.CancellationToken()
    calls = []

    def fake_run_leaf(plan, *, cancellation):
        calls.append(plan)
        if len(calls) == 1:
            cancellation.cancel()
        return len(calls)

    monkeypatch.setattr(runner, "_run_leaf", fake_run_leaf)

    with pytest.raises(axs.PlanCancelledError) as caught:
        runner.run(study, cancellation=cancellation)

    assert calls == [plans[0]]
    assert caught.value.completed.values == (1,)
    assert caught.value.pending_keys == ("task_1", "task_2")


def test_runner_rejects_unknown_plan_type():
    with pytest.raises(TypeError, match="cannot execute"):
        axs.Runner().run(object())
