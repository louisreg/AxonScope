import numpy as np

from axonscope import (
    Activation,
    ActivationObserver,
    AnalysisInputRequirement,
    AnalysisResult,
    AnalysisStatus,
    AssemblyDetailInspection,
    AxonInstance,
    AxonPopulation,
    AxonSimulation,
    BatchOptions,
    BatchRecording,
    BenchmarkReport,
    BenchmarkSession,
    Device,
    DispatchGroupInspection,
    ExecutionPolicy,
    IntracellularContext,
    IntracellularCurrentClamp,
    KernelInspection,
    LoweringInspection,
    MemoryInspection,
    MemoryEstimateItem,
    MembraneSourceInspection,
    PaddingInspection,
    PrecisionPolicy,
    ProbeInspection,
    RateTableConfig,
    RecordingPlan,
    ResultAssemblyInspection,
    SimulationInspection,
    RecordingSpatial,
    Runtime,
    Signal,
    SignalId,
    SimulationEstimate,
    SimulationEstimateGroup,
    Stimulus,
    VM_RASTER_OBSERVATION_KEY,
    VmRasterResult,
    benchmark,
    benchmark_report,
    disable_benchmark,
    enable_benchmark,
    preparation,
    reset_benchmark,
    um,
)
from axonscope import analysis, analytical, membranes, positions, results, signals
from axonscope.axons import HodgkinHuxley, MRG, RattayAberham, mrg_like_length_from_nodes
from axonscope.solvers import (
    CrankNicholson,
)


def test_public_package_imports_are_available():
    assert Stimulus.constant(1.0).y[0] == 1.0
    assert analysis.rasterize is not None
    assert analysis.ActivationCriterion is not None
    assert analysis.ActivationObserver is ActivationObserver
    assert not hasattr(analysis, "PeakVoltageObserver")
    assert not hasattr(__import__("axonscope"), "PeakVoltageObserver")
    assert analysis.AnalysisInputRequirement is AnalysisInputRequirement
    assert analysis.views.plot_spike_raster is not None
    assert not hasattr(results, "analysis")
    assert not hasattr(results, "visualization")
    assert not hasattr(results, "plot_raster")
    assert not hasattr(results, "rasterplot")
    assert Activation is __import__("axonscope").analysis.Activation
    assert AnalysisResult is __import__("axonscope").analysis.AnalysisResult
    assert AnalysisStatus.VALID.value == "VALID"
    assert not hasattr(__import__("axonscope"), "PointSourceElectrode")
    assert not hasattr(__import__("axonscope").stimulation, "PointSourceElectrode")
    electrode = analytical.PointSourceElectrode(
        x=0.0 * um,
        z=1000.0 * um,
        stimulus=Stimulus.constant(0.0),
    )
    stimulation = analytical.point_source_stimulation(
        electrode,
        np.asarray([0.0]) * um,
        sigma=0.3 * __import__("axonscope").S_per_m,
    )
    assert stimulation.drives[0].id.value == "point_source"
    for legacy_name in (
        "Electrode",
        "AnalyticalElectrode",
        "ExtracellularContext",
        "AnalyticalExtracellularContext",
        "ExtracellularStimulationContext",
        "NRVExtracellularContext",
    ):
        assert not hasattr(__import__("axonscope"), legacy_name)
        assert not hasattr(__import__("axonscope").stimulation, legacy_name)
    assert isinstance(
        IntracellularCurrentClamp(position=0.0 * um, current=Stimulus.constant(0.0)),
        IntracellularContext,
    )
    root = __import__("axonscope")
    assert not hasattr(root, "simulate")
    assert not hasattr(root, "simulate_pool")
    assert not hasattr(root, "estimate_simulation")
    assert not hasattr(root, "inspect_simulation")
    assert isinstance(signals.MEMBRANE_VOLTAGE, Signal)
    assert signals.Vm is signals.MEMBRANE_VOLTAGE
    assert signals.MEMBRANE_VOLTAGE.id == SignalId("membrane_voltage")
    assert signals.MEMBRANE_VOLTAGE.result_key == "Vm"
    assert not hasattr(Signal, "VM")
    assert RecordingSpatial.CENTER is not None
    assert RecordingPlan is not None
    assert not hasattr(analytical, "local_point_source_context")
    assert positions.PROXIMAL is not None
    assert hasattr(__import__("axonscope").positions, "DISTAL")
    assert AxonInstance is not None
    assert AxonPopulation is not None
    assert AxonSimulation is not AxonInstance
    assert BatchOptions.full().recording.is_full
    assert BatchRecording.center().label == "center"
    assert BenchmarkReport is not None
    assert BenchmarkSession is not None
    assert AssemblyDetailInspection is not None
    assert DispatchGroupInspection is not None
    assert KernelInspection is not None
    assert LoweringInspection is not None
    assert MemoryInspection is not None
    assert MembraneSourceInspection is not None
    assert PaddingInspection is not None
    assert Device.auto().kind == "auto"
    assert ExecutionPolicy(device=Device.cpu()).device == Device.cpu()
    assert MemoryEstimateItem is not None
    assert PrecisionPolicy.float32().solver_dtype == "float32"
    assert ProbeInspection is not None
    assert ResultAssemblyInspection is not None
    assert Runtime.AUTO.value == "auto"
    assert SimulationEstimate is not None
    assert SimulationEstimateGroup is not None
    assert SimulationInspection is not None
    assert VM_RASTER_OBSERVATION_KEY == "vm_raster"
    assert VmRasterResult is not None
    assert benchmark is not None
    assert benchmark_report is not None
    assert disable_benchmark is not None
    assert enable_benchmark is not None
    assert reset_benchmark is not None
    assert preparation.extracellular_stimulation_signature is not None
    assert membranes.GeneratedCodeFileInspection is not None
    assert membranes.GeneratedMembraneCodeInspection is not None
    assert membranes.GeneratedMembraneCodeReport is not None
    assert membranes.GeneratedTargetExplanation is not None
    assert membranes.MembraneEquationDependency is not None
    assert membranes.MembraneModelExplanation is not None
    assert membranes.MembraneSourceExplanation is not None
    assert membranes.MembraneSourceSection is not None
    assert membranes.MembraneSourceSymbol is not None
    assert membranes.MembraneStateUpdateExplanation is not None
    assert membranes.MembraneStepExplanation is not None
    assert membranes.explain is not None
    assert membranes.inspect_generated_code is not None
    assert not hasattr(membranes, "MembraneModel")
    assert not hasattr(membranes, "ensure_membrane_model")
    assert issubclass(membranes.HodgkinHuxley, membranes.Model)
    hh_membrane = membranes.HodgkinHuxley(celsius=6.3 * __import__("axonscope").degC)
    assert isinstance(hh_membrane, membranes.Model)
    assert hh_membrane.kind == "hodgkin_huxley"
    assert hh_membrane.params["celsius"] == 6.3
    assert HodgkinHuxley is not None
    assert RattayAberham is not None
    assert MRG is not None
    assert CrankNicholson is not None
    assert RateTableConfig(step_mV=1.0).step_mV == 1.0
    assert mrg_like_length_from_nodes(10.0 * um, 3) > 0.0
