import numpy as np

from axonscope import (
    Activation,
    ActivationObserver,
    AnalysisInputRequirement,
    AnalysisResult,
    AnalysisStatus,
    AnalyticalExtracellularContext,
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
    PaddingInspection,
    PeakVoltageObserver,
    PrecisionPolicy,
    ProbeInspection,
    RecordingPlan,
    ResultAssemblyInspection,
    SimulationInspection,
    RecordingSpatial,
    Runtime,
    Signal,
    SignalId,
    SimulationEstimate,
    Stimulus,
    VM_RASTER_OBSERVATION_KEY,
    VmRasterResult,
    benchmark,
    benchmark_report,
    disable_benchmark,
    enable_benchmark,
    estimate_simulation,
    inspect_simulation,
    preparation,
    reset_benchmark,
    simulate_pool,
    um,
)
from axonscope import analysis, analytical, positions, results, signals
from axonscope.axons import HodgkinHuxley, MRG, RattayAberham, mrg_like_length_from_nodes
from axonscope.channel_models import IonChannelModelBase, MembraneStateSpec, PassiveICM
from axonscope.icm import CompartmentMembraneLayout, HeterogeneousICMBackend
from axonscope.solvers import (
    CrankNicholson,
)


def test_public_package_imports_are_available():
    assert Stimulus.constant(1.0).y[0] == 1.0
    assert analysis.rasterize is not None
    assert analysis.ActivationCriterion is not None
    assert analysis.ActivationObserver is ActivationObserver
    assert analysis.PeakVoltageObserver is PeakVoltageObserver
    assert analysis.AnalysisInputRequirement is AnalysisInputRequirement
    assert not hasattr(results, "analysis")
    assert Activation is __import__("axonscope").analysis.Activation
    assert AnalysisResult is __import__("axonscope").analysis.AnalysisResult
    assert AnalysisStatus.VALID.value == "VALID"
    assert results.visualization.plot_raster is not None
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
    assert (
        AnalyticalExtracellularContext
        is __import__("axonscope").stimulation.AnalyticalExtracellularContext
    )
    assert isinstance(
        IntracellularCurrentClamp(position=0.0 * um, current=Stimulus.constant(0.0)),
        IntracellularContext,
    )
    assert simulate_pool is not None
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
    assert PaddingInspection is not None
    assert Device.auto().kind == "auto"
    assert ExecutionPolicy(device=Device.cpu()).device == Device.cpu()
    assert MemoryEstimateItem is not None
    assert PrecisionPolicy.float32().solver_dtype == "float32"
    assert ProbeInspection is not None
    assert ResultAssemblyInspection is not None
    assert Runtime.AUTO.value == "auto"
    assert SimulationEstimate is not None
    assert SimulationInspection is not None
    assert VM_RASTER_OBSERVATION_KEY == "vm_raster"
    assert VmRasterResult is not None
    assert estimate_simulation is not None
    assert inspect_simulation is not None
    assert benchmark is not None
    assert benchmark_report is not None
    assert disable_benchmark is not None
    assert enable_benchmark is not None
    assert reset_benchmark is not None
    assert preparation.extracellular_stimulation_signature is not None
    assert HodgkinHuxley is not None
    assert RattayAberham is not None
    assert MRG is not None
    assert CrankNicholson is not None
    assert IonChannelModelBase is not None
    assert PassiveICM is not None
    assert MembraneStateSpec("x").name == "x"
    assert CompartmentMembraneLayout is not None
    assert HeterogeneousICMBackend is not None
    assert mrg_like_length_from_nodes(10.0 * um, 3) > 0.0
