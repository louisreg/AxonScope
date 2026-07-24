import numpy as np

from axonfleet import (
    AxonInstance,
    AxonPopulation,
    AxonSimulation,
    BatchOptions,
    Device,
    ExecutionPolicy,
    IntracellularCurrentClamp,
    PrecisionPolicy,
    SimulationInspection,
    StudyExecutionError,
    StudyPlan,
    StudyResult,
    StudyTask,
    Stimulus,
    SolverPolicy,
    VM_RASTER_OBSERVATION_KEY,
    VmRasterResult,
    benchmark,
    benchmark_report,
    disable_benchmark,
    enable_benchmark,
    reset_benchmark,
    runtime,
    um,
)
from axonfleet import (
    analysis,
    analytical,
    benchmarking,
    identifiers,
    inspection,
    membranes,
    performance,
    positions,
    recording,
    signals,
    solvers,
)
from axonfleet.axons import HodgkinHuxley, MRG, RattayAberham

def test_public_package_imports_and_typed_values_are_available():
    assert Stimulus.constant(1.0).y[0] == 1.0
    assert analysis.rasterize is not None
    assert analysis.Activation is not None
    assert analysis.views.plot_spike_raster is not None
    assert analysis.AnalysisResult is not None
    assert analysis.AnalysisStatus.VALID.value == "VALID"
    electrode = analytical.PointSourceElectrode(
        x=0.0 * um,
        z=1000.0 * um,
        stimulus=Stimulus.constant(0.0),
    )
    stimulation = analytical.point_source_stimulation(
        electrode,
        np.asarray([0.0]) * um,
        sigma=0.3 * __import__("axonfleet").S_per_m,
    )
    assert stimulation.drives[0].id.value == "point_source"
    assert isinstance(signals.MEMBRANE_VOLTAGE, signals.Signal)
    assert signals.Vm is signals.MEMBRANE_VOLTAGE
    assert signals.MEMBRANE_VOLTAGE.id == identifiers.SignalId("membrane_voltage")
    assert signals.MEMBRANE_VOLTAGE.result_key == "Vm"
    assert recording.RecordingSpatial.CENTER is not None
    assert recording.RecordingPlan is not None
    assert positions.PROXIMAL is not None
    assert hasattr(__import__("axonfleet").positions, "DISTAL")
    assert AxonInstance is not None
    assert AxonPopulation is not None
    assert AxonSimulation is not AxonInstance
    assert BatchOptions.full().recording.is_full
    assert solvers.BatchRecording.center().label == "center"
    assert benchmarking.BenchmarkReport is not None
    assert benchmarking.BenchmarkSession is not None
    assert (
        runtime.jax.cpu.DoubleCableSolver.thomas().kind
        is runtime.jax.DoubleCableSolverKind.THOMAS
    )
    assert inspection.AssemblyDetailInspection is not None
    assert inspection.DispatchGroupInspection is not None
    assert inspection.KernelInspection is not None
    assert inspection.LoweringInspection is not None
    assert inspection.MemoryInspection is not None
    assert inspection.MembraneSourceInspection is not None
    assert inspection.PaddingInspection is not None
    assert Device.auto().kind == "auto"
    assert ExecutionPolicy(device=Device.cpu()).device == Device.cpu()
    assert (
        runtime.jax.gpu.DoubleCableSolver.tiled_thomas().kind
        is runtime.jax.DoubleCableSolverKind.TILED_THOMAS
    )
    assert performance.MemoryEstimateItem is not None
    assert PrecisionPolicy.float32().solver_dtype == "float32"
    assert inspection.ProbeInspection is not None
    assert inspection.ResultAssemblyInspection is not None
    assert runtime.auto.value == "auto"
    assert runtime.jax.value == "jax"
    assert performance.SimulationEstimate is not None
    assert performance.SimulationEstimateGroup is not None
    assert SimulationInspection is not None
    assert StudyExecutionError is not None
    assert StudyPlan is not None
    assert StudyResult is not None
    assert StudyTask is not None
    assert SolverPolicy(double_cable=runtime.jax.DoubleCableSolver.auto()) is not None
    assert runtime.jax.TiledThomasSolverOptions(block_b=32).block_b == 32
    assert VM_RASTER_OBSERVATION_KEY == "vm_raster"
    assert VmRasterResult is not None
    assert benchmark is not None
    assert benchmark_report is not None
    assert disable_benchmark is not None
    assert enable_benchmark is not None
    assert reset_benchmark is not None
    assert membranes.explain is not None
    assert membranes.inspect_generated_code is not None
    assert issubclass(membranes.HodgkinHuxley, membranes.Model)
    hh_membrane = membranes.HodgkinHuxley(celsius=6.3 * __import__("axonfleet").degC)
    assert isinstance(hh_membrane, membranes.Model)
    assert hh_membrane.kind == "hodgkin_huxley"
    assert hh_membrane.params["celsius"] == 6.3
    assert HodgkinHuxley is not None
    assert RattayAberham is not None
    assert MRG is not None
