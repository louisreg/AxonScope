from axonscope import (
    Activation,
    ActivationObserver,
    AnalysisInputRequirement,
    AnalysisResult,
    AnalysisStatus,
    AnalyticalExtracellularContext,
    AxonInstance,
    AxonPopulation,
    AxonSimulation,
    BenchmarkReport,
    BenchmarkSession,
    Device,
    IntracellularContext,
    IntracellularCurrentClamp,
    MemoryEstimateItem,
    PointSourceElectrode,
    PeakVoltageObserver,
    PrecisionPolicy,
    RecordingSpatial,
    Runtime,
    Signal,
    SignalId,
    SimulationEstimate,
    Stimulus,
    benchmark,
    benchmark_report,
    disable_benchmark,
    enable_benchmark,
    estimate_simulation,
    preparation,
    reset_benchmark,
    simulate_pool,
    um,
)
from axonscope import analysis, results, signals
from axonscope.axons import HodgkinHuxley, MRG, RattayAberham, mrg_like_length_from_nodes
from axonscope.channel_models import IonChannelModelBase, MembraneStateSpec, PassiveICM
from axonscope.icm import CompartmentMembraneLayout, HeterogeneousICMBackend
from axonscope.solvers import (
    CrankNicholson,
    DoubleCableKernel,
    SingleCableKernel,
    prepare_solver_runtime,
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
    electrode = PointSourceElectrode(x=0.0 * um, z=1000.0 * um, stimulus=Stimulus.constant(0.0))
    assert AnalyticalExtracellularContext(electrodes=[electrode]).sigma_S_m == 0.3
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
    assert hasattr(__import__("axonscope").positions, "DISTAL")
    assert AxonInstance is not None
    assert AxonPopulation is not None
    assert AxonSimulation is not AxonInstance
    assert BenchmarkReport is not None
    assert BenchmarkSession is not None
    assert Device.auto().kind == "auto"
    assert MemoryEstimateItem is not None
    assert PrecisionPolicy.float32().solver_dtype == "float32"
    assert Runtime.AUTO.value == "auto"
    assert SimulationEstimate is not None
    assert estimate_simulation is not None
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
    assert SingleCableKernel is not None
    assert DoubleCableKernel is not None
    assert prepare_solver_runtime is not None
    assert IonChannelModelBase is not None
    assert PassiveICM is not None
    assert MembraneStateSpec("x").name == "x"
    assert CompartmentMembraneLayout is not None
    assert HeterogeneousICMBackend is not None
    assert mrg_like_length_from_nodes(10.0 * um, 3) > 0.0
