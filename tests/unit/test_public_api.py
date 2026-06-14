from axonscope import (
    AnalyticalExtracellularContext,
    AxonInstance,
    AxonPopulation,
    AxonSimulation,
    IntracellularContext,
    IntracellularCurrentClamp,
    PointSourceElectrode,
    RecordingSpatial,
    Signal,
    Stimulus,
    simulate_pool,
    um,
)
from axonscope import results, signals
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
    assert results.analysis.rasterize is not None
    assert results.visualization.plot_raster is not None
    electrode = PointSourceElectrode(x=0.0 * um, z=1000.0 * um, stimulus=Stimulus.constant(0.0))
    assert AnalyticalExtracellularContext(electrodes=[electrode]).sigma_S_m == 0.3
    assert isinstance(
        IntracellularCurrentClamp(position=0.0 * um, current=Stimulus.constant(0.0)),
        IntracellularContext,
    )
    assert simulate_pool is not None
    assert Signal.VM is not None
    assert signals.Vm is Signal.VM
    assert RecordingSpatial.CENTER is not None
    assert hasattr(__import__("axonscope").positions, "DISTAL")
    assert AxonInstance is not None
    assert AxonPopulation is not None
    assert AxonSimulation is not AxonInstance
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
