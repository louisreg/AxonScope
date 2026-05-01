from axonscope import PointSourceElectrode, Stimulus
from axonscope.axons import HodgkinHuxley, MRG, Passive, RattayAberham
from axonscope.channel_models import IonChannelModelBase, MembraneStateSpec, PassiveICM
from axonscope.icm import CompartmentMembraneLayout, HeterogeneousICMBackend
from axonscope.morphology import get_mrg_morphology
from axonscope.solvers import (
    CrankNicholson,
    DoubleCableKernel,
    Euler,
    SingleCableKernel,
    prepare_solver_runtime,
)


def test_public_package_imports_are_available():
    assert Stimulus.constant(1.0).y[0] == 1.0
    assert PointSourceElectrode(x0_m=0.0).sigma_S_m == 0.3
    assert HodgkinHuxley is not None
    assert RattayAberham is not None
    assert Passive is not None
    assert MRG is not None
    assert CrankNicholson is not None
    assert SingleCableKernel is not None
    assert DoubleCableKernel is not None
    assert Euler is not None
    assert prepare_solver_runtime is not None
    assert IonChannelModelBase is not None
    assert PassiveICM is not None
    assert MembraneStateSpec("x").name == "x"
    assert CompartmentMembraneLayout is not None
    assert HeterogeneousICMBackend is not None
    assert get_mrg_morphology(10.0).fiberD == 10.0
