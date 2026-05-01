# AxonScope

AxonScope is an early-stage Python framework for axon simulations, with a focus on
validated peripheral nerve models, extracellular stimulation, and solver backends
that can eventually scale to large fiber populations.

The project is currently in an active refactor. The codebase now separates:

- axon descriptions and geometry
- stimulus and electrode descriptions
- membrane/channel models
- solver/runtime compilation
- NRV comparison tests
- benchmark and profiling experiments

## Current Capabilities

- Uniform and non-uniform one-dimensional cable geometries
- Passive, Hodgkin-Huxley, Rattay-Aberham, Sundt, Tigerholm, Schild94/Schild97,
  and MRG-style myelinated axons
- Intracellular current clamps via `Stimulus`
- Point-source extracellular stimulation via `Electrode` and `ExtracellularContext`
- Crank-Nicholson/Hines-style solvers with extracellular one-layer coupling
- Generic heterogeneous membrane layouts for multicompartment axons
- NRV comparison tests for morphology, numerics, intracellular stimulation,
  extracellular stimulation, and velocity trends

## Installation

This repository uses a `src/` layout and Python 3.11+.

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
pytest -q tests/unit
```

NRV comparison tests require a local NRV checkout/environment and are marked under
`tests/nrv`.

## Package Map

```text
src/axonscope/
  axons/              axon descriptions, geometry, myelinated/unmyelinated models
  channel_models/     membrane/channel models and composite dynamics
  icm/                membrane compute backends and heterogeneous layouts
  morphology/         morphology tables and interpolation helpers
  solvers/            Euler and Crank-Nicholson solver families
  stimulus.py         backend-independent stimulus descriptions
  stimulus_eval.py    NumPy evaluation helpers for stimuli and extracellular contexts
  stimulation.py      intracellular/extracellular stimulation descriptors
  electrodes.py       electrode descriptions
```

## Quick Start: Intracellular Stimulation

```python
import numpy as np

from axonscope.axons import HodgkinHuxley
from axonscope.solvers import CrankNicholson
from axonscope.stimulus import Stimulus

axon = HodgkinHuxley(L=500.0, d=0.5, Nx=41, celsius=6.3)
axon.insert_I_Clamp(
    position=250.0,
    stimulus=Stimulus.pulse(start=1.0, duration=0.5, amplitude=2.0),
)

res = CrankNicholson().solve(axon, tsim=5.0, dt=0.01)
center = int(np.argmin(np.abs(np.asarray(axon.x) - 250.0)))
print(res.t.shape, res.Vm[:, center].shape)
```

`insert_I_Clamp` still accepts the legacy `t_start`, `duration`, `amplitude`
arguments for compatibility, but new code should pass an explicit `Stimulus`.

## Quick Start: Extracellular MRG Stimulation

```python
import numpy as np

from axonscope.axons import MRG
from axonscope.electrodes import PointSourceElectrode
from axonscope.solvers import CrankNicholson
from axonscope.stimulus import Stimulus

axon = MRG(d=10.0, nodes=5)
center_x_m = float(np.asarray(axon.x)[axon.Nx // 2]) * 1e-6

electrode = PointSourceElectrode(x0_m=center_x_m, z0_m=500e-6, sigma_S_m=0.3)
stimulus = Stimulus.biphasic(
    start=0.5,
    cathodic_amplitude=80e-6,
    cathodic_duration=0.05,
    interphase=0.02,
)

axon.add_extracellular_ctx(electrode, stimulus)
res = CrankNicholson().solve(axon, tsim=2.0, dt=0.01)
```

Extracellular contexts are descriptive objects. Solver-specific JAX compilation
lives in `axonscope.solvers.stimulus_runtime`; NumPy evaluation helpers live in
`axonscope.stimulus_eval`.

## Recording Observables

Solvers can record gates, currents, conductances, and optional membrane state
variables.

```python
res = CrankNicholson().solve(
    axon,
    tsim=5.0,
    dt=0.01,
    record_observables=True,
)

print(res.recordings["gates"].keys())
print(res.recordings["currents"].keys())
```

Dynamic membrane state is model-owned. Models such as Tigerholm and Schild declare
their state variables through `MembraneStateSpec`; the base membrane model no
longer hard-codes sodium, potassium, or calcium-specific flags.

## Examples

Runnable examples live in `examples/basic/`:

```bash
python examples/basic/stimulus_demo.py
python examples/basic/point_source_electrode_demo.py
python examples/basic/intracellular_solver_demo.py
python examples/basic/mrg_extracellular_demo.py
python examples/basic/environment_info_demo.py
```

The `playground/` directory contains exploratory diagnostics and comparison
scripts. It is useful during development but is not treated as public API.

## Tests

Fast unit suite:

```bash
pytest -q tests/unit
```

NRV validation suite:

```bash
pytest -q tests/nrv
```

Useful targeted checks during solver work:

```bash
pytest -q tests/unit/solvers/test_extracellular.py
pytest -q tests/nrv/extracellular/test_extracellular_systematic_vs_nrv.py
pytest -q tests/nrv/numerics/test_mrg_morphology_vs_nrv.py
pytest -q tests/nrv/numerics/test_mrg_compartment_geometry_vs_nrv.py
```

## Benchmarks

Solver benchmarks start from the shared `Solver` API and write JSON/CSV results:

```bash
python benchmark/solver_runtime/benchmark_solver.py --list
python benchmark/solver_runtime/benchmark_solver.py --cases hh_intracellular_small --repeats 3
python benchmark/solver_runtime/benchmark_solver.py --cases all --repeats 3
```

Compare two runs after a solver refactor:

```bash
python benchmark/solver_runtime/compare_results.py \
  benchmark/results/solver_runtime/baseline.json \
  benchmark/results/solver_runtime/current.json
```

The first default workloads cover HH intracellular, Rattay-Aberham
intracellular, Schild97 intracellular, and MRG extracellular stimulation.

Reference and experimental backend comparisons remain under
`benchmark/CrankNicholson_runtime/`; those scripts are being consolidated around
the shared benchmark runner.

Generated logs and figures are ignored by git.

## Architecture Notes

- `Stimulus`, `Electrode`, `IntracellularCurrentClamp`, and
  `ExtracellularContext` are backend-independent descriptions.
- `JaxStimulus`, compiled extracellular contexts, and current-density builders
  belong to the solver runtime.
- `prepare_solver_runtime` is the first data-oriented boundary between axon
  descriptions and solver kernels. It gathers initial states, cable arrays, and
  compiled stimulation without mutating the axon.
- Extracellular Crank-Nicholson solvers precompute imposed `Vstim` samples on
  the solver time grid, then index those arrays inside the time loop. This is
  the first step toward batch-friendly `Vstim[B, Nt, Nx]` inputs.
- `AxonBase` describes geometry and attached stimuli; solvers own runtime arrays.
- `CompartmentMembraneLayout` assigns one membrane model per compartment.
- `HeterogeneousICMBackend` evaluates heterogeneous membrane layouts.
- MRG uses a generic heterogeneous layout rather than an MRG-specific masked ICM.

## Changelog

See [CHANGELOG.md](./CHANGELOG.md).
