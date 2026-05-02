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

axon.add_extracellular_context(electrode, stimulus)
res = CrankNicholson().solve(axon, tsim=2.0, dt=0.01)
```

Extracellular contexts are descriptive objects. Solver-specific JAX compilation
lives in `axonscope.solvers.stimulus_runtime`; NumPy evaluation helpers live in
`axonscope.stimulus_eval`.

Default extracellular solver policy:

- Single-cable axons use an imposed-field Vstim forcing solve. The solver
  precomputes `Vstim(t, x)` and adds the scalar axial forcing term to the cable
  solve. This keeps unmyelinated extracellular workloads close to the future
  batch/GPU data layout.
- Heterogeneous/double-cable axons such as MRG keep the dynamic Vi/Vperi
  double-cable solve.

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

Benchmarks are grouped by intent:

```text
benchmark/
  runtime/       official solver performance benchmarks
  nrv_performance/
                 AxonScope-vs-NRV compute performance comparisons
  results/       generated JSON/CSV/trace outputs, ignored by git
  reports/       generated HTML/PNG/CSV reports, ignored by git
```

Numerical validity against NRV belongs in `tests/nrv`. The benchmark workflow
below is for measuring compute performance against NRV after those tests already
guard correctness. Named suites are declared in
`benchmark/nrv_performance/suites.py`:

```bash
python benchmark/nrv_performance/run.py --list
python benchmark/nrv_performance/run.py --suite smoke --dry-run
python benchmark/nrv_performance/run.py --suite smoke
python benchmark/nrv_performance/run.py --suite mrg_extracellular_perf
python benchmark/nrv_performance/run.py --suite mrg_extracellular_gates
```

Use `mrg_extracellular_perf` for warm runtime comparisons. Use
`mrg_extracellular_gates` when you also want MRG `m`-gate diagnostics; that
suite records more data and is less representative as a pure timing baseline.

The NRV performance runner forwards additional options to
`benchmark/nrv_performance/nrv_axonscope_grid.py` after `--`, so focused runtime
sweeps stay easy to launch:

```bash
python benchmark/nrv_performance/run.py \
  --suite smoke \
  --prefix hh_dt_probe \
  -- \
  --dt 0.005 0.01 \
  --nx 21 51 \
  --tsim 1.0
```

Plot an AxonScope-vs-NRV sweep from the generated CSV:

```bash
python benchmark/nrv_performance/plot_results.py \
  benchmark/results/nrv_performance/nrv_axonscope_grid/hh_dt_probe.csv \
  --out-dir benchmark/reports/nrv_performance \
  --prefix hh_dt_probe \
  --metric warm_total \
  --x axon_nx
```

For runtime-only solver work, use the runtime suites:

```bash
python benchmark/runtime/run.py --list
python benchmark/runtime/run.py --suite smoke
python benchmark/runtime/run.py --suite full --prefix solver_runtime_current
python benchmark/runtime/run.py --suite experimental_solvers --prefix cn_experimental_compare
python benchmark/runtime/run.py --suite vstim_forcing --prefix hh_vstim_forcing_compare
python benchmark/runtime/run.py \
  --suite profiled \
  --prefix solver_runtime_current \
  -- \
  --jax-profile-name solver_runtime_current
```

Then build a static HTML report and summarize the JAX trace:

```bash
python benchmark/runtime/visualize_results.py \
  benchmark/results/runtime/solver_runtime_current.json \
  --out-dir benchmark/reports/runtime \
  --prefix solver_runtime_current

python benchmark/runtime/summarize_trace.py \
  benchmark/results/jax_profiles/solver_runtime_current \
  --timeline \
  --csv-out benchmark/reports/runtime/solver_runtime_current_trace.csv
```

The profiler trace is written under:

```text
benchmark/results/jax_profiles/solver_runtime_current/plugins/profile/...
```

JAX emits phase annotations such as `build_axon`, `first_solve`,
`measured_solve`, and `measured_materialize`, which makes it easier to separate
Python object construction, XLA compilation, blocked solver execution, and output
materialization. Add `--jax-profile-perfetto` when you want JAX to also emit a
local Perfetto trace file, if supported by the installed JAX version.

Use `summarize_trace.py` when the Perfetto timeline is too dense: it extracts
only the `benchmark/...` annotations and reports `count`, `total_ms`, `mean_ms`,
`median_ms`, `min_ms`, and `max_ms` per phase.

Example runtime output from a local optimized run:

```text
hh_intracellular_small           CrankNicholson build=0.9271s first=4.5201s compile_est=4.5017s mat=0.0002s total=4.5202s warm=0.0186s warm_total=0.0187s
rattay_intracellular_small       CrankNicholson build=0.0073s first=3.6078s compile_est=3.5876s mat=0.0001s total=3.6080s warm=0.0208s warm_total=0.0209s
schild97_intracellular_small     CrankNicholson build=0.0180s first=4.3731s compile_est=4.2839s mat=0.0001s total=4.3732s warm=0.0892s warm_total=0.0893s
mrg_extracellular_small          CrankNicholson build=0.0154s first=4.3919s compile_est=4.3310s mat=0.0001s total=4.3920s warm=0.0617s warm_total=0.0618s
```

The `vstim_forcing` suite compares the dynamic double-cable extracellular solve
against the imposed-field single-cable path used by default for non-heterogeneous
extracellular axons. On local HH extracellular runs, the Vstim path reduced warm
runtime strongly at larger `Nx` while staying within the NRV extracellular test
tolerances.

Read the columns as:

- `first`: first blocked solve, dominated by JAX compilation on new signatures.
- `compile_est`: `first` minus the fastest warm solve; useful as a tracking
  signal, not as a replacement for the profiler.
- `warm`: blocked solve on rebuilt but equivalent workloads, the main runtime
  metric for solver-loop optimizations.
- `mat`: time to materialize and summarize the output arrays.
- `warm_total`: warm solve plus materialization, closest to usable-output time.

The lower-level NRV grid script remains available when you want to bypass named
suites entirely:

```bash
python benchmark/nrv_performance/nrv_axonscope_grid.py --list
python benchmark/nrv_performance/nrv_axonscope_grid.py --profile full --dry-run
python benchmark/nrv_performance/nrv_axonscope_grid.py \
  --profile smoke
python benchmark/nrv_performance/nrv_axonscope_grid.py \
  --model mrg_extracellular --dt 0.005 0.01 --nodes 5 9 --tsim 4 \
  --repeats 4 --warmups 1
```

Compare two runtime runs after a solver refactor:

```bash
python benchmark/runtime/compare_results.py \
  benchmark/results/runtime/baseline.json \
  benchmark/results/runtime/current.json
```

The visual report can also compare several JSON files in one page:

```bash
python benchmark/runtime/visualize_results.py \
  benchmark/results/runtime/baseline.json \
  benchmark/results/runtime/current.json \
  --out-dir benchmark/reports/runtime \
  --prefix baseline_vs_current
```

The NRV/AxonScope grid reports blocked AxonScope solve timings, explicit output
materialization timings for both AxonScope and NRV, total usable-output timings,
Vm error metrics, spike timing metrics, velocity estimates, spatial alignment
error, and optional `m` gate metrics for gate-level diagnostics.

The shared solver benchmark reports solve-only, materialization, total
usable-output, and compile-estimate timings. The compile estimate is the first
blocked solve minus the fastest warm blocked solve, so it should be read as a
tracking signal rather than a standalone profiler.

The first default workloads cover HH intracellular, Rattay-Aberham
intracellular, Schild97 intracellular, and MRG extracellular stimulation.

Generated logs and figures are ignored by git.

## Architecture Notes

- `Stimulus`, `Electrode`, `IntracellularCurrentClamp`, and
  `ExtracellularContext` are backend-independent descriptions.
- `JaxStimulus`, compiled extracellular contexts, and current-density builders
  belong to the solver runtime.
- `prepare_solver_runtime` is the first data-oriented boundary between axon
  descriptions and solver kernels. It gathers initial states, cable arrays, and
  compiled stimulation without mutating the axon.
- Extracellular single-cable Crank-Nicholson solvers precompute imposed `Vstim`
  samples on the solver time grid, then add `L(Vstim)` as a known axial forcing
  term inside the time loop. This is the first step toward batch-friendly
  `Vstim[B, Nt, Nx]` inputs.
- The optimized Crank-Nicholson default path precomputes intracellular current
  density samples and calls explicit JIT-compiled VM-only single-cable or
  double-cable kernels. Recording observables still uses the more general path.
- `axonscope.solvers.experimental` intentionally keeps only a small set of
  reference/prototype solvers: dense CN, imposed-field Vstim forcing,
  semi-implicit ionic linearization, and a Newton-style implicit prototype.
- The full double-cable reference path uses scalar coefficient arrays for its
  2x2 block solve, avoiding per-step materialization of `(Nx, 2, 2)` matrices.
- `AxonBase` describes geometry and attached stimuli; solvers own runtime arrays.
- `CompartmentMembraneLayout` assigns one membrane model per compartment.
- `HeterogeneousICMBackend` evaluates heterogeneous membrane layouts.
- MRG uses a generic heterogeneous layout rather than an MRG-specific masked ICM.

## Changelog

See [CHANGELOG.md](./CHANGELOG.md).
