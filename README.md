# AxonScope

AxonScope is an early-stage Python framework for axon simulations, with a focus on
validated peripheral nerve models, extracellular stimulation, and solver backends
that can eventually scale to large axon pools.

The project is currently in an active refactor. The codebase now separates:

- axon descriptions and geometry
- stimulus and electrode descriptions
- simulation protocols that attach positions and stimulation to axons
- membrane/channel models
- solver/runtime compilation
- NRV comparison tests
- benchmark and profiling experiments

## Current Capabilities

- Uniform and non-uniform one-dimensional cable geometries
- Hodgkin-Huxley, Rattay-Aberham, Sundt, Tigerholm, Schild94/Schild97,
  and MRG-like myelinated axon templates
- Intracellular current clamps via `Stimulus` and `AxonSimulation`
- Point-source extracellular stimulation via `Electrode` and `ExtracellularContext`
- Crank-Nicholson/Hines-style solvers with extracellular one-layer coupling
- Generic heterogeneous membrane layouts for multicompartment axons
- NRV comparison tests for morphology, numerics, intracellular stimulation,
  extracellular stimulation, and velocity trends

## Installation

This repository uses a `src/` layout and Python 3.11+.

```bash
python -m pip install -e .
```

For runnable examples:

```bash
python -m pip install -e ".[examples]"
```

For development:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q tests/unit
```

Benchmark scripts use the benchmark extra:

```bash
python -m pip install -e ".[benchmark]"
```

NRV comparison tests require a local NRV checkout/environment. AxonScope declares
only the Python-side helper dependencies in the `nrv` extra; the NRV checkout and
its local logging/mod-file setup remain external:

```bash
python -m pip install -e ".[dev,nrv]"
```

Those tests are marked under `tests/nrv`.

## Package Map

```text
src/axonscope/
  axons/              axon descriptions, geometry, myelinated/unmyelinated models
  axon_simulation.py  simulation protocols attached to descriptive axons
  analysis.py         post-hoc result analysis helpers
  membranes/          runtime-independent public membrane descriptions
  channel_models/     solver-side membrane/channel implementations
  dispatcher/         axon-pool planning and execution
  icm/                membrane compute backends and heterogeneous layouts
  results/            single-axon and pool result containers
  solvers/            Crank-Nicholson runtime, scalar kernels, and batch kernels
  stimulation/        stimuli, electrodes, and physical stimulation contexts
  visualization.py    plotting helpers for results and model inspection
```

See `docs/results_recording_analysis.md` for the current `Recording`,
`SimResult`, analysis, and visualization contracts. See
`docs/pool_dispatch.md` for the current public/advanced pool split.

## Axon Package

`src/axonscope/axons/` is the descriptive model layer. It should describe what
the axon is, not how a specific run stimulates or solves it.

- `section.py` defines `Section` and `PeriaxonalLayer`: local membrane/material
  properties for one conceptual cable section.
- `layout.py` defines `LayoutElement` and `Layout`: section placement, section
  lengths, and compartment counts.
- `flattened.py` derives `FlattenedLayout`: one value per numerical compartment
  for solver/runtime code.
- `axon.py` defines `Axon`: a layout plus cable formulation, initial voltage,
  and temperature.
- `unmyelinated.py` provides single-cable templates such as `HodgkinHuxley`,
  `RattayAberham`, `Sundt`, `Tigerholm`, `Schild94`, and `Schild97`.
- `myelinated.py` provides myelinated descriptions such as `MRG` and node
  helpers.
- `templates/` contains reusable morphology/layout builders, currently the
  MRG-like double-cable node/MYSA/FLUT/STIN template.
- `formulation.py` validates or infers `single-cable` versus `double-cable`.
- `plotting.py` draws layouts with section spans, compartments, labels, and
  optional section colors.

User-facing axon APIs use short physical names and require explicit units:
`length=500 * axs.um`, `diameter=0.5 * axs.um`,
`v_init=-70 * axs.mV`, `temperature=37 * axs.degC`.
Plain numbers are rejected at public axon/section boundaries when the physical
dimension would be ambiguous.

For direct geometry authoring:

```python
import axonscope as axs

section = axs.axons.Section(
    "axon",
    membrane=axs.membranes.HodgkinHuxley(celsius=6.3 * axs.degC),
    diameter=0.5 * axs.um,
)
layout = axs.axons.Layout.single_uniform(
    section,
    length=500.0 * axs.um,
    compartments=41,
)
axon = axs.axons.Axon(layout=layout, v_init=-70.0 * axs.mV)
```

For standard models, prefer the high-level constructors and inspect the layout
when needed:

```python
axon = axs.axons.MRG(
    diameter=10.0 * axs.um,
    nodes=9,
    compartments={"node": 1, "MYSA": 1, "FLUT": 2, "STIN": 4},
)
x_um = axon.layout.position_values(unit=axs.um)
axon.layout.plot()
```

`axs.AxonSimulation(axon)` adds protocol state: global position, intracellular
clamps, and extracellular contexts. Solvers accept either a pure descriptive
axon with no stimulation, or an `AxonSimulation`.

## Quick Start: Intracellular Stimulation

```python
import axonscope as axs

axon = axs.axons.HodgkinHuxley(
    length=500.0 * axs.um,
    diameter=0.5 * axs.um,
    compartments=41,
    celsius=6.3 * axs.degC,
)
sim = axs.AxonSimulation(axon)
clamp = axs.IntracellularCurrentClamp(
    position_um=250.0 * axs.um,
    current=axs.Stimulus.pulse(
        start=1.0 * axs.ms,
        duration=0.5 * axs.ms,
        amplitude=2.0 * axs.nA,
    ),
)
sim.add_intracellular_context(context=clamp)

res = axs.solvers.CrankNicholson().solve(sim, tsim=5.0 * axs.ms, dt=0.01 * axs.ms)
center = res.nearest_position_index(250.0 * axs.um)
print(res.t.shape, res.Vm[:, center].shape)
```

## Quick Start: Extracellular MRG-Like Stimulation

```python
import axonscope as axs

axon = axs.axons.MRG(
    diameter=10.0 * axs.um,
    nodes=5,
    compartments={"node": 1, "MYSA": 1, "FLUT": 2, "STIN": 4},
)
center_x = axon.layout.position_values(unit=axs.um)[axon.n_compartments // 2] * axs.um

electrode = axs.PointSourceElectrode(
    x_um=center_x,
    z_um=500.0 * axs.um,
)
stimulus = axs.Stimulus.biphasic(
    start=0.5 * axs.ms,
    cathodic_amplitude=80.0 * axs.uA,
    cathodic_duration=0.05 * axs.ms,
    interphase=0.02 * axs.ms,
)

sim = axs.AxonSimulation(axon)
sim.add_extracellular_context(
    context=axs.AnalyticalExtracellularContext(
        electrodes=[electrode.with_stimulus(stimulus)],
        sigma=0.3 * axs.S_per_m,
    )
)
res = axs.solvers.CrankNicholson().solve(sim, tsim=2.0 * axs.ms, dt=0.01 * axs.ms)
```

Extracellular contexts are descriptive groups of stimulated electrodes.
JAX-ready compilation lives in `axonscope.stimulation.runtime`; NumPy
evaluation is available directly on contexts via `context.evaluate(...)`.

Default extracellular solver policy:

- Single-cable axons use an imposed-field Vstim forcing solve. The solver
  precomputes `Vstim(t, x)` and adds the scalar axial forcing term to the cable
  solve. This keeps unmyelinated extracellular workloads close to the future
  batch/GPU data layout.
- Heterogeneous/double-cable axons such as `MRG` keep the dynamic Vi/Vperi
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

Examples are organized by intent:

- `examples/basic/`: short, didactic scripts for one concept at a time.
- `examples/advanced/`: population and dispatcher workflows.
- `benchmark/`: timing, profiling, memory, and reproducibility utilities.

See `examples/README.md` for the intended learning path.

Basic examples:

```bash
python examples/basic/example_01_stimulus_waveforms.py
python examples/basic/example_02_point_source_electrode.py
python examples/basic/example_03_intracellular_hh.py
python examples/basic/example_04_extracellular_mrg.py
python examples/basic/example_05_pool_dispatch_basic.py
python examples/basic/example_06_velocity_vs_diameter_batch.py
```

## Tests

The default CI path runs the fast unit suite and whitespace checks. See
`docs/validation.md` for the full validation policy and current NRV reference
status.

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

For reproducibility metadata:

```bash
python benchmark/runtime/environment_info_demo.py
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
python benchmark/runtime/run.py --suite reference_solvers --prefix cn_reference_compare
python benchmark/runtime/run.py --suite vstim_forcing --prefix hh_vstim_forcing_compare
python benchmark/runtime/run.py --suite vstim_batch --prefix hh_vstim_batch_compare
python benchmark/runtime/run.py --suite double_cable_batch --prefix hh_double_cable_batch_compare
python benchmark/runtime/run.py \
  --suite profiled \
  --prefix solver_runtime_current \
  -- \
  --jax-profile-name solver_runtime_current
```

Use the precision/rate-table benchmark when changing dtype, gating equations, or
lookup-table settings. It launches one worker per dtype so JAX x64 is configured
before importing AxonScope:

```bash
python benchmark/runtime/benchmark_precision_rates.py \
  --suite smoke \
  --dtypes float32 float64 \
  --rate-table-step-mv 0.05 \
  --repeats 2 \
  --warmups 1 \
  --prefix precision_rates_smoke
```

Outputs are written to `benchmark/results/runtime/precision_rates_smoke.json`
and `.csv`. The benchmark covers HH intracellular, HH extracellular,
double-cable extracellular, Schild, and Tigerholm smoke cases; rate-table rows
are enabled by default for Schild and Tigerholm. For manual runs, set
`AXONSCOPE_DTYPE=float64` to use float64 globally in a fresh Python process.

Batch solvers consume imposed extracellular fields with a leading batch axis.
Each row in `context_batch` can be one `ExtracellularContext`, a tuple/list of
contexts that are summed, or `None` for a zero-field control row:

```python
from axonscope.dispatcher.runtime_batches import (
    build_footprint_vstim_initial_previous_batch,
    build_footprint_vstim_midpoint_batch,
    build_vstim_initial_previous_batch,
    build_vstim_midpoint_batch,
    scale_extracellular_contexts,
)
from axonscope.solvers import (
    BatchOptions,
    BatchRecording,
    DoubleCableBatchKernel,
    SingleCableVStimBatchKernel,
    prepare_solver_runtime,
)

tsim_ms = 1.2
dt_ms = 0.01
base_contexts = tuple(sim.extracellular_contexts)
context_batch = [
    base_contexts,
    scale_extracellular_contexts(base_contexts, 0.5),
    None,
]

vstim_mid = build_vstim_midpoint_batch(
    sim,
    context_batch,
    tsim_ms=tsim_ms,
    dt_ms=dt_ms,
)
```

Pass `x_positions_m` with shape `(B, Nx)` to the `Vstim` builders when each
batch row represents a fiber translated relative to the electrode.

For pool runs, the preferred fast path is to pass precomputed electrode
footprints directly. This is the shape expected from FEM/interpolation
workflows:

```python
# footprint_V_per_A has shape (B, Nx), in V/A.
vstim_mid = build_footprint_vstim_midpoint_batch(
    stimulus=stimulus,
    footprint_V_per_A=footprint_V_per_A,
    tsim_ms=tsim_ms,
    dt_ms=dt_ms,
)
vstim_previous = build_footprint_vstim_initial_previous_batch(
    stimulus=stimulus,
    footprint_V_per_A=footprint_V_per_A,
    dt_ms=dt_ms,
)
```

This builder uses a NumPy data-prep path by default, then returns a JAX array
for the solver. Use `engine="jax"` only when you specifically want JAX-side
stimulus/footprint multiplication.

For homogeneous single-cable extracellular batches, use imposed-field forcing:

```python
runtime = prepare_solver_runtime(
    sim,
    tsim_ms=tsim_ms,
    dt_ms=dt_ms,
    include_extracellular=False,
    include_area=False,
    precompute_intracellular=True,
    precompute_extracellular=False,
)

result = SingleCableVStimBatchKernel(
    runtime,
    Cm_uF_cm2=runtime.axon.Cm_uF_cm2,
).run(
    extracellular_potential_mid_mV=vstim_mid,
)
# result.Vm has shape (B, Nt, Nx)
```

For full double-cable batches, keep all cable/membrane arrays shared for now and
provide the additional previous imposed field sample required by the
extracellular state equation:

```python
runtime = prepare_solver_runtime(
    axon,
    tsim_ms=tsim_ms,
    dt_ms=dt_ms,
    include_extracellular=True,
    include_area=True,
    precompute_intracellular=True,
    precompute_extracellular=False,
)
vstim_previous = build_vstim_initial_previous_batch(
    axon,
    context_batch,
    dt_ms=dt_ms,
)

result = DoubleCableBatchKernel(runtime, Veinit_mV=axon.Veinit).run(
    extracellular_potential_mid_mV=vstim_mid,
    extracellular_potential_initial_previous_mV=vstim_previous,
)
```

For larger batches, the solver still expects materialized `Vstim` arrays, but
you can chunk the time loop and record only probes to reduce retained `Vm`
memory:

```python
options = BatchOptions(
    recording=BatchRecording.indices([0, axon.n_compartments // 2, axon.n_compartments - 1]),
    time_chunk_steps=50,
)

result = DoubleCableBatchKernel(runtime, Veinit_mV=axon.Veinit).run(
    extracellular_potential_mid_mV=vstim_mid,
    extracellular_potential_initial_previous_mV=vstim_previous,
    options=options,
)
# result.Vm has shape (B, Nt, 3)
```

For heterogeneous axon lists, use the pool wrapper. The current
implementation keeps the public API simple, groups compatible axons into batch
kernels, and falls back to scalar solves for incompatible axons while preserving
pool-order results:

```python
import axonscope as axs

sim_a = axs.AxonSimulation(axon_a, y_um=20.0 * axs.um, z_um=30.0 * axs.um)
sim_b = axs.AxonSimulation(axon_b, y_um=-40.0 * axs.um, z_um=10.0 * axs.um)
extracellular = axs.AnalyticalExtracellularContext(
    electrodes=[electrode.with_stimulus(stimulus)],
    sigma=0.3 * axs.S_per_m,
)
for sim in (sim_a, sim_b):
    sim.add_extracellular_context(context=extracellular)

result = axs.simulate_pool(
    [sim_a, sim_b],
    duration_ms=1.0 * axs.ms,
    dt_ms=0.01 * axs.ms,
    recording=axs.Recording.center("Vm"),
)
# result is list[SimResult], in the same order as the input simulations.
# Here each result.Vm has shape (Nt, 1) because center recording is selected.
```

A complete pool dispatch script is available for quick experiments:

```bash
python examples/advanced/example_01_pool_dispatch_nrv.py --fibers 8
python examples/advanced/example_01_pool_dispatch_nrv.py --source nrv --fibers 16
python examples/advanced/example_05_recording_options.py
python examples/advanced/example_06_activation_criterion.py
python examples/advanced/example_07_recruitment_curve.py
```

The pool example covers mixed models, mixed `n_compartments`, per-axon
positions, shared extracellular context, per-axon intracellular stimulation,
and public `Recording.center("Vm")` output. The NRV mode uses NRV's `axon_pool`
placement helpers when NRV is installed, then maps the generated table to
AxonScope simulations. The recording example shows single-axon observable groups
and pool Vm retention modes: `full`, `center`, `probes`, and `indices`. The
activation criterion example shows the post-hoc semantics that future
solver-side observers will share. The recruitment example uses the protocol API
for binary threshold search and sampled recruitment curves.

Use the runtime benchmark to compare the dominant tensor sizes and timings:

```bash
python benchmark/runtime/pool_memory.py \
  --mode double \
  --fibers 128 \
  --nx 201 \
  --tsim 2.0 \
  --scenarios full center center_chunked probes_chunked
```

Capture a JAX profiler trace around the same workflow with:

```bash
python benchmark/runtime/pool_batch_demo.py \
  --mode double \
  --fibers 64 \
  --nx 201 \
  --tsim 2.0 \
  --batch-only \
  --record center \
  --time-chunk-steps 50 \
  --repeats 1 \
  --warmups 0 \
  --jax-profile-dir benchmark/results/jax_profiles \
  --jax-profile-name pool_double_b64_nx201

python benchmark/runtime/summarize_trace.py \
  benchmark/results/jax_profiles/pool_double_b64_nx201 \
  --pattern pool/ \
  --timeline \
  --csv-out benchmark/reports/runtime/pool_double_b64_nx201_trace.csv
```

The profiling script builds a small pool with per-axon longitudinal offsets and
radial distances, compares scalar-loop and batched execution, and prints warm
speedups. It uses a point-source footprint as a compact debug source; pass
`--generic-vstim` to compare the slower context-based builder.

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
intracellular, Schild97 intracellular, and MRG-like extracellular stimulation.

Generated logs and figures are ignored by git.

## Architecture Notes

- `Stimulus`, `Electrode`, `IntracellularContext`, and
  `ExtracellularContext` are backend-independent descriptions.
  `IntracellularCurrentClamp` is the current point-injection context. Contexts
  own lightweight NumPy evaluation for inspection; solver compilation is
  separate.
- `JaxStimulus`, `compile_intracellular_contexts`, and
  `compile_extracellular_contexts` live in `axonscope.stimulation.runtime`;
  `add_current_clamp(...)` is intentionally kept as a compact wrapper around
  `IntracellularCurrentClamp`.
- `prepare_solver_runtime` is the first data-oriented boundary between axon
  descriptions and solver kernels. It gathers initial states, cable arrays, and
  compiled stimulation without mutating the axon.
- Extracellular single-cable Crank-Nicholson solvers precompute imposed `Vstim`
  samples on the solver time grid, then add `L(Vstim)` as a known axial forcing
  term inside the time loop. This is the first step toward batch-friendly
  `Vstim[B, Nt, Nx]` inputs.
- `SingleCableVStimBatchKernel` is the first low-level batch API for this path:
  it accepts imposed fields shaped `(B, Nt, Nx)` and returns `Vm[B, Nt, Nx]`.
- `axonscope.dispatcher.runtime_batches.build_vstim_midpoint_batch` generates
  those imposed fields directly from batched extracellular context rows,
  including per-row `(B, Nx)` fiber positions.
  `build_vstim_initial_previous_batch` provides the matching `t=-dt/2` sample
  needed by double-cable kernels.
- `DoubleCableBatchKernel` keeps the first double-cable batch policy simple:
  shared axon structure and extracellular parameters, batched `Vstim`/`Iinj`.
- The optimized Crank-Nicholson default path precomputes intracellular current
  density samples and calls explicit JIT-compiled VM-only single-cable or
  double-cable kernels. Recording observables still uses the more general path.
- `axonscope.solvers.experimental` intentionally keeps only the maintained
  reference/prototype solvers: dense CN and imposed-field Vstim forcing.
- The full double-cable reference path uses scalar coefficient arrays for its
  2x2 block solve, avoiding per-step materialization of `(Nx, 2, 2)` matrices.
- `Axon` describes section layout, cable formulation, and attached stimuli; solvers own runtime arrays.
- `CompartmentMembraneLayout` assigns one membrane model per compartment.
- `HeterogeneousICMBackend` evaluates heterogeneous membrane layouts.
- `MRG` uses the generic heterogeneous layout rather than a template-specific masked ICM.

## Changelog

See [CHANGELOG.md](./CHANGELOG.md).
