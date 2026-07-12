# P12B Runtime/JAX Cleanup - 2026-07-12

P12B continues after the P12A runtime-contract sanity gate. The goal is to make
the runtime boundary cleaner for a future NumPy/SciPy runtime while keeping the
current JAX hot paths intact.

## Scope

Keep cable-specific solvers, kernels, JAX arrays, JIT behavior, and device
placement in `src/axonscope/runtime/jax/`.

Move or formalize only semantic contracts that can apply to another runtime:
input-lowering modes, recording/observer-output semantics, execution policy
shape, benchmark vocabulary, and result assembly concepts.

## Cleanup Done In This Pass

- Shared single-cable and double-cable host preparation in
  `runtime/jax/group_runner.py` now goes through one helper for
  `runtime.prepare` and `inputs.positions`.
- Input planning no longer depends on observer-output planning. Recording and
  observer choices decide outputs, not whether an extracellular input is
  compact or dense.
- Double-cable input planning now predicts compact `shared_current` and
  `scaled_shared_waveform` factorized inputs consistently with the runtime
  lowering path, including probe/full-style recording.
- Dead parameters were removed from JAX input-lowering functions where the
  runtime did not use them.
- Guardrails now check that input planning does not reintroduce
  `observer_plan` coupling.

## Current Boundary

`runtime/input_contract.py` owns runtime-neutral semantic labels:

- cable formulation: `single-cable`, `double-cable`;
- intracellular modes: `zero`, `dense`, `sparse_current_clamp`;
- extracellular modes: `zero`, `shared_current`,
  `scaled_shared_waveform`, `current_table`, `dense`.

`runtime/jax/inputs/lowering.py` owns the current JAX implementation of those
semantics. It may use JAX-specific containers internally, but benchmark and
inspection metadata should report the runtime-neutral mode labels.

## Contract Cleanup Pass

The next P12B cleanup pass moved host-side contracts that are not inherently JAX
specific out of `runtime/jax/`:

- `OutputPlan` moved from `runtime/jax/output_plan.py` to
  `runtime/output_contract.py`. JAX batch execution still consumes it, but the
  concept describes output sinks (`vm`, `vm_raster`, `none`) and chunking, not
  JAX kernels.
- Intracellular and extracellular input-format type labels moved to
  `runtime/input_contract.py`; `runtime/jax/inputs/lowering.py` now owns only
  the JAX implementation of those labels.
- Observer-output labels and VmRaster observer compatibility moved to
  `runtime/output_contract.py`. Public estimate/inspection helpers can now ask
  `runtime.execution` for those labels without routing through
  `runtime.execution`.
- Guardrails now assert that `runtime/jax/output_plan.py` stays absent and that
  input/output labels remain runtime-neutral.
- Dense-equivalent input shape and byte-size helpers moved to
  `runtime/input_contract.py`, so JAX benchmark metadata no longer imports
  those generic memory-estimate helpers from JAX input lowering.
- Batch memory-estimate arithmetic moved to `runtime/memory_estimates.py`.
  `runtime/jax/benchmarking/metadata.py` now adapts JAX lowered payloads and
  adds optional JAX device-capacity metadata, but it no longer owns the
  runtime-neutral byte accounting for positions, dense/factorized Vstim,
  intracellular inputs, or retained Vm output.
- Dead observer-output proxy helpers were removed from the JAX benchmarking
  facade.
  The active facade for estimate/inspection code is now `runtime.execution`,
  backed by the runtime-neutral output contract for observer-output labels.
- Public dispatch-record assembly moved from JAX result helpers to
  `runtime/result_assembly.py`. `runtime/jax/recording/results.py` now keeps only
  JAX kernel-output synchronization, pending VmRaster finalization, and
  padded kernel-output trim.
- `runtime/jax/group_runner.py` now has a narrower orchestration shape:
  single-cable and double-cable input lowering remain cable-specific helpers,
  while shared progress, memory-estimate metadata, kernel compile progress, and
  retained Vm output metadata go through common helpers.
- Prepared-row input planning moved to `runtime/input_planning.py`: sampled
  footprint eligibility, factorized drive counts, planned extracellular mode,
  and scaled-shared-waveform signatures are now runtime-neutral. JAX still owns
  the actual materialized input payloads and cache behavior.
- Recording request conversion moved from `runtime/jax/recording.py` to
  `runtime/recording.py`. Public `Recording` and `RecordingPlan` lowering to
  `BatchOptions` is runtime-neutral; JAX-specific padded-row and VmRaster
  observer lowering remains in `runtime/jax/recording/lowering.py`.
- Padded recording handling also moved to `runtime/recording.py`:
  row-aware retained Vm indices, full-recording fallback for unsupported padded
  recordings, and cohort original-index tables are runtime-neutral batch
  semantics. `runtime/jax/recording/lowering.py` now only owns cached lowering
  from public observers to JAX VmRaster plans.
- Observer cache signatures moved to `runtime/output_contract.py`, so a future
  NumPy/SciPy runtime can reuse the same stable observer-definition identity
  instead of copying the JAX VmRaster-plan cache key logic.
- Estimate/inspection recording lowering now calls `runtime.recording`
  directly through `runtime.execution`; the old JAX benchmark proxy for
  `benchmark_lower_recording_options` was removed.
- Host-side cable/extracellular NumPy preparation moved to
  `runtime/host_preparation.py`: diffusion coefficients, compartment areas,
  padded space/edge/gate arrays, and double-cable extracellular host rows are
  now runtime-neutral helpers. `runtime/jax/preparation/stacking.py` owns JAX
  materialization into `CableRuntime` and `ExtracellularRuntime`.
- P12B source-pruning pass:
  test-only dense/reference Crank-Nicholson solvers moved from
  `runtime/jax/reference_solvers.py` to `tests/unit/solvers/_reference_solvers.py`;
  the P11B/P11C solver prototypes and rejected PCR-SoA probes moved under
  `benchmark/legacy/p11_solver_exploration/`; rejected or diagnostic
  candidate variants were removed from the active runtime primitive modules.
  The old `runtime/jax/kernels/common.py` bucket is gone; active shared
  primitives now live in `runtime/jax/cable_geometry.py`,
  `runtime/jax/kernels/double_cable_linear.py`, and
  `runtime/jax/kernels/block_tridiagonal.py` rather than benchmark-only probe
  code.
- Runtime-neutral stimulus current planning moved from
  `runtime/jax/inputs/extracellular.py` to `runtime/input_planning.py`: temporal
  current caching, sampled-stimulus semantic keys, rank-1 current-row planning,
  scaled-shared-waveform row planning, and cached array-content signatures are
  now reusable outside JAX. JAX input materialization is split between
  `inputs/extracellular.py` for footprints/factorized potentials and
  `inputs/intracellular.py` for dense/sparse current-density batches.
- Runtime-neutral dispatch-group preparation moved from
  `runtime/jax/preparation/runtime.py` and `runtime/jax/preparation/caches.py` to
  `runtime/group_preparation.py`: representative-row selection, runtime-context
  cache keys, dispatch-group structural signatures, prepared-cohort caches, and
  exact-group prepared-cohort reuse are no longer JAX runtime state.
  `runtime/jax/preparation/runtime.py` keeps JAX `SolverRuntime` construction,
  while `runtime/jax/preparation/stacking.py` keeps JAX array stacking, group
  `Cm` lowering, and JAX-specific membrane/cable/extracellular
  materialization. `runtime/jax/preparation/caches.py` keeps only JAX
  runtime/forcing caches.
- JAX gated/leak membrane row stacking moved from
  `runtime/jax/preparation/stacking.py` to `runtime/jax/membranes/stacking.py`.
  `runtime/jax/preparation/stacking.py` now orchestrates membrane stacking, while the
  capability-based gated/leak encoders and row caches live with the JAX
  membrane-stacking implementation. The unused `_encode_gated_leak_members`
  helper was removed rather than preserved.
- The tiny `runtime/jax/observables.py` module was removed. Its helper now lives
  in `runtime/jax/preparation/base.py`, because it packages base-runtime
  membrane outputs rather than defining an independent runtime boundary.
- A `vulture`-guided dead-code pass removed unused JAX runtime helpers that no
  active source or unit test called: `precompute_intracellular_current_density`,
  `RowIndexedMembraneBackend.init_gates_for_row`,
  `JaxModelIRLowering.source_observable_output_names`,
  `JaxMembraneProgram.gating_inf_tau`, and
  `JaxMembraneProgram.disable_rate_table`. The remaining `vulture` production
  warnings are contract or benchmark-analysis surfaces rather than deletion
  candidates for this pass.
- JAX membrane compilation and membrane-backend helpers were grouped under
  `runtime/jax/membranes/`: `backend.py`, `layout.py`,
  `model_ir_lowering.py`, `program.py`, and `stacking.py`.
- The unused rate-table option was removed completely:
  `SolverOptions.rate_table_config`, `RateTableConfig`, and the JAX
  `rate_tables.py` helper no longer exist.
- The JAX scalar fallback runner was removed. One-row public simulations now
  use the same batch route as populations (`B=1`), and unsupported dense
  observable recordings fail explicitly instead of selecting a second execution
  route.
- Compact input payload dataclasses moved from
  `runtime/jax/inputs/payloads.py` to runtime-neutral
  `runtime/input_payloads.py`. JAX now owns only the materializers that expand
  those payloads into JAX arrays; kernels and input builders import the payload
  contracts from the runtime layer.
- Prepared runtime input summaries now live in `runtime/input_contract.py`.
  The JAX group runner builds one after recording/observer and input lowering,
  validates it against the cable-specific runtime input contract, and records
  primitive benchmark metadata for cable formulation, batch shape, dtype, solver
  policy label, recording/output sink, observer count, and intracellular plus
  extracellular semantic modes before enqueueing the kernel.

Validation:

```bash
python -m compileall -q src/axonscope tests/unit
python -m pytest -q tests/unit/test_architecture_guardrails.py tests/unit/test_inspection.py tests/unit/test_performance.py --tb=short
python -m pytest -q tests/unit/test_dispatcher.py --tb=short
python -m pytest -q tests/unit/solvers/test_common.py --tb=short
python -m pytest -q tests/unit/solvers/test_batch.py tests/unit/solvers/test_cranknicholson.py tests/unit/solvers/test_extracellular.py --tb=short
python benchmark/run.py --script recruitment_curves --preset quick --platform cpu --cable single_cable --recording observer_only --n-axons 64 --nx 89 --precision fp32 --repeats 1 --warmups 1 --memory-trace rss --output benchmark/results/p12b_runtime_jax_prune_single_cpu
python benchmark/run.py --script recruitment_curves --preset quick --platform cpu --cable double_cable --recording observer_only --n-axons 64 --nx 89 --precision fp32 --repeats 1 --warmups 1 --memory-trace rss --output benchmark/results/p12b_runtime_jax_prune_double_cpu
python -m pytest -q tests/unit/test_architecture_guardrails.py tests/unit/test_dispatcher.py tests/unit/solvers/test_batch.py tests/unit/solvers/test_extracellular.py --tb=short
python benchmark/run.py --script recruitment_curves --preset quick --platform cpu --cable single_cable --recording observer_only --n-axons 64 --nx 89 --precision fp32 --repeats 1 --warmups 1 --memory-trace rss --output benchmark/results/p12b_input_planning_single_cpu
python benchmark/run.py --script recruitment_curves --preset quick --platform cpu --cable double_cable --recording observer_only --n-axons 64 --nx 89 --precision fp32 --repeats 1 --warmups 1 --memory-trace rss --output benchmark/results/p12b_input_planning_double_cpu
python -m pytest -q tests/unit/test_architecture_guardrails.py tests/unit/test_dispatcher.py tests/unit/preparation/test_cohort.py tests/unit/solvers/test_batch.py tests/unit/solvers/test_extracellular.py --tb=short
python benchmark/run.py --script recruitment_curves --preset quick --platform cpu --cable single_cable --recording observer_only --n-axons 64 --nx 89 --precision fp32 --repeats 1 --warmups 1 --memory-trace rss --output benchmark/results/p12b_group_preparation_single_cpu
python benchmark/run.py --script recruitment_curves --preset quick --platform cpu --cable double_cable --recording observer_only --n-axons 64 --nx 89 --precision fp32 --repeats 1 --warmups 1 --memory-trace rss --output benchmark/results/p12b_group_preparation_double_cpu
python -m pytest -q tests/unit/test_architecture_guardrails.py tests/unit/test_dispatcher.py tests/unit/solvers/test_batch.py tests/unit/solvers/test_extracellular.py --tb=short
python benchmark/run.py --script recruitment_curves --preset quick --platform cpu --cable single_cable --recording observer_only --n-axons 64 --nx 89 --precision fp32 --repeats 1 --warmups 1 --memory-trace rss --output benchmark/results/p12b_membrane_stacking_single_cpu
python benchmark/run.py --script recruitment_curves --preset quick --platform cpu --cable double_cable --recording observer_only --n-axons 64 --nx 89 --precision fp32 --repeats 1 --warmups 1 --memory-trace rss --output benchmark/results/p12b_membrane_stacking_double_cpu
git diff --check
python -m compileall -q src/axonscope tests/unit
python -m pytest -q tests/unit/test_architecture_guardrails.py --tb=short
python -m pytest -q tests/unit/solvers/test_cranknicholson.py tests/unit/solvers/test_runtime.py --tb=short
python -m pytest -q tests/unit/test_dispatcher.py tests/unit/solvers/test_batch.py tests/unit/solvers/test_extracellular.py --tb=short
python -m vulture src/axonscope/runtime/jax src/axonscope/runtime tests --min-confidence 60
git diff --check
python -m compileall -q src/axonscope tests/unit
python -m pytest -q tests/unit/test_architecture_guardrails.py tests/unit/model_ir/test_model_ir.py tests/unit/model_ir/test_public_membrane_compilation.py tests/unit/solvers/test_runtime.py tests/unit/axons/test_cable_heterogeneous.py tests/unit/solvers/test_cranknicholson.py --tb=short
python -m compileall -q src/axonscope tests/unit
python -m pytest -q tests/unit/test_architecture_guardrails.py --tb=short
python -m pytest -q tests/unit/test_dispatcher.py tests/unit/solvers/test_batch.py --tb=short
git diff --check
```

Results: `compileall` passed, guardrails/inspection/performance passed
`105/105`, dispatcher passed `56/56`, batch/observer runtime tests passed
`38/38`, source-pruning tests passed `128/128`, and moved-reference solver
tests passed `56/56`. The local single-cable and double-cable CPU smoke
benchmarks completed and wrote the two `p12b_runtime_jax_prune_*_cpu`
artifact directories.

For the input-planning extraction pass, `compileall` passed,
architecture guardrails passed `83/83`, dispatcher/batch/extracellular tests
passed `104/104`, and the local CPU smoke benchmarks wrote:

- `benchmark/results/p12b_input_planning_single_cpu`: `curve.simulate`
  `3245.0 ms`, `runtime.prepare` `1579.3 ms`, `inputs.extracellular`
  `22.7 ms`, `kernel.wait` `204.5 ms`.
- `benchmark/results/p12b_input_planning_double_cpu`: `curve.simulate`
  `3706.1 ms`, `runtime.prepare` `1914.0 ms`, `inputs.extracellular`
  `19.5 ms`, `kernel.wait` `211.7 ms`.

For the dispatch-group/prepared-cohort extraction pass, `compileall` passed,
architecture guardrails passed `84/84`, dispatcher plus prepared-cohort tests
passed `57/57`, batch/extracellular tests passed `48/48`, and the local CPU
smoke benchmarks wrote:

- `benchmark/results/p12b_group_preparation_single_cpu`: `curve.simulate`
  `3458.9 ms`, `runtime.prepare` `1754.7 ms`, `inputs.extracellular`
  `20.3 ms`, `kernel.wait` `209.6 ms`.
- `benchmark/results/p12b_group_preparation_double_cpu`: `curve.simulate`
  `3885.8 ms`, `runtime.prepare` `2106.2 ms`, `inputs.extracellular`
  `20.1 ms`, `kernel.wait` `228.1 ms`.

For the JAX membrane-stacking extraction pass, `compileall` passed,
architecture guardrails passed `85/85`, dispatcher tests passed `56/56`,
batch/extracellular tests passed `48/48`, and the local CPU smoke benchmarks
wrote:

- `benchmark/results/p12b_membrane_stacking_single_cpu`: `curve.simulate`
  `3434.3 ms`, `runtime.prepare` `1653.3 ms`, `inputs.extracellular`
  `18.0 ms`, `kernel.wait` `188.0 ms`.
- `benchmark/results/p12b_membrane_stacking_double_cpu`: `curve.simulate`
  `3880.1 ms`, `runtime.prepare` `2015.6 ms`, `inputs.extracellular`
  `20.3 ms`, `kernel.wait` `222.7 ms`.

For the row-output helper merge, `git diff --check` passed,
`compileall` passed, architecture guardrails passed `85/85`,
single-row batch/runtime tests passed `25/25`, and
dispatcher/batch/extracellular tests passed `104/104`.

For the `vulture`-guided dead-code pass, the actionable runtime/JAX warnings
were removed. The remaining production warnings are retained intentionally for
this pass: jax-triton solver entry points are exercised by benchmark-analysis
scripts, and `OutputPlan.to_batch_options` is part of the runtime-neutral output
contract. Test/archive warnings are outside the runtime cleanup scope.
`git diff --check` and `compileall` passed, architecture guardrails passed
`85/85`, Model IR plus runtime tests passed `81/81`, common/batch/extracellular
solver tests passed `76/76`, and dispatcher plus heterogeneous-cable tests
passed `62/62`.

For the prepared runtime input contract enforcement pass, `compileall` passed,
the targeted contract and scaled-waveform dispatcher tests passed `2/2`, and
architecture guardrails plus dispatcher tests passed `143/143`.

For the JAX subpackage layout pass, `compileall` passed after the move. The
JAX root now has 22 direct Python files; membrane compilation/lowering lives in
`runtime/jax/membranes/`, and one-row simulations use the batch route instead
of `runtime/jax/execution/scalar_runner.py`. `git diff --check` passed,
architecture guardrails passed `85/85`, Model IR/public compilation/runtime
tests passed `83/83`, heterogeneous-cable plus single-row batch tests passed
`14/14`, and dispatcher/batch/extracellular tests passed `104/104`. `vulture`
reported only the same retained benchmark/contract/test-archive warnings as
before this layout pass.

For the direct solver/scalar fallback removal pass, `git diff --check`,
`compileall`, and `vulture --min-confidence 90` passed. API and architecture
tests passed `122/122`, Model IR/runtime/single-row batch tests passed `84/84`,
batch/extracellular/performance tests passed `82/82`, and example guardrails
passed `7/7`.

For the compact input payload contract extraction, `compileall` passed,
architecture guardrails passed `87/87`, dispatcher plus batch tests passed
`87/87`, and `git diff --check` passed. The new guardrail asserts that
`runtime/input_payloads.py` has no JAX imports and that
`runtime/jax/inputs/payloads.py` owns no payload class definitions.

## Benchmark Gate

After this cleanup, run the local CPU sanity gate:

```bash
python benchmark/run.py \
  --script recruitment_curves \
  --preset quick \
  --platform cpu \
  --cable single_cable \
  --recording observer_only \
  --n-axons 64 \
  --nx 89 \
  --precision fp32 \
  --repeats 2 \
  --warmups 1 \
  --memory-trace rss \
  --output benchmark/results/p12b_runtime_cleanup_single_cpu

python benchmark/run.py \
  --script recruitment_curves \
  --preset quick \
  --platform cpu \
  --cable double_cable \
  --recording observer_only \
  --n-axons 64 \
  --nx 89 \
  --precision fp32 \
  --repeats 2 \
  --warmups 1 \
  --memory-trace rss \
  --output benchmark/results/p12b_runtime_cleanup_double_cpu
```

Only run the matching GPU smoke if the cleanup touches a GPU-sensitive path or
if the local gate shows a suspicious regression.

## CPU Gate Result

The local CPU gate was run on 2026-07-12 after the P12B cleanup.

Artifacts:

- `benchmark/results/p12b_runtime_cleanup_single_cpu`
- `benchmark/results/p12b_runtime_cleanup_double_cpu`

Comparison against the P12A CPU gate:

| Cable | Stage | P12A total | P12B total | Delta |
| --- | --- | ---: | ---: | ---: |
| single-cable | `curve.simulate` | 4115.5 ms | 3184.5 ms | -22.6% |
| single-cable | `runtime.prepare` | 2367.4 ms | 1514.1 ms | -36.0% |
| single-cable | `inputs.extracellular` | 23.5 ms | 24.2 ms | +3.0% |
| single-cable | `kernel.enqueue` | 1317.2 ms | 1264.1 ms | -4.0% |
| single-cable | `kernel.wait` | 254.5 ms | 249.1 ms | -2.1% |
| double-cable | `curve.simulate` | 3560.6 ms | 3581.1 ms | +0.6% |
| double-cable | `runtime.prepare` | 1831.9 ms | 1851.6 ms | +1.1% |
| double-cable | `inputs.extracellular` | 26.3 ms | 26.3 ms | +0.3% |
| double-cable | `kernel.enqueue` | 1278.4 ms | 1277.2 ms | -0.1% |
| double-cable | `kernel.wait` | 321.9 ms | 321.1 ms | -0.3% |

The CPU sanity gate shows no obvious regression. The double-cable path is
essentially unchanged, and the shared preparation helper preserved the original
benchmark span names.

## Recording-Contract CPU Gate Result

After moving runtime-neutral recording conversion, padded recording lowering,
row-aware retained Vm indices, and cohort original-index tables to
`runtime/recording.py`, the local CPU sanity gate was repeated on 2026-07-12.

Artifacts:

- `benchmark/results/p12b_runtime_recording_contract_single_cpu`
- `benchmark/results/p12b_runtime_recording_contract_double_cpu`

Comparison against the previous P12B CPU gate:

| Cable | Stage | Previous total | Recording-contract total | Delta |
| --- | --- | ---: | ---: | ---: |
| single-cable | `curve.simulate` | 3184.5 ms | 3293.4 ms | +3.4% |
| single-cable | `runtime.prepare` | 1514.1 ms | 1557.2 ms | +2.9% |
| single-cable | `inputs.extracellular` | 24.2 ms | 24.5 ms | +1.2% |
| single-cable | `observer.plan` | 31.5 ms | 33.0 ms | +4.9% |
| single-cable | `kernel.enqueue` | 1264.1 ms | 1315.1 ms | +4.0% |
| single-cable | `kernel.wait` | 249.1 ms | 257.3 ms | +3.3% |
| single-cable | `kernel.finalize_observer` | 1.4 ms | 1.4 ms | +0.0% |
| double-cable | `curve.simulate` | 3581.1 ms | 3623.1 ms | +1.2% |
| double-cable | `runtime.prepare` | 1851.6 ms | 1858.6 ms | +0.4% |
| double-cable | `inputs.extracellular` | 26.3 ms | 26.9 ms | +2.1% |
| double-cable | `observer.plan` | 20.0 ms | 20.3 ms | +1.4% |
| double-cable | `kernel.enqueue` | 1277.2 ms | 1299.3 ms | +1.7% |
| double-cable | `kernel.wait` | 321.1 ms | 329.9 ms | +2.8% |
| double-cable | `kernel.finalize_observer` | 1.5 ms | 1.5 ms | +0.2% |

This is a smoke gate, not a policy benchmark. The moved recording-contract code
does not show a targeted observer/finalization regression, and the total deltas
are within the expected noise range for these small cold-heavy CPU runs.

## Host-Preparation CPU Gate Result

After moving runtime-neutral host-array preparation to
`runtime/host_preparation.py`, the local CPU sanity gate was repeated on
2026-07-12.

Artifacts:

- `benchmark/results/p12b_host_preparation_contract_single_cpu`
- `benchmark/results/p12b_host_preparation_contract_double_cpu`

Comparison against the recording-contract CPU gate:

| Cable | Stage | Previous total | Host-prep total | Delta |
| --- | --- | ---: | ---: | ---: |
| single-cable | `curve.simulate` | 3293.4 ms | 3186.6 ms | -3.2% |
| single-cable | `runtime.prepare` | 1557.2 ms | 1524.7 ms | -2.1% |
| single-cable | `inputs.extracellular` | 24.5 ms | 23.7 ms | -3.2% |
| single-cable | `observer.plan` | 33.0 ms | 31.6 ms | -4.3% |
| single-cable | `kernel.enqueue` | 1315.1 ms | 1252.3 ms | -4.8% |
| single-cable | `kernel.wait` | 257.3 ms | 251.2 ms | -2.4% |
| single-cable | `kernel.finalize_observer` | 1.4 ms | 1.4 ms | +1.0% |
| double-cable | `curve.simulate` | 3623.1 ms | 3618.7 ms | -0.1% |
| double-cable | `runtime.prepare` | 1858.6 ms | 1872.2 ms | +0.7% |
| double-cable | `inputs.extracellular` | 26.9 ms | 26.0 ms | -3.5% |
| double-cable | `observer.plan` | 20.3 ms | 20.1 ms | -1.0% |
| double-cable | `kernel.enqueue` | 1299.3 ms | 1291.5 ms | -0.6% |
| double-cable | `kernel.wait` | 329.9 ms | 323.8 ms | -1.9% |
| double-cable | `kernel.finalize_observer` | 1.5 ms | 1.5 ms | -2.7% |

This gate shows no local CPU regression from extracting the host-array helpers.
The JAX runtime still owns materialization into JAX runtime containers.

## Prepared-Input Contract CPU Gate Result

After enforcing the prepared runtime input contract in the JAX group runner, the
local CPU sanity gate was repeated on 2026-07-12 with the same quick recruitment
configuration as the recording-contract gate: `Naxons=64`, `Nx=89`, fp32,
observer-only recording, `repeats=2`, `warmups=1`, and RSS tracing.

Artifacts:

- `benchmark/results/p12_current_repeats2_single_cpu`
- `benchmark/results/p12_current_repeats2_double_cpu`

Superseded exploratory artifacts from the same pass used `repeats=1` and should
not be used for apples-to-apples comparison:

- `benchmark/results/p12_current_single_cpu`
- `benchmark/results/p12_current_double_cpu`

Comparison against the recording-contract CPU gate:

| Cable | Stage | Recording-contract total | Current total | Delta |
| --- | --- | ---: | ---: | ---: |
| single-cable | `curve.simulate` | 3293.4 ms | 3476.4 ms | +5.6% |
| single-cable | `runtime.prepare` | 1557.2 ms | 1576.9 ms | +1.3% |
| single-cable | `inputs.extracellular` | 24.5 ms | 26.0 ms | +6.5% |
| single-cable | `kernel.dispatch_jax` | 663.2 ms | 617.2 ms | -6.9% |
| single-cable | `kernel.wait` | 257.3 ms | 242.3 ms | -5.8% |
| single-cable | `results.split_batch` | 3.3 ms | 3.3 ms | +0.1% |
| double-cable | `curve.simulate` | 3623.1 ms | 3760.2 ms | +3.8% |
| double-cable | `runtime.prepare` | 1858.6 ms | 1963.3 ms | +5.6% |
| double-cable | `inputs.extracellular` | 26.9 ms | 29.5 ms | +9.7% |
| double-cable | `kernel.dispatch_jax` | 979.2 ms | 966.7 ms | -1.3% |
| double-cable | `kernel.wait` | 329.9 ms | 331.2 ms | +0.4% |
| double-cable | `results.split_batch` | 3.4 ms | 3.5 ms | +3.3% |

The rerun does not point to solver degradation: single-cable dispatch and wait
improved, and double-cable dispatch/wait are effectively flat. The remaining
P12 optimization target is the cold/preparation side, especially
`runtime.prepare.base_runtime`, membrane init/compile/backend setup, and small
extracellular input preparation overheads. Keep this as a local CPU guardrail;
it does not close the broader P12 performance-loss claim until the relevant
P11 hot-path and GPU slices are rerun.

## Uniform Membrane Init Optimization

The first P12 optimization from the current CPU guardrail targets the
single-cable cold path. Uniform stateless Model IR membranes now build initial
`Vm0`, `gates0`, and background-current arrays through the NumPy interpreter
instead of using the JAX backend for initial gate values. Solver execution still
uses the JAX membrane backend; this only removes an avoidable cold initialization
cost.

Artifacts:

- `benchmark/results/p12_opt_uniform_init_single_cpu`
- `benchmark/results/p12_opt_uniform_init_double_cpu`

Comparison against the prepared-input contract CPU gate:

| Cable | Stage | Before total | After total | Delta |
| --- | --- | ---: | ---: | ---: |
| single-cable | `curve.simulate` | 3476.4 ms | 2901.4 ms | -16.5% |
| single-cable | `runtime.prepare` | 1576.9 ms | 979.5 ms | -37.9% |
| single-cable | `runtime.prepare.base_runtime` | 1570.2 ms | 972.7 ms | -38.0% |
| single-cable | `runtime.prepare.membrane_init` | 708.2 ms | 3.3 ms | -99.5% |
| single-cable | `kernel.dispatch_jax` | 617.2 ms | 694.8 ms | +12.6% |
| single-cable | `kernel.wait` | 242.3 ms | 257.6 ms | +6.3% |
| double-cable | `curve.simulate` | 3760.2 ms | 4001.8 ms | +6.4% |
| double-cable | `runtime.prepare` | 1963.3 ms | 2005.9 ms | +2.2% |
| double-cable | `runtime.prepare.membrane_init` | 943.6 ms | 914.5 ms | -3.1% |
| double-cable | `kernel.dispatch_jax` | 966.7 ms | 1086.7 ms | +12.4% |
| double-cable | `kernel.wait` | 331.2 ms | 346.9 ms | +4.7% |

The single-cable result is the accepted signal: benchmark metadata changed from
`membrane_init_source=backend_jax` to `membrane_init_source=uniform_numpy`, and
the cold membrane initialization span nearly disappears. The double-cable run
continues to report `membrane_init_source=heterogeneous_numpy`, so the modest
double-cable total increase is not attributed to this patch and should be
rechecked in the broader P11/GPU performance pass before making a global claim.

## Heterogeneous Membrane Init Optimization

The next P12 optimization applies the same host-side principle to stateless
heterogeneous Model IR membranes, which covers the current MRG/double-cable
path. Heterogeneous groups still keep their existing JAX backend for solver
execution, but cold `Vm0`, `gates0`, and background-current arrays are now
constructed through `NumpyModelInterpreter` when every group is a
`JaxMembraneProgram`. Non-Model-IR heterogeneous groups keep the previous
fallback path.

Artifact:

- `benchmark/results/p12_opt_heterogeneous_init_double_cpu`

Comparison against the uniform-init double-cable CPU guard:

| Cable | Stage | Before total | After total | Delta |
| --- | --- | ---: | ---: | ---: |
| double-cable | `curve.simulate` | 4001.8 ms | 3115.8 ms | -22.1% |
| double-cable | `runtime.prepare` | 2005.9 ms | 1211.6 ms | -39.6% |
| double-cable | `runtime.prepare.base_runtime` | 2003.5 ms | 1209.0 ms | -39.7% |
| double-cable | `runtime.prepare.membrane_init` | 914.5 ms | 4.6 ms | -99.5% |
| double-cable | `runtime.prepare.membrane_compile` | 195.8 ms | 193.2 ms | -1.3% |
| double-cable | `inputs.extracellular` | 33.4 ms | 26.9 ms | -19.5% |
| double-cable | `kernel.enqueue` | 1471.3 ms | 1363.5 ms | -7.3% |
| double-cable | `kernel.dispatch_jax` | 1086.7 ms | 1015.4 ms | -6.6% |
| double-cable | `kernel.wait` | 346.9 ms | 326.1 ms | -6.0% |

Benchmark metadata now records
`membrane_init_source=heterogeneous_model_ir_numpy`. This is a cold-run
preparation optimization only; it does not change the public runtime boundary
or solver path, and it leaves compile/backend, dispatch/enqueue, and GPU
solver-bound work as the remaining P12 performance targets.

## Host Cable/Extracellular Runtime Optimization

The following P12 pass removes another cold-start cost from the shared
double-cable base runtime. The one-row cable coefficients and double-cable
extracellular absolute arrays now reuse the runtime-neutral NumPy host
preparation helpers before one compact JAX materialization step. This mirrors
the existing parameter-batch stacking path and keeps the public runtime
contract unchanged.

Artifact:

- `benchmark/results/p12_opt_host_double_only_double_cpu_serial`
- `benchmark/results/p12_opt_host_double_only_double_cpu_serial2`

Comparison against the heterogeneous-init double-cable CPU guard:

| Cable | Stage | Before total | After total | Delta |
| --- | --- | ---: | ---: | ---: |
| double-cable | `curve.simulate` | 3115.8 ms | 2826.1-3040.1 ms | -9.3% to -2.4% |
| double-cable | `runtime.prepare` | 1211.6 ms | 341.0-353.9 ms | -71.9% to -70.8% |
| double-cable | `runtime.prepare.base_runtime` | 1209.0 ms | 337.5-351.0 ms | -72.1% to -71.0% |
| double-cable | `runtime.prepare.membrane_compile` | 193.2 ms | 205.2-205.8 ms | +6.2% to +6.5% |
| double-cable | `runtime.prepare.membrane_init` | 4.6 ms | 4.1-5.1 ms | -12.4% to +10.2% |

Benchmark metadata records `cable_runtime_source=numpy` and
`extracellular_runtime_source=numpy`. The remaining large costs in this local
guardrail are now mostly `kernel.enqueue`/`kernel.dispatch_jax`, cold membrane
compile, and kernel state/observer preparation rather than base runtime array
construction. The host cable path is currently limited to the double-cable
runtime (`include_area=True`); single-cable keeps the previous JAX cable
preparation path until a separate benchmark shows that changing it improves
both cold preparation and kernel timing.

## Final P12 GPU Gate Result

The matching Kaggle P100 GPU smoke gate was rerun on commit `aab5384` after the
uniform membrane-initialization optimization. The single-cable run used the
normal GPU environment; the double-cable tiled-Thomas route required
`jax-triton`, so the first double-cable submission without that package failed
before producing a benchmark and was replaced by the `-jt` run below.

Artifacts:

- `benchmark/results/p12_final_gpu_single_aab5384/outputs/benchmark/results/recruitment_curves_gpu_smoke_gpu_20260712_234919`
- `benchmark/results/p12_final_gpu_double_jt_aab5384/outputs/benchmark/results/recruitment_curves_gpu_smoke_gpu_20260712_235200`
- Failed dependency probe:
  `benchmark/results/p12_final_gpu_double_aab5384`
  (`RuntimeError: Python package 'jax-triton' is not installed.`)

All-phase totals compared with the host-preparation Kaggle gate:

| Cable | Stage | Host-prep total | Current total | Delta |
| --- | --- | ---: | ---: | ---: |
| single-cable | `curve.simulate` | 4043.9 ms | 3360.2 ms | -16.9% |
| single-cable | `runtime.prepare` | 1788.6 ms | 1390.7 ms | -22.2% |
| single-cable | `runtime.prepare.membrane_init` | 640.0 ms | 4.2 ms | -99.3% |
| single-cable | `kernel.enqueue` | 1626.7 ms | 1608.1 ms | -1.1% |
| single-cable | `kernel.dispatch_jax` | 820.5 ms | 805.9 ms | -1.8% |
| single-cable | `kernel.wait` | 61.6 ms | 0.3 ms | -99.4% |
| single-cable | `inputs.extracellular` | 106.0 ms | 29.9 ms | -71.8% |
| double-cable | `curve.simulate` | 9369.7 ms | 9632.3 ms | +2.8% |
| double-cable | `runtime.prepare` | 2344.4 ms | 2268.6 ms | -3.2% |
| double-cable | `runtime.prepare.membrane_init` | 963.3 ms | 936.2 ms | -2.8% |
| double-cable | `kernel.enqueue` | 6395.5 ms | 7044.9 ms | +10.2% |
| double-cable | `kernel.dispatch_jax` | 6067.6 ms | 6726.4 ms | +10.9% |
| double-cable | `kernel.wait` | 102.1 ms | 33.6 ms | -67.1% |
| double-cable | `inputs.extracellular` | 119.7 ms | 31.9 ms | -73.3% |

Warm repeat means use only `phase=repeat` and `iteration>0` simulations:

| Cable | Stage | Host-prep mean | Current mean | Delta |
| --- | --- | ---: | ---: | ---: |
| single-cable | `curve.simulate` | 35.226 ms | 27.764 ms | -21.2% |
| single-cable | `runtime.prepare` | 0.044 ms | 0.062 ms | +40.1% |
| single-cable | `inputs.extracellular` | 1.982 ms | 1.998 ms | +0.8% |
| single-cable | `observer.plan` | 0.119 ms | 0.139 ms | +17.2% |
| single-cable | `kernel.enqueue` | 13.683 ms | 16.830 ms | +23.0% |
| single-cable | `kernel.dispatch_jax` | 4.404 ms | 6.393 ms | +45.2% |
| single-cable | `kernel.wait` | 6.888 ms | 0.029 ms | -99.6% |
| double-cable | `curve.simulate` | 38.747 ms | 23.077 ms | -40.4% |
| double-cable | `runtime.prepare` | 0.055 ms | 0.043 ms | -21.1% |
| double-cable | `inputs.extracellular` | 2.749 ms | 2.215 ms | -19.4% |
| double-cable | `observer.plan` | 0.137 ms | 0.118 ms | -14.1% |
| double-cable | `kernel.enqueue` | 11.598 ms | 11.472 ms | -1.1% |
| double-cable | `kernel.dispatch_jax` | 4.772 ms | 4.639 ms | -2.8% |
| double-cable | `kernel.wait` | 12.935 ms | 4.241 ms | -67.2% |

Interpretation:

- The single-cable cold path validates the uniform NumPy initialization
  optimization on GPU: `runtime.prepare.membrane_init` is now about 4 ms and
  records `membrane_init_source=uniform_numpy`. Warm single-cable is still not
  solver-bound: `kernel.wait` is near zero while dispatch/enqueue dominate.
- Double-cable with `jax-triton` is the closer warm solver-bound target in this
  small smoke case. Warm dispatch and wait are the same order of magnitude
  (`kernel.dispatch_jax` about 4.6 ms, `kernel.wait` about 4.2 ms), but the cold
  first compile/enqueue path still dominates all-phase totals.
- The final P12 performance target is therefore two-sided: make warm GPU runs
  increasingly solver-bound by reducing dispatch/enqueue/input/observer overhead,
  and reduce cold-run latency via preparation, membrane/runtime compilation, and
  cache policy improvements.

## Host-Preparation Kaggle Gate Result

The matching Kaggle smoke gate was run on commit `b5d88b2` with `Naxons=1024`,
`Nx=89`, fp32, observer-only recording, `repeats=2`, `warmups=1`, and RSS memory
tracing. CPU and GPU runs used the same Kaggle P100 environment; CPU runs forced
the benchmark `--platform cpu`, while GPU runs used `--platform gpu`.

Artifacts:

- `benchmark/results/kaggle/20260712_132342_recruitment_curves_quick_cpu_NvidiaTeslaP100_axs-p12b-hostprep-single-cpu-b5d88b2`
- `benchmark/results/kaggle/20260712_132359_recruitment_curves_quick_cpu_NvidiaTeslaP100_axs-p12b-hostprep-double-cpu-b5d88b2`
- `benchmark/results/kaggle/20260712_132637_recruitment_curves_gpu_smoke_gpu_NvidiaTeslaP100_axs-p12b-hostprep-single-gpu-r2-b5d88b2`
- `benchmark/results/kaggle/20260712_132649_recruitment_curves_gpu_smoke_gpu_NvidiaTeslaP100_axs-p12b-hostprep-double-gpu-jt-r2-b5d88b2`

Key phase totals:

| Platform | Cable | Solver | `curve.simulate` | `runtime.prepare` | `inputs.extracellular` | `kernel.enqueue` | `kernel.dispatch_jax` | `kernel.wait` |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | single-cable | auto | 7896.6 ms | 1246.6 ms | 107.3 ms | 1398.3 ms | 775.9 ms | 4695.0 ms |
| CPU | double-cable | thomas | 7704.9 ms | 1784.9 ms | 117.5 ms | 1605.4 ms | 1222.4 ms | 3783.6 ms |
| GPU | single-cable | auto | 4043.9 ms | 1788.6 ms | 106.0 ms | 1626.7 ms | 820.5 ms | 61.6 ms |
| GPU | double-cable | tiled-thomas b64 | 9369.7 ms | 2344.4 ms | 119.7 ms | 6395.5 ms | 6067.6 ms | 102.1 ms |

This is still a smoke/non-regression gate, not a solver policy benchmark. It
validates that the host-preparation extraction runs on CPU and GPU with both
cable formulations. The CPU runs remain mostly solver-wait dominated at this
scale. The GPU runs are not solver-wait dominated: the remaining cost is mostly
runtime preparation and JAX/Triton dispatch/enqueue plumbing, especially for the
double-cable tiled-thomas path.

## Earlier GPU Smoke Gate Result

Because the shared preparation helper also touches the GPU execution path, two
small Kaggle P100 smoke runs were launched at commit `deb6954`. This predates
the host-preparation extraction gate above.

Artifacts:

- `benchmark/results/kaggle/20260712_115203_recruitment_curves_gpu_smoke_gpu_NvidiaTeslaP100_axs-p12b-single-gpu-deb6954`
- `benchmark/results/kaggle/20260712_115217_recruitment_curves_gpu_smoke_gpu_NvidiaTeslaP100_axs-p12b-double-gpu-jt-deb6954`

Comparison artifacts:

- P12A single-cable GPU smoke:
  `benchmark/results/kaggle/20260712_111722_recruitment_curves_gpu_smoke_gpu_NvidiaTeslaP100_axonscope-p12a-runtime-contract-single-gpu-6e9a0f5`
- P12A double-cable GPU smoke with `jax-triton`:
  `benchmark/results/kaggle/20260712_112604_recruitment_curves_gpu_smoke_gpu_NvidiaTeslaP100_axs-p12a-double-gpu-jt-6e9a0f5`

All-phase totals from `summary.csv`:

| Cable | Stage | P12A total | P12B total | Delta |
| --- | --- | ---: | ---: | ---: |
| single-cable | `curve.simulate` | 4074.7 ms | 4476.4 ms | +9.9% |
| single-cable | `runtime.prepare` | 1843.7 ms | 1968.7 ms | +6.8% |
| single-cable | `inputs.extracellular` | 108.9 ms | 136.1 ms | +25.0% |
| single-cable | `kernel.enqueue` | 1592.7 ms | 1790.5 ms | +12.4% |
| single-cable | `kernel.wait` | 61.8 ms | 55.7 ms | -9.7% |
| double-cable | `curve.simulate` | 9098.9 ms | 9064.4 ms | -0.4% |
| double-cable | `runtime.prepare` | 2340.5 ms | 2283.8 ms | -2.4% |
| double-cable | `inputs.extracellular` | 118.6 ms | 114.9 ms | -3.1% |
| double-cable | `kernel.enqueue` | 6144.6 ms | 6167.4 ms | +0.4% |
| double-cable | `kernel.wait` | 103.8 ms | 103.0 ms | -0.8% |

Steady repeat means use only `phase=repeat` and `iteration>0` simulations:

| Cable | Stage | P12A mean | P12B mean | Delta |
| --- | --- | ---: | ---: | ---: |
| single-cable | `curve.simulate` | 36.716 ms | 39.198 ms | +6.8% |
| single-cable | `runtime.prepare` | 0.056 ms | 0.051 ms | -8.9% |
| single-cable | `inputs.extracellular` | 2.047 ms | 2.291 ms | +11.9% |
| single-cable | `kernel.enqueue` | 14.185 ms | 15.555 ms | +9.7% |
| single-cable | `kernel.wait` | 6.908 ms | 6.655 ms | -3.7% |
| double-cable | `curve.simulate` | 37.578 ms | 38.232 ms | +1.7% |
| double-cable | `runtime.prepare` | 0.060 ms | 0.048 ms | -20.9% |
| double-cable | `inputs.extracellular` | 2.520 ms | 2.606 ms | +3.4% |
| double-cable | `kernel.enqueue` | 11.363 ms | 11.870 ms | +4.5% |
| double-cable | `kernel.wait` | 13.048 ms | 13.024 ms | -0.2% |

Interpretation:

- The double-cable GPU smoke is effectively unchanged and passes the P12B
  cleanup gate.
- The single-cable GPU smoke is still runnable, but this small case shows a
  modest enqueue/dispatch-side increase. Since `kernel.wait` did not worsen,
  this is not solver degradation, but it should be watched in the broader P11
  hot-path slices before claiming no GPU performance loss.
- The current cleanup therefore remains acceptable as a runtime-boundary
  cleanup, but it does not close the broader P12 performance-loss claim.

## Remaining Cleanup

- Continue auditing `runtime/jax/` for dead or duplicated host-side code.
- Keep test-only dense/reference routes under `tests/`, not in
  `runtime/jax`. Do not promote those dense/reference routes into public
  examples or stable runtime policy.
- Keep diagnostic solver routes out of public examples and stable docs.
- Do not choose a new solver policy in P12B.
- Do not optimize cold start until the runtime contract and hot path are stable.
