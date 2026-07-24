# P20 Runnable Plan And Runner Audit

Status: active migration audit, 2026-07-23.

## Existing Ownership

The public workflow entered `AxonSimulation.run()`, which owned a private
dispatch-plan cache and the simulation setup/result lifecycle. It then called
`dispatcher.run_pool()`, which could independently build the same dispatch
plan. Concrete group preparation and execution continued through
`runtime.execution` into the JAX group runner.

Numeric-axis protocols reused an `AxonSimulation` instance and called its
private `_run_numeric_axis()` method. That method expanded the private
dispatch plan before returning to the same population execution function.
Inspection and estimation independently rebuilt host planning descriptions.

## Problems To Remove

- Planning state had two owners: `AxonSimulation` and the global dispatcher
  plan cache.
- Public `simulation.py` owned execution-context entry, dispatcher invocation,
  recording filtering, and public result assembly.
- A protocol composed work by invoking a private execution method instead of
  producing a composed plan.
- There was no object that could execute several plans while intentionally
  sharing planning and future scheduling state.

## Convergence Direction

`RunnablePlan` is the backend-neutral immutable envelope. `SimulationPlan`
describes one population run and `NumericAxisPlan` composes a typed dynamic
axis over it. These plans contain no dispatch groups, solver axons, NumPy/JAX
runtime arrays, device placement, compiled functions, or result buffers.

`Runner` owns dispatch materialization, reusable dispatch state, execution
context, runtime invocation, and canonical result assembly. The existing
`DispatchPlan` remains a runner-internal lowering contract during migration;
it is not a second public plan type. `AxonSimulation.run()` remains the
canonical convenience interface and delegates its generated plan to a runner.

The second migration slice adds generic `SweepPlan`. Public pool and
recruitment helpers now only build this plan; typed
numeric-axis preparation, value chunking, per-value execution, progress, and
generic observation assembly happen inside `Runner`. A generic plan-level
result factory converts recruitment observations to the canonical
`RecruitmentCurve` before the runner returns. The private
`AxonSimulation._run_numeric_axis()` route and the protocol-specific numeric
sweep plan were removed.

The third migration slice adds `ThresholdPlan`. The protocol describes the
source simulation, row labels, bounds, stopping policy, update, and decoder;
`Runner` resolves callable bounds lazily and owns per-row bisection, one-shot
solver progress, execution, and canonical `ThresholdCurve` assembly.

The fourth migration slice moves synchronous and bounded asynchronous group
scheduling into `Runner`. The duplicate `dispatcher/execution.py` facade and
its raw `run_pool()` entry point were removed; backend group execution remains
behind `runtime.execution`.

The fifth migration slice adds `PopulationPlan`. `AxonSimulation` now freezes
only its ordered descriptive inputs; `Runner` materializes and caches the
canonical `AxonPopulation` when executing, estimating, inspecting, or serving
an explicit population access. Sweep and threshold composition reuse the same
runner-owned population instead of normalizing rows before execution.

The sixth migration slice replaces tuple execution with a generic named
dependency composition. `Runner` validates and executes a stable
topological order, emits one benchmark span per task, preserves completed
results in structured fail-fast errors, and supports cooperative cancellation
between tasks and protocol iterations without interrupting an in-flight kernel.

The seventh migration slice extends `Runner.estimate()` and `Runner.inspect()`
to numeric axes, sweeps, thresholds, and composed studies. Plain simulations
retain their existing report types. Composed reports separate peak memory for
one currently scheduled execution from repeated simulation-work bounds:
numeric axes are one compact execution, sweeps count value batches, thresholds
report two bound evaluations plus up to `max_iterations`, and sequential graph
peak is the largest task peak rather than the sum of all task estimates.

The eighth migration slice gives that composition its canonical public
vocabulary: `StudyTask`, `StudyPlan`, and `StudyResult`. It replaces the
provisional graph-named API instead of wrapping it. A study is a named DAG of
existing leaf plans and executes through the same runner scheduler; retention,
serialization, result persistence, and result-dependent task generation remain
separate future contracts.

The ninth migration slice re-audits execution ownership after study
composition. It removes the unused observer-path functions that still executed
an `AxonSimulation` directly. Public protocol functions produce leaf plans;
`Runner` is their sole executor. `AxonSimulation.run()` is the one retained
single-simulation convenience and delegates to its runner. Explicit
estimate/inspection builders remain the only intentional host-planning paths
outside runtime execution.

The tenth migration slice replaces the process-global prepared-cohort caches
with one bounded `PreparedCohortCache` per `Runner`. The runner injects that
cache through `runtime.execution`; JAX group enqueue rejects valid cable groups
when it is absent, so there is no hidden global fallback. `Runner.clear()` now
drops population, dispatch, and prepared-cohort state together. Immutable
compiled membrane, JAX executable, and persistent Triton/XLA caches remain
runtime-owned and shareable across runners.

The eleventh migration slice removes eager membrane compilation from the
descriptive object graph. `Model` construction no longer normalizes source
parameters, `Composite` retains its original Python components, and
`SectionLayout` plus `Section` perform structural validation only.
`FlattenedLayout` carries those descriptions without resolving them, so geometry
queries and plots cannot trigger generated-code loading. `build_solver_axon()`
resolves each distinct description once inside Runner-owned dispatch
preparation. Invalid compiler-level parameters or units now fail at
`run()`, `estimate()`, `inspect()`, or explicit membrane introspection.

Explicit estimation and inspection may materialize the population, dispatch
descriptions, host preparation metadata, generated membrane-source metadata,
and typed numeric-axis waveforms needed to describe shapes. They do not enter
an execution context, place runtime arrays on a device, compile kernels,
allocate result buffers, or execute solver work.

The final public-surface slice removes the duplicate executing protocol
functions and their `_plan` counterparts. `pool_sweep()`,
`recruitment_sweep()`, and `find_threshold()` now have one meaning: describe
lazy work. It also removes unused `Runner.run_many()` and
`StudyPlan.from_plans()` shortcuts, keeps technical leaf-plan classes under
`axs.plans`, and leaves only `Runner`, `StudyPlan`, and `StudyTask` at the
package root. Examples now teach direct `AxonSimulation.run()` for one simple
simulation and explicit `plan + Runner` for protocols and composed work.

## Validation Evidence

The acceptance campaign covers simple and mixed populations, native numeric
axes, recruitment and threshold plans, named studies, cold/warm reuse,
`Runner.clear()`, cache invalidation, cancellation, and compact 1024/4096-row
scaling. The retained 2026-07-24 CPU result is
`benchmark/results/p20_runner_validation_cpu_api_convergence_20260724`; the
matching P100 result is under
`benchmark/results/kaggle/20260724_132155_runner_plan_validation_gpu_smoke_gpu_NvidiaTeslaP100_axonfleet-p20-runner-validation`,
and the accepted comparison is
`benchmark/results/p20_runner_validation_cpu_gpu_api_convergence_20260724`.
Numerical acceptance uses `rtol=1e-4` and `0.025 mV`
absolute tolerance per voltage sample, including a shape-scaled tolerance for
reported sums.

## Deferred Work

Add result-dependent study factories only when their lazy callable and
reproducibility contracts are defined; do not add a second scheduler.

Multi-GPU, HPC, remote failure recovery, distributed cache policy, and bounded
cross-worker overlap are explicitly deferred to the future distributed-runner
phase in `todo.md`. That phase starts with a fresh Dask-oriented design review
and requires representative infrastructure before implementation; it must reuse
the canonical plan, Runner, and result vocabulary rather than creating a second
public scheduler.
