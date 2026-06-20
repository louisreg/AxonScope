# AxonScope Dispatch Scheduling Note: Larger Buckets and Optional Async Group Execution

## Purpose

This note proposes improvements to AxonScope's current pool dispatch strategy so GPU workloads are better saturated when simulating many axons, especially for the target regime:

```text
many fibers
small-to-medium Nx
multiple compatible or semi-compatible models
GPU execution
```

The key idea is **not** to rely primarily on launching many small GPU jobs concurrently. The preferred strategy is:

```text
1. Build larger compatible batches.
2. Bucket semi-compatible models into static padded shapes.
3. Launch fewer, larger JAX calls.
4. Optionally enqueue multiple independent groups asynchronously.
5. Synchronize once at the end or at controlled memory-pressure points.
```

This is a scheduling / dispatch improvement. It does not replace the double-cable solver roadmap. It should complement PCR, split-iterative, associative scan, or Pallas backends.

---

## Current behavior summary

AxonScope already has useful batching infrastructure.

The current dispatcher:

```text
simulate_pool(...)
  -> run_pool(...)
  -> build_dispatch_plan(...)
  -> for each DispatchGroup:
       run batch group if possible
       otherwise scalar group
       wait for the group kernel
       split results
```

The important point is that AxonScope already groups compatible axons and executes each group through a batched kernel. But the current execution loop waits for each group before moving to the next group.

In `dispatcher/execution.py`, `_run_pool_checked(...)` iterates over `plan.groups` and calls `_run_batch_group(...)` or `_run_scalar_group(...)` group by group. Each batch group eventually records `kernel.enqueue`, then calls `benchmark_wait(out.Vm)` inside `kernel.wait`, then splits the batch output into per-axon results.

That means the current dispatch model is essentially:

```python
for group in plan.groups:
    out = run_group(group)
    wait(out)
    results.extend(split(out))
```

rather than:

```python
pending = []
for group in plan.groups:
    pending.append(enqueue_group(group))

wait_all(pending)
results = split_all(pending)
```

This matters because JAX dispatch to accelerators is asynchronous: Python can enqueue computations and continue before device execution is complete, unless a result is inspected or `.block_until_ready()` is called. However, asynchronous enqueueing does not guarantee true simultaneous kernel execution; the GPU may still serialize work internally. Therefore, async group execution is a throughput optimization to test, not a substitute for larger batches.

---

## Main recommendation

Implement a two-layer strategy:

```text
Layer 1: Better batching
    Group more axons into fewer, larger JAX calls using static bucket keys.

Layer 2: Optional async group scheduling
    For remaining independent groups, enqueue multiple group kernels before waiting.
```

The expected impact is:

```text
largest gain:
    fewer, larger calls from bucketed batching

secondary gain:
    less host/device idle time from enqueue-all / wait-later scheduling
```

---

# Phase A — Add dispatch bucket keys

## Goal

Make grouping more GPU-oriented by grouping axons by a **compilable execution signature** instead of only exact model compatibility.

A bucket key should represent everything that affects JIT compilation and kernel shape.

## Proposed bucket key

Add a structure such as:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ExecutionBucketKey:
    mode: str                  # "single" or "double"
    solver: str                # "single", "thomas", "pcr_soa", "split_iterative", ...
    nx_bucket: int             # exact Nx or padded bucket: 32, 64, 128, ...
    dtype: str                 # "float32" or "float64"
    recording_mode: str        # "full", "center", "observer", "none"
    iinj_kind: str             # "none", "dense_zero", "nonzero"
    vext_kind: str             # "dense", "factorized", "none"
    membrane_signature: str    # exact or row-indexed membrane compatibility
    geometry_kind: str         # "shared", "parameter_batched", "padded"
```

The first implementation can be smaller:

```python
@dataclass(frozen=True)
class ExecutionBucketKey:
    mode: str
    nx_bucket: int
    dtype: str
    recording_mode: str
    geometry_kind: str
```

Then expand only as needed.

## Why this helps

The GPU usually prefers:

```text
one group of B = 2048
```

over:

```text
four groups of B = 512
```

especially for small `Nx`.

For double-cable, the solver may already be under-occupying the GPU, so increasing effective batch size can matter a lot.

---

# Phase B — Add Nx bucketing and padding

## Goal

Group axons whose `Nx` differs slightly by padding to a common bucket.

Suggested buckets:

```text
Nx <= 32   -> 32
Nx <= 64   -> 64
Nx <= 128  -> 128
Nx <= 256  -> 256, only if needed
```

For the target regime:

```text
Nx = 30-100
```

this means nearly all cases land in:

```text
32, 64, 128
```

## Padding rules

For cable arrays:

```text
diag padding:
    identity / stable no-op row

lower/upper padding:
    zero coupling into padded compartments

state padding:
    edge or rest value depending on variable

output:
    slice back to original Nx during result unpacking
```

For full trace recording, padded rows may need full padded output internally and slicing afterward.

For observer-only recording, observers should ignore padded compartments.

## Existing AxonScope relevance

AxonScope already has padding helpers in the dispatch path for some parameter-batched cases, including space-array padding and edge-array padding. The improvement here is to make padding/bucketing a first-class GPU scheduling policy rather than a special case.

---

# Phase C — Prefer concat-by-bucket over concurrent small calls

## Goal

If multiple groups have the same `ExecutionBucketKey`, merge them into one larger executable batch.

## Proposed execution shape

Instead of:

```text
group A, B=256, Nx=64
group B, B=512, Nx=64
group C, B=128, Nx=64
```

run:

```text
merged bucket, B=896, Nx=64
```

## Implementation sketch

Add a planning stage after the current dispatch plan:

```python
def coalesce_dispatch_groups(plan: DispatchPlan, options: BatchSchedulingOptions):
    buckets: dict[ExecutionBucketKey, list[DispatchGroup]] = {}

    for group in plan.groups:
        key = execution_bucket_key(group, options)
        buckets.setdefault(key, []).append(group)

    return [
        merge_groups_if_safe(key, groups)
        for key, groups in buckets.items()
    ]
```

The first version does not need to physically merge all metadata deeply. It can construct a **super-group** that contains:

```text
original groups
row mapping
target Nx bucket
shared execution runtime if possible
per-row runtime if needed
```

## Safety rules

Only merge groups if:

```text
same mode
same solver backend
same dtype
same recording mode
same Nx bucket
compatible membrane runtime path
compatible geometry path
compatible stimulation representation
```

Do not merge if:

```text
one group requires scalar fallback
one group has unsupported stateful membrane batching
one group needs full traces and another observer-only
one group has nonzero Iinj and another has no-Iinj specialization
```

Start conservative. Expand merging rules after correctness tests.

---

# Phase D — Optional async group scheduling

## Goal

For groups that still cannot be merged, optionally enqueue multiple group kernels before waiting.

This exploits JAX's asynchronous dispatch behavior.

## Current pattern to avoid in async mode

```python
for group in groups:
    out = run_group_kernel(group)
    jax.block_until_ready(out)
    split_results(out)
```

## Proposed async pattern

```python
pending = []

for group in groups:
    prepared = prepare_group_inputs(group)

    out = run_group_kernel(prepared)  # enqueue only
    pending.append(PendingGroup(group=group, out=out, prepared=prepared))

    if memory_pressure_too_high(pending):
        wait_and_split_some(pending)

wait_and_split_all(pending)
```

## Important warning

This does **not** guarantee that kernels run simultaneously. It only avoids unnecessary host-side synchronization between groups. The GPU may still execute kernels sequentially.

This is still useful if the old path introduced bubbles like:

```text
enqueue group 0
wait group 0
split group 0
prepare group 1
enqueue group 1
wait group 1
...
```

The async path allows:

```text
prepare group 0
enqueue group 0
prepare group 1 while group 0 may execute
enqueue group 1
...
wait once
```

## Memory pressure

Async scheduling keeps inputs and outputs alive for multiple groups. Add a memory budget.

```python
@dataclass(frozen=True)
class BatchSchedulingOptions:
    coalesce_groups: bool = True
    async_groups: bool = False
    max_pending_groups: int = 4
    max_pending_output_bytes: int | None = None
    nx_buckets: tuple[int, ...] = (32, 64, 128)
    prefer_bucket_padding: bool = True
```

If full `Vm[B, Nt, Nx]` is recorded, do **not** enqueue too many groups. Full traces can dominate memory.

For observer-only / compact outputs, async scheduling is much safer.

---

# Phase E — Add `PendingGroup` abstraction

## Goal

Separate enqueueing from waiting and result splitting.

## Proposed dataclass

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class PendingGroup:
    group: DispatchGroup
    output: Any
    t: Any
    method: str
    batch_options: BatchOptions
    kernel_batch_options: BatchOptions
    record_indices: tuple[int, ...] | None
    estimated_output_bytes: int
```

For single and double cable batch groups, `_run_single_cable_batch_group` and `_run_double_cable_batch_group` can be split into:

```python
_prepare_single_cable_batch_group(...)
_enqueue_single_cable_batch_group(...)
_finalize_single_cable_batch_group(...)

_prepare_double_cable_batch_group(...)
_enqueue_double_cable_batch_group(...)
_finalize_double_cable_batch_group(...)
```

The current functions can remain as synchronous wrappers:

```python
def _run_single_cable_batch_group(...):
    pending = _enqueue_single_cable_batch_group(...)
    _wait_pending_group(pending)
    return _finalize_pending_group(pending)
```

This minimizes API disruption.

---

# Phase F — Refactor execution path

## Current synchronous path

```python
def _run_pool_checked(...):
    results = [None] * len(plan.items)

    for group in plan.groups:
        if _can_run_batch_group(group):
            group_results = _run_batch_group(...)
        else:
            group_results = _run_scalar_group(...)

        for result in group_results:
            results[result.index] = result

    return tuple(results)
```

## Proposed path

```python
def _run_pool_checked(...):
    options = resolve_batch_scheduling_options(...)
    plan = build_dispatch_plan(axons)

    if options.coalesce_groups:
        groups = coalesce_dispatch_groups(plan.groups, options)
    else:
        groups = plan.groups

    if options.async_groups:
        return _run_pool_async_groups(groups, ...)
    else:
        return _run_pool_sync_groups(groups, ...)
```

## Async implementation sketch

```python
def _run_pool_async_groups(groups, ...):
    results = [None] * len(plan.items)
    pending: list[PendingGroup] = []

    for group in groups:
        if not _can_run_batch_group(group):
            flush_pending(pending, results)
            group_results = _run_scalar_group(group, ...)
            store_results(results, group_results)
            continue

        pending_group = _enqueue_batch_group(group, ...)
        pending.append(pending_group)

        if should_flush_pending(pending):
            flush_pending(pending, results)

    flush_pending(pending, results)
    return tuple(result for result in results if result is not None)
```

`flush_pending`:

```python
def flush_pending(pending, results):
    if not pending:
        return

    # Wait once for all pending outputs.
    jax.block_until_ready([p.output.Vm for p in pending])

    for p in pending:
        group_results = _finalize_pending_group(p)
        store_results(results, group_results)

    pending.clear()
```

---

# Phase G — Benchmark the scheduler separately

## Goal

Prove that the scheduler improves throughput independent of solver changes.

## Add benchmark

Create:

```text
benchmark/dispatcher/bench_group_scheduling.py
```

Test modes:

```text
sync_current
async_groups
coalesce_buckets
coalesce_buckets_async
```

Test inputs:

```text
many small compatible groups
many semi-compatible Nx groups
mixed single/double groups
full output vs observer output
small B per group vs large B per group
```

Matrix:

```text
groups:       4, 8, 16, 32
B_per_group: 64, 128, 256, 512
Nx:          32, 51, 64, 96, 100
recording:   full, center, observer/none
mode:        single, double
```

Metrics:

```text
total wall time
kernel wait time
input preparation time
result splitting time
peak output bytes
compile count
number of JIT calls
effective B per JIT call
```

## Success criteria

Keep coalescing if:

```text
coalesce_buckets improves total wall time by >20%
or reduces JIT call count substantially without memory regressions
```

Keep async scheduling if:

```text
async_groups improves total wall time by >10%
and peak memory remains acceptable
```

Do not enable async by default until it is stable.

---

# Phase H — Instrumentation and profiling

## Required metadata

Add benchmark metadata for:

```text
dispatch_group_count
bucket_count
coalesced_group_count
original_group_count
async_pending_max
async_flush_count
jit_call_count
effective_batch_size_per_call
nx_bucket_distribution
estimated_output_bytes_per_group
```

## JAX profiling

Add `TraceAnnotation` around:

```text
prepare_group_inputs
kernel_enqueue
kernel_wait
result_split
async_flush
coalesce_groups
```

Example:

```python
with jax.profiler.TraceAnnotation("dispatch.async_flush"):
    jax.block_until_ready([p.output.Vm for p in pending])
```

---

# Phase I — Interaction with solver backends

The scheduler should be solver-aware.

For double-cable:

```text
THOMAS:
    coalescing helps because B is the main parallel axis

PCR_SOA:
    coalescing helps, but PCR also exposes Nx parallelism

SPLIT_ITERATIVE:
    coalescing helps a lot because it turns double-cable into many scalar tridiagonal solves

ASSOCIATIVE_SCAN:
    coalescing helps if prefix scans are batched over B

PALLAS:
    coalescing helps because Pallas kernels can be designed around fixed B/Nx buckets
```

The scheduler should not assume one solver. Include solver in the bucket key.

---

# Phase J — Multi-device future

If multiple GPUs or TPUs are available, shard by batch axis `B`.

Preferred:

```text
device 0: subset of fibers
device 1: subset of fibers
device 2: subset of fibers
...
```

Avoid sharding over `Nx` or `Nt` initially.

Future APIs:

```text
shard_map
pjit / named sharding
```

But do not add multi-device complexity before single-device coalescing and async scheduling are validated.

---

# Implementation order

## Step 1 — add options

Add:

```python
@dataclass(frozen=True)
class BatchSchedulingOptions:
    coalesce_groups: bool = True
    async_groups: bool = False
    max_pending_groups: int = 4
    max_pending_output_bytes: int | None = None
    nx_buckets: tuple[int, ...] = (32, 64, 128)
    prefer_bucket_padding: bool = True
```

Expose through:

```python
simulate_pool(..., batch_scheduling_options=...)
```

or integrate into existing `BatchOptions` if that is the preferred API.

## Step 2 — split enqueue/finalize

Refactor batch execution functions so they can enqueue without waiting.

Keep synchronous behavior identical by default.

## Step 3 — implement async mode only

Before coalescing, test:

```text
current groups
async enqueue
wait at end
```

This isolates the benefit of removing per-group waits.

## Step 4 — implement bucket coalescing

Add conservative coalescing for groups with exactly matching shapes first.

Then add Nx padding buckets.

## Step 5 — add scheduler benchmark

Benchmark:

```text
sync current
async only
coalesce only
coalesce + async
```

## Step 6 — update AUTO policy

If benchmarks show consistent benefit, set:

```text
coalesce_groups = True by default
async_groups = False by default initially
```

Then consider enabling async automatically only for compact-output modes.

---

# Default policy recommendation

Initial defaults:

```python
BatchSchedulingOptions(
    coalesce_groups=True,
    async_groups=False,
    max_pending_groups=4,
    max_pending_output_bytes=None,
    nx_buckets=(32, 64, 128),
    prefer_bucket_padding=True,
)
```

Reason:

```text
coalescing is likely to help and is easier to reason about
async scheduling can increase memory and complicate debugging
```

Possible later default:

```text
async_groups=True only when:
    output is compact
    no scalar fallback groups are interleaved
    estimated pending output bytes is below budget
    benchmarking confirms benefit
```

---

# Risks

## Risk 1 — memory explosion

Async groups keep multiple outputs alive. This is especially dangerous with:

```text
Vm[B, Nt, Nx]
```

Mitigation:

```text
limit pending groups
estimate output bytes
flush when budget exceeded
prefer observer-only outputs
```

## Risk 2 — no true GPU concurrency

JAX async dispatch may enqueue multiple calls, but the GPU may execute them sequentially.

Mitigation:

```text
treat async as optional
benchmark carefully
do not rely on it for core speedups
prioritize coalescing
```

## Risk 3 — recompilation due to too many bucket keys

If bucket keys include too many varying fields, the scheduler may create many compiled variants.

Mitigation:

```text
use Nx buckets
use persistent compilation cache
log compile counts
keep bucket key minimal
```

## Risk 4 — padded computation overhead

Padding from Nx=65 to Nx=128 may double work.

Mitigation:

```text
benchmark bucket boundaries
consider buckets 32/64/96/128 if needed
do not overpad very small groups
```

## Risk 5 — result ordering bugs

Coalesced groups must preserve original result indices.

Mitigation:

```text
explicit row mapping
unit tests with shuffled inputs
test mixed group ordering
```

---

# Required tests

Add tests for:

```text
1. Synchronous path unchanged by default.
2. Async path returns same results as sync.
3. Coalesced path returns same results as non-coalesced.
4. Output order matches input order.
5. Padded rows are sliced correctly.
6. Full recording and compact recording both work.
7. Scalar fallback groups flush pending GPU work safely.
8. Errors in one pending group are surfaced clearly.
9. Progress reporting remains correct or is explicitly disabled in async mode.
```

---

# Expected impact

## Best case

```text
large number of small groups
compact outputs
same Nx bucket
GPU underoccupied

Expected:
    1.2x-2x end-to-end throughput improvement
```

## Moderate case

```text
few large groups already
GPU reasonably saturated

Expected:
    0-20% improvement
```

## Bad case

```text
full Vm outputs dominate memory
groups have incompatible shapes
host-side input preparation dominates
GPU already saturated by one group

Expected:
    little or no improvement
```

---

# Bottom line

The current AxonScope dispatcher already performs important batching, but it is still group-synchronous.

The most valuable next step is not to force many GPU calls to run simultaneously. It is to:

```text
1. coalesce more work into larger static buckets,
2. reduce the number of JIT calls,
3. avoid waiting after every group,
4. optionally enqueue multiple independent groups before waiting,
5. keep memory under control.
```

This scheduler work should be developed alongside the double-cable solver roadmap. It will help all solver backends, but it will be especially useful for double-cable workloads where small `Nx` and moderate `B` can leave the GPU underoccupied.

---

# Sources

## AxonScope

- `dispatcher/execution.py`
  - Current `run_pool` implementation iterates over dispatch groups, launches one group at a time, waits on `out.Vm`, and then splits batch results.
  - https://raw.githubusercontent.com/louisreg/AxonScope/main/src/axonscope/dispatcher/execution.py

- `dispatcher/plan.py`
  - Current dispatch planning and grouping logic.
  - https://raw.githubusercontent.com/louisreg/AxonScope/main/src/axonscope/dispatcher/plan.py

- `solvers/batch_kernels.py`
  - Current JAX batch kernels for single- and double-cable workloads.
  - https://raw.githubusercontent.com/louisreg/AxonScope/main/src/axonscope/solvers/batch_kernels.py

## JAX

- Asynchronous dispatch
  - JAX can enqueue accelerator work asynchronously; Python may continue before device execution completes until synchronization is forced.
  - https://docs.jax.dev/en/latest/async_dispatch.html

- Benchmarking JAX code
  - Timings must separate compilation/runtime and use `.block_until_ready()` to measure accelerator execution.
  - https://docs.jax.dev/en/latest/benchmarking.html
