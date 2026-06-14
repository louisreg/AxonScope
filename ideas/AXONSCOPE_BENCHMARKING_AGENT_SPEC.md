# AxonScope Built-in Benchmarking and Hotpath Instrumentation

## Implementation Specification for an Autonomous Coding Agent

## 1. Mission

Implement a lightweight, opt-in benchmarking system directly inside AxonScope.

The target user experience is deliberately minimal:

```python
import axonscope as axs

axs.enable_benchmark("benchmarks/run_001")

results = axs.simulate_pool(
    simulations,
    duration_ms=20 * axs.ms,
    dt_ms=0.01 * axs.ms,
    recording=axs.Recording.center("Vm"),
)

axs.disable_benchmark()
```

When benchmarking is enabled, AxonScope must:

1. time the main end-to-end hotpaths;
2. distinguish host-side preparation from JAX device execution;
3. correctly handle asynchronous GPU execution;
4. print a concise hierarchical report;
5. save raw events and aggregate summaries;
6. record backend, device, shapes, dtypes, and estimated array sizes;
7. add negligible overhead when disabled;
8. preserve existing numerical behavior;
9. avoid unnecessary GPU synchronization.

The feature must make it easy to answer questions such as:

- Is dispatch planning expensive?
- Is extracellular preprocessing the dominant cost?
- Is the solver kernel actually faster on GPU?
- Are large `(B, Nt, Nx)` tensors being materialized?
- Is result packaging expensive for large pools?
- Is the first call dominated by JAX compilation?
- Is the workload running on the expected device?
- Does a recording mode select an unexpectedly slow path?

This is primarily a developer and advanced-user diagnostic feature.

---

## 2. Design principles

### 2.1 Opt-in only

Benchmarking must be disabled by default.

When disabled:

- do not create files;
- do not print reports;
- do not start a profiler;
- do not synchronize JAX arrays;
- do not calculate array sizes;
- do not inspect devices;
- do not alter execution order.

The disabled path should cost approximately:

```python
session = _ACTIVE_BENCHMARK_SESSION.get()
if session is None:
    ...
```

### 2.2 Instrument pipeline stages, not every helper

The first version should instrument only high-value boundaries:

```text
simulation.total
simulation.pool.total
dispatch.build_plan
dispatch.group.total
runtime.prepare
inputs.positions
inputs.intracellular
inputs.extracellular
kernel.enqueue
kernel.wait
results.split_batch
results.to_public
```

These stages are sufficient to classify the dominant cost as:

- planning;
- preprocessing;
- input materialization;
- solver execution;
- synchronization;
- postprocessing.

### 2.3 Preserve asynchronous execution

JAX dispatch is asynchronous on accelerators.

A timer around:

```python
output = kernel.run(...)
```

may measure only submission time.

Therefore, kernel measurement must be split:

```python
with benchmark_span("kernel.enqueue"):
    output = kernel.run(...)

with benchmark_span("kernel.wait"):
    output.Vm.block_until_ready()
```

Do not synchronize after every JAX operation.

Synchronize only:

- at the main kernel boundary;
- optionally at the root simulation boundary;
- in an explicit detailed diagnostic mode.

### 2.4 Hierarchical timing

Events must support nesting:

```text
simulation.pool.total
└── dispatch.group.total
    ├── runtime.prepare
    ├── inputs.intracellular
    ├── inputs.extracellular
    ├── kernel.enqueue
    ├── kernel.wait
    └── results.split_batch
```

Reports must distinguish:

- **inclusive time**: complete span duration;
- **self time**: span duration minus direct child durations.

### 2.5 Save raw and aggregated data

Recommended output:

```text
output_dir/
├── events.jsonl
├── summary.csv
├── metadata.json
└── jax-trace/       # optional
```

Use JSONL for raw events so each record is independently parseable.

---

## 3. Public API

Expose the following at top-level package scope.

### 3.1 `enable_benchmark`

Minimal use:

```python
axs.enable_benchmark("benchmarks/run_001")
```

Recommended signature:

```python
def enable_benchmark(
    output_dir: str | Path,
    *,
    print_summary: bool = True,
    save: bool = True,
    reset: bool = True,
    sync_device: bool = True,
    record_shapes: bool = True,
    record_memory: bool = True,
    level: str = "hotpaths",
    jax_trace: bool = False,
) -> BenchmarkSession:
    ...
```

Required behavior:

- create and activate a `BenchmarkSession`;
- create the output directory when saving;
- collect environment metadata;
- optionally clear previous in-memory events;
- return the active session;
- reject incompatible nested sessions.

Suggested future levels:

```text
minimal
hotpaths
detailed
```

The first implementation may support only `hotpaths`.

### 3.2 `disable_benchmark`

Recommended signature:

```python
def disable_benchmark(
    *,
    print_summary: bool | None = None,
    save: bool | None = None,
) -> BenchmarkReport | None:
    ...
```

Required behavior:

- finalize active events;
- aggregate events;
- print a report when configured;
- save files when configured;
- stop optional JAX tracing;
- deactivate the session;
- return a report object.

Calling it with no active session should safely return `None`.

### 3.3 `benchmark_report`

```python
report = axs.benchmark_report()
```

Recommended signature:

```python
def benchmark_report(
    *,
    print_report: bool = True,
    save: bool = False,
) -> BenchmarkReport:
    ...
```

This must aggregate the active session without disabling it.

### 3.4 `reset_benchmark`

```python
axs.reset_benchmark()
```

Clear events, counters, and first-call signatures while preserving configuration.

### 3.5 Context manager

Also provide:

```python
with axs.benchmark("benchmarks/run_001"):
    results = axs.simulate_pool(...)
```

Suggested implementation:

```python
@contextmanager
def benchmark(output_dir: str | Path, **options):
    session = enable_benchmark(output_dir, **options)
    try:
        yield session
    finally:
        disable_benchmark()
```

---

## 4. Module structure

Create:

```text
src/axonscope/benchmarking.py
```

Suggested types:

```text
BenchmarkConfig
BenchmarkSession
BenchmarkEvent
BenchmarkReport
BenchmarkSummaryRow
```

Responsibilities:

```text
benchmarking.py
├── active-session management
├── span context manager
├── metadata normalization
├── array metadata
├── event aggregation
├── console formatting
├── JSONL/CSV/JSON writing
└── optional JAX trace management
```

Keep instrumentation calls lightweight inside existing modules.

---

## 5. Session storage

Use `ContextVar`, not a plain global variable.

```python
from contextvars import ContextVar

_ACTIVE_BENCHMARK_SESSION: ContextVar[BenchmarkSession | None] = ContextVar(
    "axonscope_active_benchmark_session",
    default=None,
)
```

Reasons:

- thread isolation;
- test isolation;
- compatibility with nested execution contexts;
- reduced global-state leakage.

Suggested session structure:

```python
@dataclass
class BenchmarkSession:
    config: BenchmarkConfig
    events: list[BenchmarkEvent]
    stack: list[ActiveSpan]
    metadata: dict[str, Any]
    simulation_counter: int
    event_counter: int
    signatures_seen: set[Hashable]
    active: bool
```

---

## 6. Event model

Recommended event:

```python
@dataclass(frozen=True)
class BenchmarkEvent:
    event_id: int
    run_id: str
    simulation_id: int | None

    name: str
    parent_event_id: int | None
    depth: int

    start_ns: int
    end_ns: int
    duration_ns: int

    metadata: dict[str, Any]

    @property
    def duration_ms(self) -> float:
        return self.duration_ns / 1e6
```

Use unique event IDs for parent relationships.

Do not use event names as parent identifiers because names repeat.

### Metadata fields

Collect when relevant:

```text
backend
device
device_kind
jax_version
batch_size
group_size
group_id
nt
nx
dtype
recording_mode
recorded_columns
shared_geometry
first_call
input_shape
output_shape
input_nbytes
output_nbytes
solver_name
membrane_model
```

All metadata must be JSON serializable.

Normalize:

- `Path` to `str`;
- NumPy scalars to Python scalars;
- dtype to string;
- device to string;
- shapes to integer lists;
- unknown objects to `repr`.

---

## 7. Span API

Implement:

```python
@contextmanager
def benchmark_span(
    name: str,
    **metadata: Any,
) -> Iterator[None]:
    ...
```

Disabled behavior:

```python
session = _ACTIVE_BENCHMARK_SESSION.get()

if session is None:
    yield
    return
```

Enabled behavior:

1. allocate an event ID;
2. find parent from the stack;
3. record `perf_counter_ns()`;
4. push an active span;
5. execute wrapped code;
6. finalize in `finally`;
7. pop the stack;
8. append the event;
9. preserve the original exception.

If an exception occurs, optionally attach:

```text
failed=true
exception_type="..."
```

Never swallow the original exception.

---

## 8. Array metadata helper

Implement:

```python
def record_array_metadata(
    name: str,
    array: Any,
    *,
    role: str | None = None,
) -> None:
    ...
```

Collect without reading values:

```text
shape
dtype
size
itemsize
nbytes
device
sharding
```

Example:

```python
shape = tuple(int(dim) for dim in array.shape)
size = math.prod(shape)
itemsize = int(array.dtype.itemsize)
nbytes = size * itemsize
```

Do not:

- convert to NumPy;
- call `block_until_ready`;
- copy the array;
- inspect contents.

Preferred representation inside event metadata:

```json
{
  "arrays": {
    "vstim_mid": {
      "shape": [500, 2000, 51],
      "dtype": "float32",
      "nbytes": 204000000,
      "device": "cuda:0"
    }
  }
}
```

---

## 9. Required instrumentation points

The agent must inspect the current repository and place spans at the actual current boundaries.

### 9.1 Public simulation

Instrument single simulation:

```text
simulation.total
```

Instrument pool simulation:

```text
simulation.pool.total
```

Metadata:

```text
simulation_count
duration_ms
dt_ms
recording_mode
```

Do not include report formatting or file writing inside the root timer.

### 9.2 Dispatch plan

Instrument:

```text
dispatch.build_plan
```

around plan creation.

Metadata:

```text
simulation_count
group_count
```

### 9.3 Dispatch group

Instrument:

```text
dispatch.group.total
```

around each group execution.

Metadata:

```text
group_id
group_kind
group_size
nx
shared_geometry
recording_mode
```

### 9.4 Runtime preparation

Instrument:

```text
runtime.prepare
```

around solver/runtime preparation.

Metadata:

```text
group_size
nx
dtype
solver_name
membrane_model
```

### 9.5 Position preparation

Instrument:

```text
inputs.positions
```

Record output shapes and estimated memory.

### 9.6 Intracellular input

Instrument:

```text
inputs.intracellular
```

around the complete batch input construction.

Record:

```text
shape
dtype
nbytes
context_count
```

If absence of stimulation is already known from metadata, record:

```text
all_zero=true
```

Do not scan the full array to determine this.

### 9.7 Extracellular input

Instrument:

```text
inputs.extracellular
```

around the complete extracellular potential construction.

This is a high-priority hotpath.

Record:

```text
shape
dtype
nbytes
context_count
electrode_count
factorized
```

For a future factorized path, also record:

```text
footprint_shape
waveform_shape
```

### 9.8 Kernel enqueue

Instrument:

```text
kernel.enqueue
```

only around the call launching or returning the JAX computation.

Do not synchronize inside this span.

Metadata:

```text
backend
device
batch_size
nt
nx
recording_mode
first_call
signature
```

### 9.9 Kernel wait

Instrument:

```text
kernel.wait
```

immediately afterward.

When `sync_device=True`:

```python
primary_output.block_until_ready()
```

or:

```python
jax.block_until_ready(output)
```

Record:

```text
synchronized=true
```

When synchronization is disabled:

```text
synchronized=false
```

and print a warning that GPU timings are incomplete.

### 9.10 Result splitting

Instrument:

```text
results.split_batch
```

around conversion from batched solver output to per-fiber internal results.

Metadata:

```text
result_count
recorded_columns
```

### 9.11 Public results

Instrument:

```text
results.to_public
```

around creation of public `SimResult` objects.

Metadata:

```text
result_count
```

---

## 10. First-call and compilation classification

The first call for a static signature may include tracing and compilation.

Track approximate first-call status with:

```python
signature = (
    backend,
    device_kind,
    dtype,
    batch_size,
    nt,
    nx,
    recording_mode,
    solver_name,
    membrane_model,
    shared_geometry,
)
```

Store signatures in:

```python
session.signatures_seen
```

Before kernel execution:

```python
first_call = signature not in session.signatures_seen
session.signatures_seen.add(signature)
```

Attach:

```text
first_call=true|false
```

Document that this is an approximation.

It indicates that the benchmark session has not seen the signature before; it does not prove that JAX compiled.

---

## 11. Optional JAX trace

When:

```python
jax_trace=True
```

start:

```python
jax.profiler.start_trace(trace_dir)
```

Stop during finalization:

```python
jax.profiler.stop_trace()
```

Use `try/finally`.

Trace directory:

```text
output_dir/jax-trace/
```

Add named regions where useful:

```python
with jax.named_scope("axonscope/membrane_update"):
    ...

with jax.named_scope("axonscope/tridiagonal_solve"):
    ...

with jax.named_scope("axonscope/record_vcenter"):
    ...
```

Do not attempt Python timing of individual `lax.scan` iterations.

---

## 12. Aggregation

Aggregate by:

```text
event name
parent name
backend
device
batch size
Nt
Nx
recording mode
first-call status
```

Recommended summary columns:

```text
run_id
event
parent
count
total_ms
self_ms
mean_ms
median_ms
min_ms
max_ms
p50_ms
p95_ms
percentage_of_root
backend
device
batch_size
nt
nx
dtype
recording_mode
first_call
```

### Self time

For each event:

```text
self_time = duration - sum(direct_child_durations)
```

Subtract only direct children.

Clamp tiny negative values caused by timer precision to zero.

### Percentage

For a single simulation report, calculate percentage relative to:

```text
simulation.pool.total
```

or:

```text
simulation.total
```

---

## 13. Console report

Default report example:

```text
AxonScope benchmark — simulate_pool #1
Backend: gpu | Device: Tesla T4
B=500 | Nt=2000 | Nx=51 | Recording: Vcenter

TOTAL                                      70.675 s  100.0%
├── dispatch.build_plan                     8.214 s   11.6%
├── dispatch.group.total                   61.892 s   87.6%
│   ├── runtime.prepare                     0.183 s    0.3%
│   ├── inputs.intracellular                7.492 s   10.6%
│   │   └── output: [500, 2000, 51], 204 MB
│   ├── inputs.extracellular               50.346 s   71.2%
│   │   └── output: [500, 2000, 51], 204 MB
│   ├── kernel.enqueue                      0.012 s    0.0%
│   ├── kernel.wait                         3.621 s    5.1%
│   └── results.split_batch                 0.238 s    0.3%
└── results.to_public                       0.326 s    0.5%

Hotpath: inputs.extracellular (71.2%)
Saved to: benchmarks/run_001
```

Formatting:

- use seconds for values ≥ 1000 ms;
- otherwise use milliseconds;
- align names and values;
- show percentages to one decimal;
- include large array shapes and human-readable sizes;
- mark first calls;
- limit default depth to 3;
- avoid printing hundreds of rows.

---

## 14. Output formats

### `events.jsonl`

One event per line:

```json
{
  "event_id": 15,
  "run_id": "2026-06-13T14-32-18-4f82",
  "simulation_id": 1,
  "name": "inputs.extracellular",
  "parent_event_id": 10,
  "depth": 2,
  "start_ns": 123456789,
  "end_ns": 173802989,
  "duration_ms": 50346.2,
  "metadata": {
    "backend": "gpu",
    "device": "Tesla T4",
    "batch_size": 500,
    "shape": [500, 2000, 51],
    "dtype": "float32",
    "nbytes": 204000000
  }
}
```

### `summary.csv`

Recommended columns:

```text
run_id
event
parent
count
total_ms
self_ms
mean_ms
median_ms
min_ms
max_ms
p50_ms
p95_ms
percentage
backend
device
batch_size
nt
nx
dtype
recording_mode
first_call
```

### `metadata.json`

Record:

```text
run_id
started_at
finished_at
axonscope_version
axonscope_git_commit
jax_version
jaxlib_version
python_version
platform
backend
devices
jax_enable_x64
benchmark_config
```

Git commit retrieval is best effort and must not fail benchmarking.

---

## 15. Naming rules

Use stable dot-separated event names.

Required:

```text
simulation.total
simulation.pool.total
dispatch.build_plan
dispatch.group.total
runtime.prepare
inputs.positions
inputs.intracellular
inputs.extracellular
kernel.enqueue
kernel.wait
results.split_batch
results.to_public
```

Do not include dynamic IDs in names.

Use:

```text
group_id=2
```

as metadata, not:

```text
dispatch.group.2.total
```

---

## 16. Error handling

Benchmarking must never hide the original AxonScope exception.

Requirements:

- finalize spans in `finally`;
- re-raise original exceptions;
- report-writing failures should warn, not replace simulation results;
- trace-start failures should warn and continue;
- output-directory creation failures should raise immediately;
- non-serializable metadata should fall back to `repr`.

Nested sessions should be rejected:

```text
RuntimeError: An AxonScope benchmark session is already active.
```

---

## 17. Performance constraints

When disabled:

- no synchronization;
- no disk writes;
- no profiler calls;
- no array reads;
- no expensive metadata construction.

When enabled:

- use `perf_counter_ns()`;
- synchronize only at explicit kernel boundaries;
- do not convert arrays to NumPy for metadata;
- keep report formatting and file writing outside root simulation spans.

Preferred sequence:

```text
start simulation root span
run simulation
end root span
format report
write files
```

---

## 18. Implementation order

### Phase 1: framework

Implement:

- configuration;
- event model;
- session model;
- `ContextVar`;
- `benchmark_span`;
- public API;
- simple report;
- JSONL/CSV/metadata output.

### Phase 2: hotpath instrumentation

Instrument:

- public simulation;
- dispatch plan;
- group total;
- runtime preparation;
- intracellular input;
- extracellular input;
- kernel enqueue;
- kernel wait;
- result splitting;
- public result conversion.

### Phase 3: metadata

Add:

- backend/device;
- dimensions;
- recording mode;
- shapes/dtypes/nbytes;
- first-call signatures.

### Phase 4: JAX trace

Add optional profiler trace and selected `jax.named_scope` annotations.

### Phase 5: tests and documentation

Add unit tests, integration tests, examples, and user documentation.

---

## 19. Test plan

### Disabled behavior

Verify:

- spans execute normally;
- no events are stored;
- no files are created;
- exceptions are unchanged.

### Nested events

Create synthetic parent/child spans.

Verify:

- parent IDs;
- depths;
- positive durations;
- self-time calculation.

### Exception behavior

Raise inside a span.

Verify:

- original exception propagates;
- event finalizes;
- session remains recoverable.

### Context isolation

Verify that separate contexts do not share events.

### Array metadata

Test NumPy and JAX arrays.

Verify shape, dtype, and nbytes without value transfer.

### Minimal AxonScope integration

Run a small simulation and verify the presence of the main events.

### GPU integration

Skip when no GPU is available.

Verify:

- backend/device metadata;
- `kernel.wait`;
- explicit synchronization;
- no CPU fallback.

### Serialization

Verify:

- each JSONL line parses;
- CSV loads;
- metadata JSON parses.

### Nested session rejection

Enable twice and verify a clear error.

### Context manager cleanup

Raise inside:

```python
with axs.benchmark(...):
    raise ValueError(...)
```

Verify session cleanup and original exception propagation.

---

## 20. Documentation requirements

Add a user guide section titled:

```text
Performance benchmarking
```

Include:

```python
import axonscope as axs

axs.enable_benchmark("benchmarks/example")

results = axs.simulate_pool(
    simulations,
    duration_ms=20 * axs.ms,
    dt_ms=0.01 * axs.ms,
    recording=axs.Recording.center("Vm"),
)

axs.disable_benchmark()
```

Also include:

```python
with axs.benchmark("benchmarks/example"):
    results = axs.simulate_pool(...)
```

Explain:

- asynchronous GPU timing;
- `kernel.enqueue`;
- `kernel.wait`;
- first-call classification;
- output files;
- overhead;
- difference between lightweight hotpath reports and JAX profiler traces.

---

## 21. Interpretation examples

If the report shows:

```text
inputs.extracellular   50.3 s
kernel.wait             3.6 s
```

the primary bottleneck is likely extracellular preprocessing or tensor materialization.

If it shows:

```text
kernel.wait            62.0 s
inputs.extracellular    2.0 s
```

the solver kernel is the main target.

If it shows:

```text
dispatch.build_plan    20.0 s
kernel.wait             4.0 s
```

investigate grouping, compatibility checks, geometry hashing, and repeated planning.

If it shows:

```text
results.split_batch    15.0 s
kernel.wait             5.0 s
```

investigate per-fiber Python object creation and consider a lazy batched result.

---


## 22. Legacy benchmark traces and backward compatibility

The agent is explicitly allowed to remove all existing or legacy benchmark
tracing code, benchmark trace formats, helper functions, output files, and
compatibility layers that predate this specification.

There is **no requirement to preserve backward compatibility** with previous
benchmark traces or benchmark-related APIs.

The agent may:

- delete obsolete benchmark instrumentation;
- remove deprecated trace writers and readers;
- remove old benchmark-specific environment variables;
- remove legacy JSON, CSV, text, or profiler output formats;
- rename or replace internal benchmark helpers;
- replace previous benchmark event names;
- remove compatibility shims for old benchmark reports;
- update or delete tests that only validate the legacy benchmark system;
- update examples and documentation to use only the new API;
- simplify the implementation instead of maintaining parallel old and new
  tracing systems.

The agent should prefer a clean replacement over incremental compatibility
layers.

Only normal AxonScope simulation behavior and public non-benchmark APIs must
remain compatible unless another task explicitly permits broader breaking
changes.

The repository should contain one canonical benchmark implementation after this
work. Legacy benchmark code that is no longer used should be deleted rather
than left dormant.

Existing benchmark output files created by older versions do not need to remain
readable by the new implementation.

## 23. Non-goals

The first version must not attempt to:

- replace JAX profiler tooling;
- time every function;
- time individual `lax.scan` iterations with Python;
- automatically optimize the solver;
- inspect CUDA kernels directly;
- estimate occupancy;
- support distributed multi-host traces;
- provide a browser UI;
- require pandas at runtime;
- enable benchmarking by default.

---

## 24. Acceptance criteria

The implementation is complete when:

### Public API

These work:

```python
axs.enable_benchmark(...)
axs.disable_benchmark()
axs.benchmark_report()
axs.reset_benchmark()

with axs.benchmark(...):
    ...
```

### Hotpaths

A normal pool simulation records the available main stages:

```text
simulation.pool.total
dispatch.build_plan
dispatch.group.total
runtime.prepare
inputs.intracellular
inputs.extracellular
kernel.enqueue
kernel.wait
results.split_batch
results.to_public
```

### GPU correctness

When `sync_device=True`, `kernel.wait` synchronizes actual device work.

### Output

Saving produces:

```text
events.jsonl
summary.csv
metadata.json
```

### Report

The report shows:

- total time;
- hierarchical stages;
- inclusive and/or self time;
- percentages;
- backend/device;
- dimensions;
- large array sizes;
- first-call status.

### Disabled mode

Normal AxonScope behavior is unchanged when benchmarking is disabled.

### Tests

Core unit tests and a minimal integration test pass.

---

## 25. Final target experience

```python
import axonscope as axs

axs.enable_benchmark(
    "benchmarks/500_fibers",
    print_summary=True,
    save=True,
)

results = axs.simulate_pool(
    simulations,
    duration_ms=20 * axs.ms,
    dt_ms=0.01 * axs.ms,
    recording=axs.Recording.center("Vm"),
)

axs.disable_benchmark()
```

Expected report:

```text
AxonScope benchmark — simulate_pool #1
Backend: gpu | Device: Tesla T4
B=500 | Nt=2000 | Nx=51 | Recording: Vcenter

TOTAL                                      70.675 s  100.0%
├── dispatch.build_plan                     8.214 s   11.6%
├── runtime.prepare                         0.183 s    0.3%
├── inputs.intracellular                    7.492 s   10.6%
├── inputs.extracellular                   50.346 s   71.2%
├── kernel.enqueue                          0.012 s    0.0%
├── kernel.wait                             3.621 s    5.1%
├── results.split_batch                     0.238 s    0.3%
└── results.to_public                       0.326 s    0.5%

Hotpath: inputs.extracellular
Saved to: benchmarks/500_fibers
```

The core product requirement is:

> One line to enable benchmarking, one line to disable it, and enough internal instrumentation to identify where execution time and memory pressure are actually concentrated.
