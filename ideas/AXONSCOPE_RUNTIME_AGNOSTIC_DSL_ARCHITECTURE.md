# AxonScope Runtime-Agnostic Model DSL Architecture

## Executive summary

AxonScope should adopt **runtime-agnostic scientific semantics** while remaining
**JAX-first in execution**.

The public model language, units, types, states, parameters, dependency graph,
canonical hashing, recording semantics, and domain-specific optimizations
should belong entirely to AxonScope.

JAX should remain the first production runtime because it already provides:

- JIT compilation;
- CPU and GPU execution;
- vectorization;
- functional control flow;
- automatic differentiation;
- accelerator support;
- a mature XLA compilation path.

However, JAX must not define the scientific meaning of an AxonScope model.

The recommended near-term architecture is deliberately pragmatic:

```text
restricted Python model DSL
→ typed AxonScope Model IR
→ validation and domain optimization
├── NumPy reference interpreter
└── JAX lowering
    → fused model + solver program
    → compilation and execution
```

A general runtime-neutral `Kernel IR` should not be mandatory in the first
implementation.

It should be introduced only when a real second backend, native code generator,
or shared low-level optimization requirement demonstrates that it is needed.

The first engineering priority should also be broader than the DSL itself.

AxonScope should first make the solver:

- functional;
- batch-oriented;
- free of Python loops in numerical hotpaths;
- recording-aware;
- memory-efficient;
- compilable as one coherent program;
- capable of fusing membrane updates with the cable solve.

The long-term principle is:

> AxonScope owns the semantics and scientific optimization.
> JAX initially owns high-performance execution.
> Future backends may replace execution without changing model definitions.

This can be summarized as:

```text
semantic portability now
execution portability later
```

The architecture should preserve a credible path toward future runtimes without
delaying present performance work behind an oversized multi-backend compiler
framework.

---

---

# 1. Background

An earlier AxonScope prototype explored a symbolic DSL using Python operator
overloading.

The prototype included concepts such as:

```text
Const
Var
Param
BinOp
UnaryOp
Call
Function
Gate
Channel
IonModel
```

Users could write expressions in a Python-like style, while overloaded
operators constructed a symbolic graph.

The prototype also included:

- canonical model serialization;
- deterministic model hashing;
- generated JAX source code;
- filesystem caching;
- dynamically imported generated modules;
- generated gate and current functions;
- a fused `step_ion(...)` function intended for future solvers.

The prototype demonstrated that the core idea is viable:

```text
domain model
→ symbolic representation
→ generated JAX function
→ fused execution
```

Its strongest architectural decisions should be retained.

Its prototype-specific implementation choices should not necessarily become the
production design.

---

---

# 2. Evaluation of the tested DSL approach

## 2.1 Strengths

### Explicit domain IR

The prototype did not directly bind model definitions to JAX operations.

Instead, it created symbolic nodes representing model expressions.

This is a strong foundation because an explicit IR enables:

- validation before runtime lowering;
- deterministic hashing;
- unit checking;
- shape checking;
- source-level diagnostics;
- common-subexpression elimination;
- dead-code elimination;
- constant folding;
- output pruning;
- backend-independent optimization;
- model inspection;
- model serialization;
- reproducibility.

### Python-like model authoring

The model equations remained close to their mathematical definitions.

For example, the conceptual style:

```python
@model.function
def vtrap(x, y):
    z = x / y
    return where(
        abs_(z) < 1e-7,
        y * (1.0 - z / 2.0),
        x / (exp(z) - 1.0),
    )
```

is much more readable than directly writing low-level backend kernels.

### Canonical hashing and caching

The prototype generated a canonical representation and a deterministic hash.

This is essential for:

- compilation caching;
- reproducibility;
- invalidation;
- artifact identity;
- regression tests;
- distributed reuse of compiled models.

### Fused model-step API

The prototype's fused API was one of its best ideas:

```python
def step_ion(V, gates, dt):
    gates_new = update_gates(gates, V, dt)
    Iion, Gtot = currents_and_gtot(V, gates_new)
    return gates_new, Iion, Gtot
```

This gives a compiler visibility over the full membrane update instead of
forcing multiple opaque calls.

That visibility enables:

- inlining;
- common-subexpression elimination;
- fusion;
- reduced intermediate materialization;
- better backend scheduling;
- integration with the cable solver.

### Separation from the public descriptive model

The current AxonScope architecture already treats the public membrane model as
a descriptive object rather than a compiled backend object.

That separation is aligned with a future DSL compiler.

---

## 2.2 Limitations

### It was not fully "normal Python"

The prototype relied on symbolic execution and operator overloading.

That supports arithmetic expressions well, but it does not support arbitrary
Python semantics.

Potentially unsupported or ambiguous constructs include:

```python
if V > threshold:
    ...

for gate in gates:
    ...

values.append(x)

np.exp(x)

external_function(x)
```

A production system should describe the language honestly as:

> A restricted, pure numerical Python subset for model equations.

It should not promise arbitrary Python execution.

### The intrinsic function set was explicit and limited

Functions such as:

```text
exp
abs
where
```

had to be recognized by the DSL.

A production system needs an explicit intrinsic registry with:

- semantic definitions;
- type rules;
- unit rules;
- backend lowering rules;
- differentiability metadata;
- numerical-stability variants.

### Generated Python files should not be the primary backend interface

Writing generated `.py` files is useful for:

- debugging;
- inspection;
- prototypes;
- reproducibility.

It is less attractive as the primary compiler pipeline because it introduces:

- module-loading complexity;
- cache invalidation problems;
- source-generation security concerns;
- unstable generated names;
- temporary-file management;
- duplicated representations;
- weaker source mapping.

Generated source should remain an optional debug artifact, not the core IR.

### Parameters were too static

The prototype embedded numerical parameter values in generated source and model
hashes.

That can cause unnecessary recompilation when only parameter values change.

A production system should separate:

```text
model structure
```

from:

```text
runtime parameter values
```

For example:

```python
def model_step(voltage, state, dt, params):
    ...
```

rather than compiling every conductance value as a constant.

Some parameters may still be static if they alter graph structure or shapes.

### Fusion was manually exposed, not compiler-driven

The fused `step_ion` function was valuable, but the prototype did not yet have
a full optimization pipeline.

A production compiler should explicitly implement:

- function inlining;
- common-subexpression elimination;
- dead-code elimination;
- constant folding;
- algebraic simplification;
- state packing;
- output pruning;
- fusion grouping;
- backend capability-aware lowering.

### Types and physical units were not first-class

A scientific model DSL should be able to reject invalid expressions such as:

```text
voltage + conductance
```

or an exponent applied to a dimensional quantity.

The IR therefore needs:

- scalar and tensor shapes;
- dtypes;
- units;
- semantic roles;
- static versus dynamic classification;
- state versus parameter classification.

### Source provenance was limited

Good compiler diagnostics require source information:

```text
filename
line
column
function
component
expression
```

A production compiler should preserve source provenance through all IR stages.

---

---

# 3. Primary architectural principle: runtime independence

The DSL must not lower directly from Python syntax into JAX-specific model
semantics.

AxonScope should own a typed, immutable `Model IR` that represents the
scientific meaning of a model independently of any execution runtime.

The initial production design should use:

```text
Model IR
├── NumPy reference interpretation
└── JAX lowering
```

A separate lower-level `Kernel IR` is a possible future extension, not a
first-release requirement.

The purpose of the architecture is to isolate:

```text
model meaning
```

from:

```text
runtime execution strategy
```

## 3.1 Model IR

The Model IR represents scientific meaning.

It should contain entities such as:

```text
Model
Parameter
State
Gate
Channel
Current
Observable
Function
Expression
Unit
Shape
SourceLocation
Dependency
UpdateRule
InitialCondition
```

Example conceptual Model IR:

```text
ModelIR
├── parameters
│   ├── gnabar
│   ├── gkbar
│   ├── ena
│   └── ek
├── states
│   ├── m
│   ├── h
│   └── n
├── functions
│   └── vtrap
├── gates
│   ├── m
│   ├── h
│   └── n
├── channels
│   ├── sodium
│   └── potassium
├── currents
│   └── leak
└── observables
    ├── Iion
    └── Gtot
```

The Model IR must not contain:

- JAX arrays;
- JAX primitives;
- PyTorch tensors;
- Triton code;
- XLA objects;
- backend-specific device handles.

## 3.2 Optional future Kernel IR

A future Kernel IR may represent executable numerical structure without
committing to a specific runtime.

It could express operations such as:

```text
load parameter
load state
broadcast
elementwise arithmetic
select
exp
log
expm1
state update
reduction
tridiagonal solve
scan
map
record output
```

A Kernel IR should be introduced only when concrete requirements justify it,
for example:

- a second serious backend;
- a native CPU or GPU code generator;
- shared memory-planning passes;
- portable executable serialization;
- explicit cross-backend scan, map, and reduction semantics;
- a need for lower-level control than direct JAX lowering provides.

Its design should be informed by at least two real backend targets.

Otherwise, it risks becoming a custom IR that merely mirrors JAX without
delivering genuine portability.

---

# 4. Recommended pragmatic architecture

## 4.1 Guiding strategy

Use:

```text
runtime-agnostic semantics
```

with:

```text
JAX-first execution
```

Do not attempt to provide equal support for hypothetical runtimes from the
first release.

The public DSL and Model IR must remain backend-neutral, but the first optimized
implementation may rely heavily on JAX.

## 4.2 Solver-first development order

Before building the full model compiler, restructure the solver so that it can
be compiled as one coherent numerical program.

The first milestones should be:

1. remove Python loops from numerical hotpaths;
2. represent fibers as an explicit batch dimension;
3. represent time integration as a backend scan;
4. separate preparation, compilation, and execution;
5. factorize extracellular stimulation;
6. avoid unnecessary `(B, Nt, Nx)` materialization;
7. avoid building full zero tensors when no stimulation is present;
8. return batched results instead of eagerly creating one Python object per
   fiber;
9. define a pure `solver_step`;
10. expose stable compilation and profiling boundaries.

The DSL should compile into this solver contract rather than forcing the solver
to adapt to a model-specific runtime object.

## 4.3 Pure solver step

The central executable interface should resemble:

```python
def solver_step(carry, inputs):
    voltage, model_state = carry

    (
        model_state_next,
        ionic_current,
        total_conductance,
        observables,
    ) = model_step(
        voltage,
        model_state,
        inputs.dt,
        inputs.model_parameters,
    )

    voltage_next = cable_step(
        voltage,
        ionic_current,
        total_conductance,
        inputs,
    )

    recorded = record_requested_values(
        voltage_next,
        model_state_next,
        observables,
        inputs.recording,
    )

    return (
        voltage_next,
        model_state_next,
    ), recorded
```

The backend should see the membrane update and cable solve as one program
whenever possible.

## 4.4 Homogeneous cohort compilation

Fibers should be grouped into homogeneous cohorts.

A cohort should normally share:

```text
model structure
number of compartments
dtype
integration method
recording policy
solver strategy
```

Compile one executable per cohort:

```python
compiled = compile_cohort(
    model=model_ir,
    solver=solver_spec,
    batch_size=batch_size,
    compartments=nx,
    recording=recording_spec,
    runtime="jax",
)
```

Dynamic inputs should include:

```text
initial state
dynamic parameters
fiber positions
stimulus waveform
extracellular footprints
runtime values
```

## 4.5 Factorized stimulation

Do not make the solver contract depend on a fully materialized tensor:

```text
Vstim[B, Nt, Nx]
```

Prefer a factorized representation:

```text
waveform[Nt]
footprint[B, Nx]
```

Then compute the forcing required for one time step inside the scan:

```python
forcing_t = waveform[t] * forcing_footprint
```

This is an architectural requirement because it reduces allocations, transfers,
and memory pressure.

## 4.6 Batched and lazy results

The primary solver output should remain batched:

```python
PoolResult(
    voltage=(B, Nt, Nrecorded),
    final_state=(B, Nx, S),
    observables=...,
    metadata=...,
)
```

Per-fiber result objects should be lightweight views created on demand:

```python
fiber_result = pool_result.fiber(42)
```

This avoids eager construction of hundreds or thousands of Python objects.

---

# 5. Proposed compiler pipeline

The compiler pipeline should be introduced incrementally.

## Stage 1: Python front-end capture

Users write model definitions using a restricted Python subset.

Recommended first implementation:

```text
operator-overloading symbolic DSL
```

AST analysis may be added for:

- source locations;
- syntax validation;
- diagnostics;
- forbidden-feature detection.

The first implementation should not attempt to compile arbitrary Python.

## Stage 2: typed Model IR

Each expression receives:

```text
dtype
shape
unit
semantic role
variability
source location
```

The Model IR is immutable and canonicalizable.

## Stage 3: semantic validation

Validate:

- units;
- shapes;
- states;
- parameters;
- purity;
- dependency cycles;
- initial conditions;
- outputs;
- solver compatibility.

## Stage 4: canonicalization and hashing

Normalize equivalent models into stable forms.

Produce a deterministic:

```text
model_hash
```

The hash should identify semantics, not incidental object identity.

## Stage 5: domain optimization

Perform runtime-neutral passes such as:

```text
constant folding
algebraic simplification
common-subexpression elimination
dead-code elimination
function inlining
dependency pruning
recording-aware output pruning
state packing
unit erasure after validation
```

## Stage 6: reference execution

Interpret the optimized Model IR with a NumPy backend.

This provides:

- a semantic oracle;
- readable execution;
- correctness tests;
- cross-backend comparisons;
- easier debugging.

## Stage 7: JAX lowering

Lower optimized Model IR to pure JAX functions.

The JAX lowering should support:

```text
jax.numpy
jax.lax.scan
explicit batch dimensions
jit compilation
CPU execution
GPU execution
named scopes for profiling
```

The model function should be inlined into the solver program when practical.

## Stage 8: compilation and cache

Separate:

```text
capture
validation
optimization
lowering
compilation
execution
```

Cache identity should include model semantics, compiler version, backend,
target, static shapes, and recording policy.

## Stage 9: optional future Kernel IR

Only after a second backend or native compiler path exists, determine whether a
shared Kernel IR is useful.

Do not design it solely from hypothetical requirements.

---

# 6. Backend interface

Each runtime backend should implement a stable protocol.

Example conceptual interface:

```python
class RuntimeBackend(Protocol):
    name: str
    version: str

    def capabilities(self) -> BackendCapabilities:
        ...

    def lower_model(
        self,
        model_ir: OptimizedModelIR,
        solver: SolverSpec,
        recording: RecordingSpec,
        *,
        options: CompileOptions,
    ) -> BackendProgram:
        ...

    def compile(
        self,
        program: BackendProgram,
        *,
        target: TargetSpec,
    ) -> CompiledExecutable:
        ...

    def execute(
        self,
        executable: CompiledExecutable,
        inputs: RuntimeInputs,
    ) -> RuntimeOutputs:
        ...

    def synchronize(
        self,
        outputs: RuntimeOutputs,
    ) -> None:
        ...
```

## 5.1 Backend capabilities

A backend must declare features:

```python
BackendCapabilities(
    supports_jit=True,
    supports_aot=False,
    supports_autodiff=True,
    supports_vmap=True,
    supports_scan=True,
    supports_dynamic_shapes=False,
    supports_gpu=True,
    supports_cpu=True,
    supports_custom_kernels=True,
    supports_tridiagonal_solve=True,
)
```

Optimization and lowering can adapt to capabilities.

## 5.2 Target specification

```python
TargetSpec(
    runtime="jax",
    platform="gpu",
    device_kind="NVIDIA T4",
    dtype="float32",
    batch_size=500,
    spatial_size=51,
    time_steps=2000,
)
```

## 5.3 Compile options

```python
CompileOptions(
    optimization_level=2,
    fusion_strategy="aggressive",
    recording_mode="center_vm",
    enable_autodiff=False,
    debug_symbols=False,
)
```

---

---

# 7. Recommended initial backends

## 6.1 NumPy reference backend

Implement a simple NumPy backend first.

Purpose:

- semantic reference;
- correctness tests;
- debugging;
- deterministic small-model evaluation;
- backend comparison;
- IR validation.

It does not need to be fast.

A reference backend is extremely valuable because it prevents the JAX backend
from becoming the definition of correctness.

## 6.2 JAX production backend

The first optimized backend should be JAX.

Responsibilities:

- lower Kernel IR to pure JAX functions;
- use `jax.numpy`;
- use `jax.lax.scan`;
- use `jax.vmap` where appropriate;
- use `jax.jit`;
- expose explicit compilation boundaries;
- preserve named scopes for profiling;
- support CPU and GPU;
- optionally support ahead-of-time lowering.

The JAX backend should not receive Python model objects directly.

It should receive optimized Kernel IR.

## 6.3 Future native or MLIR backend

A future backend could lower Kernel IR to:

- MLIR;
- LLVM;
- CUDA;
- ROCm;
- Triton;
- another scientific compiler.

The existence of this future path is the reason the IR and backend protocol
must remain independent from JAX.

---

---

# 8. Runtime selection

Users should normally not need to know compiler internals.

Suggested API:

```python
axs.set_runtime("jax")
```

or:

```python
with axs.runtime("jax", platform="gpu"):
    results = axs.simulate_pool(...)
```

Automatic selection may be supported:

```python
axs.set_runtime("auto")
```

The runtime selector may consider:

- installed backends;
- available devices;
- model features;
- batch size;
- recording policy;
- expected memory;
- user preference;
- benchmark history.

A future autotuner could select different backends for different workloads.

---

---

# 9. Parameter model

Separate parameters into categories.

## Dynamic parameters

Values that can change without recompilation:

```text
maximal conductance
reversal potential
temperature factor
stimulation amplitude
initial state
```

Pass them as runtime inputs.

## Static parameters

Values that alter structure:

```text
number of gates
channel topology
state count
selected integration scheme
recorded output structure
```

Include them in compile keys.

## Specializable parameters

Values that can be either dynamic or static.

Example:

```text
temperature
```

Dynamic mode:

- fewer compilations;
- more general executable.

Static mode:

- more constant folding;
- potentially faster executable.

Expose policy through compile options.

---

---

# 10. Cache architecture

Compilation cache identity should include:

```text
canonical model hash
IR schema version
compiler version
optimization level
backend name
backend version
target platform
device kind
dtype
static shapes
recording policy
solver coupling mode
static parameters
```

Suggested key:

```text
artifact_hash = hash(
    model_hash,
    compiler_version,
    backend_id,
    target_spec,
    compile_options,
)
```

Cache layers:

```text
Model IR cache
Optimized IR cache
Backend program cache
Compiled executable cache
```

Do not assume all backends serialize executables.

Each backend should declare cache capabilities.

---

---

# 11. Fusion strategy

## 10.1 Domain-level fusion

AxonScope should fuse based on semantic dataflow.

Examples:

```text
gate rates + gate update
conductance + current
all channels + total reduction
model step + cable solver step
```

## 10.2 Backend-level fusion

The backend compiler should perform low-level fusion.

For JAX/XLA, this includes:

- operation fusion;
- buffer optimization;
- scheduling;
- device-specific code generation.

AxonScope should not attempt to reproduce XLA-level optimizations.

## 10.3 Preserve backend freedom

Fusion groups should not be hard barriers.

Represent them as:

```text
preferred fusion regions
```

rather than mandatory kernels.

A backend may:

- merge regions;
- split regions;
- ignore hints;
- generate custom kernels.

---

---

# 12. Recording-aware compilation

Requested outputs should influence optimization.

Example:

```python
axs.Recording.center("Vm")
```

may allow the compiler to avoid:

- storing full voltage history;
- materializing unrequested observables;
- returning all channel currents;
- preserving intermediate states after use.

The compiler should receive a `RecordingSpec`.

Example:

```python
RecordingSpec(
    voltage_indices=[center_index],
    observables=[],
    retain_final_state=True,
)
```

Output pruning should occur before backend lowering.

The benchmark and profiler must not force generation of otherwise unused
outputs.

---

---

# 13. Solver integration

The most important performance objective is not merely generating a fast
membrane function.

It is generating a model representation that can be fused with the solver.

Avoid this architecture:

```text
solver kernel
→ opaque model executable call
→ materialized intermediate current
→ opaque cable executable call
```

Prefer:

```text
one solver program
├── membrane update
├── channel currents
├── conductance reduction
├── cable solve
└── requested recording
```

This allows:

- fewer launches;
- fewer intermediate buffers;
- less device memory traffic;
- global common-subexpression elimination;
- solver-aware output pruning;
- better temporal fusion.

---

---

# 14. DSL syntax recommendation

## 13.1 Use operator overloading first

Recommended first production front-end:

```text
operator-overloading symbolic expressions
```

Advantages:

- relatively simple implementation;
- controlled language;
- direct IR construction;
- predictable semantics;
- easy backend independence;
- clear operation registry;
- no need to compile arbitrary Python.

## 13.2 Use AST only as support

AST analysis can provide:

- source locations;
- syntax validation;
- better diagnostics;
- forbidden-feature detection;
- documentation extraction.

Do not make full Python AST compilation the first target.

A full Python compiler would need to handle:

- closures;
- mutation;
- imports;
- exceptions;
- generators;
- comprehensions;
- arbitrary calls;
- object identity;
- version-specific syntax.

That complexity is unnecessary for model equations.

## 13.3 Restricted-language contract

Document the supported subset explicitly.

Supported examples:

```text
arithmetic
comparisons
registered intrinsics
pure helper functions
conditional expressions
immutable local bindings
```

Unsupported examples:

```text
mutation
I/O
dynamic object creation
arbitrary library calls
data-dependent Python loops
side effects
```

---

---

# 15. Intrinsic registry

Create a runtime-neutral intrinsic registry.

Example:

```python
Intrinsic(
    name="exp",
    type_rule=...,
    unit_rule=dimensionless_input,
    evaluator=...,
    differentiable=True,
    lowerings={
        "numpy": ...,
        "jax": ...,
        "torch": ...,
    },
)
```

Core intrinsics:

```text
exp
expm1
log
log1p
sqrt
abs
minimum
maximum
clip
where
sigmoid
tanh
pow
```

Numerically stable domain-specific intrinsics may include:

```text
vtrap
safe_exp
ghk_current
rush_larsen_update
crank_nicolson_gate_update
```

A backend may implement a custom lowering.

---

---

# 16. Type and unit system

Each expression should have:

```text
dtype
shape
unit
semantic role
variability
source location
```

Example roles:

```text
voltage
current
conductance
time
rate
dimensionless
state
parameter
observable
```

Validation examples:

```text
voltage + voltage          valid
voltage + conductance      invalid
exp(dimensionless)         valid
exp(voltage)               invalid
rate × time                dimensionless
conductance × voltage      current
```

Units should be removed or normalized before low-level runtime execution.

The backend should operate on normalized numerical values.

---

---

# 17. Debugging and introspection

Provide user-facing inspection tools:

```python
model.show_ir()
model.show_optimized_ir()
model.show_dependencies()
model.show_generated_code(runtime="jax")
model.explain_fusion()
model.explain_cache_key()
```

Generated source should be optional:

```python
artifact.emit_debug_source(path)
```

The canonical source of truth remains the IR.

---

---

# 18. Diagnostics

Errors should reference user code.

Example:

```text
ModelCompilationError:
  Invalid unit in sodium.alpha

  models/hh.py:42
      return exp(-(V + 65 * mV) / 18)

  exp() requires a dimensionless input.
  Received: mV
```

Store source provenance on IR nodes:

```text
filename
line
column
function
component
```

---

---

# 19. Benchmarking integration

The runtime-agnostic compiler should integrate with the benchmarking framework.

Reserved compiler events:

```text
model.capture
model.ir.build
model.ir.validate
model.ir.optimize
model.fusion.plan
model.kernel_ir.lower
model.cache.lookup
model.backend.lower
model.compile
kernel.enqueue
kernel.wait
```

Example report:

```text
model.capture                    2.1 ms
model.ir.build                   1.4 ms
model.ir.validate                1.2 ms
model.ir.optimize                4.8 ms
model.fusion.plan                0.9 ms
model.cache.lookup               0.2 ms  miss
model.backend.lower              3.7 ms
model.compile                  842.0 ms
kernel.wait [first]             18.2 ms
kernel.wait [steady]             3.1 ms
```

Runtime-neutral metadata:

```text
model_hash
ir_hash
backend
backend_version
target
cache_hit
optimization_level
fusion_strategy
```

Do not insert host timers or synchronization inside generated kernels.

Use backend-specific profiler annotations where available.

---

---

# 20. Role of jaxpr, StableHLO, and MLIR

## 20.1 jaxpr

Do not use `jaxpr` as the canonical AxonScope model representation.

`jaxpr` is useful for:

- inspecting JAX lowering;
- diagnosing unexpected operations;
- validating the generated JAX program;
- checking whether fusion opportunities remain visible;
- profiling backend output.

It is not the scientific source of truth because it does not naturally preserve
domain concepts such as:

```text
gate
channel
reversal potential
recording policy
compartment
extracellular footprint
```

## 20.2 StableHLO

StableHLO may be useful as:

- an export artifact;
- a portable compiled-program representation;
- an interface to OpenXLA-compatible tools;
- a deployment format.

It should not replace the Model IR because it is too low-level to preserve
scientific semantics and domain optimization opportunities.

## 20.3 MLIR

MLIR is a credible long-term option if AxonScope eventually requires:

- custom compiler dialects;
- native CPU or GPU code generation;
- several progressive lowering stages;
- explicit memory planning;
- custom solver kernels;
- targets beyond what JAX supports effectively.

Do not adopt MLIR in the first production implementation.

Its toolchain, build complexity, and maintenance cost are justified only when
JAX becomes a demonstrated limitation or native multi-target compilation
becomes a concrete product requirement.

---

# 21. Advantages of the proposed architecture

## Runtime independence

The model semantics are not tied to JAX.

A new backend can be added without changing model definitions.

## Better scientific validation

Units, states, dependencies, and equations can be checked before execution.

## Better optimization

AxonScope can perform domain-specific optimization that a generic runtime
cannot infer easily.

## Maximal fusion opportunity

The model can be fused into the solver instead of executed as an opaque module.

## Reproducibility

Canonical IR and hashes make compiled artifacts identifiable.

## Testing

The same model can run on:

```text
NumPy reference
JAX CPU
JAX GPU
future runtime
```

and outputs can be compared.

## Debuggability

Users can inspect IR, optimization, dependencies, and generated programs.

## Future portability

Backends can target CPUs, GPUs, web runtimes, or specialized accelerators.

## Better cache control

Cache keys can be based on semantic structure and target configuration.

---

---

# 22. Costs and limitations

## Compiler complexity

A runtime-agnostic compiler requires more engineering than direct JAX model
functions.

It needs:

- IR definitions;
- validation;
- optimization;
- lowering;
- backend interfaces;
- cache management;
- source mapping;
- tests.

## Restricted Python subset

Users cannot use arbitrary Python.

The language contract must be explicit and documented.

## Backend feature mismatch

Not every backend will support:

- autodiff;
- dynamic shapes;
- scans;
- custom linear solves;
- GPU execution;
- equivalent numerical precision.

The compiler needs capability checks and graceful errors.

## Numerical differences

Different backends may produce slightly different results because of:

- floating-point ordering;
- fused operations;
- math-library implementations;
- precision defaults;
- solver implementations.

Cross-backend tolerances must be defined.

## Cache invalidation

Compiler and backend upgrades require careful artifact invalidation.

## Fusion trade-offs

Aggressive fusion can:

- improve execution;
- increase compilation time;
- increase register pressure;
- make debugging harder;
- produce larger kernels.

The compiler needs configurable optimization levels.

## Dynamic model features

Highly dynamic model structures may reduce specialization and optimization.

---

---

# 23. Development roadmap

## Phase 0: benchmark the current implementation

- implement built-in hotpath benchmarking;
- separate preparation, compilation, execution, and postprocessing;
- synchronize device execution correctly;
- measure large array materialization;
- establish CPU and GPU baselines.

## Phase 1: make the solver compilable as one block

- define a pure `solver_step`;
- move time integration into a scan;
- remove per-fiber Python loops from hotpaths;
- make the batch dimension explicit;
- factorize extracellular stimulation;
- avoid full zero stimulation tensors;
- return batched results;
- separate `prepare`, `compile`, and `run`.

## Phase 2: define model semantics

- define the restricted Python subset;
- implement immutable Model IR;
- add types and units;
- preserve source provenance;
- classify dynamic and static parameters;
- implement canonical serialization and hashing.

## Phase 3: implement reference execution

- implement the NumPy interpreter;
- port representative membrane models;
- compare against existing implementations;
- define numerical tolerances;
- add semantic tests.

## Phase 4: implement JAX lowering

- lower Model IR to pure JAX functions;
- integrate model and solver steps;
- use `lax.scan`;
- preserve explicit batching;
- implement compilation caching;
- test CPU and GPU;
- optionally expose StableHLO export.

## Phase 5: add domain optimization

- constant folding;
- common-subexpression elimination;
- dead-code elimination;
- function inlining;
- recording-aware pruning;
- state packing;
- specialization policies;
- fusion planning.

## Phase 6: evaluate a second backend

Only when a concrete second backend exists:

- identify shared backend requirements;
- decide whether a Kernel IR is justified;
- implement the smallest sufficient common IR;
- formalize backend capabilities;
- compare lowering quality and performance;
- avoid speculative abstractions.

---

# 24. Recommended concrete API

Example model:

```python
class HodgkinHuxley(axs.ModelDSL):
    gnabar = axs.parameter(
        120.0,
        unit=axs.mS / axs.cm**2,
    )

    ena = axs.parameter(
        50.0,
        unit=axs.mV,
    )

    @axs.function
    def vtrap(x, y):
        z = x / y
        return axs.where(
            axs.abs(z) < 1e-7,
            y * (1.0 - z / 2.0),
            x / axs.expm1(z),
        )

    m = axs.gate(
        alpha=lambda V: 0.1 * vtrap(
            -(V + 40.0 * axs.mV),
            10.0 * axs.mV,
        ),
        beta=lambda V: 4.0 * axs.exp(
            -(V + 65.0 * axs.mV)
            / (18.0 * axs.mV)
        ),
    )

    sodium = axs.channel(
        conductance=gnabar * m**3,
        reversal=ena,
    )
```

Compile:

```python
model = HodgkinHuxley()

artifact = axs.compile_model(
    model,
    runtime="jax",
    target="gpu",
    optimization_level=2,
    fusion="aggressive",
)
```

Use in solver:

```python
axon = axs.Axon(
    membrane=artifact,
)
```

Or allow automatic compilation:

```python
axon = axs.Axon(
    membrane=HodgkinHuxley(),
)
```

with runtime selection at simulation time.

---

---

# 25. Recommended internal interfaces

## Model compiler

```python
class ModelCompiler:
    def capture(self, model_definition) -> ModelIR:
        ...

    def validate(self, model_ir: ModelIR) -> ValidationReport:
        ...

    def optimize(
        self,
        model_ir: ModelIR,
        options: OptimizationOptions,
    ) -> OptimizedModelIR:
        ...

    def lower(
        self,
        model_ir: OptimizedModelIR,
        solver: SolverSpec,
        recording: RecordingSpec,
        backend: "RuntimeBackend",
    ) -> BackendProgram:
        ...

    # Add lower_to_kernel_ir(...) only when a shared Kernel IR is justified.

```

## Runtime backend

```python
class RuntimeBackend:
    def capabilities(self) -> BackendCapabilities:
        ...

    def lower(
        self,
        model_ir: OptimizedModelIR,
        solver: SolverSpec,
        recording: RecordingSpec,
        options: CompileOptions,
    ) -> BackendProgram:
        ...

    def compile(
        self,
        program: BackendProgram,
        target: TargetSpec,
    ) -> CompiledExecutable:
        ...
```

## Compiled artifact

```python
class CompiledModel:
    model_hash: str
    kernel_ir_hash: str
    backend_name: str
    backend_version: str
    target: TargetSpec
    executable: Any
    metadata: dict[str, Any]

    def run(self, inputs):
        ...
```

---

---

# 26. What should remain backend-neutral

The following must remain independent of JAX:

```text
public model syntax
model semantics
states
parameters
gates
channels
units
shape validation
source locations
canonical hashing
domain optimization
dependency analysis
recording requirements
solver-model contract
benchmark event model
```

The following may be backend-specific:

```text
array type
device selection
JIT mechanism
AOT mechanism
scan implementation
vectorization strategy
linear solver implementation
custom kernels
executable serialization
synchronization
profiler integration
```

---

---

# 27. Final recommendation

Retain the strongest ideas from the original prototype:

```text
Python-like symbolic model definitions
→ explicit scientific representation
→ deterministic hashing
→ fused numerical execution
```

Replace the prototype-specific path:

```text
symbolic expressions
→ generated JAX Python file
→ dynamically imported module
```

with a pragmatic architecture:

```text
restricted Python DSL
→ typed AxonScope Model IR
→ validation and domain optimization
├── NumPy reference interpreter
└── JAX lowering
    → fused model + solver program
    → compiled executable
```

Do not require a general Kernel IR initially.

Introduce one only when a concrete second backend or native code-generation
path provides enough evidence to design it correctly.

The recommended ownership split is:

```text
AxonScope owns:
    model semantics
    units and types
    states and parameters
    dependency analysis
    scientific validation
    domain optimization
    solver contract
    recording semantics
    cache identity

JAX initially owns:
    array execution
    scans
    batching
    JIT compilation
    CPU/GPU lowering
    device execution

Future backends may replace:
    backend lowering
    compilation
    synchronization
    device runtime

without changing:
    model definitions
    scientific meaning
    validation
    optimization intent
```

The immediate priority should be:

```text
solver-first performance work
→ Model IR
→ NumPy reference
→ JAX lowering
→ fused solver execution
→ second backend evaluation
```

This architecture provides genuine future portability without forcing
AxonScope to build a large multi-runtime compiler before it has the evidence or
backend requirements needed to design one well.

The final principle is:

> Runtime-agnostic semantics, JAX-first execution, evidence-driven
> generalization.
