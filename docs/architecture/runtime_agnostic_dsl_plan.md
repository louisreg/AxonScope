# Runtime-Agnostic DSL And Model IR Plan

Status on 2026-07-03: architecture plan and implementation history. The P7
membrane cleanup described here is complete for the current JAX path. Use
`docs/membranes.md`, `README.md`, and the active examples for user-facing
membrane authoring. Use `GUIDELINES.md` and `todo.md` for current architecture
direction and remaining work.

The long-form rationale lives in
`ideas/AXONSCOPE_RUNTIME_AGNOSTIC_DSL_ARCHITECTURE.md`.

IR means "Intermediate Representation": the structured internal form between
plain Python membrane source/public membrane descriptions and backend execution. In
AxonScope, Model IR is the canonical scientific representation of membrane
semantics: states, parameters, gates, currents, units, observables, hashes,
validation, and optimization metadata.

## Direction

AxonScope should own model semantics independently from any execution runtime:
units, states, parameters, gates, channels, currents, observables, dependency
graphs, validation, canonical hashes, recording semantics, and domain
optimization belong to AxonScope.

Execution stays JAX-first for now. JAX owns array execution, scans, batching,
JIT compilation, CPU/GPU lowering, and device runtime. JAX must not define the
scientific meaning of a membrane model.

The target sequence is:

```text
plain Python membrane source
-> typed AxonScope Model IR
-> validation and domain optimization
-> NumPy reference interpreter for model semantics
-> JAX lowering for production execution
-> fused membrane + cable solver programs
```

A general low-level Kernel IR is not part of the first implementation. It
should be introduced only after a concrete second backend or native code
generation path proves that a shared executable IR is necessary.

The first P7 cleanup target is done: the historical `channel_models/` and
`icm/` runtime packages are no longer active package paths. The remaining exit
target is stricter: backend kernels should consume Model IR-derived/generated
step programs directly through structural backend contracts. The active
JAX runtime now consumes `JaxMembraneProgram` directly; the former
`ModelIRMembrane` adapter and `CompiledMembrane` inheritance surface have been
deleted from active runtime code.

P7 is also a user-facing membrane authoring phase. The rejected builder-style
manifest DSL is not the target. Users should be able to write membrane
equations as ordinary Python source with visible intermediate assignments; the
compiler should parse that source, validate it, build the internal model graph,
then generate JAX or NumPy model-step code according to the selected runtime.
Users define membrane models, not Model IR.

## Why This Replaces The NumPy Runtime Slice

`axs.runtime.numpy` remains reserved and non-executable. Passive, Hodgkin-Huxley,
Rattay-Aberham, Sundt, AxNode, Tigerholm, and Schild semantics now have a
backend-neutral Model IR, a NumPy model-step interpreter, and JAX lowering. A
NumPy/SciPy full
simulation runtime still needs its own cable-solver implementation, execution
facade integration, inspection, estimates, docs, examples, and validation
before it can become publicly executable.

The right order is:

1. define backend-neutral model semantics;
2. validate and canonicalize those semantics;
3. add a NumPy reference interpreter for the Model IR;
4. lower the same Model IR to JAX;
5. only then expose a public NumPy/SciPy runtime for tiny simulations if it is still
   useful.

## Phase Contract

The public membrane contract starts from `axs.membranes.Model` subclasses.
Built-in names such as `axs.membranes.HodgkinHuxley` are model classes, not
wrapper functions around string registries. Instantiating the class returns a
descriptive model instance with autocomplete-friendly constructor parameters.
Flattening converts public model instances to the internal normalized
`MembraneModel` descriptor consumed by solver preparation.
Public examples should not use raw `axonscope.model_ir` dataclasses, JAX arrays,
backend classes, or the removed builder-style manifest DSL.

The first deliverable is internal architecture:

- immutable Model IR dataclasses;
- a restricted Python source contract;
- intrinsic registry;
- semantic validation;
- canonical serialization and hashing;
- reference interpretation for model steps;
- JAX lowering for model steps;
- a solver-model interface that allows fusion.

The second deliverable is the Python-source compiler layer:

- a tiny, explicit source contract for model files and functions;
- public membrane classes backed by exactly one source file per model;
- AST parsing of ordinary assignments, helper calls, dictionaries/returns, and
  source locations;
- graph construction into AxonScope's internal representation;
- generated JAX and NumPy model-step implementations from the same graph;
- examples that cover a passive model, a HH-style gated model, readable
  intermediate equations, recordings/plots, and a deliberately invalid model.

## Current Boundary To Audit

The current public descriptive layer is now split at the desired boundary:

- `axonscope.membranes.Model` is the public authoring/constructor base;
- `Passive`, `HodgkinHuxley`, `RattayAberham`, `Sundt`, and `AxNode` are
  source-backed `Model` classes;
- axon `Section` objects keep the user model instance;
- layout flattening normalizes model instances into internal `MembraneModel`
  descriptors for solver preparation.

The backend-dependent layer is now reduced to JAX lowering and structural
backend execution contracts:

- public membrane descriptors compile to `JaxMembraneProgram`;
- JAX runtime preparation wraps programs in uniform, heterogeneous, or
  structural gated/leak stack membrane backends;
- solver kernels call backend methods inside JAX scans while deeper
  recording-aware fusion/pruning is being built.

The remaining compiler work is to make those backend calls more recording-aware
and optimizable, so scientific semantics stay in AxonScope's neutral graph and
JAX owns only execution.

Adding a membrane family must not require solver/runtime edits. Model
semantics belong in `src/axonscope/membranes/models` and compiler metadata; the
solver consumes only structural `MembraneProgram`/backend contracts. Any future
fast path must be triggered by IR capabilities or optimized graph structure,
not by names such as AxNode, Rattay, Schild, Tigerholm, or Passive.

## Current Stack Audit

The current P7 implementation keeps runtime behavior stable while removing the
historical membrane packages and the per-model backend adapter layer.

- Public `Model` instances are backend-neutral. Flattening normalizes them to
  internal `MembraneModel` descriptors, and
  `runtime/jax/membranes/compile.py::compile_axon_membrane` translates those
  descriptors to `JaxMembraneProgram` through the single Model IR path.
- The JAX membrane backend protocol exposes the hot solver terms today:
  `init_gates`, `cn_gate_update`, `currents`, `total_conductance`,
  `membrane_conductance_terms`, and `background_current`.
- Single-cable and double-cable kernels call those backend methods inside JAX
  scans, then assemble cable matrices from `Gm`, `GE`, explicit outward
  current, correction current, and retained state.
- Membrane cache identity currently combines membrane/program signatures,
  solver options, geometry shape, dtype, initial voltage, and runtime policy
  context. It does not yet use a backend-neutral optimized Model IR hash as the
  primary key.
- Recording and observer lowering already happen before batch execution, but
  membrane observables are still packaged after opaque backend calls.

The DSL must keep lowering to visible solver terms, not to an opaque
`Iion(V, state)` callback. The model-step contract exposes state updates,
total outward current, total conductance, conductance-reversal sum,
explicit/correction currents, and requested observables.

## Current Implementation Layout

The membrane implementation is now deliberately split by responsibility:

- `axonscope.membranes.Model` is the public class base for membrane models;
- `axonscope.membranes.builtins` re-exports source-backed public model classes
  without carrying equations, defaults, or migration bridges;
- `axonscope.membranes.models` owns human-authored plain-Python source
  equations for built-ins. Each file defines a `Model` subclass with typed
  fields, public aliases, decorated section methods, decorator-local
  output/state metadata, and any derived default logic needed by that model;
- `axonscope.membranes.types` owns semantic annotation markers such as
  `Voltage`, `Gate`, `Rate`, `ConductanceDensity`, and `CurrentDensity`;
- `axonscope.membranes.compiler` owns the internal descriptor-to-source bridge:
  it maps a membrane kind to `axonscope.membranes.models/{kind}.py` and passes
  already-normalized source-unit parameter defaults to the generic compiler;
- each built-in source file is intentionally standalone: tiny equation helpers
  stay local to the file, canonical unit labels come from
  `axonscope.utils.units`, and the same file owns model equations,
  unit-bearing defaults, public aliases, and derived parameter logic;
- `axonscope.model_ir` contains no model-family-specific built-ins, registries,
  or parameter defaults; it owns only the backend-neutral representation,
  compiler machinery, validation, composition, serialization, and program
  derivation;
- `axonscope.model_ir.unit_algebra` owns only compiler-side unit
  normalization/multiplication/division rules;
- `axonscope.model_ir.program` derives the backend-neutral `MembraneProgram`
  execution contract from `ModelIR`: names, raw/grouped current and
  conductance columns, auxiliary state names, diagnostics, final gate update
  policy, and stable hashes;
- `axonscope.runtime.jax.membranes.program.JaxMembraneProgram` is the
  executable contract for Model IR membranes. It owns JAX lowering, rate-table
  policy, traces, state updates, diagnostics, and static signatures;
- `UniformMembraneBackend`, `HeterogeneousMembraneBackend`, and the structural
  `GatedLeakStackMembraneBackend` consume `JaxMembraneProgram` directly;
- `axonscope.runtime.jax.membranes.model_ir_lowering` lowers Model IR expressions and
  step programs to JAX callables;
- `axonscope.runtime.jax.preparation.base` is intentionally model-family agnostic:
  it no longer carries Rattay/AxNode/passive membrane equations or model-name
  fast paths.

The active public-to-runtime path is:

```text
axs.membranes public Model instance
-> lower_membrane_model_to_ir(...)
-> source compiler through membranes/models/{kind}.py
-> ModelIR validation and hashes
-> backend-neutral MembraneProgram
-> optional future IR optimization passes
-> JAX lowering
-> JaxMembraneProgram
-> solver kernels
-> requested recordings, observations, and result views
```

The next optimization target is a fused, recording-aware solver program:

```text
ModelIR
-> validation
-> backend-neutral graph optimization
-> JAX target lowering
-> fused membrane/cable step program
-> requested outputs only
```

The source-codegen checkpoint is now broad enough to cover all built-in
membrane families: Passive, Hodgkin-Huxley, Rattay-Aberham, Sundt, AxNode,
Tigerholm, Schild94, and Schild97 are authored as ordinary Python equation
files with named model sections, semantic quantity annotations, root
unit aliases directly in equations, decorator-local public outputs, and
source-owned non-gate state initialization/update programs where needed.
Generated code now comes from the compiled graph rather than raw source text,
so readable physical equations can still lower to canonical numeric units. The
open work is no longer "convert built-ins" or "remove runtime adapters"; it is
optimizing, explaining, and making generated-code caches first-class.

## Generalized IR Direction

The separated model files make the next IR generalization clearer. The v1 IR
should avoid treating the current gate/current split as the only possible
representation. It should preserve these concepts separately:

- physical equations: states, parameters, algebraic bindings, currents, pumps,
  exchangers, buffers, and source provenance;
- solver-facing terms: total outward current, conductance linearization,
  reversal-weighted conductance sum, explicit current, correction current, and
  state-update programs;
- output policy: named observables, diagnostics, gate/state traces, recording
  requirements, and pruning rules;
- dependency graph: expression dependencies, update ordering, cycles, static
  versus dynamic parameters, and source locations;
- backend metadata: optimized IR hash, target lowering key, dtype, static
  shapes, recording policy, compiler version, and optimization level.

Optimization should happen in two layers:

- backend-neutral Model IR graph passes before runtime preparation for common
  subexpression elimination, dead observable pruning, dependency analysis,
  source/provenance diagnostics, and cache identity;
- JAX-specific lowering passes during backend lowering for gate/current fusion,
  conductance/current linearization fusion, state-update fusion, scan-friendly
  scheduling, and avoiding transport of unrequested intermediate arrays.

## Initial Implementation

The internal package `axonscope.model_ir` now contains the first neutral layer:

- immutable expression nodes used by the internal compiler graph;
- a runtime-neutral intrinsic registry including elementary functions, `vtrap`,
  and gate-update intrinsics;
- immutable Model IR dataclasses for inputs, parameters, states, gates,
  currents, observables, functions, units, shapes, dtypes, and variability;
- canonical serialization with separate structural and parameterized hashes;
- semantic validation before backend lowering;
- adapters for Passive, Hodgkin-Huxley, Rattay-Aberham, Sundt, AxNode,
  Tigerholm, and Schild public membrane descriptions;
- composition of Model IR-covered membrane components into one graph with
  deterministic internal symbol renaming and duplicate current/conductance
  aggregation;
- a NumPy reference interpreter for model-step semantics;
- JAX lowering and `JaxMembraneProgram` runtime execution for Passive,
  Hodgkin-Huxley, Rattay-Aberham, Sundt, AxNode, Tigerholm, Schild, and
  covered composites;
- gate states split from auxiliary membrane states, so ion pools and buffers are
  not exposed as HH gates;
- a first `StepProgram` contract for prepare-state updates, explicit outward
  current, correction current, linearization gate selection, diagnostics, and
  post-solve finalization; Tigerholm uses this path for Na/K concentration
  pools, and Schild uses it for calcium pools, KCa finalization, diagnostics,
  and pump/exchanger corrections;
- a `ModelStepContract` and `OutputPruningPlan` for future fused
  cable/membrane programs.

This is still internal architecture, not a public DSL API. It deliberately has
no JAX imports in `axonscope.model_ir`. Public `AxonSimulation` APIs are
unchanged, but covered built-in membrane equations now execute through the
Model IR JAX lowering rather than the previous hand-written concrete membrane
classes.

## Initial Scope

The first runtime-backed built-ins are:

- passive leak;
- Hodgkin-Huxley;
- Rattay-Aberham;
- Sundt through composed NaHH, BorgKDR, and passive Model IR components;
- AxNode/MRG nodal dynamics, with the structural gated/leak stack batch
  fast path preserved after runtime adapter removal;
- Tigerholm C-fiber dynamics with explicit Na/K concentration states,
  background pump current, and correction current;
- Schild 1994/1997 DRG C-fiber dynamics with explicit calcium pool states,
  background pump/exchanger currents, correction current, diagnostics, and
  post-solve KCa finalization.

Defer the hard cases until the skeleton is real:

- composition rules for stateful `StepProgram` components;
- generated debug source;
- Kernel IR;
- broader public source-model examples.

## Model IR Skeleton

The first IR should represent:

- parameters and their static/dynamic variability;
- states and initial conditions;
- functions and expression trees;
- gates and update rules;
- channels and current equations;
- observables;
- units, dtypes, shapes, and semantic roles;
- source locations;
- dependency edges.

The IR must not contain:

- JAX arrays;
- JAX primitives;
- backend device handles;
- dynamically imported generated modules;
- runtime-specific executable objects.

## Plain Python Source Contract

The accepted front-end should be a restricted subset of normal Python source,
not an operator-overloaded builder API. Raw Model IR dataclasses remain the
compiler target, not the intended user surface. AST parsing is the primary
front-end so equations, intermediate assignments, source locations, and helper
calls stay visible.

Supported:

- arithmetic;
- comparisons;
- immutable local bindings;
- registered intrinsics;
- pure helper functions;
- conditional expressions represented by `where`.

Unsupported:

- mutation;
- data-dependent Python loops;
- arbitrary NumPy/JAX calls;
- I/O;
- object creation inside equations;
- side effects.

## Intrinsic Registry

Intrinsics need semantic rules before backend lowering:

- name;
- type rule;
- unit rule;
- differentiability metadata;
- NumPy evaluator;
- JAX lowering;
- optional numerical-stability notes.

Initial intrinsics:

- `exp`, `expm1`, `log`, `log1p`, `sqrt`, `abs`;
- `minimum`, `maximum`, `clip`, `where`;
- `pow`, `tanh`, `sigmoid`;
- domain intrinsics such as `vtrap` and gate-update helpers.

## Validation

Validation should reject invalid models before any backend lowering:

- incompatible units such as voltage plus conductance;
- dimensional arguments to dimensionless intrinsics such as `exp`;
- shape mismatches;
- missing initial states;
- dependency cycles;
- unsupported intrinsics;
- state/parameter role misuse;
- recording requests for unavailable observables.

Diagnostics should point back to source locations when available.

## Hashing And Cache Identity

Canonical hashes should separate model structure from runtime parameter values.

Structural hash:

- graph topology;
- states;
- equations;
- static parameters;
- units and normalized roles;
- compiler schema version.

Runtime parameter values:

- conductances;
- reversal potentials;
- temperature-like values when dynamic;
- initial states;
- stimulation/runtime inputs.

Backend cache keys later add:

- backend name/version;
- target platform/device;
- dtype;
- static shapes;
- recording policy;
- solver coupling mode;
- optimization level;
- compiler version.

Generated code should have a persistent cache so unchanged models do not
recompile. The default project-local cache is:

```text
.axonscope_cache/model_codegen/<cache_key>/
  manifest.json
  source_snapshot.py
  graph.json
  optimized_graph.json
  jax_model.py
  numpy_model.py
  __pycache__/
```

The source hash should include normalized source/AST, helper function sources,
source contract version, compiler version, unit schema version, intrinsic
registry version, and static model metadata. The generated-code cache key then
adds optimized graph hash, target backend, dtype/precision policy, static
shapes needed by the model-step signature, recording/output policy,
optimization level, and helper/dependency hashes.

`ModelIR`, `MembraneProgram`, and backend programs may carry stable provenance
and cache identity: source contract/compiler version, source hash, source
function names, generated-code cache key, target list, manifest name, and
generated file names. They must not carry volatile hit/miss state because that
would make cold and warm runs produce different model signatures.

On a cache hit, AxonScope imports requested generated `.py` modules from the
cache and skips AST parsing, graph optimization, and code emission. The source
compiler uses a source-text index to find the cache key, validates the manifest,
reloads `optimized_graph.json` into `ModelIR`, applies dynamic parameter
overrides after reload, and lets the JAX runtime request `jax_model.py`.
Inspection and benchmark metadata report cache hit/miss status, cache key,
source hash, generated module path, compiler version, loaded target names, and
the reason for invalidation when known.

The generated modules are now first-class cache artifacts and the first
executable path is active: generated modules expose `ARG_NAMES` and
`OUTPUT_NAMES`, source codegen prunes unused dependencies before emission, and
`JaxModelIRLowering.current_matrix(...)` uses the generated JAX `model_step`
for standalone source models when the requested current/output path is covered.
Uncovered cases fall back to the IR evaluator.

The P7 supported class subset now has generated conductance/current outputs,
state prepare/finalize updates, diagnostics, and requested-output pruning. The
next optimization step is deeper fusion: common-subexpression elimination,
composite generated programs, recording-aware schedule pruning, and direct
solver-kernel fusion where possible. These optimization passes may lower a
compiled model into an internal representation that no longer follows the
public `Model` class contract; the stable contract is the user-authored class
source, not the optimized runtime plan.

Current inspection exposes source/codegen identity and current cache status per
dispatch group: membrane kinds, unique membrane count, source count, cache
statuses, miss reasons, and shortened cache keys. Dedicated generated-code
reports now start from public membrane models through
`membrane.inspect_generated_code()` or `axs.membranes.inspect_generated_code(...)`.
They expose manifest contents, cache key, source hash, generated files, and
optional selected generated JAX/NumPy module text without making the internal
representation a user-facing API.
The companion source explanation report starts from the same public model via
`membrane.explain()` or `axs.membranes.explain(membrane)` and summarizes source
sections, equation dependencies, unit roles, source outputs, generated backend
targets, cache identity, retained generated assignments, and intermediates
pruned from generated `model_step` targets.
The source compiler now validates equation dependencies before lowering: local
equations may be written out of order, then topologically ordered for generated
code; missing symbols, duplicate assignments, export duplicates/overlap,
dependency cycles, unsupported helpers/statements, and unit/default mistakes
are reported with source location context where available.

## Solver Integration

The Model IR should lower to a model-step contract, not to an opaque external
function call. The solver should be able to see:

- gate/rate calculations;
- current calculations;
- total conductance or linearization terms;
- state updates;
- requested observables.

That makes it possible to fuse membrane work with the cable step and to prune
unrequested outputs before backend lowering.

The first fused JAX target should keep the same membrane terms for both
single-cable and double-cable assembly:

```text
model_step(Vm, state, dt, params, requested_outputs)
-> state_next
-> total_outward_current
-> total_conductance
-> conductance_reversal_sum
-> explicit_outward_current
-> correction_current
-> requested_observables
```

Single-cable and double-cable lowering should differ in cable assembly and
linear-solver shape, not in the scientific membrane semantics.

## Testing Plan

Add tests in layers:

- IR node construction and immutability;
- unit/type/shape validation failures;
- canonical serialization/hash stability;
- model-step NumPy interpretation for tiny deterministic inputs;
- JAX lowering equivalence against the NumPy interpreter;
- equivalence against current built-in behavior for small Passive and HH cases;
- cache-key stability and invalidation;
- guardrails preventing JAX primitives from entering Model IR definitions.

## Exit Criteria

P7 is not complete until:

- built-in membrane families can be represented as Model IR;
- users can author at least scalar HH-style custom membrane models as ordinary
  Python equation source compiled by AxonScope;
- custom model examples cover successful simulation, recording/plotting, and
  readable validation failures;
- validation catches common scientific mistakes;
- canonical hashes are stable;
- a NumPy model-step interpreter exists;
- JAX lowering runs equivalent model-step tests;
- solver integration has a clear path to use Model IR without changing public
  axon APIs;
- `axs.runtime.numpy` remains reserved until a future bonus NumPy/SciPy reference
  runtime implements executable behavior through the same public
  `AxonSimulation` lifecycle.
