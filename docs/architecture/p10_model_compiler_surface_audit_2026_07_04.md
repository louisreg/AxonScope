# P10 Model/Compiler Surface Audit - 2026-07-04

This note records the first P10 audit pass after the post-P7 class-based
membrane cleanup. It is a working architecture note, not a user tutorial.

## Scope Checked

- Public membrane authoring in `src/axonscope/membranes/`.
- Built-in source-backed models in `src/axonscope/membranes/models/`.
- Internal compiler/runtime semantics in `src/axonscope/model_ir/`.
- JAX model lowering in `src/axonscope/backends/jax/model_ir_lowering.py`.
- JAX membrane execution facade in `src/axonscope/backends/jax/membrane_program.py`.
- Generated NumPy/JAX model-step cache code in `src/axonscope/model_ir/source.py`.
- Custom membrane example and model-codegen benchmark surface at a high level.

## Current Shape

- The public user vocabulary is mostly aligned with the target direction:
  users write `axs.membranes.Model` classes, parameters, equations, gates,
  currents, observables, and state updates.
- Model IR remains internal. Public docs and examples no longer need users to
  construct internal graph objects for membrane authoring.
- Built-in membrane truth lives in the model source files. The compiler and
  JAX backend consume the compiled semantic graph instead of reintroducing
  built-in formulas.
- The generated-code cache records source/compiler/schema/helper identity and
  emits target-specific NumPy/JAX modules with stable output names.
- `explain()` already exposes source sections, symbols, generated targets,
  cache state, source hashes, output pruning, and target summaries.

## Gaps

- Rejected-source diagnostics now give targeted source-location messages for
  mutation, loops, statement-level conditionals, imports inside equation
  functions, arbitrary NumPy/JAX calls, object construction, hidden globals,
  I/O, and side effects.
- The helper surface is intentionally conservative. Scalar helpers exist for
  common math, and tau/inf gate conversion is exposed publicly as
  `rates_from_tau_inf`. Model-specific formulas should stay in the model source
  or in a model-family common module rather than becoming global helpers.
- `rates_from_tau_inf` is now the canonical public tau/inf helper. Model source
  uses tuple assignment, while the compiler lowers that syntax to scalar
  internal alpha/beta expressions.
- Conductance/reversal inference still accepts the readable linear form
  `I_x = g_x * (Vm - E_x)`. Currents whose conductance or reversal cannot be
  inferred can now declare explicit source metadata with
  `@currents(conductances={"I_x": "g_x"}, reversals={"I_x": "E_x"})`.
- Mechanism boundaries are preserved in source sections, compiled metadata, and
  generated-program reports. `explain()` now aggregates `@mechanism(...)`
  sections with their produced assignments and external dependencies.
- Generated-artifact identity is still mostly source/cache oriented. The full
  target-specialized identity should include graph hashes, lowering key, static
  shapes, recording policy, dtype/precision, optimization level, helper
  versions, and dependency hashes.
- Recording-aware output pruning is visible in reports, but still needs to be
  treated as a compiler plan before backend lowering.
- Semantic validation now checks current linearization term units,
  source-backed metadata consistency, and component-qualified public names for
  composite states/observables. Generic recording outputs use qualified
  component names; physical currents/conductances keep explicit aggregate
  semantics.
- Backend-neutral optimization closeout remains open: CSE, unused diagnostic
  pruning, optimized-graph hashes, and before/after summaries.
- JAX-specific fusion closeout remains open: generated conductance terms,
  state updates, diagnostics, requested-observable pruning, composite generated
  programs or explicit fail-fast boundaries, and avoiding unrequested
  intermediate transport.

## Completed Slices

The candidate scalar `boltzmann(x, midpoint, slope)` helper was rejected for
the public surface because no built-in model, public example, or benchmark uses
it directly. Source models should keep one-off sigmoid/Boltzmann-like formulas
inline until the same operation is needed by multiple independent model
families.

The tuple helper `rates_from_tau_inf(x_inf, tau)` is now supported in source
model assignments:

```python
alpha_m, beta_m = rates_from_tau_inf(m_inf, tau_m)
```

The compiler lowers the tuple assignment to scalar internal alpha/beta
expressions, so the rest of the Model IR and backend lowering paths remain
scalar.

Rejected Python constructs now fail with specific messages instead of the old
generic unsupported-statement fallback for the common authoring mistakes P10
identified.

Currents whose conductance/reversal terms cannot be inferred from the simple
linear form now have explicit source syntax on `@currents(...)`. Both
`conductances` and `reversals` are required for each current, and the compiler
rejects unknown current names before lowering.

Concentration/current conversion formulas were audited across the active
source-backed built-ins. Tigerholm has dynamic Na/K Nernst terms, Na/K pump
terms, and current-to-concentration factors tied to its diameter and
periaxonal-volume assumptions. Schild 94/97 share Nernst, Ca-pool,
Na/Ca-exchanger, Na/K-pump, and Ca-pump formulas inside one model family. None
of these formulas are shared across independent model families in a way that
justifies a public membrane helper today, so the current policy is:
Schild-family duplication belongs in `schild_common.py`, Tigerholm-specific
logic stays inline in `tigerholm.py`, and public helpers should wait for at
least two independent families needing the same operation.

Mechanism sections are now preserved as first-class report metadata:
`source_sections` records every source section, `source_mechanisms` records
named `@mechanism(...)` groups, and `explain()` prints a `mechanisms:` block
with assignments plus dependencies outside each mechanism boundary. This keeps
the authoring shape visible for future optimization/fusion work without making
Model IR a user-facing concept.

Model IR semantic validation now rejects malformed source-backed metadata and
wrong current linearization terms. `source_outputs` must be internally
consistent, source provenance must match the top-level source metadata, source
sections/mechanisms must have well-formed names and dependencies, and current
conductance/reversal expressions must carry `mS/cm2` and `mV` respectively.

Composite recording identity is explicit. Public `Composite({...})` mappings
provide component labels, non-ambiguous sequences derive labels from model
kind, and duplicate model kinds in a sequence fail early. Public gate, state,
and generic-observable names are qualified as `component_label.name`; current
and conductance groups remain the only automatic duplicate-name aggregates.

## Next P10 Slices

1. Extend `explain()` and generated-artifact identity around optimization,
   recording policy, dtype/precision, and target-specialized lowering.
