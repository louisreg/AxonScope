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
- Conductance/reversal inference still expects the readable linear form
  `I_x = g_x * (Vm - E_x)`. Currents whose conductance or reversal cannot be
  inferred need either explicit public syntax or fail-fast diagnostics.
- Mechanism boundaries are preserved in source sections, but optimization and
  generated-program reports should make those boundaries more visible.
- Generated-artifact identity is still mostly source/cache oriented. The full
  target-specialized identity should include graph hashes, lowering key, static
  shapes, recording policy, dtype/precision, optimization level, helper
  versions, and dependency hashes.
- Recording-aware output pruning is visible in reports, but still needs to be
  treated as a compiler plan before backend lowering.
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

## Next P10 Slices

1. Audit concentration/current conversion formulas. Keep model-family-specific
   electrochemical helpers local, following the Schild pattern in
   `src/axonscope/membranes/models/schild_common.py`.
2. Add explicit current metadata syntax or a stricter diagnostic for
   non-inferable conductance/reversal terms.
3. Extend `explain()` and generated-artifact identity around optimization,
   recording policy, dtype/precision, and target-specialized lowering.
