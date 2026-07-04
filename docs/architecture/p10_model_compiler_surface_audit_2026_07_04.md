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

- Rejected-source diagnostics have line/column support for many failures, but
  mutation, loops, object construction, hidden globals, dynamic imports, and
  side-effect-oriented code still need clearer user messages.
- The helper surface is not yet fully flat. Scalar helpers exist for common
  math and gate-rate conversion, but Nernst/concentration/current conversion
  helpers need a deliberate unit contract before becoming public.
- The TODO name `rates_from_tau_inf` conflicts with the implemented scalar
  helpers `alpha_from_inf_tau` and `beta_from_inf_tau`. P10 should decide
  whether to keep the explicit scalar names, add tuple-valued syntax, or expose
  both with one canonical public story.
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

## First Slice Done

The scalar `boltzmann(x, midpoint, slope)` helper is now supported across the
public membrane math surface, internal intrinsic registry, semantic validation,
NumPy interpreter, JAX lowering, generated NumPy/JAX model-step code, and unit
tests. The convention is:

```text
1 / (1 + exp((x - midpoint) / slope))
```

The signed slope keeps activation and inactivation forms explicit without
adding duplicate helper names.

## Next P10 Slices

1. Tighten diagnostics for rejected source constructs.
2. Decide the public rate-helper story around `alpha_from_inf_tau`,
   `beta_from_inf_tau`, and possible `rates_from_tau_inf` tuple syntax.
3. Define Nernst and concentration/current conversion helpers with units before
   exposing them.
4. Add explicit current metadata syntax or a stricter diagnostic for
   non-inferable conductance/reversal terms.
5. Extend `explain()` and generated-artifact identity around optimization,
   recording policy, dtype/precision, and target-specialized lowering.
