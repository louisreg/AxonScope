# P11B Double-Cable Solver And Compiler Audit - 2026-07-07

This note is a P11B working decision record. It reconciles the current code,
fresh benchmark evidence, and the older GPU solver idea documents before any
new double-cable solver implementation work.

## Scope

Read and checked:

- `ideas/axonscope_gpu_tridiagonal_solver_literature_synthesis.md`
- `ideas/axonscope_double_cable_exact_gpu_solver_roadmap.md`
- `ideas/axonscope_single_double_cable_gpu_solver_options_with_precompute.md`
- Current JAX solver dispatch, double-cable batch kernels, runtime
  preparation, membrane compiler/lowering, generated-code cache, and benchmark
  reports around P11B.

Fresh evidence anchor:

- Real workflow solver-choice report:
  `benchmark/results/p11b_real_solver_choice_cpu_gpu_ff78c4f/solver_choice/solver_choice_report.md`
- Real double-cable stage profile reports:
  `docs/architecture/p11b_real_double_cable_stage_profile_2026_07_07.md` and
  `docs/architecture/p11b_real_double_cable_stage_profile_512_2026_07_07.md`
- GPU solver lowering report:
  `docs/architecture/p11b_double_cable_solver_lowering_audit_2026_07_07.md`

That report confirms the current next target:

- CPU double-cable warm runs are solver dominated once preparation confounders
  are removed. CPU `auto` resolves to Thomas and forced `pcr_soa` is far worse.
- GPU double-cable warm runs are solver-sensitive. GPU `auto` resolves through
  `pcr_adaptive`/PCR-SoA and forced Thomas is much worse.
- GPU is not yet pure solver time. Solver work must stay separated from launch,
  compile, finalization, result assembly, and remaining runtime preparation.

## Current Code Shape

The public double-cable solver surface is still narrow and acceptable:

- `BatchOptions.double_cable_block_solver` supports only `auto`, `thomas`,
  `pcr`, `pcr_soa`, and `pcr_adaptive`.
- `auto` resolves to Thomas on CPU-like platforms and `pcr_adaptive` on GPU-like
  platforms.
- In the current JAX batch runtime, `pcr_adaptive` selects batch-native
  `pcr_soa` for batches up to the configured threshold, otherwise matrix PCR.

The runtime/preparation side already contains important P11B fixes:

- Batch runtime, static runtime, prepared-cohort, recording-plan, footprint, and
  single-cable forcing caches are present.
- Double-cable extracellular stacking now builds NumPy rows on host, caches by
  cable signature, and transfers batched blocks instead of many per-row arrays.
- Double-cable membrane stacking has a structural gated/leak fast path that
  keeps model-specific simplification in the compiler/backend layer rather than
  in user-workflow branches.
- `JaxMembraneProgram` memoizes derived static values such as `g_bar`, `E_rev`,
  membrane state specs, and static signatures. This was the key fix for the
  previous `runtime.prepare.stack_membrane` bottleneck.

The membrane compiler/lowering side is also more mature than the old roadmap
assumes:

- Class-based membrane source is parsed into internal Model IR.
- Generated JAX/NumPy artifacts are cached with source/compiler/schema/helper
  identity.
- `inspect_generated_code()` and `explain()` expose generated files, graph
  hashes, optimized graph hashes, cache keys, target metadata, and recording
  output plans.
- Public composite names are component-qualified for gates, states, and generic
  observables. Current/conductance aggregation is explicit and separate.
- Generated model steps are active for single-source membranes; multi-source
  composites still fall back to interpreter-style lowering.

## Critical Reading Of Solver Ideas

### Keep Active

1. Exact double-cable solving remains the right target.

   The scientific and numerical direction still favors exact block systems for
   double-cable MRG-like workloads. Split or approximate physiology-changing
   routes should not enter runtime defaults.

2. CPU Thomas / GPU PCR-SoA is the current supported policy.

   Fresh real workflow evidence confirms the existing policy direction. CPU
   should keep Thomas-style solves for now. GPU should keep PCR-SoA-style solves
   until a better exact GPU route survives realistic benchmarks.

3. Compiler/backend precompute remains highly relevant.

   Static coefficient extraction, row caching, footprint factorization,
   gated/leak stacking, and generated-code pruning are the right kind of work.
   They belong behind runtime/compiler abstractions, not as model-family
   branches in public workflows.

4. A strong batched Thomas-family GPU prototype is worth considering, but only
   with a new integration gate.

   The literature note argues for a memory-coalesced, many-small-systems
   Thomas/PTA-style GPU baseline. The old Triton/JAX-Triton evidence also found
   a real solver-only and partial E2E win, but validation/routing was not ready.
   Reopening this line should start as benchmark-only and must pass agreement
   gates before any `auto` policy change.

### Keep As Standby

1. Associative backward.

   It was correct and interesting as a Thomas-family optimization, but not
   better than the active GPU PCR-SoA path in the documented regime. Keep it as
   a standby candidate if a future trace shows a Thomas-like GPU region.

2. Pallas/CuTe/custom kernels.

   Prior Pallas attempts hit current-stack/hardware limitations on P100/T4.
   CuTe requires newer GPUs. Custom kernels should wait until a narrow,
   evidence-backed candidate justifies the integration cost.

3. Chunked amplitude batching and high-level workflow scheduling.

   These may matter later, but they are high-level optimization axes. Keep them
   in the TODO and do not use them to mask low-level solver/runtime costs.

### Do Not Reopen Without New Evidence

1. Split iterative solvers.

   They showed performance promise but failed the exact/physiology agreement
   gate. They are not candidates for the exact double-cable runtime path.

2. Associative transfer dense.

   The archived result marks it as numerically unstable. It should remain a
   diagnostic idea unless a genuinely new stable formulation is proposed.

3. PCR-SoA variants already tried: transposed, padded, nomask, shift, simple
   hybrids, and reference/diagnostic layouts.

   These are documented as neutral or slower in the old campaigns. Repeating
   them inside active runtime work would add churn without a new hypothesis.

## Compiler And Lowering Audit

The next optimizer work should treat generated membrane code as part of the
low-level system, not as a black box. Current likely useful gates are:

1. Measure real generated membrane work inside the double-cable time step.

   The synthetic solver-stage profiler already includes `vm_gate_update`,
   `assemble_system`, `block_solve`, observer write, and a compact full-step
   proxy. The next useful extension is to tie those stages to the real
   generated MRG/AxNode path and confirm how much time is gate/rate/conductance
   work versus the linear solve.

2. Lower generated conductance terms directly when recording does not need full
   current/conductance matrices.

   P10 already defined solver-required versus recording-requested outputs.
   P11B should benchmark whether direct generated `Gm`/`GE` terms reduce hot
   path work or HLO size for double-cable workloads.

3. Keep model-specific simplifications structural.

   The current gated/leak stack is the right pattern: classify capabilities
   from compiled membrane structure and encode row parameters. Do not add
   runtime branches that know about a specific membrane model family.

4. Improve composite/generated boundaries only with measurements.

   Single-source generated model steps are active. Multi-source composites
   still fall back. A composite generated program may help, but only if the
   real double-cable MRG path proves membrane lowering is material after the
   solver split.

5. Prune unrequested diagnostics/observables before transport and result
   assembly.

   This should flow from the compiler output-pruning plan and recording policy,
   not from ad hoc runtime conditions.

## First MRG Hot-Step Decomposition

The first compiler/solver split gate is now implemented in
`benchmark/analysis/double_cable_real_stage_profile.py`. The profiler still uses
real public MRG/double-cable workloads and prepared backend inputs, but its
report now exposes:

- hot-step group shares for membrane, forcing, assembly, solver, and observer
  stages;
- primary block-solver share relative to the one-step proxy;
- membrane backend metadata: backend kind, model kind, max gates/channels,
  backend branches, and gated/leak compartment counts where available;
- MRG gated/leak-stack diagnostic rows:
  `membrane_gate_update_gated_only`,
  `membrane_conductance_terms_gated_only`, and
  `membrane_conductance_terms_mask_mix`.

The lowering audit now includes all `membrane_*` sub-stages when
`--include-membrane-stages` is enabled, and `hlo_fusion_summary.py` recognizes
the new stage names.

Local CPU smoke artifacts:

- `benchmark/results/p11b_mrg_hot_step_profile_smoke_diff`
- `benchmark/results/p11b_mrg_hot_step_lowering_smoke`

Small-smoke result, for `Naxons=8`, different diameters, requested `Nx=31`,
actual kernel `Nx=56`, fp32, observer-only, and `pcr_soa`: the prepared MRG
path uses `GatedLeakStackMembraneBackend` with one backend branch, `6` gated
compartments and `50` leak compartments. The isolated PCR/SoA block solve is
about `53%` of the one-step proxy; full membrane gate+conductance work is about
`15%`; assembly, observer write, and forcing are each below that on this CPU
smoke. This is not a speed claim, but it confirms that the current next GPU
benchmark should inspect both PCR/SoA and generated MRG membrane lowering before
opening a new solver route.

## Runtime Source Hygiene

Several historical solver candidates still live in active JAX solver source as
internal functions, even though they are not public solver choices. The public
surface is clean, so this is not urgent, but it is a real cleanup item:

- keep only production solver implementations in active runtime modules;
- move failed or benchmark-only variants to `benchmark/`, `benchmark/legacy/`,
  or a clearly named internal diagnostic module;
- keep tests for active choices and archival evidence for rejected choices.

This is housekeeping. It should not block the next measurement/prototype gate.

## Proposed Next Sequence

1. Continue current PCR/SoA inspection before adding a solver route.

   The real stage profiler now shows GPU PCR/SoA as the immediate low-level
   target, but the lowering audit shows `pcr_soa_batched` and
   `pcr_soa_vmap` become nearly identical after XLA GPU optimization. The next
   useful step is therefore not a vmap-vs-batched runtime route; it is a
   deeper inspection of current PCR/SoA fusion bodies, memory layout, and
   one-step composition.

2. Add a CPU/generated-membrane lowering view for the different-diameter case.

   CPU remains Thomas-first, but different-diameter CPU stage profiles still
   show visible generated membrane, conductance, and assembly costs. Keep this
   in the compiler/backend layer rather than adding model-family branches to
   runtime policy.

3. Add a focused exact solver candidate only after that inspection.

   The most plausible first candidate is a benchmark-only batched
   Thomas-family GPU route inspired by the literature/PTA note and prior
   Triton/JAX-Triton results. It must start outside public `auto` routing and
   compare against current GPU PCR-SoA on solver-only and real curve workloads.

4. Add correctness gates before keeping any solver candidate.

   Compare against Thomas float64 on generated real double-cable systems,
   subthreshold traces, suprathreshold traces with margin, activation counts,
   and threshold/recruitment outcomes. Do not use spike-timing-sensitive traces
   alone as the first rejection/acceptance oracle.

5. Revisit GPU fusion/launch overhead only after solver candidate evidence.

   The latest maps show GPU still has non-solver boundaries. If solve time
   shrinks, kernel fusion/launch/finalization can become the next bottleneck.

6. Defer policy.

   Do not choose adaptive time-chunk or high-level amplitude batching defaults
   in this step. Keep them tracked as later workflow optimizations after the
   low-level solver/compiler map is stable.

## Acceptance For The Next Slice

The next P11B slice is complete when:

- a tracked benchmark/report can separate real double-cable generated membrane
  work from system assembly, block solve, observer write, and result
  finalization;
- the report includes CPU/GPU, platform metadata, memory mode, git metadata,
  and cold/warm labels;
- the TODO names the first solver candidate gate or explicitly says no solver
  candidate should be attempted yet;
- no public solver routing or workflow policy changes are made without fresh
  real-workflow artifacts.
