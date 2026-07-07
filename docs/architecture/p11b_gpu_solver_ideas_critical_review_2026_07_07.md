# P11B GPU Solver Ideas Critical Review - 2026-07-07

This note is a GPU-only critical reading of the older solver idea documents.
It is not a new roadmap by itself. Its purpose is to decide which ideas still
deserve benchmark work now that P11B has fresh real double-cable GPU evidence.

Reviewed documents:

- `ideas/axonscope_gpu_tridiagonal_solver_literature_synthesis.md`
- `ideas/axonscope_double_cable_exact_gpu_solver_roadmap.md`
- `ideas/axonscope_single_double_cable_gpu_solver_options_with_precompute.md`

Current evidence anchor:

- P100 MRG hot-step, `Naxons=512`, actual `Nx=89`, fp32, observer-only:
  fused one-step is about `0.488-0.499 ms`; isolated `pcr_soa_batched` is about
  `0.466-0.499 ms`.
- GPU lowering: `pcr_soa_batched` is `2267` optimized-HLO lines, `7` fusions,
  `184` gathers, `117` selects, no transposes.
- GPU fusion I/O estimate: repeated PCR/SoA `loop_select_subtract*` fusions
  each carry about `3.82 MiB` of HLO-shaped inputs and `3.82 MiB` of tuple
  outputs, about `7.65 MiB` estimated I/O per fusion.

## Executive Decision

For the immediate P11B GPU work, keep the active target narrow:

1. Optimize current exact GPU PCR/SoA lowering and stage state pressure first.
2. Do not reopen public solver policy or model-specific runtime paths.
3. Keep Thomas/PTA/custom-kernel ideas as benchmark-only follow-ups, not as the
   next runtime change.
4. Treat MRG as the realistic validation workload for generic solver/compiler
   improvements, not as a reason to specialize the runtime.

## Document 1 - Literature Synthesis

The literature synthesis remains useful for its engineering principles:

- optimize many short independent systems, not one fiber;
- use SoA rather than tiny dense `2x2` block arrays;
- treat layout and memory coalescing as first-order GPU issues;
- compare solver-only and end-to-end behavior separately;
- preserve exact double-cable solving as the reference path.

The document's main recommendation, "test PTA-style batched block Thomas first",
is now only partially current. It was reasonable before the P11B evidence, but
the present GPU path is already clearly PCR/SoA-first for real MRG double-cable
hot steps. A new Thomas/PTA prototype may still be valuable, especially through
a custom-kernel route, but it should not displace the current JAX PCR/SoA
inspection.

Critical update:

- `PTA_BLOCK_THOMAS_*` should remain a benchmark-only candidate family.
- The immediate work should not be "Thomas versus PCR from scratch"; it should
  be "why do PCR/SoA stages carry large fusion tuples, and can that state be
  reduced or staged differently?"
- The literature's memory-layout warning is still directly relevant, but now
  the concrete object to inspect is the current PCR/SoA `loop_select_subtract*`
  fusion family.

## Document 2 - Double-Cable Exact GPU Solver Roadmap

This roadmap is explicitly archival, and much of it has already been tested.
The useful status is:

- Active and relevant:
  - current public solver choices stay `auto`, `thomas`, `pcr`, `pcr_soa`,
    `pcr_adaptive`;
  - GPU `auto` remains PCR/SoA-oriented for now;
  - batch-native `solve_block_tridiagonal_2x2_pcr_soa_batched(...)` is the
    current exact JAX GPU path worth inspecting.
- Standby only:
  - `thomas_batched` was not a GPU steady-state win versus the existing Thomas
    route;
  - `assoc_backward` improved Thomas-family solves, but did not beat PCR/SoA
    generally on JAX 0.10.2 P100;
  - `jax_triton_block_thomas` had strong T4 performance signals, but validation
    did not clear the agreement gate and the code now lives in legacy/archive.
- Do not reopen without a new hypothesis:
  - `pcr_soa_nomask`, `pcr_soa_shift`, `pcr_soa_transposed`,
    `pcr_soa_padded`, simple PCR/Thomas hybrids, `layout_auto`, and JAX refs
    were neutral or slower in earlier evidence;
  - dense associative transfer was unstable;
  - current-stack Pallas on P100/T4 was blocked by toolchain/hardware lowering
    constraints.

Critical update:

- The roadmap already contains enough evidence to avoid another broad solver
  zoo sweep.
- The next JAX-native GPU candidate should change the PCR/SoA stage state or
  staging model, not just remove masks/gathers, transpose arrays, add padding,
  or wrap the same body in a different JIT layout contract.
- Custom kernels are a later line. Before reviving `jax-triton` or FFI, the
  validation oracle must separate solver numerical agreement from
  threshold-sensitive spike timing.

## Document 3 - Single/Double-Cable Options With Precompute

This document is the most relevant for current production cleanup, but many
items are already partially done by P11B:

- Runtime/preparation caches now cover batch runtime, static runtime,
  prepared cohorts, recording plans, footprints, and double-cable membrane and
  extracellular stacking.
- Diameter rounding and row-cache work increased cache hits for realistic
  different-diameter cohorts.
- Factorized and cached extracellular preparation is already a major part of
  the current implementation.
- The gated/leak membrane backend and generated-code cache address part of the
  "partial precompute for active membranes" direction.

Still relevant:

- Keep separating static geometry/linear-system terms from dynamic membrane and
  RHS terms.
- Build benchmark evidence around partial active-node/static-row reuse only if
  it reduces the fused hot step or HLO size, not just isolated preparation
  cost.
- Store future GPU-specific packed layouts in backend-owned preparation data,
  never in public model or simulation APIs.

Less relevant for the immediate GPU slice:

- Full Thomas pre-factorization is mainly useful for passive/linear benchmark
  cases. It is not the main path for fully active MRG-like double-cable nodes.
- Scheduler coalescing, amplitude batching, and high-level policy changes may
  matter later, but they should not hide the current low-level PCR/SoA fusion
  pressure.

Critical update:

- Precompute is no longer an abstract future direction. The remaining useful
  question is narrower: can static structure reduce the live PCR/SoA stage
  state or the data moved between GPU fusions?

## What To Do Now

Immediate GPU-only benchmark slice:

1. Add a PCR/SoA stage-state audit that reports, per stride, the carried arrays,
   output tuple shape, estimated I/O, and which values are structurally required
   by the next stride. Implemented in
   `benchmark/analysis/pcr_soa_stage_state_audit.py`; the first P100-shape
   offline report is under `benchmark/results/p11b_gpu_pcr_soa_stage_state_audit`.
   For `B=512`, actual `Nx=89`, fp32, the algorithmic PCR/SoA state is 14
   batch-space arrays (`~2.43 MiB`), while optimized HLO rows commonly expose
   22 fusion outputs and up to `~7.65 MiB` estimated fusion I/O.
2. Prototype one benchmark-only stage-state variant that aims to reduce live
   PCR output state without changing public routing. Candidate directions:
   carry fewer structurally redundant block components, split a stage to reduce
   tuple pressure only if the extra launch is measured, or recompute cheap
   intermediates when it removes large carried arrays.
3. Validate on a tiny local correctness case against current PCR/SoA and Thomas,
   then run a P100 GPU lowering/hot-step gate on the existing MRG shape.
4. Keep or reject based on full evidence: HLO tuple/I/O pressure, isolated
   block-solve time, fused one-step time, and real curve-level behavior if the
   hot-step result is promising.

Do not do now:

- no new public solver enum or `auto` policy;
- no membrane-model-specific runtime branches;
- no broad rerun of old PCR micro-variants;
- no Pallas/P100 work on the current stack;
- no revival of split iterative solvers for exact runtime routing.

## Next Acceptance Gate

The next GPU slice is complete when a benchmark artifact answers:

- did the candidate reduce the dominant PCR/SoA fusion tuple or estimated I/O?
- did that reduction produce a hot-step or one-step win on P100?
- did correctness match current exact solvers on generated double-cable inputs?
- is the change generic to solver/compiler structure rather than MRG-specific?

If the answer is no, keep the candidate as benchmark-only evidence and move to
the next low-level family rather than promoting it into runtime.
