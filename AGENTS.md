---
name: AxonFleet Agent
description: Working guide for the AxonFleet peripheral nerve simulation repository
instructions: |
  You are working in AxonFleet, a pre-release Python framework for
  one-dimensional peripheral nerve axon simulations.

  ## Source Of Truth

  - `GUIDELINES.md` is the project philosophy and target architecture
    reference. Update it only when product boundary, public API direction, or
    backend architecture changes.
  - `todo.md` is the living operational checklist. Read it before cleanup,
    docs, benchmark, or roadmap work. Do not remove unfinished tasks unless
    they are completed, rejected, or moved to a named tracking document.
  - `src/`, `tests/`, runnable `examples/`, and fresh benchmark/validation
    outputs are the implementation truth for current behavior.
  - Markdown docs can be uneven. Before treating a claim as current, compare it
    with source, tests, and active examples.

  ## Product Boundary

  AxonFleet owns 1D axon models, membrane dynamics, stimulation along axons,
  simulation execution, recording, analyses, thresholds, recruitment, sweeps,
  validation, and performance evidence.

  AxonFleet does not own nerve/fascicle geometry, segmentation, 3D axon
  trajectories, anatomical frames, electrode CAD, surgical placement, or FEM
  field solving. External packages should provide sampled extracellular
  footprints; AxonFleet combines those footprints with temporal stimuli and
  runs the cable and membrane dynamics.

  Intrinsic axon coordinates belong in AxonFleet. World coordinates,
  orientation, trajectories, and field-generation geometry stay outside the
  core architecture.

  ## Cleanup Policy

  AxonFleet is pre-release with one active user. Prefer clean convergence over
  compatibility shims, aliases, deprecated wrappers, or parallel old/new public
  paths. The target is one concept, one public name, one execution path, and
  one canonical public result model.

  Performance work follows the same convergence rule. Prototype alternatives
  may exist only in benchmark tooling while they are being measured; a promoted
  optimization replaces the production path it improves instead of adding an
  `optimized`/`legacy` branch, hidden fallback, duplicate kernel route, or
  second public option. If the optimized design requires a material change to
  canonical results, serialized output, user-visible progress, or the public
  workflow/UI, stop and discuss that product change with the user before
  implementing it.

  ## Current Public Workflow

  The canonical user path is:

  ```text
  membranes + axons + stimulation
          -> AxonInstance or AxonPopulation
          -> AxonSimulation(...).run()
          -> AxonSimulationResult / AxonResultView
  ```

  Prefer `AxonSimulation(...).estimate()` and
  `AxonSimulation(...).inspect()` over standalone root helpers. Public examples
  must use public APIs, not solver/backend internals.

  ## Membrane Direction

  User-facing membrane vocabulary is "membrane model", "equation",
  "parameter", "gate", "current", and "observable". "Model IR" and
  "intermediate representation" are internal compiler/runtime terms.

  Built-in membrane model truth lives in
  `src/axonfleet/membranes/models/`. Each model file owns equations,
  unit-bearing defaults, aliases, and derived parameter logic. Axon templates
  may keep ergonomic model-specific kwargs, but they must forward those kwargs
  to the membrane source compiler instead of owning defaults or formulas.

  ## Runtime Boundary

  Public orchestration enters concrete JAX code through
  `axonfleet.runtime.execution`. Public `simulation.py`, `performance.py`,
  and `inspection.py` must not import `axonfleet.runtime.jax.*` directly.
  Runtime-owned estimate/inspection facts route through `runtime.execution`
  and concrete support under `src/axonfleet/runtime/jax/benchmarking/`.

  Runtime/device/precision public values are `axs.runtime.*`, `axs.Device`,
  `axs.PrecisionPolicy`, and `axs.ExecutionPolicy`. Do not add string-primary
  public APIs such as `"gpu"` or `"float32"`.

  ## Observers And Recording

  The active solver-side observer path is the strict VmRaster route:
  compatible threshold-style definitions may produce
  `observations["vm_raster"]`. Activation, latency, velocity, threshold,
  recruitment summaries, and peak voltage remain result-side post-processing
  unless a dedicated fast path is designed, tested, and benchmarked.

  Do not reintroduce a broad generic solver-side observer fallback.

  ## Examples And Docs

  Examples are executable documentation. Every public option, workflow,
  runtime mode, inspection view, analysis concept, or solver-facing user
  concept should be documented in a runnable example or removed/archived.

  Put introductory scripts in `examples/basic/`, advanced public workflows in
  `examples/advanced/<area>/`, NRV integration examples in `examples/with_nrv/`,
  notebook mini-courses in `examples/tutorials/`, and profiling/benchmark
  material under `benchmark/`.

  Keep proposal and roadmap docs clearly labelled so users do not run future
  API snippets as current behavior.

  ## Validation And Benchmarks

  Fast local checks do not require NRV:

  ```bash
  git diff --check
  MPLBACKEND=Agg python -m compileall -q src tests/unit
  MPLBACKEND=Agg python -m pytest -q tests/unit --tb=short
  ```

  Run `tests/nrv` only for numerical behavior, solver semantics, stimulation
  semantics, membrane dynamics, or NRV integration changes. Record NRV results
  only after a fresh NRV-ready run.

  Re-run hotpath/realistic benchmarks only when making timing or memory claims.
  Use `benchmark/README.md` as the benchmark surface map.

keywords:
  - nerve simulation
  - computational neuroscience
  - JAX
  - solver
  - batch processing
  - GPU acceleration
  - validation

applyTo:
  - language: python
    patterns:
      - src/axonfleet/**/*.py
      - examples/**/*.py
      - benchmark/**/*.py
      - tests/**/*.py

rules:
  - name: Follow project guidelines
    description: Use GUIDELINES.md for product and architecture direction.
  - name: Keep TODO current
    description: Update todo.md checkboxes only after source, docs, examples, and relevant tests are checked.
  - name: Prefer clean APIs
    description: Do not preserve compatibility aliases or shims unless explicitly requested.
  - name: Replace optimized paths
    description: Promote optimizations by replacing the existing production path; discuss material result, output, or UI changes before implementation.
  - name: Respect runtime boundaries
    description: Public simulation, estimate, and inspection modules route through axonfleet.runtime.execution.
  - name: Validate proportionally
    description: Use fast unit checks for cleanup, NRV for numerical changes, and benchmarks for performance claims.

helpfulLinks:
  - path: GUIDELINES.md
    description: Product philosophy and target architecture.
  - path: todo.md
    description: Living cleanup, benchmark, docs, and roadmap checklist.
  - path: README.md
    description: Current user-facing package entry point.
  - path: docs/validation.md
    description: Fast checks and NRV validation policy.
  - path: examples/README.md
    description: Executable learning path.
  - path: benchmark/README.md
    description: Supported benchmark surface and outputs.
---

# AxonFleet Development Agent

This file is intentionally short. It is a working guide for agents, not a
roadmap ledger. Keep detailed architecture direction in `GUIDELINES.md`, active
tasks in `todo.md`, and historical plans under clearly labelled docs.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
