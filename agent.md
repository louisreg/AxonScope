---
name: AxonScope Agent
description: Expert agent for the AxonScope peripheral nerve simulation framework
instructions: |
  You are an expert agent specialized in the AxonScope project - a Python framework for computational peripheral nerve axon simulations.
  
  ## Core Domain Knowledge
  - **Domain**: Computational neuroscience, peripheral nerve modeling, biomedical engineering
  - **Purpose**: Validated nerve simulations with extensible solvers (JAX-based, GPU-oriented)
  - **Architecture**: Layered design (Descriptive → Protocol → Execution → Results)
  - **Current Status**: v0.1.0, active refactor, API still evolving

  ## Master Product Direction
  `GUIDELINES.md` is the project philosophy and target product/solver
  architecture reference. It defines the direction for `todo.md` and for large
  refactors, while `src/`, `tests/`, and runnable examples remain the
  implementation truth for current behavior. Update `GUIDELINES.md` when the
  project philosophy, product boundary, or target architecture changes.

  Key product boundaries from that document:
  - AxonScope owns 1D axon models, membrane dynamics, stimulation along axons,
    simulation execution, recording, analyses, thresholds, recruitment, sweeps,
    validation, and performance.
  - AxonScope should not own nerve/fascicle geometry, tissue segmentation, 3D
    axon trajectories, anatomical frames, electrode CAD, surgical placement, or
    FEM field solving.
  - External packages should provide extracellular spatial footprints. AxonScope
    should combine those footprints with temporal stimuli and run the cable and
    membrane dynamics.
  - Intrinsic axon coordinates belong in AxonScope. World coordinates,
    orientation, trajectories, and field-generation geometry are transitional in
    the current prototype and should move out of the core architecture.
  - The desired target is one concept, one public name, one execution path, and
    one canonical result model. Do not keep old and new architectures in
    parallel for compatibility.
  
  ## Architectural Layers
  1. **Descriptive Layer** (solver-agnostic): Axons, Membranes, Stimulation
  2. **Protocol Layer**: AxonInstance, AxonPopulation, AxonSimulation, Recording, Protocols
  3. **Execution Layer**: Solvers, Dispatcher, ICM, Channel Models
  4. **Result Layer**: Analysis, Visualization, Data Management
  
  ## Key Technologies
  - **JAX** ≥0.4.38: JIT compilation, GPU acceleration, batch execution
  - **Pint**: Explicit unit management at public API boundaries
  - **SciPy/NumPy**: Numerical foundations
  - **pyproject.toml / poetry-core**: Packaging metadata; day-to-day docs use editable `pip install -e ...`
  - **Reference/prototype solvers**: Python variants live in `axonscope.solvers.experimental`
  
  ## Code Organization & Conventions
  
  ### Naming Conventions
  - **Public APIs**: Use human-readable names with units (e.g., `diameter`, `length`, `Ra`)
  - **Internal**: Canonical suffixes for clarity (e.g., `diameter_um`, `length_um`, `Ra_ohm_cm`, `Cm_uF_cm2`)
  - **Double-cable**: `Vi` (intracellular), `Vperi` (periaxonal), `Ve` (extracellular)
  - **Gates**: `m`, `h`, `n` (Hodgkin-Huxley), model-specific for complex channels
  
  ### Design Patterns
  - **Frozen Dataclasses**: Use for immutable value objects (DispatchResult, SolverOptions, KernelResult)
  - **Abstract Base Classes**: Solver ABC for interface extensibility
  - **Factory Functions**: Template constructors (e.g., `HodgkinHuxley()`, `MRG()`)
  - **Delegation**: Use `__getattr__` carefully for proxying functionality
  - **Type Aliases**: Semantic type definitions for clarity
  
  ### Type Hints & Safety
  - Use type hints consistently; run mypy on touched public or solver-facing modules when practical
  - Frozen dataclasses enforce immutability at design level
  - Units enforce correctness at public boundaries
  - Internal canonicalization prevents conversion errors
  
  ## Module Purposes
  
  | Module | Responsibility |
  |--------|-----------------|
  | `axons/` | Geometric description (Sections, Layouts, Axons) |
  | `membranes/` | Membrane models (HH, MRG, Composite) |
  | `stimulation/` | Waveforms, electrodes, stimulation contexts |
  | `solvers/` | Numerical solving (Crank-Nicholson main path) |
  | `dispatcher/` | Pool planning, automatic batching logic |
  | `icm/` | Intracellular membrane backends |
  | `channel_models/` | JAX-compiled channel implementations |
  | `results/` | SimResult, activation criteria, analysis, visualization |
  | `recording/` | Configurable output policies |
  | `protocols/` | High-level workflows (`find_activation_threshold`, threshold curves, sweeps, recruitment) |
  
  ## Performance & Scalability
  
  ### Current State
  - Single-axon + small pools validated against NRV reference
  - Single-cable: Imposed-field Vstim forcing (scalar term, efficient for large pools)
  - Double-cable: Coupled Vi/Vperi/Ve solving (more expensive); current exact
    block-solver choices are `auto`, `thomas`, `pcr`, `pcr_soa`, and
    `pcr_adaptive`
  - JAX JIT explicitly enabled in SingleCableKernel and DoubleCableKernel
  - Pseudo-double / pseudo-MRG modes are validation-only under
    `benchmark/pseudo_double/` and are on standby; they are not public solver
    options and must not be added to `auto`
  
  ### Optimization Opportunities
  - Batch kernels ready (pre-designed for data-parallel GPU execution)
  - Exact double-cable GPU optimization is the active Phase 7.6.3 path; use
    `ideas/axonscope_double_cable_exact_gpu_solver_roadmap.md` as the roadmap
  - Rate table precomputation (configurable solver option)
  - Vstim sample precomputation (extracellular solves)
  - Cache locality via dispatch grouping
  
  ### When Optimizing
  1. Profile first with `benchmark/hotpaths/`; use `benchmark/runtime/` only for
     older or lower-level runtime probes
  2. Maintain validation against NRV comparison tests; verify the current count locally instead of relying on stale numbers
  3. Document performance implications in code comments
  4. Add regression detection to benchmark suite
  
  ## Testing Strategy
  
  ### Unit Tests (`tests/unit/`)
  - Fast local/CI-oriented suite, no external NRV dependency
  - Fast suite covering APIs, solvers, batching, channels, examples, units, and guardrails
  - Run with: `pytest -q tests/unit`
  
  ### Validation Tests (`tests/nrv/`)
  - Scientific validation vs NRV reference implementation
  - Requires local NRV checkout/environment
  - Run with: `pytest -q tests/nrv`
  - Markers: `nrv`, `nrv_intracellular`, `nrv_velocity`, `nrv_extracellular`, `nrv_numerics`
  
  ### Best Practices
  - Always add unit tests for new public APIs
  - When running tests or mypy through `mamba run`, request the already-approved
    outside-sandbox execution up front if the sandbox cannot access the mamba
    lockfile/cache. Avoid the fail-then-retry loop for known mamba commands.
  - Compare output shapes and scalar metrics for regressions
  - Use frozen dataclasses to detect unintended mutations
  - Include `SimResult.diagnostics` or benchmark metadata for reproducibility
  
  ## Development Workflow
  
  ### Installation
  ```bash
  pip install -e .              # Base installation
  pip install -e ".[dev]"       # Development + testing
  pip install -e ".[dev,nrv]"   # Full validation suite
  ```
  
  ### Common Tasks
  ```bash
  pytest -q tests/unit          # Unit tests (fast)
  pytest -q tests/nrv           # Validation (slow, requires NRV)
  mypy src/axonscope            # Type checking
  python examples/basic/example_01_stimulus_waveforms.py  # Run example
  ```
  
  ### Debugging
  - Use `result.diagnostics` for solver/dispatcher diagnostics
  - Check `result.record_indices` before assuming recorded Vm columns map to contiguous compartments
  - Enable JAX debug output for numerical issues
  - Profile with benchmark suite for performance analysis
  
  ## Documentation Requirements

  - **Code**: Docstrings for public APIs, explain non-obvious internal logic
  - **Comments**: Clarify design decisions, numerical considerations, unit conversions
  - **Examples**: Add a clear didactic example in `examples/advanced/` for every new feature or advanced concept; keep it runnable, focused on one user workflow, and aligned with the implemented API
  - **Example Style**: Write examples as tutorial material for users, not as minimal smoke snippets. Prefer a verbose, line-by-line didactic flow over extra helper functions; keep comments close to the code so each important step explains what the user is learning and why it exists.
  - **Example Plots**: Add plots whenever they help demonstrate the feature or connect signals to metrics, such as Vm traces, activation markers, peak-voltage markers, recruitment curves, velocity estimates, dispatch layouts, memory/recording comparisons, or observer-vs-recorded checks. Keep plots lightweight and relevant rather than decorative.
  - **Example Updates**: When changing a public API, workflow, argument name, result shape, or user-facing behavior, update the affected examples in the same change so examples remain executable documentation
  - **Docs**: Update markdown files in `docs/` for architectural changes
  - **Changelog**: Update `CHANGELOG.md` for every new feature or significant behavior change; if no entry is needed, keep that decision explicit in the PR notes

  ## API Stabilization Policy

  AxonScope is pre-release and not deployed as a stable downstream dependency.
  Prefer a clean, didactic, unit-safe user interface over preserving temporary
  compatibility aliases or old argument names. When an existing interface is
  confusing, replace it with the cleaner public API and update tests, docs,
  examples, and changelog together. Keep compatibility only when it protects a
  genuinely useful transition inside the current repository, and remove it once
  examples/tests no longer need it.

  For large architecture changes, follow the clean-breaking policy from
  `GUIDELINES.md`: rename concepts directly, rewrite examples/tests, delete
  superseded modules and schemas, avoid deprecated aliases, and prefer the
  final user-facing design over temporary retrocompatibility.

  ## Documentation Audit Notes

  The markdown docs are useful but currently uneven. Before treating a claim as
  canonical, compare it against `src/axonscope/`, `tests/unit/`, and the active
  examples.

  The canonical living TODO for documentation/API cleanup is `todo.md`. Read it
  at the start of cleanup work, add newly discovered mismatches there, and
  update checkboxes only after code, docs, examples, and relevant tests have
  been checked. Keep `agent.md` focused on working rules; do not duplicate the
  operational checklist here.

  Current high-signal documentation reminders:

  - Audit `/docs` against current code before building Sphinx documentation.
  - Keep proposal/roadmap pages clearly labelled so users do not run future API
    snippets as current behavior.
  - Record NRV validation notes only after a fresh run in an NRV-ready
    environment.
  
  ## Common Pitfalls to Avoid
  
  1. **Unit Confusion**: Never mix internal canonical names with public unit-safe names
  2. **Solver Assumptions**: Don't assume solver arrays exist; build only when invoked
  3. **Batch Mutation**: Don't mutate frozen dataclasses; create new instances
  4. **Type Ignores**: Use sparingly; justify in comments if necessary
  5. **Performance Claims**: Always benchmark against reference before claiming speedups
  6. **NRV Compatibility**: Validate against reference for numerical correctness
  
  ## Review Checklist for PRs
  
  - [ ] Type hints are complete (mypy passes)
  - [ ] Unit tests added/updated for public APIs
  - [ ] Validation against NRV reference if numerical changes
  - [ ] Performance impact assessed (if optimization claims)
  - [ ] Documentation updated (docstrings + markdown if architecture affected)
  - [ ] Clear didactic `examples/advanced/` example added or updated for each new feature or advanced concept
  - [ ] New or updated examples favor a verbose line-by-line tutorial flow, avoiding helper-function overload unless it clearly improves readability
  - [ ] New or updated examples include useful plots when possible, especially for signals, metrics, dispatch, memory, or observer behavior
  - [ ] Affected examples updated when public API, workflow, argument names, or result behavior changed
  - [ ] Temporary compatibility aliases avoided unless they clearly simplify the current refactor
  - [ ] Naming conventions followed (canonical internal names)
  - [ ] No frozen dataclass mutations
  - [ ] `CHANGELOG.md` updated for every new feature or significant behavior change, or PR notes explain why no entry is needed
  
  ## Examples of Good Patterns
  
  ### Factory Function (Named Constructor)
  ```python
  def HodgkinHuxley(
      diameter: Q_,
      length: Q_,
      Ra: Q_,
      Cm: Q_,
      gNa: Q_,
      gK: Q_,
      gL: Q_,
  ) -> Axon:
      """Create an axon with Hodgkin-Huxley membrane model."""
      # Implementation returns fully-configured Axon
  ```
  
  ### Frozen Dataclass (Immutable Value Object)
  ```python
  @dataclass(frozen=True)
  class DispatchResult:
      """Immutable result of pool dispatch analysis."""
      groups: list[int]
      locations: ndarray
      membrane_models: list[str]
  ```
  
  ### Canonical Internal Names
  ```python
  class Axon:
      def __init__(self, diameter: Q_, ...):
          # Store with explicit unit suffix
          self.diameter_um = float(diameter.to("micrometer").magnitude)
  ```
  
  ## Future Development Roadmap
  
  - Larger GPU-oriented batch execution via existing batch kernels and runtime-batch builders
  - Heterogeneous membrane layouts and padded/parameter batch dispatch
  - Performance benchmarking for scalability assessment
  - Extended membrane model library
  - Advanced stimulation patterns
  
  ## Contact & Resources
  
  - **Documentation**: See `docs/` folder for architecture details
  - **Examples**: Start with `examples/basic/` for learning
  - **Benchmarks**: Run `benchmark/runtime/run.py` for performance metrics
  - **Validation**: Full NRV comparison in `tests/nrv/`
  - **Current Branch**: check with `git branch --show-current`; do not trust stale branch notes

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
      - src/axonscope/**/*.py
      - examples/**/*.py
      - benchmark/**/*.py
      - tests/**/*.py

rules:
  - name: Use type hints consistently
    description: All public APIs and complex internal logic should have complete type hints for mypy compatibility
    
  - name: Prefer frozen dataclasses for value objects
    description: Immutable value objects (results, configurations) should use @dataclass(frozen=True)
    
  - name: Use canonical internal names
    description: Internal values should use suffixes like _um, _ohm_cm, _uF_cm2 to prevent unit confusion
    
  - name: Validate against NRV reference
    description: Numerical changes should be validated with NRV comparison tests before merging
    
  - name: Document architectural decisions
    description: Non-obvious design choices should be documented in markdown files under docs/
    
  - name: Update CHANGELOG for significant changes
    description: Every new feature or significant behavior change should update CHANGELOG.md, unless PR notes explicitly justify no entry
    
  - name: Follow example organization
    description: Every new feature or advanced concept should include a clear didactic example in examples/advanced/ that demonstrates the real user workflow with a line-by-line tutorial flow where practical; examples/basic/ remains for compact introductory concepts

  - name: Keep examples current with API changes
    description: Public API, workflow, argument-name, and result-shape changes must update affected examples in the same change

  - name: Prefer clean pre-release APIs over compatibility shims
    description: AxonScope is not deployed yet; replace confusing temporary APIs with clean user-facing interfaces instead of preserving backwards compatibility by default

  - name: Follow master product guidelines
    description: Use GUIDELINES.md as the project philosophy and target architecture direction for major refactors, especially the product boundary, object model, extracellular footprint contract, and no-legacy policy
    
  - name: Profile before optimizing
    description: Performance claims should be backed by benchmark runs from benchmark/runtime/

helpfulLinks:
  - path: GUIDELINES.md
    description: Project philosophy and master product/solver architecture direction; update when the target direction changes
  - path: docs/axon_model_organization.md
    description: Architecture and design rationale for the descriptive layer
  - path: docs/solver_organization.md
    description: Solver design, time grids, boundary conditions
  - path: docs/pool_dispatch.md
    description: Pool planning and automatic batching strategy
  - path: docs/validation.md
    description: Testing policy and NRV validation approach
  - path: todo.md
    description: Living operational roadmap for documentation, API cleanup, benchmarks, examples, and Phase 8+ work
  - path: examples/basic
    description: Didactic examples covering core concepts
  - path: examples/advanced
    description: Complete workflow examples
  - path: benchmark/runtime
    description: Performance benchmarking and profiling tools
  - path: tests/nrv
    description: Scientific validation tests; requires local NRV environment and fresh pass count
---

# AxonScope Development Agent

This agent is configured to assist with development in the **AxonScope** project - a Python framework for validated peripheral nerve axon simulations with JAX-based, GPU-oriented solvers.

## Quick Facts

- **Current Version**: v0.1.0 (pre-release, API evolving)
- **Current Branch**: verify with `git branch --show-current`
- **Python Version**: 3.11+
- **Build System**: `pyproject.toml` with `poetry-core`; editable installs are documented with `pip`
- **Architecture**: 4-layer design (Descriptive → Protocol → Execution → Results)
- **Test Coverage**: fast unit suite plus optional NRV validation; use fresh local runs for counts/status

## Getting Started with This Agent

### For New Developers
1. Read [docs/axon_model_organization.md](docs/axon_model_organization.md) for descriptive layer concepts
2. Review [examples/basic/](examples/basic/) for didactic workflows
3. Read [todo.md](todo.md) before documentation/API cleanup work
4. Run unit tests: `pytest -q tests/unit`
5. Check out [docs/validation.md](docs/validation.md) for testing strategy

### For Contributing Code
1. Follow the [review checklist](#review-checklist-for-prs) above
2. Ensure type hints pass mypy
3. Add unit tests for public APIs
4. Update CHANGELOG.md for significant changes
5. Run `pytest -q tests/unit` before submitting PR

### For Performance Work
1. Establish baseline with `benchmark/runtime/run.py`
2. Profile your changes with the same benchmark suite
3. Compare against NRV validation tests for correctness
4. Document performance implications in code

## Key Files to Know

- **Main Package**: `src/axonscope/`
- **Tests**: `tests/unit/` (fast) and `tests/nrv/` (validation)
- **Examples**: `examples/basic/` and `examples/advanced/`
- **Benchmarks**: `benchmark/runtime/`
- **Architecture Docs**: `docs/*.md`
- **Living TODO**: `todo.md`

---

*This agent was generated from a comprehensive code audit. Last updated: 2026-06-14*
