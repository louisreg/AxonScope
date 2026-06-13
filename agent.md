---
name: AxonScope Agent
description: Expert agent for the AxonScope peripheral nerve simulation framework
instructions: |
  You are an expert agent specialized in the AxonScope project - a Python framework for computational peripheral nerve axon simulations.
  
  ## Core Domain Knowledge
  - **Domain**: Computational neuroscience, peripheral nerve modeling, biomedical engineering
  - **Purpose**: Validated nerve simulations with extensible solvers (JAX-based, GPU-ready)
  - **Architecture**: Layered design (Descriptive → Protocol → Execution → Results)
  - **Current Status**: v0.1.0, active refactor, API still evolving
  
  ## Architectural Layers
  1. **Descriptive Layer** (solver-agnostic): Axons, Membranes, Stimulation
  2. **Protocol Layer**: AxonSimulation, Recording, Protocols
  3. **Execution Layer**: Solvers, Dispatcher, ICM, Channel Models
  4. **Result Layer**: Analysis, Visualization, Data Management
  
  ## Key Technologies
  - **JAX** ≥0.4.38: JIT compilation, GPU acceleration, batch execution
  - **Pint**: Explicit unit management at public API boundaries
  - **SciPy/NumPy**: Numerical foundations
  - **Poetry**: Build and dependency management
  - **Rust** (optional): Reference solver implementation
  
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
  - Full mypy coverage required; use type hints consistently
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
  | `results/` | SimResult, analysis, visualization |
  | `recording/` | Configurable output policies |
  | `protocols/` | High-level workflows (ThresholdSearch, RecruitmentCurve) |
  
  ## Performance & Scalability
  
  ### Current State
  - Single-axon + small pools validated against NRV reference
  - Single-cable: Imposed-field Vstim forcing (scalar term, efficient for large pools)
  - Double-cable: Coupled Vi/Vperi/Ve solving (more expensive)
  - JAX JIT explicitly enabled in SingleCableKernel and DoubleCableKernel
  
  ### Optimization Opportunities
  - Batch kernels ready (pre-designed for data-parallel GPU execution)
  - Rate table precomputation (configurable solver option)
  - Vstim sample precomputation (extracellular solves)
  - Cache locality via dispatch grouping
  
  ### When Optimizing
  1. Profile first with benchmark suite in `benchmark/runtime/`
  2. Maintain validation against NRV comparison tests (116 passing)
  3. Document performance implications in code comments
  4. Add regression detection to benchmark suite
  
  ## Testing Strategy
  
  ### Unit Tests (`tests/unit/`)
  - Fast CI suite, no external dependencies
  - 37 modules covering APIs, solvers, batching, channels
  - Run with: `pytest -q tests/unit`
  
  ### Validation Tests (`tests/nrv/`)
  - Scientific validation vs NRV reference implementation
  - 116 tests, all passing, requires local NRV checkout
  - Run with: `pytest -q tests/nrv`
  - Markers: `nrv`, `nrv_intracellular`, `nrv_velocity`, `nrv_extracellular`, `nrv_numerics`
  
  ### Best Practices
  - Always add unit tests for new public APIs
  - Compare output shapes and scalar metrics for regressions
  - Use frozen dataclasses to detect unintended mutations
  - Include diagnostic metadata in SimResult for reproducibility
  
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
  - Use `result.metadata` for solver diagnostics
  - Check `SimResult.output_guard` for shape validation
  - Enable JAX debug output for numerical issues
  - Profile with benchmark suite for performance analysis
  
  ## Documentation Requirements
  
  - **Code**: Docstrings for public APIs, explain non-obvious internal logic
  - **Comments**: Clarify design decisions, numerical considerations, unit conversions
  - **Examples**: Add to `examples/` if introducing new functionality
  - **Docs**: Update markdown files in `docs/` for architectural changes
  - **Changelog**: Record all significant changes (v0.1.0+ tracking)
  
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
  - [ ] Naming conventions followed (canonical internal names)
  - [ ] No frozen dataclass mutations
  - [ ] CHANGELOG updated with significant changes
  
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
  
  - Large GPU batch execution via pre-designed batch kernels
  - Heterogeneous membrane layouts (infrastructure ready)
  - Performance benchmarking for scalability assessment
  - Extended membrane model library
  - Advanced stimulation patterns
  
  ## Contact & Resources
  
  - **Documentation**: See `docs/` folder for architecture details
  - **Examples**: Start with `examples/basic/` for learning
  - **Benchmarks**: Run `benchmark/runtime/run.py` for performance metrics
  - **Validation**: Full NRV comparison in `tests/nrv/`
  - **Current Branch**: `solver-benchmark-profiling` (default: `main`)

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
    description: Version tracking is important; significant changes should appear in CHANGELOG.md
    
  - name: Follow example organization
    description: New examples should fit into either examples/basic/ (didactic) or examples/advanced/ (workflows)
    
  - name: Profile before optimizing
    description: Performance claims should be backed by benchmark runs from benchmark/runtime/

helpfulLinks:
  - path: docs/axon_model_organization.md
    description: Architecture and design rationale for the descriptive layer
  - path: docs/solver_organization.md
    description: Solver design, time grids, boundary conditions
  - path: docs/pool_dispatch.md
    description: Pool planning and automatic batching strategy
  - path: docs/validation.md
    description: Testing policy and NRV validation approach
  - path: examples/basic
    description: Didactic examples covering core concepts
  - path: examples/advanced
    description: Complete workflow examples
  - path: benchmark/runtime
    description: Performance benchmarking and profiling tools
  - path: tests/nrv
    description: Scientific validation tests (116 passing)
---

# AxonScope Development Agent

This agent is configured to assist with development in the **AxonScope** project - a Python framework for validated peripheral nerve axon simulations with JAX-based, GPU-ready solvers.

## Quick Facts

- **Current Version**: v0.1.0 (pre-release, API evolving)
- **Current Branch**: `solver-benchmark-profiling` (main: `main`)
- **Python Version**: 3.11+
- **Build System**: Poetry
- **Architecture**: 4-layer design (Descriptive → Protocol → Execution → Results)
- **Test Coverage**: 37 unit modules + 116 NRV validation tests (all passing)

## Getting Started with This Agent

### For New Developers
1. Read [docs/axon_model_organization.md](docs/axon_model_organization.md) for descriptive layer concepts
2. Review [examples/basic/](examples/basic/) for didactic workflows
3. Run unit tests: `pytest -q tests/unit`
4. Check out [docs/validation.md](docs/validation.md) for testing strategy

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

---

*This agent was generated from a comprehensive code audit. Last updated: 2026-06-13*
