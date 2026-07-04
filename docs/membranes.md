# Membranes

`axonscope.membranes` is the public namespace for runtime-independent membrane
descriptions.

Membrane objects are descriptive. They do not own JAX functions, solver
backends, time stepping, or compiled state. Solvers translate covered built-ins
and supported custom classes through AxonScope's internal compiler/runtime path.
Public code should not import or construct compiler/runtime objects to describe
a membrane.

```text
axs.membranes.Model -> axon Section/Layout -> source compiler -> backend runtime
```

## Public Surface

```python
import axonscope as axs

membrane = axs.membranes.HodgkinHuxley(celsius=6.3)
```

Built-in membrane names are classes that inherit `Model`. The class constructor
gives autocomplete-friendly parameter names and validates units. The scientific
definition lives in the matching standalone source file under
`src/axonscope/membranes/models/`.

```python
assert issubclass(axs.membranes.HodgkinHuxley, axs.membranes.Model)

hh_standard = axs.membranes.HodgkinHuxley(celsius=6.3 * axs.degC)
hh_hotspot = axs.membranes.HodgkinHuxley(
    celsius=6.3 * axs.degC,
    gnabar=0.18 * axs.S_per_cm2,
    gkbar=0.045 * axs.S_per_cm2,
    gl=0.0003 * axs.S_per_cm2,
)
```

Built-in model classes:

- `Passive`
- `HodgkinHuxley`
- `RattayAberham`
- `Sundt`
- `Tigerholm`
- `Schild94`
- `Schild97`
- `AxNode`
- `Composite`
- `SectionLayout`

The package layout mirrors these responsibilities:

```text
src/axonscope/membranes/
  model.py           Model base class, runtime descriptors, decorators
  builtins.py        public re-exports of source-backed model classes
  compiler.py        internal source loader and public-parameter normalizer
  models/*.py        one standalone source of truth per built-in model
  section_layout.py  named section-to-membrane assignment
  __init__.py        public facade
```

Each built-in model is owned by exactly one file in
`src/axonscope/membranes/models/`. That file must contain the model equations,
unit-bearing parameter defaults, public aliases, and any derived parameter
logic needed for construction. `builtins.py` must not duplicate equations or
defaults; it only re-exports the public classes.

## Units

Most scalar membrane parameters may be passed as plain numbers in AxonScope's
canonical membrane units. Pint quantities are accepted and converted at the
public construction boundary. Geometry-like public parameters such as
`diameter` must carry length units. `Model.params` reports the normalized
plain-float values that solver preparation receives after validation.

Canonical units:

| Parameter family | Canonical unit |
| --- | --- |
| membrane voltage, reversal potential | `millivolt` |
| temperature | `degree_Celsius` |
| diameter | `micrometer` |
| public conductance density inputs | `siemens / centimeter ** 2` |
| passive membrane resistance `Rm` | `ohm * centimeter ** 2` |
| Tigerholm concentrations | `millimolar` |
| Tigerholm pump current density | `milliampere / centimeter ** 2` |

Standalone source models specify their own units directly in annotations and
defaults, for example `gnabar: ConductanceDensity = 120.0 * axs.mS_per_cm2`.
Parameters not visible in equation signatures, such as geometry inputs used to
derive defaults, should use semantic annotations too, for example
`diameter_um: Length = 1.0 * axs.um`.
After public boundary conversion, internal parameter keys mirror the source
names (`gnabar`, `ena`, `diameter_um`) instead of relying on suffixes to carry
unit meaning.

```python
import axonscope as axs

hh = axs.membranes.HodgkinHuxley(
    gnabar=120.0 * axs.mS_per_cm2,
    ena=50.0 * axs.mV,
    celsius=6.3 * axs.degC,
)

passive = axs.membranes.Passive(
    Rm=10_000.0 * axs.ohm_cm2,
    EL=-70.0 * axs.mV,
)

tigerholm = axs.membranes.Tigerholm(
    diameter=1.0 * axs.um,
    celsius=37.0 * axs.degC,
)
```

Built-in axon templates keep their ergonomic model-specific keyword API, but
those keywords are only forwarded to the matching membrane source. Valid units,
defaults, aliases, equations, and derived parameter logic are defined by the
matching file in `src/axonscope/membranes/models/`, not by the axon template.

```python
axon = axs.axons.HodgkinHuxley(
    length=1.0 * axs.mm,
    diameter=0.5 * axs.um,
    compartments=101,
    celsius=6.3 * axs.degC,
    gnabar=120.0 * axs.mS_per_cm2,
    ena=50.0 * axs.mV,
)
```

## Generated Code Inspection

Generated-code inspection starts from the public membrane model. It reports the
source file, source hash, generated-code cache key, manifest, cache hit/miss
reason, and generated files without exposing the internal representation as a
user API.

```python
import axonscope as axs

model = axs.membranes.HodgkinHuxley(celsius=6.3 * axs.degC)
report = model.inspect_generated_code()
print(report.format())

jax_source = report.sources[0].file("jax_model.py").read_text()
```

For a higher-level source explanation, use `explain()`. It reports source
sections, equation dependencies, unit roles, public outputs, generated backend
targets, cache identity, and intermediates pruned from the generated
`model_step`.

```python
explanation = model.explain()
print(explanation.format())

same_report = axs.membranes.explain(model)
```

To include generated module text directly in the report:

```python
report = axs.membranes.inspect_generated_code(
    model,
    include_text=True,
    files=("jax_model.py", "numpy_model.py"),
)
```

Generated artifacts are cached under
`.axonscope_cache/model_codegen/<cache_key>/` unless
`AXONSCOPE_MODEL_CODEGEN_CACHE` points elsewhere. The cache key includes the
model source hash, selected class name, source/compiler/schema/helper/decorator
versions, generated targets, and static source metadata. Runtime parameter
values do not force regeneration unless a future parameter is declared
structural/static. Corrupt manifests, missing generated files, source changes,
or compiler/schema/helper/decorator version bumps are treated as misses and the
directory is regenerated. It is always safe to delete `.axonscope_cache/`; the
next inspection or run rebuilds missing artifacts.

## Composite Membranes

`Composite` combines several membrane mechanisms into one membrane assigned to
one section.

```python
membrane = axs.membranes.Composite(
    [
        axs.membranes.HodgkinHuxley(celsius=6.3 * axs.degC),
        axs.membranes.Passive(Rm=10_000.0 * axs.ohm_cm2, EL=-70.0 * axs.mV),
    ]
)

section = axs.axons.Section(
    "axon",
    membrane=membrane,
    diameter=0.5 * axs.um,
    Ra=100.0 * axs.ohm_cm,
    Cm=1.0 * axs.uF_per_cm2,
)

layout = axs.axons.Layout.single_uniform(section, length=1000.0 * axs.um, compartments=101)
```

Conceptually, this is still one membrane description for one section. Spatial
length and compartment count live in `Layout`. The solver composes supported
components through the internal compiler path with aggregated public current
and conductance names. Unsupported components fail at compile time; public
composites do not fall back to a separate old composite runtime path.

## Custom Membranes

Custom membrane authoring uses the same class shape as built-ins: subclass
`axs.membranes.Model`, declare typed parameter fields, write equation sections
as plain Python methods, and pass an instance anywhere a membrane is accepted.
AxonScope compiles the class source into a backend-ready semantic graph and
generated runtime artifacts.

The rejected builder-style surface and the old module-level
`model = Model(...)` manifest are gone. Users define models, not Model IR. The
internal graph is compiler/runtime machinery and should not appear in examples
except through inspection/explain reports.

```python
from axonscope.membranes.types import CurrentDensity, ResistanceArea, Voltage


class Leak(axs.membranes.Model):
    Rm: ResistanceArea = 10_000.0 * axs.ohm_cm2
    EL: Voltage = -70.0 * axs.mV

    @axs.membranes.currents
    def currents(self, Vm: Voltage):
        I_l: CurrentDensity = (Vm - self.EL) / self.Rm
        return I_l


section = axs.axons.Section(
    "custom leak",
    membrane=Leak(),
    diameter=0.5 * axs.um,
    Ra=100.0 * axs.ohm_cm,
    Cm=1.0 * axs.uF_per_cm2,
)
```

Complex models may define a plain module-level `derive_parameters(...)`
function when a small set of public fields expands into many solver constants.
The compiler discovers it automatically; model files should not need
`staticmethod(...)` boilerplate for this.

When a section needs a state, special solver symbol, or expression produced by
another section, reference it as `self.<name>` inside the equation. The compiler
resolves these model symbols; source files should not carry `TYPE_CHECKING` /
`cast(...)` placeholder blocks just to satisfy an editor.

Complex section metadata belongs on the section decorator, next to the code that
produces the symbols. Use `@currents(outputs=..., observables=..., internal=...)`
to mark public currents, public observables, and retained internal values. Use
`@initials(updates=...)` and `@step(prepare=..., finalize=..., diagnostics=...)`
to map state updates and solver diagnostics. Built-in model files should not
use class-level `exports = {...}` or `dynamics = {...}` manifests.

The current authoring subset is intentionally small:

- model classes inherit `axs.membranes.Model`;
- model parameters are annotated class fields with unit-bearing defaults;
- annotations use semantic types from `axonscope.membranes.types`;
- defaults and equations may use direct units, `units.<name>`, or `axs.<name>`
  for compiler-known unit aliases;
- equation bodies use assignments, annotated assignments, returns,
  arithmetic, comparisons, and supported helpers from `axonscope.membranes.math`;
- `self.<field>` reads model parameters and `self.<symbol>` reads compiler
  symbols produced by another section;
- `@rates`, `@currents`, `@mechanism(...)`, `@initials(...)`, and `@step(...)`
  define source sections;
- `state(...)` declares non-gate state and `Gate` arguments are inferred from
  alpha/beta rate names;
- `self.keep(...)` retains named intermediates in generated `model_step`;
- unsupported Python constructs, missing unit annotations, unknown symbols,
  duplicate assignments, cycles, and stale manifest fields fail at compile
  time with source locations when available.

Supported equation helpers currently live in `axonscope.membranes.math`:
`exp`, `expm1`, `log`, `log1p`, `sqrt`, `abs`, `minimum`, `maximum`, `clip`,
`where`, `tanh`, `sigmoid`, `boltzmann`, `vtrap`, `q10`,
`rates_from_tau_inf`, and `safe_exp`.
`boltzmann(x, midpoint, slope)` uses the signed-slope convention
`1 / (1 + exp((x - midpoint) / slope))`.
`alpha_x, beta_x = rates_from_tau_inf(x_inf, tau_x)` expands a steady-state
gate value and time constant into the corresponding forward/backward rates.

Currently unsupported: arbitrary Python side effects, loops, mutation, dynamic
attribute creation, class-level `exports`/`dynamics`, manual construction of
internal runtime descriptors, direct imports from `axonscope.model_ir`,
backend-local extension classes, and stateful `Composite` components. See
`examples/advanced/axon_models/05_custom_membrane_authoring.py` for the accepted
class style.

Backend classes, JAX arrays, and compiler dataclasses are internal details.
User code should describe a membrane model with `axs.membranes` equations.

## Section Layouts

`SectionLayout` assigns membrane models to named anatomical section kinds. It
is useful for templates such as MRG-like myelinated fibers, where a cable
contains node, MYSA, FLUT, and STIN sections.

```python
section_membranes = axs.membranes.SectionLayout(
    node=axs.membranes.AxNode(),
    mysa=axs.membranes.Passive(Rm=1e6 * axs.ohm_cm2, EL=-80.0 * axs.mV),
    flut=axs.membranes.Passive(Rm=1e6 * axs.ohm_cm2, EL=-80.0 * axs.mV),
    stin=axs.membranes.Passive(Rm=1e6 * axs.ohm_cm2, EL=-80.0 * axs.mV),
)

template = axs.axons.MRGLikeDoubleCableTemplate(
    diameter=10.0 * axs.um,
    nodes=5,
    compartments={"node": 1, "MYSA": 1, "FLUT": 2, "STIN": 4},
)
axon = axs.axons.Myelinated(layout=template.layout(membranes=section_membranes))
```

`SectionLayout` is descriptive. It does not create solver-side heterogeneous
compute layouts. That responsibility stays in backend preparation and JAX
membrane lowering.

## Runtime Boundary

Public axon construction must use `axonscope.membranes` descriptions. Passive,
Hodgkin-Huxley, Rattay-Aberham, Sundt, AxNode, Tigerholm, Schild, and supported
composites compile through AxonScope's internal compiler path before reaching
the active backend runtime.

The runtime bridge is structural: backend kernels consume solver-facing terms
such as gate updates, ionic current, conductance linearization, state updates,
diagnostics, and traces. Backend/runtime classes are not public extension
points. New custom membrane semantics should go through the plain-Python source
compiler instead of backend-local classes or temporary builder surfaces.

The generated-execution boundary is semantic, not the final performance
ceiling. Generated model-step modules already cover currents, conductances,
state prepare/finalize updates, diagnostics, observable pruning, and source
cache inspection for the supported class subset. Direct cable-solver kernel
fusion, more aggressive common-subexpression elimination, and target-specific
layout rewrites may use stricter optimized representations after compilation,
even if those representations no longer look like the public `Model` class.
