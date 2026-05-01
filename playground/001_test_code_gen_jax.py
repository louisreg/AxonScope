# mini_symbolic_model_codegen.py

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt

from axonscope.axons.generic import GenericAxon
from axonscope.solvers.CrankNicholson import CrankNicholson


# ============================================================
# Symbolic expression system
# ============================================================

def as_expr(x):
    if isinstance(x, Expr):
        return x
    return Const(float(x))


class Expr:
    def __add__(self, other): return BinOp("+", self, as_expr(other))
    def __radd__(self, other): return BinOp("+", as_expr(other), self)

    def __sub__(self, other): return BinOp("-", self, as_expr(other))
    def __rsub__(self, other): return BinOp("-", as_expr(other), self)

    def __mul__(self, other): return BinOp("*", self, as_expr(other))
    def __rmul__(self, other): return BinOp("*", as_expr(other), self)

    def __truediv__(self, other): return BinOp("/", self, as_expr(other))
    def __rtruediv__(self, other): return BinOp("/", as_expr(other), self)

    def __pow__(self, other): return BinOp("**", self, as_expr(other))
    def __rpow__(self, other): return BinOp("**", as_expr(other), self)

    def __neg__(self): return UnaryOp("-", self)

    def __lt__(self, other): return BinOp("<", self, as_expr(other))
    def __le__(self, other): return BinOp("<=", self, as_expr(other))
    def __gt__(self, other): return BinOp(">", self, as_expr(other))
    def __ge__(self, other): return BinOp(">=", self, as_expr(other))


@dataclass(frozen=True)
class Const(Expr):
    value: float


@dataclass(frozen=True)
class Var(Expr):
    name: str


@dataclass(frozen=True)
class Param(Expr):
    name: str
    value: float


@dataclass(frozen=True)
class BinOp(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass(frozen=True)
class UnaryOp(Expr):
    op: str
    value: Expr


@dataclass(frozen=True)
class Call(Expr):
    name: str
    args: tuple[Expr, ...]


def exp(x):
    return Call("exp", (as_expr(x),))


def abs_(x):
    return Call("abs", (as_expr(x),))


def where(cond, a, b):
    return Call("where", (as_expr(cond), as_expr(a), as_expr(b)))


V = Var("V")


# ============================================================
# Model objects
# ============================================================

@dataclass
class FunctionDef:
    name: str
    args: tuple[str, ...]
    body: Expr

    def __call__(self, *args):
        return Call(self.name, tuple(as_expr(a) for a in args))


@dataclass
class Gate:
    alpha: Expr
    beta: Expr


@dataclass
class Channel:
    gbar: Expr
    erev: Expr
    conductance: Expr


@dataclass
class IonModel:
    name: str
    params: dict[str, Param] = field(default_factory=dict)
    functions: dict[str, FunctionDef] = field(default_factory=dict)
    gates: dict[str, Gate] = field(default_factory=dict)
    channels: dict[str, Channel] = field(default_factory=dict)

    def param(self, name: str, value: float) -> Param:
        p = Param(name, float(value))
        self.params[name] = p
        return p

    def gate_var(self, name: str) -> Var:
        if name not in self.gates:
            # Allowed before gate registration too, but this helps catch typos
            # in more mature versions. For this prototype, we keep it permissive.
            pass
        return Var(name)

    def function(self, fn):
        arg_names = fn.__code__.co_varnames[: fn.__code__.co_argcount]
        symbolic_args = [Var(a) for a in arg_names]
        body = fn(*symbolic_args)

        f = FunctionDef(fn.__name__, tuple(arg_names), as_expr(body))
        self.functions[f.name] = f
        return f

    def gate(self, name: str, alpha: Expr, beta: Expr):
        self.gates[name] = Gate(as_expr(alpha), as_expr(beta))

    def channel(self, name: str, gbar: Expr, erev: Expr, conductance: Expr):
        self.channels[name] = Channel(as_expr(gbar), as_expr(erev), as_expr(conductance))


# ============================================================
# Codegen helpers
# ============================================================

def emit_expr(e: Expr) -> str:
    if isinstance(e, Const):
        return repr(e.value)

    if isinstance(e, Var):
        return e.name

    if isinstance(e, Param):
        return e.name

    if isinstance(e, UnaryOp):
        return f"({e.op}{emit_expr(e.value)})"

    if isinstance(e, BinOp):
        return f"({emit_expr(e.left)} {e.op} {emit_expr(e.right)})"

    if isinstance(e, Call):
        args = ", ".join(emit_expr(a) for a in e.args)

        if e.name == "exp":
            return f"jnp.exp({args})"

        if e.name == "abs":
            return f"jnp.abs({args})"

        if e.name == "where":
            return f"jnp.where({args})"

        return f"{e.name}({args})"

    raise TypeError(type(e))


def canonical_model(model: IonModel) -> dict:
    return {
        "name": model.name,
        "params": {k: v.value for k, v in sorted(model.params.items())},
        "functions": {
            name: {
                "args": f.args,
                "body": emit_expr(f.body),
            }
            for name, f in sorted(model.functions.items())
        },
        "gates": {
            name: {
                "alpha": emit_expr(g.alpha),
                "beta": emit_expr(g.beta),
            }
            for name, g in sorted(model.gates.items())
        },
        "channels": {
            name: {
                "gbar": emit_expr(c.gbar),
                "erev": emit_expr(c.erev),
                "conductance": emit_expr(c.conductance),
            }
            for name, c in sorted(model.channels.items())
        },
    }


def model_hash(model: IonModel) -> str:
    payload = json.dumps(canonical_model(model), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


# ============================================================
# JAX code generation
# ============================================================

def generate_jax_code(model: IonModel) -> str:
    gate_names = tuple(model.gates.keys())
    channel_names = tuple(model.channels.keys())

    param_lines = [
        f"{name} = dtype({param.value!r})"
        for name, param in model.params.items()
    ]

    func_blocks = []
    for f in model.functions.values():
        args = ", ".join(f.args)
        body = emit_expr(f.body)
        func_blocks.append(
            f"""
def {f.name}({args}):
    return {body}
"""
        )

    alpha_lines = []
    beta_lines = []
    for name, gate in model.gates.items():
        alpha_lines.append(f"    alpha_{name} = {emit_expr(gate.alpha)}")
        beta_lines.append(f"    beta_{name} = {emit_expr(gate.beta)}")

    alpha_stack = ", ".join(f"alpha_{name}" for name in gate_names)
    beta_stack = ", ".join(f"beta_{name}" for name in gate_names)

    gate_unpack = "\n".join(
        f"    {name} = gates[:, {i}]"
        for i, name in enumerate(gate_names)
    )

    channel_g_lines = []
    gbar_values = []
    erev_values = []

    for name, ch in model.channels.items():
        channel_g_lines.append(f"    g_{name} = {emit_expr(ch.conductance)}")
        gbar_values.append(emit_expr(ch.gbar))
        erev_values.append(emit_expr(ch.erev))

    g_stack = ", ".join(f"g_{name}" for name in channel_names)

    code = f"""
from __future__ import annotations

import jax
import jax.numpy as jnp

try:
    from axonscope.settings import dtype
except Exception:
    dtype = jnp.float32


GATE_NAMES = {gate_names!r}
CHANNEL_NAMES = {channel_names!r}

{chr(10).join(param_lines)}

q10 = dtype(3.0 ** ((celsius - 6.3) / 10.0))

g_bar = jnp.asarray([{", ".join(gbar_values)}], dtype=dtype)
E_rev = jnp.asarray([{", ".join(erev_values)}], dtype=dtype)


{chr(10).join(func_blocks)}


@jax.named_call
def rates(V):
    V = jnp.asarray(V, dtype=dtype)

{chr(10).join(alpha_lines)}

{chr(10).join(beta_lines)}

    alpha = jnp.stack([{alpha_stack}], axis=1)
    beta = jnp.stack([{beta_stack}], axis=1)
    return alpha, beta


def alpha_funcs(V):
    alpha, _ = rates(V)
    return alpha


def beta_funcs(V):
    _, beta = rates(V)
    return beta


def init_gates(V0_mV=None, V=None):
    if V is None:
        V = V0_mV
    alpha, beta = rates(V)
    return alpha / jnp.maximum(alpha + beta, dtype(1e-12))


def cn_gate_update(gates, V, dt):
    alpha, beta = rates(V)
    alpha = q10 * alpha
    beta = q10 * beta

    inv_dt = dtype(1.0) / dtype(dt)
    ab = alpha + beta
    denom = jnp.maximum(inv_dt + dtype(0.5) * ab, dtype(1e-12))

    return alpha / denom + ((inv_dt - dtype(0.5) * ab) / denom) * gates


def conductances(V, gates):
{gate_unpack}

{chr(10).join(channel_g_lines)}

    return jnp.stack([{g_stack}], axis=1)


def currents_and_gtot(V, gates):
    g = conductances(V, gates)
    Iion = jnp.sum(g * (V[:, None] - E_rev[None, :]), axis=1)
    Gtot = jnp.sum(g, axis=1)
    return Iion, Gtot


def currents(V, gates):
    Iion, _ = currents_and_gtot(V, gates)
    return Iion


def total_conductance(V, gates):
    _, Gtot = currents_and_gtot(V, gates)
    return Gtot


def I_background(V, gates=None):
    return jnp.zeros_like(V, dtype=dtype)


# ============================================================
# Optimized fused API for future solvers
# ============================================================

@jax.named_call
def step_ion(V, gates, dt):
    gates_new = cn_gate_update(gates, V, dt)
    Iion, Gtot = currents_and_gtot(V, gates_new)
    return gates_new, Iion, Gtot


# ============================================================
# Compatibility layer with IonChannelModelBase-style solvers
# ============================================================

def g_funcs(gates, g_bar_arg=None):
    # Existing AxonScope solvers call:
    #     g_func(gates, g_bar)
    #
    # The generated model already embeds its channel conductance parameters.
    # g_bar_arg is accepted for compatibility but not used.
    return conductances(None, gates)
"""
    return textwrap.dedent(code).strip() + "\n"


def compile_model(model: IonModel, cache_dir=".axonscope_cache", force=False):
    h = model_hash(model)
    module_name = f"{model.name.lower()}_{h}_jax"

    out_dir = Path(cache_dir) / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"{module_name}.py"

    if force or not path.exists():
        path.write_text(generate_jax_code(model), encoding="utf-8")

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import generated module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module, path


# ============================================================
# Example: full Hodgkin-Huxley model
# ============================================================

def build_hodgkin_huxley():
    model = IonModel("HodgkinHuxley")

    # Use mS/cm² here to match your existing AxonScope g_bar convention.
    gnabar = model.param("gnabar", 0.12 * 1e3)
    gkbar = model.param("gkbar", 0.036 * 1e3)
    gl = model.param("gl", 0.0003 * 1e3)

    ena = model.param("ena", 50.0)
    ek = model.param("ek", -77.0)
    el = model.param("el", -54.3)

    celsius = model.param("celsius", 6.3)

    @model.function
    def vtrap(x, y):
        z = x / y
        return where(
            abs_(z) < 1e-7,
            y * (1.0 - z / 2.0),
            x / (exp(z) - 1.0),
        )

    model.gate(
        "m",
        alpha=0.1 * vtrap(-(V + 40.0), 10.0),
        beta=4.0 * exp(-(V + 65.0) / 18.0),
    )

    model.gate(
        "h",
        alpha=0.07 * exp(-(V + 65.0) / 20.0),
        beta=1.0 / (exp(-(V + 35.0) / 10.0) + 1.0),
    )

    model.gate(
        "n",
        alpha=0.01 * vtrap(-(V + 55.0), 10.0),
        beta=0.125 * exp(-(V + 65.0) / 80.0),
    )

    m = model.gate_var("m")
    h = model.gate_var("h")
    n = model.gate_var("n")

    model.channel(
        "Na",
        gbar=gnabar,
        erev=ena,
        conductance=gnabar * m**3 * h,
    )

    model.channel(
        "K",
        gbar=gkbar,
        erev=ek,
        conductance=gkbar * n**4,
    )

    model.channel(
        "Leak",
        gbar=gl,
        erev=el,
        conductance=gl * (m * 0.0 + 1.0),
    )

    return model


# ============================================================
# Quick test
# ============================================================

if __name__ == "__main__":
    import jax
    import jax.numpy as jnp

    model = build_hodgkin_huxley()
    ion, path = compile_model(model, force=True)

    print("Generated module:", path)

    Vtest = jnp.linspace(-80.0, 40.0, 8)

    gates0 = ion.init_gates(Vtest)
    Iion, Gtot = ion.currents_and_gtot(Vtest, gates0)
    gates1 = ion.cn_gate_update(gates0, Vtest, dt=0.001)

    print("V:")
    print(Vtest)

    print("gates0:")
    print(gates0)

    print("Iion:")
    print(Iion)

    print("Gtot:")
    print(Gtot)

    print("gates1:")
    print(gates1)

    @jax.jit
    def jitted_call(V):
        gates = ion.init_gates(V)
        return ion.currents_and_gtot(V, gates)

    Ijit, Gjit = jitted_call(Vtest)

    print("JIT Iion:")
    print(Ijit)

    print("JIT Gtot:")
    print(Gjit)

    # ------------------------------------------------------------
    # AxonScope compatibility test
    # ------------------------------------------------------------

    L = 1000.0
    tsim = 10.0
    dt = 0.001

    t_start = 1.0
    duration = 1.0
    amplitude = 2.0

    axon = GenericAxon(ion, L=L)
    axon.insert_I_Clamp(
        position=L / 2,
        t_start=t_start,
        duration=duration,
        amplitude=amplitude,
    )

    solver = CrankNicholson()
    res = solver.solve(axon, tsim, dt)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(
        res.Vm.T,
        aspect="auto",
        extent=[0, tsim, 0, L],
        origin="lower",
        cmap="viridis",
    )

    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Position [µm]")
    ax.set_title("Generated Hodgkin-Huxley model — AxonScope simulation")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Vm [mV]")

    fig.tight_layout()
    plt.show()