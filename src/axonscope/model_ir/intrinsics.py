"""Runtime-neutral intrinsic registry for model expressions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .expressions import Call, Expression, as_expression


@dataclass(frozen=True, slots=True)
class Intrinsic:
    """Semantic declaration for a function known by the Model IR compiler."""

    name: str
    arity: int | tuple[int, ...]
    unit_rule: str
    numpy_name: str
    jax_name: str
    differentiable: bool = True
    note: str = ""

    def accepts(self, argc: int) -> bool:
        if isinstance(self.arity, int):
            return argc == self.arity
        return argc in self.arity


class IntrinsicRegistry:
    """Immutable-ish lookup table for supported model intrinsics."""

    def __init__(self, intrinsics: tuple[Intrinsic, ...]) -> None:
        self._by_name = {intrinsic.name: intrinsic for intrinsic in intrinsics}
        if len(self._by_name) != len(intrinsics):
            raise ValueError("Intrinsic names must be unique.")

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def get(self, name: str) -> Intrinsic:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise KeyError(f"Unknown Model IR intrinsic {name!r}.") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_name))


DEFAULT_INTRINSICS = IntrinsicRegistry(
    (
        Intrinsic("abs", 1, "same", "absolute", "abs"),
        Intrinsic("clip", 3, "first", "clip", "clip"),
        Intrinsic("exp", 1, "dimensionless_to_dimensionless", "exp", "exp"),
        Intrinsic("expm1", 1, "dimensionless_to_dimensionless", "expm1", "expm1"),
        Intrinsic("log", 1, "dimensionless_to_dimensionless", "log", "log"),
        Intrinsic("log1p", 1, "dimensionless_to_dimensionless", "log1p", "log1p"),
        Intrinsic("maximum", 2, "matching_args", "maximum", "maximum"),
        Intrinsic("minimum", 2, "matching_args", "minimum", "minimum"),
        Intrinsic("pow", 2, "pow", "power", "power"),
        Intrinsic(
            "boltzmann",
            3,
            "boltzmann",
            "boltzmann",
            "boltzmann",
            note="x, midpoint, slope -> 1 / (1 + exp((x - midpoint) / slope)).",
        ),
        Intrinsic(
            "alpha_from_inf_tau",
            2,
            "dimensionless_over_time_to_rate",
            "alpha_from_inf_tau",
            "alpha_from_inf_tau",
        ),
        Intrinsic(
            "beta_from_inf_tau",
            2,
            "dimensionless_over_time_to_rate",
            "beta_from_inf_tau",
            "beta_from_inf_tau",
        ),
        Intrinsic(
            "q10",
            3,
            "q10",
            "q10",
            "q10",
            note="base, celsius, reference_degC -> temperature factor.",
        ),
        Intrinsic(
            "safe_exp",
            1,
            "dimensionless_to_dimensionless",
            "safe_exp",
            "safe_exp",
            note="exp(x), but returns zero for x < -100 to mirror legacy nodal rates.",
        ),
        Intrinsic("sigmoid", 1, "dimensionless_to_dimensionless", "sigmoid", "sigmoid"),
        Intrinsic("sqrt", 1, "sqrt", "sqrt", "sqrt"),
        Intrinsic("tanh", 1, "dimensionless_to_dimensionless", "tanh", "tanh"),
        Intrinsic("vtrap", 2, "matching_args", "vtrap", "vtrap"),
        Intrinsic("where", 3, "where", "where", "where", differentiable=False),
        Intrinsic(
            "rush_larsen_gate",
            4,
            "dimensionless_gate",
            "rush_larsen_gate",
            "rush_larsen_gate",
            note="gate_prev, alpha, beta, dt -> gate_next",
        ),
        Intrinsic(
            "cn_gate",
            4,
            "dimensionless_gate",
            "cn_gate",
            "cn_gate",
            note="gate_prev, alpha, beta, dt -> gate_next",
        ),
    )
)


def call(name: str, *args: Any) -> Call:
    return Call(name, tuple(as_expression(arg) for arg in args))


def abs_(x: Any) -> Call:
    return call("abs", x)


def clip(x: Any, low: Any, high: Any) -> Call:
    return call("clip", x, low, high)


def exp(x: Any) -> Call:
    return call("exp", x)


def expm1(x: Any) -> Call:
    return call("expm1", x)


def log(x: Any) -> Call:
    return call("log", x)


def log1p(x: Any) -> Call:
    return call("log1p", x)


def maximum(left: Any, right: Any) -> Call:
    return call("maximum", left, right)


def minimum(left: Any, right: Any) -> Call:
    return call("minimum", left, right)


def pow_(left: Any, right: Any) -> Call:
    return call("pow", left, right)


def boltzmann(x: Any, midpoint: Any, slope: Any) -> Call:
    return call("boltzmann", x, midpoint, slope)


def q10(base: Any, celsius: Any, reference: Any) -> Call:
    return call("q10", base, celsius, reference)


def alpha_from_inf_tau(x_inf: Any, tau: Any) -> Call:
    return call("alpha_from_inf_tau", x_inf, tau)


def beta_from_inf_tau(x_inf: Any, tau: Any) -> Call:
    return call("beta_from_inf_tau", x_inf, tau)


def safe_exp(x: Any) -> Call:
    return call("safe_exp", x)


def sigmoid(x: Any) -> Call:
    return call("sigmoid", x)


def sqrt(x: Any) -> Call:
    return call("sqrt", x)


def tanh(x: Any) -> Call:
    return call("tanh", x)


def vtrap(x: Any, y: Any) -> Call:
    return call("vtrap", x, y)


def where(condition: Any, if_true: Any, if_false: Any) -> Call:
    return call("where", condition, if_true, if_false)
