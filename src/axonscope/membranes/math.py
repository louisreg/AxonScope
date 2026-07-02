"""Small scalar helpers mirrored by the membrane source compiler."""

from __future__ import annotations

import builtins
import math
from typing import Any


def abs(x: Any) -> Any:
    return builtins.abs(x)


def clip(x: Any, low: Any, high: Any) -> Any:
    return min(max(x, low), high)


def exp(x: Any) -> Any:
    return math.exp(x)


def expm1(x: Any) -> Any:
    return math.expm1(x)


def log(x: Any) -> Any:
    return math.log(x)


def log1p(x: Any) -> Any:
    return math.log1p(x)


def maximum(left: Any, right: Any) -> Any:
    return max(left, right)


def minimum(left: Any, right: Any) -> Any:
    return min(left, right)


def pow(left: Any, right: Any) -> Any:
    return builtins.pow(left, right)


def q10(base: Any, celsius: Any, reference: Any) -> Any:
    return base ** ((celsius - reference) / 10.0)


def alpha_from_inf_tau(x_inf: Any, tau: Any) -> Any:
    return x_inf / tau


def beta_from_inf_tau(x_inf: Any, tau: Any) -> Any:
    return (1.0 - x_inf) / tau


def safe_exp(x: Any) -> Any:
    if x < -100.0:
        return 0.0
    return math.exp(x)


def sigmoid(x: Any) -> Any:
    return 1.0 / (1.0 + math.exp(-x))


def sqrt(x: Any) -> Any:
    return math.sqrt(x)


def tanh(x: Any) -> Any:
    return math.tanh(x)


def where(condition: Any, if_true: Any, if_false: Any) -> Any:
    if condition:
        return if_true
    return if_false


def vtrap(x: Any, y: Any) -> Any:
    z = x / y
    if abs(z) < 1e-6:
        return y * (1.0 - z / 2.0)
    return x / (math.exp(z) - 1.0)


__all__ = [
    "abs",
    "alpha_from_inf_tau",
    "beta_from_inf_tau",
    "clip",
    "exp",
    "expm1",
    "log",
    "log1p",
    "maximum",
    "minimum",
    "pow",
    "q10",
    "safe_exp",
    "sigmoid",
    "sqrt",
    "tanh",
    "vtrap",
    "where",
]
