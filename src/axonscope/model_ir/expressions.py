"""Immutable expression nodes for the runtime-agnostic model IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal as TypingLiteral

from axonscope.utils.units import DIMENSIONLESS

from .unit_algebra import is_dimensionless, normalize_unit, quantity_literal


BinaryOperator = TypingLiteral[
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "lt",
    "le",
    "gt",
    "ge",
]
UnaryOperator = TypingLiteral["neg"]


class Expression:
    """Base class that gives symbolic expressions Python arithmetic syntax."""

    def __add__(self, other: Any) -> "BinaryOp":
        return BinaryOp("add", self, as_expression(other))

    def __radd__(self, other: Any) -> "BinaryOp":
        return BinaryOp("add", as_expression(other), self)

    def __sub__(self, other: Any) -> "BinaryOp":
        return BinaryOp("sub", self, as_expression(other))

    def __rsub__(self, other: Any) -> "BinaryOp":
        return BinaryOp("sub", as_expression(other), self)

    def __mul__(self, other: Any) -> "BinaryOp":
        return BinaryOp("mul", self, as_expression(other))

    def __rmul__(self, other: Any) -> "BinaryOp":
        return BinaryOp("mul", as_expression(other), self)

    def __truediv__(self, other: Any) -> "BinaryOp":
        return BinaryOp("div", self, as_expression(other))

    def __rtruediv__(self, other: Any) -> "BinaryOp":
        return BinaryOp("div", as_expression(other), self)

    def __pow__(self, other: Any) -> "BinaryOp":
        return BinaryOp("pow", self, as_expression(other))

    def __rpow__(self, other: Any) -> "BinaryOp":
        return BinaryOp("pow", as_expression(other), self)

    def __neg__(self) -> "UnaryOp":
        return UnaryOp("neg", self)

    def __lt__(self, other: Any) -> "BinaryOp":
        return BinaryOp("lt", self, as_expression(other))

    def __le__(self, other: Any) -> "BinaryOp":
        return BinaryOp("le", self, as_expression(other))

    def __gt__(self, other: Any) -> "BinaryOp":
        return BinaryOp("gt", self, as_expression(other))

    def __ge__(self, other: Any) -> "BinaryOp":
        return BinaryOp("ge", self, as_expression(other))


@dataclass(frozen=True, slots=True)
class Literal(Expression):
    """A scalar literal embedded in model structure."""

    value: int | float | bool
    unit: str = DIMENSIONLESS


@dataclass(frozen=True, slots=True)
class Symbol(Expression):
    """Reference to a named parameter, state, input, or local binding."""

    name: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Symbol name cannot be empty.")


@dataclass(frozen=True, slots=True)
class UnaryOp(Expression):
    op: UnaryOperator
    operand: Expression


@dataclass(frozen=True, slots=True)
class BinaryOp(Expression):
    op: BinaryOperator
    left: Expression
    right: Expression


@dataclass(frozen=True, slots=True)
class Call(Expression):
    """Call to a registered runtime-neutral intrinsic."""

    intrinsic: str
    args: tuple[Expression, ...]

    def __post_init__(self) -> None:
        if not self.intrinsic:
            raise ValueError("Intrinsic name cannot be empty.")
        object.__setattr__(self, "args", tuple(self.args))


def as_expression(value: Any) -> Expression:
    if isinstance(value, Expression):
        return value
    if isinstance(value, bool):
        return Literal(value)
    if isinstance(value, int | float):
        return Literal(value)
    if hasattr(value, "to") and hasattr(value, "magnitude"):
        magnitude = value.magnitude
        if isinstance(magnitude, Expression):
            unit = normalize_unit(value)
            if is_dimensionless(unit):
                return magnitude
            return BinaryOp("mul", magnitude, Literal(1.0, unit=unit))
    quantity = quantity_literal(value)
    if quantity is not None:
        magnitude, unit = quantity
        return Literal(magnitude, unit=unit)
    raise TypeError(f"Cannot convert {type(value).__name__} to a Model IR expression.")


def symbol(name: str) -> Symbol:
    return Symbol(name)


def literal(value: int | float | bool, *, unit: str = DIMENSIONLESS) -> Literal:
    return Literal(value, unit=unit)
