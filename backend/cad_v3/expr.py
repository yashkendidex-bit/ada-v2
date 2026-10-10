"""Equations — the thing that turns a pile of numbers into a parametric design.

Before this module a spec could say the carrier's pin circle is 46 mm and the sun
gear has 24 teeth at module 2, and nothing anywhere noticed that those two
statements have to agree. They agreed in the shipped gearbox by coincidence,
which is another way of saying the gearbox was correct by accident and one edit
away from silently wrong.

So every numeric field in a spec may now be an expression over named variables:

    "vars": {"module": 2.0, "sun_t": 24, "planet_t": 18,
             "pcd": "module * (sun_t + planet_t) / 2"}
    ...
    "params": {"pin_circle_d": "2 * pcd"}

Deliberately not `eval`. The grammar below is arithmetic, comparison-free, and
call-restricted to a fixed table, so a spec — which in the general case arrives
from a language model — cannot reach the interpreter, the filesystem or the
network through a parameter value.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

# The whole callable surface a dimension ever needs.
FUNCS: dict[str, Any] = {
    "sin": lambda d: math.sin(math.radians(d)),
    "cos": lambda d: math.cos(math.radians(d)),
    "tan": lambda d: math.tan(math.radians(d)),
    "asin": lambda v: math.degrees(math.asin(max(-1.0, min(1.0, v)))),
    "acos": lambda v: math.degrees(math.acos(max(-1.0, min(1.0, v)))),
    "atan2": lambda y, x: math.degrees(math.atan2(y, x)),
    "sqrt": lambda v: math.sqrt(max(0.0, v)),
    "hypot": math.hypot,
    "abs": abs,
    "min": min,
    "max": max,
    "round": lambda v, n=0: round(v, int(n)),
    "floor": lambda v: float(math.floor(v)),
    "ceil": lambda v: float(math.ceil(v)),
    "sign": lambda v: (v > 0) - (v < 0),
    "clamp": lambda v, lo, hi: max(lo, min(hi, v)),
    "deg": math.degrees,
    "rad": math.radians,
    "pow": lambda a, b: float(a) ** float(b),
}

CONSTS: dict[str, float] = {"pi": math.pi, "tau": math.tau, "e": math.e}

_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

MAX_LEN = 400  # an expression longer than this is not a dimension


class ExprError(ValueError):
    """Raised with a message the repair loop can show verbatim."""


def is_expression(value: Any) -> bool:
    """True when a field carries a formula rather than a plain number."""
    if isinstance(value, (int, float, bool)):
        return False
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    try:
        float(text)
    except ValueError:
        return True
    return False


def evaluate(text: str, scope: dict[str, float]) -> float:
    """One expression against one already-resolved scope."""
    src = str(text).strip()
    if src.startswith("="):
        src = src[1:].strip()
    if not src:
        raise ExprError("empty expression")
    if len(src) > MAX_LEN:
        raise ExprError(f"expression longer than {MAX_LEN} characters")
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as exc:
        raise ExprError(f"'{text}' is not an expression ({exc.msg})") from None
    return float(_walk(tree.body, scope))


def _walk(node: ast.AST, scope: dict[str, float]) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ExprError(f"{node.value!r} is not a number")
        return float(node.value)
    if isinstance(node, ast.Name):
        key = node.id
        if key in scope:
            return float(scope[key])
        if key in CONSTS:
            return CONSTS[key]
        raise ExprError(f"unknown variable '{key}'")
    if isinstance(node, ast.BinOp):
        fn = _BINOPS.get(type(node.op))
        if fn is None:
            raise ExprError(f"operator {type(node.op).__name__} is not allowed")
        left, right = _walk(node.left, scope), _walk(node.right, scope)
        try:
            return float(fn(left, right))
        except ZeroDivisionError:
            raise ExprError("division by zero") from None
        except OverflowError:
            raise ExprError("overflow") from None
    if isinstance(node, ast.UnaryOp):
        fn = _UNARYOPS.get(type(node.op))
        if fn is None:
            raise ExprError(f"unary {type(node.op).__name__} is not allowed")
        return float(fn(_walk(node.operand, scope)))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCS:
            name = getattr(node.func, "id", "?")
            raise ExprError(f"unknown function '{name}'; allowed: {', '.join(sorted(FUNCS))}")
        if node.keywords:
            raise ExprError("keyword arguments are not allowed in an expression")
        args = [_walk(a, scope) for a in node.args]
        try:
            return float(FUNCS[node.func.id](*args))
        except ExprError:
            raise
        except Exception as exc:
            raise ExprError(f"{node.func.id}(...) failed: {exc}") from None
    raise ExprError(f"{type(node).__name__} is not allowed in an expression")


def names_in(text: str) -> set[str]:
    """Which variables an expression reads — used to order the resolution."""
    src = str(text).strip().lstrip("=")
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError:
        return set()
    return {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and n.id not in FUNCS and n.id not in CONSTS}


def resolve(raw: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    """Turn a `vars` block, whose values may reference each other, into numbers.

    Order is derived, not required of the author: a designer thinks "the pitch
    circle follows from the tooth counts" without caring which line they wrote
    first. A genuine cycle is reported by name rather than by recursion depth,
    because `a = b + 1, b = a - 1` is a modelling mistake and the message has to
    say which two lines are arguing.
    """
    out: dict[str, float] = {}
    errors: list[str] = []
    pending: dict[str, Any] = {}

    for key, val in (raw or {}).items():
        if not isinstance(key, str) or not key.isidentifier():
            errors.append(f"vars: '{key}' is not a usable variable name")
            continue
        if key in CONSTS:
            errors.append(f"vars: '{key}' is a reserved constant")
            continue
        if is_expression(val):
            pending[key] = val
        else:
            try:
                out[key] = float(val)
            except (TypeError, ValueError):
                errors.append(f"vars.{key}: {val!r} is neither a number nor an expression")

    # Repeated sweeps rather than a topological sort: the dependency graph of a
    # dimension set is tiny, and a sweep that makes no progress is exactly the
    # cycle test we want anyway.
    while pending:
        progressed = False
        for key in list(pending):
            need = names_in(pending[key])
            if need - set(out) - set(CONSTS):
                continue
            try:
                out[key] = evaluate(pending.pop(key), out)
                progressed = True
            except ExprError as exc:
                errors.append(f"vars.{key}: {exc}")
                pending.pop(key, None)
                progressed = True
        if not progressed:
            stuck = sorted(pending)
            unknown: set[str] = set()
            for key in stuck:
                unknown |= names_in(pending[key]) - set(out) - set(CONSTS) - set(pending)
            if unknown:
                errors.append(f"vars {stuck}: reference undefined name(s) {sorted(unknown)}")
            else:
                errors.append(f"vars {stuck}: form a circular definition")
            break
    return out, errors


def number(value: Any, scope: dict[str, float], *, where: str = "") -> tuple[float, str]:
    """One field's numeric value. Returns (value, error) with error '' on success."""
    if isinstance(value, bool):
        return (float(value), "")
    if isinstance(value, (int, float)):
        return (float(value), "")
    if is_expression(value):
        try:
            return (evaluate(value, scope), "")
        except ExprError as exc:
            return (0.0, f"{where}: {exc}" if where else str(exc))
    try:
        return (float(str(value).strip()), "")
    except (TypeError, ValueError):
        return (0.0, f"{where}: {value!r} is not a number" if where else f"{value!r} is not a number")


def bake(value: Any, scope: dict[str, float]) -> Any:
    """Substitute expressions for their values, leaving everything else untouched.

    Applied to a whole params dict so the vocabulary never has to know that
    equations exist — a skill still receives plain floats.
    """
    if isinstance(value, dict):
        return {k: bake(v, scope) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [bake(v, scope) for v in value]
    if is_expression(value):
        try:
            return evaluate(value, scope)
        except ExprError:
            return value
    return value
