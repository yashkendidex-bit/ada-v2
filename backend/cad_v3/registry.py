"""The closed vocabulary. A skill that isn't registered cannot be requested.

This module is the reason coordinate hallucination is structurally impossible:
the brain may only name skills that exist here, may only pass parameters these
signatures declare, and may only reference connectors these skills publish.
Everything else is rejected at spec-validation time, before the kernel runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass
class ParamDef:
    name: str
    type: type
    default: Any = None
    required: bool = False
    min: float | None = None
    max: float | None = None
    choices: tuple = ()
    doc: str = ""

    def describe(self) -> str:
        bits = [self.type.__name__ if self.type is not list else "list"]
        if self.choices:
            bits.append("|".join(str(c) for c in self.choices))
        elif self.min is not None or self.max is not None:
            bits.append(f"{self.min if self.min is not None else '-inf'}..{self.max if self.max is not None else 'inf'}")
        if not self.required:
            bits.append(f"default={self.default!r}")
        tail = f"  # {self.doc}" if self.doc else ""
        return f"{self.name}: {', '.join(bits)}{tail}"


def p(
    name: str,
    type_: type,
    default: Any = None,
    *,
    required: bool = False,
    min: float | None = None,
    max: float | None = None,
    choices: Sequence = (),
    doc: str = "",
) -> ParamDef:
    return ParamDef(name, type_, default, required, min, max, tuple(choices), doc)


@dataclass
class Skill:
    name: str
    fn: Callable
    params: dict[str, ParamDef]
    connectors: tuple[str, ...]
    category: str
    doc: str
    source: str = "ada"


REGISTRY: dict[str, Skill] = {}


def skill(
    name: str,
    *,
    connectors: Sequence[str],
    category: str,
    params: Sequence[ParamDef],
    source: str = "ada",
):
    def deco(fn: Callable) -> Callable:
        head = (fn.__doc__ or "").strip().splitlines()
        REGISTRY[name] = Skill(
            name=name,
            fn=fn,
            params={d.name: d for d in params},
            connectors=tuple(connectors),
            category=category,
            doc=head[0] if head else "",
            source=source,
        )
        return fn

    return deco


def get(name: str) -> Skill | None:
    return REGISTRY.get(str(name))


def names() -> list[str]:
    return sorted(REGISTRY)


# ── what each seat actually looks like, measured not declared ───────────────
# `Skill.connectors` is a tuple of names, which is all the brain used to be told.
# A name does not say whether a seat faces along the axis or out of it, so a spec
# could mate a part that needs to reach outward onto a seat that points up the
# axis — legal, buildable, and wrong, with nothing in the loop able to say why.
# The frame knows; it just was never asked. So build each skill once at its own
# defaults, read the facing straight off the connector frames, and cache it.
# Deriving it means it cannot drift from the geometry the way a written note would.

_SEATS: dict[str, list[str]] = {}


def seat_lines(name: str) -> list[str]:
    """`['bolt_circle (ring of 6, axial+)', ...]`, or [] if the skill will not probe.

    Cached per skill: one build at defaults, reused by every later prompt. A skill
    with a required param and no default cannot be probed, and a skill that raises
    is the gate's problem, not this function's — either way the caller falls back
    to bare connector names rather than losing the catalogue entry.
    """
    if name in _SEATS:
        return _SEATS[name]
    sk = get(name)
    out: list[str] = []
    if sk is not None:
        try:
            kwargs, errs = coerce_params(sk, {n: d.default for n, d in sk.params.items()
                                              if d.default is not None})
            if not errs:
                res = sk.fn(**kwargs)
                conns = getattr(res, "connectors", None) or {}
                out = [conns[c].tag for c in sk.connectors if c in conns]
        except Exception:
            out = []
    _SEATS[name] = out
    return out


def forget_seats(name: str | None = None) -> None:
    """Drop the probe cache — for a skill synth has just rewritten, or all of them."""
    if name is None:
        _SEATS.clear()
    else:
        _SEATS.pop(name, None)


def seat_direction(skill_name: str, connector: str) -> str:
    """`axial+`, `radial`, ... for one seat, or '' when it cannot be probed."""
    for tag in seat_lines(skill_name):
        head, _, rest = tag.partition("(")
        if head == connector:
            return rest.rstrip(")").split(", ")[-1]
    return ""


def seat_layout(skill_name: str, connector: str) -> str:
    """`single`, `ring of N`, `stack of N`, or '' when it cannot be probed."""
    for tag in seat_lines(skill_name):
        head, _, rest = tag.partition("(")
        if head == connector:
            bits = rest.rstrip(")").split(", ")
            return bits[0] if len(bits) > 1 else "single"
    return ""


def coerce_params(sk: Skill, raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Type-check and range-check raw spec params. Returns (kwargs, errors)."""
    raw = dict(raw or {})
    errors: list[str] = []

    unknown = [k for k in raw if k not in sk.params]
    if unknown:
        errors.append(
            f"unknown params {unknown} for skill '{sk.name}'. Valid: {sorted(sk.params)}"
        )

    kwargs: dict[str, Any] = {}
    for name, d in sk.params.items():
        if name not in raw:
            if d.required:
                errors.append(f"missing required param '{name}' ({d.describe()})")
            else:
                kwargs[name] = d.default
            continue
        val = raw[name]
        try:
            if d.type is list:
                if not isinstance(val, (list, tuple)):
                    raise TypeError("expected a list")
                val = list(val)
            elif d.type is bool:
                val = bool(val)
            elif d.type is str:
                val = str(val)
            elif d.type is int:
                val = int(round(float(val)))
            else:
                val = float(val)
        except Exception:
            errors.append(f"param '{name}' = {val!r} is not a {d.type.__name__}")
            continue
        if d.choices and val not in d.choices:
            errors.append(f"param '{name}' = {val!r} not in {list(d.choices)}")
            continue
        if d.type in (int, float):
            if d.min is not None and val < d.min:
                errors.append(f"param '{name}' = {val} below minimum {d.min}")
                continue
            if d.max is not None and val > d.max:
                errors.append(f"param '{name}' = {val} above maximum {d.max}")
                continue
        kwargs[name] = val

    return kwargs, errors


def signature_text(name: str) -> str:
    sk = get(name)
    if sk is None:
        return f"{name}: <UNKNOWN SKILL>"
    lines = [f"{sk.name}  [{sk.category}]  — {sk.doc}"]
    lines += [f"    {d.describe()}" for d in sk.params.values()]
    seats = seat_lines(name)
    if seats:
        lines.append(f"    connectors: {', '.join(seats)}")
    else:
        lines.append(f"    connectors: {', '.join(sk.connectors)}")
    return "\n".join(lines)


def vocabulary_digest(categories: Sequence[str] | None = None) -> str:
    """The full catalogue, formatted for the brain's system prompt."""
    out: list[str] = []
    for cat in sorted({s.category for s in REGISTRY.values()}):
        if categories and cat not in categories:
            continue
        out.append(f"### {cat}")
        for n in sorted(s.name for s in REGISTRY.values() if s.category == cat):
            out.append(signature_text(n))
        out.append("")
    return "\n".join(out)
