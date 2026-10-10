"""Spec validation — the anti-hallucination gate.

Every rule here exists because a specific class of LLM error would otherwise
reach the CAD kernel and produce either a crash or, worse, a plausible-looking
wrong model. The error text is written for the repair loop to consume: it names
the offending path, the code, and the legal alternatives.

One rule is deliberately switchable. `allow_unknown=True` stops rule 2 from
rejecting a skill the vocabulary does not have yet, because an unknown skill is
not a hallucination — it is a *work order* (see synth.py). Everything else still
applies, so the spec is judged complete in every respect except the words that
have yet to be written. Callers that are about to build, rather than about to
synthesise, must leave it False: by then every named skill really does exist,
as either a synthesised part or a placeholder envelope.

The rule set grew when the schema did. A spec can now declare variables with
equations between them, reference frames to measure from, dimensions the finished
model must hit, and engineering conditions it must satisfy — and every one of
those is a new way for a spec to be wrong on paper, which is the cheapest place
to catch it.
"""

from __future__ import annotations

from . import graph, registry
from .schema import (BANNED_PARAM_KEYS, CHECK_KINDS, FITS, FLIP_MODES, MATE_KINDS,
                     MEASURES, ORIENTS, STATION_ORDERS, WORLD, MateSpec, Spec,
                     SpecError)

# What each engineering condition needs before conditions.py can evaluate it.
CHECK_ARGS: dict[str, tuple[str, ...]] = {
    "planetary": ("sun", "planet", "ring"),
    "gear_mesh": ("a", "b"),
    "bolt_circle": ("a", "b"),
    "fit": ("shaft", "bore"),
    "coaxial": ("a", "b"),
    "stack_length": ("parts",),
    "count": ("part",),
    "clearance": ("a", "b"),
}


def unknown_skills(spec: Spec) -> list[str]:
    """The vocabulary gap this spec implies, whatever the caller does about it."""
    return sorted({p.skill for p in spec.parts if registry.get(p.skill) is None})


def validate(spec: Spec, *, allow_unknown: bool = False) -> list[SpecError]:
    errs: list[SpecError] = []
    if not spec.baked:
        spec.resolve()

    # ── rule 0: the equations have to work before anything else can ─────
    for msg in spec.var_errors:
        errs.append(SpecError("vars", "BAD_EQUATION", msg,
                              hint="an expression may use the other variables, "
                                   "arithmetic and sin/cos/sqrt/min/max"))

    if not spec.parts:
        errs.append(SpecError("parts", "EMPTY", "a spec must contain at least one part"))
        return errs

    errs += _check_parts(spec, allow_unknown)
    errs += _check_datums(spec)
    errs += _check_mates(spec, allow_unknown)
    errs += _check_reach(spec)
    errs += _check_patterns(spec, allow_unknown)
    errs += _check_expectations(spec)
    return errs


# ── rule 1-4: the parts themselves ──────────────────────────────────────────

def _check_parts(spec: Spec, allow_unknown: bool) -> list[SpecError]:
    errs: list[SpecError] = []
    seen: set[str] = set()
    frames = set(spec.frame_ids)
    for i, part in enumerate(spec.parts):
        if not part.id:
            errs.append(SpecError(f"parts[{i}].id", "MISSING_ID", "every part needs a stable id"))
        elif part.id in seen:
            errs.append(SpecError(f"parts[{i}].id", "DUPLICATE_ID",
                                  f"id '{part.id}' is used more than once"))
        elif part.id == WORLD or part.id.startswith("@"):
            errs.append(SpecError(f"parts[{i}].id", "RESERVED_ID",
                                  f"'{part.id}' is reserved",
                                  hint="'@' introduces a datum and WORLD is the origin"))
        elif part.id in frames:
            errs.append(SpecError(f"parts[{i}].id", "ID_COLLIDES_WITH_DATUM",
                                  f"'{part.id}' is also a datum id",
                                  hint="rename one of them; a mate reference must be unambiguous"))
        seen.add(part.id)
        if part.count < 1:
            errs.append(SpecError(f"parts[{i}].count", "BAD_COUNT", "count must be >= 1"))

        sk = registry.get(part.skill)
        if sk is None:
            if not allow_unknown:
                errs.append(SpecError(
                    f"parts[{i}].skill", "UNKNOWN_SKILL",
                    f"'{part.skill}' is not a registered skill",
                    hint=f"choose one of: {', '.join(registry.names())}",
                ))
            continue

        # ── no coordinates may enter through params ─────────────────────
        smuggled = sorted(set(k.lower() for k in part.params) & BANNED_PARAM_KEYS)
        if smuggled:
            errs.append(SpecError(
                f"parts[{i}].params", "COORDINATE_IN_PARAMS",
                f"params may not contain positional keys {smuggled}",
                hint="a param is a dimension of the part, never a position of it; "
                     "use dx/dy/dz on a mate or declare a datum",
            ))

        # ── params type/range check, after the equations are resolved ───
        _, perrs = registry.coerce_params(sk, spec.params_of(part))
        for msg in perrs:
            errs.append(SpecError(f"parts[{i}].params", "BAD_PARAM", msg,
                                  hint=registry.signature_text(part.skill)))

        if part.anchor and part.anchor not in sk.connectors:
            errs.append(SpecError(f"parts[{i}].anchor", "UNKNOWN_CONNECTOR",
                                  f"skill '{part.skill}' publishes no connector '{part.anchor}'",
                                  hint=f"available: {', '.join(sk.connectors)}"))
    return errs


# ── rule 5: declared reference frames ───────────────────────────────────────

def _check_datums(spec: Spec) -> list[SpecError]:
    """A datum chain that does not reach the origin positions nothing.

    Cycles are the interesting failure: `a` measured from `b` and `b` from `a` is a
    perfectly readable pair of sentences and an unsolvable pair of constraints, and
    the traversal would simply never place either of them.
    """
    errs: list[SpecError] = []
    ids = set(spec.ids)
    seen: set[str] = set()
    for i, dat in enumerate(spec.datums):
        if not dat.id:
            errs.append(SpecError(f"datums[{i}].id", "MISSING_ID", "every datum needs an id"))
            continue
        if dat.id in seen:
            errs.append(SpecError(f"datums[{i}].id", "DUPLICATE_ID",
                                  f"datum id '{dat.id}' is used more than once"))
        seen.add(dat.id)
        if dat.kind not in ("frame", "plane", "axis", "point"):
            errs.append(SpecError(f"datums[{i}].kind", "BAD_KIND", f"'{dat.kind}' invalid",
                                  hint="one of frame, plane, axis, point"))
        parent = dat.parent
        if parent == WORLD:
            continue
        if parent in ids:
            part = spec.part(parent)
            sk = registry.get(part.skill) if part else None
            if dat.parent_seat and sk and dat.parent_seat not in sk.connectors:
                errs.append(SpecError(
                    f"datums[{i}].on", "UNKNOWN_CONNECTOR",
                    f"skill '{part.skill}' publishes no connector '{dat.parent_seat}'",
                    hint=f"available: {', '.join(sk.connectors)}"))
            continue
        if parent not in seen and parent not in set(spec.frame_ids):
            errs.append(SpecError(f"datums[{i}].on", "UNKNOWN_ANCHOR",
                                  f"'{dat.on}' is neither a part nor a declared datum",
                                  hint=f"parts: {', '.join(sorted(ids))}; "
                                       f"datums: {', '.join('@' + d for d in spec.frame_ids)}"))

    for cycle in _datum_cycles(spec):
        errs.append(SpecError("datums", "DATUM_CYCLE",
                              f"these datums are defined in terms of each other: {cycle}",
                              hint="every datum chain has to end at a part or at the origin"))
    return errs


def _datum_cycles(spec: Spec) -> list[list[str]]:
    """Datum ids that can never be resolved because their chain never terminates."""
    parent = {d.id: d.parent for d in spec.datums}
    frames = set(parent)
    bad: list[list[str]] = []
    for start in sorted(frames):
        walk: list[str] = []
        node = start
        while node in frames and node not in walk:
            walk.append(node)
            node = parent.get(node, WORLD)
        if node in walk:
            loop = walk[walk.index(node):]
            if sorted(loop) not in [sorted(b) for b in bad]:
                bad.append(loop)
    return bad


# ── rule 6: the mates ───────────────────────────────────────────────────────

def _resolve_side(spec: Spec, ref: str, allow_unknown: bool) -> tuple[str, str, str]:
    """(node, connector, error_code) for one mate side. '' means it resolved."""
    if MateSpec.is_frame(ref):
        fid = MateSpec.frame_id(ref)
        if spec.frame(fid) is None:
            return (fid, "", "UNKNOWN_DATUM")
        if "." in str(ref):
            return (fid, "", "DATUM_HAS_NO_CONNECTOR")
        return (fid, "", "")
    pid, conn = MateSpec.split(ref)
    if pid not in set(spec.ids):
        return (pid, conn, "UNKNOWN_PART")
    if not conn:
        return (pid, conn, "MISSING_CONNECTOR")
    part = spec.part(pid)
    sk = registry.get(part.skill) if part else None
    if sk is None:
        return (pid, conn, "" if allow_unknown else "UNKNOWN_SKILL")
    if conn not in sk.connectors:
        return (pid, conn, "UNKNOWN_CONNECTOR")
    return (pid, conn, "")


def _check_mates(spec: Spec, allow_unknown: bool) -> list[SpecError]:
    errs: list[SpecError] = []
    ids = set(spec.ids)
    for i, m in enumerate(spec.mates):
        nodes: list[str] = []
        for side, ref in (("a", m.a), ("b", m.b)):
            node, conn, code = _resolve_side(spec, ref, allow_unknown)
            nodes.append(node)
            if not code:
                continue
            if code == "UNKNOWN_PART":
                errs.append(SpecError(f"mates[{i}].{side}", code,
                                      f"'{node}' is not a part id",
                                      hint=f"known ids: {', '.join(sorted(ids))}"))
            elif code == "MISSING_CONNECTOR":
                errs.append(SpecError(f"mates[{i}].{side}", code,
                                      f"'{ref}' must be of the form part_id.connector"))
            elif code == "UNKNOWN_CONNECTOR":
                part = spec.part(node)
                sk = registry.get(part.skill) if part else None
                errs.append(SpecError(
                    f"mates[{i}].{side}", code,
                    f"skill '{part.skill if part else '?'}' publishes no connector '{conn}'",
                    hint=f"available: {', '.join(sk.connectors) if sk else '-'}"))
            elif code == "UNKNOWN_DATUM":
                errs.append(SpecError(f"mates[{i}].{side}", code,
                                      f"'{ref}' names no declared datum",
                                      hint=f"declared: {', '.join('@' + d for d in spec.frame_ids) or 'none'}"))
            elif code == "DATUM_HAS_NO_CONNECTOR":
                errs.append(SpecError(f"mates[{i}].{side}", code,
                                      f"'{ref}' — a datum is a single frame",
                                      hint=f"write '@{node}'"))
        if m.kind not in MATE_KINDS:
            errs.append(SpecError(f"mates[{i}].kind", "BAD_KIND",
                                  f"'{m.kind}' invalid", hint=f"one of {list(MATE_KINDS)}"))
        if m.fit not in FITS:
            errs.append(SpecError(f"mates[{i}].fit", "BAD_FIT",
                                  f"'{m.fit}' invalid", hint=f"one of {list(FITS)}"))
        if m.orient not in ORIENTS:
            errs.append(SpecError(f"mates[{i}].orient", "BAD_ORIENT",
                                  f"'{m.orient}' invalid", hint=f"one of {list(ORIENTS)}"))
        if m.flip not in FLIP_MODES:
            errs.append(SpecError(f"mates[{i}].flip", "BAD_FLIP",
                                  f"'{m.flip}' invalid", hint=f"one of {list(FLIP_MODES)}"))
        if m.station_order not in STATION_ORDERS:
            errs.append(SpecError(f"mates[{i}].station_order", "BAD_STATION_ORDER",
                                  f"'{m.station_order}' invalid",
                                  hint=f"one of {list(STATION_ORDERS)}"))
        if nodes[0] == nodes[1] and nodes[0]:
            errs.append(SpecError(f"mates[{i}]", "SELF_MATE",
                                  f"'{nodes[0]}' cannot be mated to itself"))
        if MateSpec.is_frame(m.a) and MateSpec.is_frame(m.b):
            errs.append(SpecError(f"mates[{i}]", "FRAME_TO_FRAME",
                                  "a mate between two datums moves nothing",
                                  hint="datums are positioned by their own `on`; "
                                       "mate a part to one of them instead"))
    return errs


# ── rule 7: everything must reach the origin ────────────────────────────────

def _check_reach(spec: Spec) -> list[SpecError]:
    """A part with no chain back to the origin has no position at all.

    Reachability is measured over parts *and* datums, from `WORLD`, because a part
    mated to a declared frame is properly positioned even though no part-to-part
    chain connects it to the datum part.
    """
    errs: list[SpecError] = []
    ids = set(spec.ids)
    if not spec.datum:
        errs.append(SpecError("datum", "MISSING_DATUM",
                              "name the part that defines the assembly origin",
                              hint=f"e.g. '{spec.parts[0].id}'"))
    elif spec.datum not in ids:
        errs.append(SpecError("datum", "UNKNOWN_DATUM", f"'{spec.datum}' is not a part id",
                              hint=f"known ids: {', '.join(sorted(ids))}"))
    if not spec.datum or spec.datum not in ids:
        return errs

    orphans = [n for n in graph.unreachable(WORLD, spec.nodes, spec.graph_edges())
               if n in ids]
    if orphans:
        errs.append(SpecError(
            "mates", "FLOATING_PARTS",
            f"{len(orphans)} part(s) have no mate path to the datum: {orphans[:8]}",
            hint="every part must be positioned by a mate chain or by a datum; "
                 "add a mate for each",
        ))
    return errs


# ── rule 8: a patterned part needs a patterned seat ─────────────────────────

def _check_patterns(spec: Spec, allow_unknown: bool) -> list[SpecError]:
    errs: list[SpecError] = []
    for i, part in enumerate(spec.parts):
        if part.count <= 1:
            continue
        host = None
        for m in spec.mates:
            if part.id not in (spec.node_of(m.a), spec.node_of(m.b)):
                continue
            other = m.b if spec.node_of(m.a) == part.id else m.a
            if MateSpec.is_frame(other):
                # A ring of parts hung off one declared frame is legitimate when the
                # mate itself spreads them, which is what `stations` is for.
                if m.stations:
                    host = (other, "")
                    break
                continue
            opid, oconn = MateSpec.split(other)
            opart = spec.part(opid)
            osk = registry.get(opart.skill) if opart else None
            if osk and oconn in osk.connectors:
                host = (opid, oconn)
                break
            # A host whose skill has not been written yet cannot be judged;
            # synth.py sizes that seat from this very count.
            if osk is None and opart is not None and oconn and allow_unknown:
                host = (opid, oconn)
                break
        if host is None:
            errs.append(SpecError(
                f"parts[{i}]", "PATTERN_WITHOUT_SEAT",
                f"count={part.count} but '{part.id}' has no mate to a patterned connector",
                hint="mate it to a bolt_circle / vane_ring / seats connector whose seat "
                     "count matches, or give the mate an explicit `stations` list",
            ))
    return errs


# ── rule 9: what the finished assembly is claimed to measure ────────────────

def _check_expectations(spec: Spec) -> list[SpecError]:
    """Dimensions and conditions have to name things that exist.

    These checks are cheap and their absence was expensive: a dimension that quietly
    refers to a part id with a typo would be skipped at verify time, and skipping a
    check reads exactly like passing it.
    """
    errs: list[SpecError] = []
    for i, dim in enumerate(spec.expect.dims):
        for side, ref in (("of", dim.of), ("to", dim.to)):
            if not ref:
                continue
            if not _anchor_exists(spec, ref):
                errs.append(SpecError(f"expect.dims[{i}].{side}", "UNKNOWN_ANCHOR",
                                      f"'{ref}' is neither a part, a connector nor a datum",
                                      hint=f"parts: {', '.join(sorted(spec.ids))}"))
        if dim.measure not in MEASURES:
            errs.append(SpecError(f"expect.dims[{i}].measure", "BAD_MEASURE",
                                  f"'{dim.measure}' is not measurable",
                                  hint=f"one of {list(MEASURES)}"))
        if dim.tol <= 0:
            errs.append(SpecError(f"expect.dims[{i}].tol", "BAD_TOLERANCE",
                                  "a tolerance of zero can never be met",
                                  hint="state the tolerance the drawing states, e.g. 0.2"))
        if dim.measure in ("distance", "distance_xy", "axial", "radial", "gap", "angle") \
                and not dim.to:
            errs.append(SpecError(f"expect.dims[{i}].to", "MISSING_ANCHOR",
                                  f"'{dim.measure}' is a measurement between two things",
                                  hint="name the second anchor in `to`"))
    for i, chk in enumerate(spec.checks):
        if chk.kind not in CHECK_KINDS:
            errs.append(SpecError(f"checks[{i}].kind", "BAD_CHECK",
                                  f"'{chk.kind}' is not a known condition",
                                  hint=f"one of {list(CHECK_KINDS)}"))
            continue
        missing = [k for k in CHECK_ARGS.get(chk.kind, ()) if chk.get(k) in (None, "", [])]
        if missing:
            errs.append(SpecError(f"checks[{i}]", "CHECK_INCOMPLETE",
                                  f"'{chk.kind}' needs {missing}",
                                  hint=f"required: {list(CHECK_ARGS.get(chk.kind, ()))}"))
    return errs


def _anchor_exists(spec: Spec, ref: str) -> bool:
    if MateSpec.is_frame(ref):
        return spec.frame(ref) is not None
    pid, conn = MateSpec.split(ref)
    part = spec.part(pid)
    if part is None:
        return False
    if not conn:
        return True
    sk = registry.get(part.skill)
    return sk is None or conn in sk.connectors


def summary(errs: list[SpecError]) -> str:
    if not errs:
        return "spec OK"
    return "\n".join(str(e) for e in errs)


# ── deterministic near-miss repair ──────────────────────────────────────────
# Every rule above computes the legal alternatives before it rejects anything. If
# the word the model wrote is one edit away from a legal one — `bore` for
# `bore_front`, `sliding` for `linear` — then rejecting the spec spends a whole
# model call to be told what this module already knows. So fix that class here,
# for free, and let the model's one retry be spent on the errors that are real
# design mistakes. Nothing here invents geometry: it only substitutes a legal
# token for an illegal one, or deletes something the schema forbids outright.

def _closest(word: str, candidates, cutoff: float = 0.62) -> str | None:
    """The candidate a near-miss unambiguously meant, or None to leave it alone."""
    import difflib

    word = (word or "").strip()
    cands = [c for c in candidates if c]
    if not word or not cands:
        return None
    for c in cands:
        if c.lower() == word.lower():
            return c
    # `bore` -> `bore_front` is a containment signal, which beats edit distance.
    # Several candidates can share the stem (`bore_front`, `bore_rear`); take the
    # first in declaration order, because that is the skill author's own default
    # and the geometry repair loop measures the result either way.
    stem = [c for c in cands if c.lower().startswith(word.lower())
            or c.lower().endswith(word.lower())]
    if stem:
        return stem[0]
    hit = difflib.get_close_matches(word.lower(), [c.lower() for c in cands],
                                    n=1, cutoff=cutoff)
    if not hit:
        return None
    return next(c for c in cands if c.lower() == hit[0])


def _fix_mates(spec: Spec, notes: list[str], *, allow_unknown: bool) -> None:
    """Legal-token substitution on every mate, plus removal of self-loops."""
    ids = set(spec.ids)
    frames = set(spec.frame_ids)
    keep = []
    for i, m in enumerate(spec.mates):
        node_a, node_b = spec.node_of(m.a), spec.node_of(m.b)
        if node_a and node_a == node_b:
            notes.append(f"mates[{i}]: dropped a self-mate on '{node_a}'")
            continue
        for side in ("a", "b"):
            ref = getattr(m, side)
            if MateSpec.is_frame(ref):
                fid = MateSpec.frame_id(ref)
                if fid not in frames:
                    better = _closest(fid, sorted(frames), cutoff=0.8)
                    if better:
                        notes.append(f"mates[{i}].{side}: datum '@{fid}' -> '@{better}'")
                        setattr(m, side, f"@{better}")
                else:
                    setattr(m, side, f"@{fid}")
                continue
            pid, conn = MateSpec.split(ref)
            if pid not in ids:
                # A wrong part id silently rewires the machine, so only a very
                # close typo is corrected; anything looser goes to the model.
                better = _closest(pid, sorted(ids), cutoff=0.84)
                if better is None:
                    continue
                notes.append(f"mates[{i}].{side}: part '{pid}' -> '{better}'")
                pid = better
            part = spec.part(pid)
            sk = registry.get(part.skill) if part else None
            if sk and conn and conn not in sk.connectors:
                better = _closest(conn, sk.connectors)
                if better:
                    notes.append(f"mates[{i}].{side}: '{part.skill}.{conn}' "
                                 f"-> '{better}'")
                    conn = better
            setattr(m, side, f"{pid}.{conn}" if conn else pid)
        for name, table, fallback in (("kind", MATE_KINDS, "rigid"),
                                      ("fit", FITS, "slip"),
                                      ("orient", ORIENTS, "seat"),
                                      ("flip", FLIP_MODES, "auto"),
                                      ("station_order", STATION_ORDERS, "world")):
            value = getattr(m, name)
            if value in table:
                continue
            better = _closest(str(value), table) or fallback
            notes.append(f"mates[{i}].{name}: '{value}' -> '{better}'")
            setattr(m, name, better)
        if m.stations:
            child = spec.part(spec.node_of(m.b)) or spec.part(spec.node_of(m.a))
            limit = max(1, child.count if child else 1)
            trimmed = [int(s) for s in m.stations][:limit]
            if trimmed != list(m.stations):
                notes.append(f"mates[{i}].stations: {list(m.stations)} -> {trimmed} "
                             f"(the moved part has {limit} instance(s))")
                m.stations = trimmed
        keep.append(m)
    spec.mates = keep


def _fix_parts(spec: Spec, notes: list[str]) -> None:
    """Delete what the schema forbids; clamp what it bounds; fix near-miss names."""
    from .materials import MATERIALS

    for i, part in enumerate(spec.parts):
        if part.count < 1:
            notes.append(f"parts[{i}].count: {part.count} -> 1")
            part.count = 1
        if part.material not in MATERIALS:
            better = _closest(part.material, sorted(MATERIALS))
            if better:
                notes.append(f"parts[{i}].material: '{part.material}' -> '{better}'")
                part.material = better
        smuggled = sorted(k for k in part.params if k.lower() in BANNED_PARAM_KEYS)
        for key in smuggled:
            part.params.pop(key, None)
        if smuggled:
            notes.append(f"parts[{i}].params: deleted coordinate key(s) {smuggled} "
                         "— pose comes from mates and datums")
        sk = registry.get(part.skill)
        if sk is None:
            continue
        for key in [k for k in part.params if k not in sk.params]:
            better = _closest(key, sorted(sk.params))
            if better and better not in part.params:
                part.params[better] = part.params.pop(key)
                notes.append(f"parts[{i}].params: '{key}' -> '{better}'")
            else:
                part.params.pop(key, None)
                notes.append(f"parts[{i}].params: dropped unknown '{key}' "
                             f"for skill '{part.skill}'")
        if part.anchor and part.anchor not in sk.connectors:
            better = _closest(part.anchor, sk.connectors)
            notes.append(f"parts[{i}].anchor: '{part.anchor}' -> '{better or '(dropped)'}'")
            part.anchor = better or ""
        baked = spec.params_of(part)
        for key, val in list(part.params.items()):
            d = sk.params.get(key)
            raw = baked.get(key, val)
            if d is None or d.type not in (int, float) or not isinstance(raw, (int, float)):
                continue
            lo, hi = d.min, d.max
            # A formula is clamped by rewriting the field to the bound, which loses
            # the equation — but an equation whose value the skill cannot build is
            # not a design intent worth preserving, and the note says what happened.
            if lo is not None and raw < lo:
                part.params[key] = d.type(lo)
                notes.append(f"parts[{i}].params.{key}: {raw} -> {lo} (declared minimum)")
            elif hi is not None and raw > hi:
                part.params[key] = d.type(hi)
                notes.append(f"parts[{i}].params.{key}: {raw} -> {hi} (declared maximum)")


def _fix_datums(spec: Spec, notes: list[str]) -> None:
    """Point every datum at something that exists, and break any cycle found."""
    ids = set(spec.ids)
    frames = [d.id for d in spec.datums]
    for i, dat in enumerate(spec.datums):
        parent = dat.parent
        if parent == WORLD or parent in ids or parent in frames:
            continue
        better = _closest(parent, sorted(ids) + frames, cutoff=0.8)
        if better is None:
            notes.append(f"datums[{i}].on: '{dat.on}' does not exist -> the origin")
            dat.on = ""
            continue
        prefix = "@" if better in frames and better not in ids else ""
        seat = f".{dat.parent_seat}" if dat.parent_seat and not prefix else ""
        notes.append(f"datums[{i}].on: '{dat.on}' -> '{prefix}{better}{seat}'")
        dat.on = f"{prefix}{better}{seat}"
    for cycle in _datum_cycles(spec):
        victim = cycle[0]
        for dat in spec.datums:
            if dat.id == victim:
                notes.append(f"datums['{victim}'].on: '{dat.on}' -> the origin "
                             f"(broke the cycle {cycle})")
                dat.on = ""


def _fix_datum(spec: Spec, notes: list[str]) -> None:
    """A missing or misspelled datum is never a design decision worth a model call.

    The datum is by definition the part everything else hangs off, so the most
    heavily mated part is the right answer; ties break on spec order, which is the
    order the model itself thought in.
    """
    ids = set(spec.ids)
    if spec.datum in ids:
        return
    guess = None
    if spec.datum:
        guess = _closest(spec.datum, sorted(ids), cutoff=0.8)
    if guess is None:
        degree = {p.id: 0 for p in spec.parts}
        for m in spec.mates:
            for pid in (spec.node_of(m.a), spec.node_of(m.b)):
                if pid in degree:
                    degree[pid] += 1
        guess = max(spec.parts, key=lambda p: degree.get(p.id, 0)).id
    notes.append(f"datum: '{spec.datum or '(missing)'}' -> '{guess}' "
                 "(the most heavily mated part)")
    spec.datum = guess


def autofix(spec: Spec, *, allow_unknown: bool = False) -> list[str]:
    """Repair the near-misses in `spec` in place. Returns what was changed.

    Called before the model is asked to fix anything, so the retry it does get is
    spent on real design errors rather than on spelling. Idempotent: running it on
    a clean spec changes nothing and returns [].
    """
    notes: list[str] = []
    if not spec.parts:
        return notes
    if not spec.baked:
        spec.resolve()
    _fix_parts(spec, notes)
    _fix_datums(spec, notes)
    _fix_mates(spec, notes, allow_unknown=allow_unknown)
    _fix_datum(spec, notes)
    if notes:
        spec.resolve()
    return notes



