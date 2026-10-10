"""Export the placed assembly without ever copying the B-rep.

Every variant (X-ray colouring, exploded layout) is produced by temporarily
mutating the instances and restoring them afterwards. That matters: a 90-part
turbocharger is ~1.2 MB of STEP but deep-copying its solids to build a second
Compound costs more than the export itself, and `Compound(children=...)`
re-parents its children, which would silently empty the original assembly.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from build123d import Color, Location, Vector, export_gltf, export_step, export_stl

from .. import materials
from ..assembler import Assembly, inner_keys


@dataclass
class ExportResult:
    files: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    def text(self) -> str:
        lines = []
        for path, size in self.files.items():
            lines.append(f"  {Path(path).name:34s} {size / 1024.0:9.1f} KB")
        lines.append(f"  export time: {self.seconds:.2f}s")
        return "\n".join(lines + [f"  note: {n}" for n in self.notes])


# ── reversible scopes ───────────────────────────────────────────────────
@contextmanager
def xray_scope(asm: Assembly):
    """Casings go translucent, whatever they enclose stays hot and opaque."""
    saved = [(i.solid, getattr(i.solid, "color", None)) for i in asm.instances]
    inner = inner_keys(asm)
    try:
        for inst in asm.instances:
            feature = inst.key in inner
            r, g, b, a = materials.rgba(inst.material, xray=True, feature=feature)
            inst.solid.color = Color(r, g, b, a)
        yield asm
    finally:
        for solid, col in saved:
            if col is not None:
                solid.color = col


@contextmanager
def explode_scope(asm: Assembly, factor: float = 0.6):
    """Push every part away from the assembly centroid, proportionally."""
    saved = [(i.solid, i.solid.location) for i in asm.instances]
    try:
        centres: list[Vector] = []
        for inst in asm.instances:
            try:
                centres.append(inst.solid.bounding_box().center())
            except Exception:
                centres.append(Vector(0, 0, 0))
        n = max(1, len(centres))
        hub = Vector(sum(c.X for c in centres) / n,
                     sum(c.Y for c in centres) / n,
                     sum(c.Z for c in centres) / n)
        for inst, c in zip(asm.instances, centres):
            d = c - hub
            try:
                inst.solid.location = Location(d * float(factor)) * inst.solid.location
            except Exception:
                pass
        yield asm
    finally:
        for solid, loc in saved:
            solid.location = loc


# ── single-file writers ─────────────────────────────────────────────────
def export_step_file(asm: Assembly, path: str | Path) -> int:
    export_step(asm.compound, str(path))
    return Path(path).stat().st_size if Path(path).exists() else 0


def export_gltf_file(asm: Assembly, path: str | Path, *, linear: float = 0.06,
                     angular: float = 0.25) -> int:
    export_gltf(asm.compound, str(path), binary=True,
                linear_deflection=linear, angular_deflection=angular)
    return Path(path).stat().st_size if Path(path).exists() else 0


def export_stl_file(asm: Assembly, path: str | Path, *, tol: float = 0.06) -> int:
    export_stl(asm.compound, str(path), tolerance=tol, angular_tolerance=0.25)
    return Path(path).stat().st_size if Path(path).exists() else 0


# ── the bundle the viewer wants ─────────────────────────────────────────
def export_all(asm: Assembly, outdir: str | Path, *, name: str | None = None,
               step: bool = True, gltf: bool = True, xray: bool = True,
               exploded: float = 0.0, stl: bool = False,
               linear: float = 0.06) -> ExportResult:
    t0 = time.time()
    res = ExportResult()
    if asm.compound is None:
        res.notes.append("nothing to export: assembly has no compound")
        return res
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    stem = name or asm.spec.name.replace(" ", "_").lower()

    def _try(label: str, fn):
        try:
            p = out / label
            res.files[str(p)] = fn(p)
        except Exception as exc:
            res.notes.append(f"{label} failed: {type(exc).__name__}: {exc}")

    if step:
        _try(f"{stem}.step", lambda p: export_step_file(asm, p))
    if gltf:
        _try(f"{stem}.glb", lambda p: export_gltf_file(asm, p, linear=linear))
    if stl:
        _try(f"{stem}.stl", lambda p: export_stl_file(asm, p, tol=linear))
    if xray:
        with xray_scope(asm):
            _try(f"{stem}_xray.glb", lambda p: export_gltf_file(asm, p, linear=linear))
    if exploded and exploded > 0:
        with explode_scope(asm, exploded):
            _try(f"{stem}_exploded.glb", lambda p: export_gltf_file(asm, p, linear=linear))

    res.seconds = time.time() - t0
    return res
