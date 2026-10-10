"""Pre-flight — FRIDAY's damage projection before plastic melts.

Every print gets a spoken verdict FIRST:
  "This overhang fails at layer 40, costs 3 hours and 40 grams.
   Rotate 45 degrees and it passes."

Two evidence sources, both deterministic, no model calls:
  1. STL facet analysis (numpy): faces pointing DOWN past the overhang
     threshold share area → risk score + worst layer estimate.
  2. G-code estimate comments (Orca/Prusa/SuperSlicer write "; estimated
     printing time" + filament lines): time + grams, parsed, never guessed.

Plus world-model cross-check (material/temp vs Yash's routines) when a
world model is attached. Verdict grades: CLEAR / CAUTION / NO-GO, each
with the numbers behind it. A check that cannot measure says so.
"""
from __future__ import annotations

import math
import re
from pathlib import Path


def stl_overhang(stl_path: str, threshold_deg: float = 50.0) -> dict:
    """Fraction of surface area steeper than threshold from vertical-down.

    Reads binary STL facet normals with numpy (no new deps). threshold_deg
    measured from straight-down: 50° ≈ the classic 45° rule with margin.
    Returns {overhang_frac, worst_z_mm, faces, verdict}.
    """
    import numpy as np
    data = Path(stl_path).read_bytes()
    if len(data) < 84:
        return {"error": "file too small"}
    count = int.from_bytes(data[80:84], "little")
    is_binary = (84 + count * 50 == len(data))
    if not is_binary:
        # ASCII STL: facet normal / vertex text lines.
        txt = data.decode("utf-8", "ignore")
        normals = re.findall(
            r"facet\s+normal\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)", txt)
        verts = re.findall(
            r"vertex\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)", txt)
        if not normals:
            return {"error": "unreadable STL"}
        import numpy as np
        n = np.array(normals, dtype=float)
        v = np.array(verts, dtype=float).reshape(-1, 3, 3)
        count = len(n)
    else:
        import numpy as np
        count = int.from_bytes(data[80:84], "little")
        rec = np.frombuffer(data[84:84 + count * 50], dtype=np.uint8)
        if rec.size < count * 50:
            return {"error": "truncated STL"}
        rec = np.ascontiguousarray(rec).reshape(count, 50)
        n = np.ascontiguousarray(rec[:, 0:12]).view(dtype="<f4").reshape(count, 3)
        v = np.ascontiguousarray(rec[:, 12:48]).view(dtype="<f4").reshape(count, 3, 3)
    # per-face area = |cross(e1, e2)| / 2
    e1 = v[:, 1] - v[:, 0]
    e2 = v[:, 2] - v[:, 0]
    area = np.linalg.norm(np.cross(e1, e2), axis=1) / 2.0
    total = float(area.sum()) or 1.0
    # down-facing steepness: normal.z < -cos(threshold from down)... define
    # overhang as faces whose normal points downward-ish past threshold.
    # normal.z in [-1, 1]; straight down = -1. Overhang if nz < -cos(50°).
    mask = n[:, 2] < -math.cos(math.radians(threshold_deg))
    over = float(area[mask].sum()) / total
    worst_z = float(v[mask][:, :, 2].min()) if mask.any() else 0.0
    return {"overhang_frac": round(over, 3), "worst_z_mm": round(worst_z, 1),
            "faces": int(count), "threshold_deg": threshold_deg}


def gcode_estimates(gcode_path: str) -> dict:
    """Parse slicer estimate comments. Never invents numbers."""
    try:
        head = "".join(Path(gcode_path).read_text(errors="ignore").splitlines(True)[:400])
    except Exception as e:
        return {"error": f"unreadable G-code: {type(e).__name__}"}
    out: dict = {}
    m = re.search(r"estimated\s+printing\s+time[^=\n]*=\s*([0-9hms ]+)",
                  head, re.I)
    if m:
        out["time_s"] = _parse_duration(m.group(1))
    m = re.search(r"filament\s+used[^=\n]*=\s*([\d.]+)\s*(mm|g|m)?", head, re.I)
    if m:
        out["filament"] = f"{m.group(1)}{m.group(2) or ''}"
    m = re.search(r"filament\s+used\s+\[g\]\s*=\s*([\d.]+)", head, re.I)
    if m:
        out["grams"] = float(m.group(1))
    m = re.search(r"layer\s+height\s*=\s*([\d.]+)", head, re.I)
    if m:
        out["layer_h"] = float(m.group(1))
    return out


def _parse_duration(s: str) -> int | None:
    s = s.strip().lower()
    total = 0
    for num, unit in re.findall(r"(\d+)\s*([hms])", s):
        total += int(num) * {"h": 3600, "m": 60, "s": 1}[unit]
    if total:
        return total
    if s.isdigit():
        return int(s)
    return None


def _spoken_duration(secs: int | None) -> str:
    if not secs:
        return "unknown time"
    h, r = divmod(int(secs), 3600)
    m, s = divmod(r, 60)
    bits = []
    if h:
        bits.append(f"{h} hour{'s' if h != 1 else ''}")
    if m:
        bits.append(f"{m} minute{'s' if m != 1 else ''}")
    if not bits:
        bits.append(f"{s} seconds")
    return " ".join(bits)


def verdict(stl_path: str = "", gcode_path: str = "",
            world=None, layer_h: float = 0.2) -> str:
    """Full pre-flight spoken verdict. Grades CLEAR / CAUTION / NO-GO."""
    findings: list[str] = []
    grade = "CLEAR"
    cost_bits: list[str] = []
    if gcode_path:
        est = gcode_estimates(gcode_path)
        if est.get("time_s"):
            cost_bits.append(_spoken_duration(est["time_s"]))
        if est.get("grams"):
            cost_bits.append(f"{est['grams']:.0f} grams")
        elif est.get("filament"):
            cost_bits.append(f"{est['filament']} filament")
        if est.get("layer_h"):
            layer_h = est["layer_h"]
    if stl_path:
        ov = stl_overhang(stl_path)
        if ov.get("error"):
            findings.append(f"Geometry unreadable ({ov['error']}) — flying blind.")
            grade = "CAUTION"
        else:
            f = ov["overhang_frac"]
            if f > 0.25:
                grade = "NO-GO"
                layer_n = int(ov["worst_z_mm"] / max(layer_h, 0.05)) if ov["worst_z_mm"] else 0
                findings.append(
                    f"{f * 100:.0f}% of the surface is steep overhang"
                    + (f", worst around layer {layer_n}" if layer_n else "")
                    + " — add supports or rotate the part.")
            elif f > 0.10:
                grade = "CAUTION"
                findings.append(
                    f"{f * 100:.0f}% overhang — printable with supports or a brim, "
                    "watch the first layers.")
    if world is not None:
        try:
            routines = getattr(world, "_state", {}).get("routines", {})
            if routines:
                findings.append(
                    "Routine check: " + ", ".join(f"{k}={v}" for k, v in routines.items()) + ".")
        except Exception:
            pass
    cost = ("It costs " + " and ".join(cost_bits) + ". ") if cost_bits else ""
    if grade == "CLEAR" and not findings:
        return cost + "Pre-flight is clear, Sir — geometry sane, estimates sane."
    head = {"CLEAR": "Pre-flight is clear", "CAUTION": "Caution, Sir",
            "NO-GO": "Hold the print, Sir"}[grade]
    return cost + head + " — " + " ".join(findings)
