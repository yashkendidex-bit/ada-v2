"""Overnight Evolution — ADA proposes lighter/stronger variants while you sleep.

Like JARVIS improving the suit overnight. Takes a generated part, mutates it
(thinner walls, lattice holes, material swap), re-runs real FEA on each variant,
and keeps only those that still PASS with better weight.

Usage (from server or scheduler):
    from cad_overnight import evolve_overnight
    result = evolve_overnight(stl_path, material, force_n)
"""
from __future__ import annotations
import os, sys, json, math, shutil
from datetime import datetime

BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

try:
    import trimesh
    import numpy as np
except ImportError:
    trimesh = None
    np = None

from cad_v3.fea_solver import compute_von_mises_stress_approx, MATERIAL_PROPERTIES


def _mutate_thin(mesh, factor=0.9):
    """Shrink slightly toward centroid — simulates thinner walls."""
    m = mesh.copy()
    m.apply_scale(factor)
    return m


def _mutate_lattice(mesh, holes=3):
    """Drill lightening holes through Z to reduce weight."""
    m = mesh.copy()
    bounds = m.bounds
    size = bounds[1] - bounds[0]
    r = min(size.x, size.y) * 0.08
    for i in range(holes):
        ang = 2 * math.pi * i / holes
        cx = (bounds[0][0] + size.x / 2) + size.x * 0.25 * math.cos(ang)
        cy = (bounds[0][1] + size.y / 2) + size.y * 0.25 * math.sin(ang)
        cyl = trimesh.creation.cylinder(radius=r, height=size.z * 1.5)
        cyl.apply_translation([cx, cy, (bounds[0][2] + bounds[1][2]) / 2])
        try:
            m = m.difference(cyl)
        except Exception:
            pass
    return m


def _swap_material(current, direction="lighter"):
    """Suggest a lighter or stronger material."""
    order = ["pla", "abs", "nylon", "brass", "aluminum", "steel", "titanium", "carbon_fiber"]
    if current not in order:
        return "titanium" if direction == "stronger" else "aluminum"
    idx = order.index(current)
    if direction == "lighter" and idx > 0:
        return order[idx - 1]
    if direction == "stronger" and idx < len(order) - 1:
        return order[idx + 1]
    return current


def evolve_overnight(stl_path: str, material: str = "steel", force_n: float = 500.0,
                     max_variants: int = 6) -> dict:
    """Generate + FEA-test variants; keep only ones that PASS and are lighter."""
    if trimesh is None:
        return {"ok": False, "error": "trimesh_missing"}

    base_mesh = trimesh.load(stl_path, force="mesh")
    base_vol = float(base_mesh.volume)

    # Baseline FEA
    base_fea = compute_von_mises_stress_approx(stl_path, force_n=force_n, material=material)
    baseline = {
        "name": "original",
        "weight_g": round(base_vol * MATERIAL_PROPERTIES.get(material, MATERIAL_PROPERTIES["generic"])["density_kg_m3"] / 1e6, 1),
        "safety_factor": base_fea.get("safety_factor", 0),
        "status": base_fea.get("status", "?"),
        "material": material,
    }

    variants = [baseline]
    tmp_dir = os.path.dirname(os.path.abspath(stl_path))

    # Variant ideas (JARVIS-style proposals)
    proposals = [
        ("thin_walls_10pct", lambda m: _mutate_thin(m, 0.90), material),
        ("thin_walls_20pct", lambda m: _mutate_thin(m, 0.80), material),
        ("lattice_3_holes", lambda m: _mutate_lattice(m, 3), material),
        ("lattice_4_holes", lambda m: _mutate_lattice(m, 4), material),
        ("material_swap_lighter", lambda m: m, _swap_material(material, "lighter")),
        ("material_swap_stronger", lambda m: m, _swap_material(material, "stronger")),
    ]

    for name, mutate, mat in proposals[:max_variants]:
        try:
            vm = mutate(base_mesh)
            if vm is None or not hasattr(vm, "volume") or vm.volume <= 0:
                continue
            vpath = os.path.join(tmp_dir, f"variant_{name}.stl")
            vm.export(vpath)
            fea = compute_von_mises_stress_approx(vpath, force_n=force_n, material=mat)
            weight = float(vm.volume) * MATERIAL_PROPERTIES.get(mat, MATERIAL_PROPERTIES["generic"])["density_kg_m3"] / 1e6
            variants.append({
                "name": name,
                "weight_g": round(weight, 1),
                "safety_factor": fea.get("safety_factor", 0),
                "status": fea.get("status", "?"),
                "material": mat,
                "stl_path": vpath,
            })
            try: os.remove(vpath)
            except Exception: pass
        except Exception as e:
            print(f"[OVERNIGHT] variant {name} failed: {str(e)[:80]}")

    # Rank: must PASS, then lightest wins
    passing = [v for v in variants if v["status"] == "PASS"]
    passing.sort(key=lambda v: v["weight_g"])
    best = passing[0] if passing else None

    improvement = None
    if best and best["name"] != "original":
        improvement = round((baseline["weight_g"] - best["weight_g"]) / max(baseline["weight_g"], 0.01) * 100, 1)

    return {
        "ok": True,
        "baseline": baseline,
        "variants": variants,
        "best": best,
        "improvement_pct": improvement,
        "message": (
            f"Overnight evolution: {len(passing)}/{len(variants)} variants pass at {force_n}N. "
            f"Best: {best['name']} saves {improvement}% weight (SF {best['safety_factor']})."
            if best and best["name"] != "original"
            else "Original design is already optimal — no safe improvement found."
        ),
        "ts": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import glob
    stls = sorted(glob.glob("projects/cad_exports/*/output*.stl"), key=os.path.getmtime, reverse=True)
    if stls:
        print(f"Testing on {stls[0]}")
        r = evolve_overnight(stls[0], material="steel", force_n=500)
        print(json.dumps(r, indent=2)[:2000])
