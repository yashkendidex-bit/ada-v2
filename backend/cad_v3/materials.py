"""Material table: density for mass properties, colour for render, alpha for X-ray.

Densities are g/cm^3 at room temperature. `xray_alpha` is what makes reference
image 3 possible — a housing drops to 0.12 opacity while whatever it encloses stays
solid, so the internals glow through without any per-model hand tuning. Which parts
those are is measured on the boxes (`assembler.inner_keys`), never named here.
"""

from __future__ import annotations

MATERIALS: dict[str, dict] = {
    "steel":        dict(rho=7.85, color=(0.62, 0.64, 0.67), xray_alpha=0.28, metal=0.95, rough=0.35),
    "stainless":    dict(rho=7.90, color=(0.72, 0.74, 0.76), xray_alpha=0.30, metal=0.95, rough=0.22),
    "cast_iron":    dict(rho=7.20, color=(0.34, 0.34, 0.36), xray_alpha=0.18, metal=0.70, rough=0.72),
    "inconel":      dict(rho=8.44, color=(0.55, 0.52, 0.48), xray_alpha=0.30, metal=0.92, rough=0.42),
    "aluminium":    dict(rho=2.70, color=(0.80, 0.82, 0.85), xray_alpha=0.16, metal=0.90, rough=0.30),
    "aluminum":     dict(rho=2.70, color=(0.80, 0.82, 0.85), xray_alpha=0.16, metal=0.90, rough=0.30),
    "titanium":     dict(rho=4.51, color=(0.66, 0.66, 0.70), xray_alpha=0.26, metal=0.92, rough=0.40),
    "brass":        dict(rho=8.50, color=(0.85, 0.68, 0.32), xray_alpha=0.40, metal=0.95, rough=0.24),
    "bronze":       dict(rho=8.80, color=(0.72, 0.52, 0.30), xray_alpha=0.38, metal=0.95, rough=0.34),
    "copper":       dict(rho=8.96, color=(0.86, 0.50, 0.30), xray_alpha=0.42, metal=0.96, rough=0.26),
    "gold":         dict(rho=19.3, color=(0.95, 0.76, 0.30), xray_alpha=0.95, metal=1.00, rough=0.14),
    "nickel":       dict(rho=8.90, color=(0.74, 0.72, 0.68), xray_alpha=0.32, metal=0.94, rough=0.28),
    "graphite":     dict(rho=2.20, color=(0.20, 0.20, 0.22), xray_alpha=0.35, metal=0.20, rough=0.85),
    "carbon_fibre": dict(rho=1.60, color=(0.14, 0.15, 0.17), xray_alpha=0.30, metal=0.15, rough=0.55),
    "nylon":        dict(rho=1.15, color=(0.90, 0.90, 0.88), xray_alpha=0.14, metal=0.02, rough=0.62),
    "rubber":       dict(rho=1.30, color=(0.12, 0.12, 0.13), xray_alpha=0.55, metal=0.02, rough=0.94),
    "nbr":          dict(rho=1.30, color=(0.14, 0.13, 0.16), xray_alpha=0.60, metal=0.02, rough=0.92),
    "ceramic":      dict(rho=3.90, color=(0.92, 0.92, 0.90), xray_alpha=0.22, metal=0.05, rough=0.40),
    "glass":        dict(rho=2.50, color=(0.85, 0.92, 0.95), xray_alpha=0.08, metal=0.02, rough=0.05),
}

DEFAULT = MATERIALS["steel"]

# Which parts stay opaque and hot in X-ray mode is a question about the assembly, not
# about this table, so it is answered by `assembler.inner_keys` — measured enclosure
# rather than a list of names. The list that used to live here held seven, and all
# seven were a turbocharger's.


def get(name: str) -> dict:
    return MATERIALS.get(str(name).strip().lower().replace("-", "_"), DEFAULT)


def density(name: str) -> float:
    return get(name)["rho"]


def mass_g(name: str, volume_mm3: float) -> float:
    """volume in mm^3 -> mass in grams (rho is g/cm^3, 1 cm^3 = 1000 mm^3)."""
    return density(name) * float(volume_mm3) / 1000.0


def rgba(name: str, *, xray: bool = False, feature: bool = False) -> tuple[float, float, float, float]:
    m = get(name)
    r, g, b = m["color"]
    if not xray:
        return (r, g, b, 1.0)
    if feature:
        return (min(r * 1.25, 1.0), min(g * 1.1, 1.0), b * 0.65, 1.0)
    return (r, g, b, m["xray_alpha"])


def names() -> list[str]:
    return sorted(MATERIALS)
