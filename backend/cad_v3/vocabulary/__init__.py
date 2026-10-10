"""Importing this package registers the entire vocabulary.

Order matters only for readability; the registry is a flat namespace and skill
names are unique. Adding a new engineering capability to ADA means adding a
module here, not touching the brain, the assembler, or the schema.
"""

from __future__ import annotations

from . import (  # noqa: F401
    bearings,
    fasteners,
    profiles,
    power,
    retaining,
    rotary,
    shape,
    structure,
    turbo_rotor,
    turbo_static,
)

MODULES = (
    "profiles",
    "turbo_rotor",
    "turbo_static",
    "power",
    "bearings",
    "fasteners",
    "retaining",
    "structure",
    "rotary",
    "shape",
)
