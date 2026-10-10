"""Skills ADA synthesised for herself. Imported by bootstrap(), never by hand.

One file per skill, each admitted by the geometry gate before it was written
here. A file that fails to import is skipped and named in FAILED rather than
being allowed to take the whole vocabulary down with it.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from ...registry import REGISTRY

FAILED: list[tuple[str, str]] = []


def load() -> list[str]:
    """Import every generated module; return the skill names that appeared."""
    before = set(REGISTRY)
    here = Path(__file__).resolve().parent
    for mod in sorted(m.name for m in pkgutil.iter_modules([str(here)])):
        try:
            importlib.import_module(f"{__name__}.{mod}")
        except Exception as exc:
            FAILED.append((mod, f"{type(exc).__name__}: {exc}"))
    fresh = sorted(set(REGISTRY) - before)
    for name in fresh:
        REGISTRY[name].source = "synth"
    return fresh
