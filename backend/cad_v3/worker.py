"""In-process build pool with a parameter-hash cache.

cad_v2 launched one Python subprocess per part — roughly 0.9 s of interpreter and
OCCT import cost before any geometry happened, which is 85 s of pure overhead on
a 94-part assembly. Here the kernel is imported once and identical parts (all 12
M6 screws in a bolt circle) are built once and instanced.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from . import registry
from .frames import SkillResult


@dataclass
class BuildStat:
    skill: str
    seconds: float
    cached: bool
    volume: float = 0.0
    error: str = ""


@dataclass
class Worker:
    cache: dict[str, SkillResult] = field(default_factory=dict)
    stats: list[BuildStat] = field(default_factory=list)
    hits: int = 0
    misses: int = 0

    @staticmethod
    def key(skill_name: str, params: dict[str, Any]) -> str:
        blob = json.dumps({"s": skill_name, "p": params}, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    def build(self, skill_name: str, params: dict[str, Any]) -> SkillResult:
        """Build (or fetch) one part. Raises on skill error — the caller repairs."""
        sk = registry.get(skill_name)
        if sk is None:
            raise KeyError(f"unknown skill '{skill_name}'")
        kwargs, errs = registry.coerce_params(sk, params)
        if errs:
            raise ValueError("; ".join(errs))

        k = self.key(skill_name, kwargs)
        if k in self.cache:
            self.hits += 1
            self.stats.append(BuildStat(skill_name, 0.0, True, self.cache[k].solid.volume))
            return self.cache[k]

        self.misses += 1
        t0 = time.time()
        try:
            res = sk.fn(**kwargs)
        except Exception as exc:
            self.stats.append(BuildStat(skill_name, time.time() - t0, False, 0.0, f"{type(exc).__name__}: {exc}"))
            raise
        dt = time.time() - t0
        vol = 0.0
        try:
            vol = float(res.solid.volume)
        except Exception:
            pass
        self.stats.append(BuildStat(skill_name, dt, False, vol))
        self.cache[k] = res
        return res

    def instance(self, res: SkillResult) -> Any:
        """A fresh, independently movable copy of a cached solid."""
        return copy.deepcopy(res.solid)

    # ── reporting ───────────────────────────────────────────────────────
    @property
    def build_seconds(self) -> float:
        return sum(s.seconds for s in self.stats)

    def slowest(self, n: int = 6) -> list[BuildStat]:
        uniq: dict[str, BuildStat] = {}
        for s in self.stats:
            if s.cached:
                continue
            if s.skill not in uniq or s.seconds > uniq[s.skill].seconds:
                uniq[s.skill] = s
        return sorted(uniq.values(), key=lambda s: -s.seconds)[:n]

    def summary(self) -> str:
        return (
            f"{self.misses} unique parts built, {self.hits} instanced from cache, "
            f"{self.build_seconds:.2f}s of geometry"
        )
