"""World Model — prophecy, not parlor tricks (ComBodied PWM, honest edition).

A Personal World Model predicts Yash's state–event trajectory under
alternative scenarios. This V1 is deliberately NOT a neural simulator:
it is routines + pace math + deviation checks, all inspectable, all
correctable, uncertainty stated as ranges. It never invents physics;
when it has no data it says so (metamemory applies here too).

Stores ada_kernel/world.json:
  routines: {material, nozzle_c, bed_c, slicer, units, ...} — print/shop defaults
  projects: {name: {deadline_ts, milestones_total, milestones_done[],
                    notes[]}} — pace math lives here
  deviations: log of flagged mismatches (for the diary, not nagging)

Three verbs:
  note()    — record a routine default, project, milestone, deadline
  check()   — compare a planned action against routines + pace
  compare() — two options side by side from stored facts + pace math
  watch()   — proactive signals (deadline risk), for the signal loop
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import time
from pathlib import Path


# LLM extraction gate. Stdlib path is the default and fully functional;
# cloud extraction costs money, so it stays OFF unless the env flag is "1"
# AND a callable was supplied to WorldModel(llm=...).
LLM_EXTRACT_ENV = "ADA_WORLD_LLM_EXTRACT"


def _parse_temp(text: str) -> dict:
    """Pull nozzle/bed temps + material mentions out of free text."""
    t = (text or "").lower()
    out = {}
    m = re.search(r"nozzle\D{0,8}(\d{2,3})\s*°?c?", t)
    if m:
        out["nozzle_c"] = int(m.group(1))
    m = re.search(r"bed\D{0,8}(\d{2,3})\s*°?c?", t)
    if m:
        out["bed_c"] = int(m.group(1))
    m = re.search(r"\b(petg|pla|abs|asa|tpu|nylon|pc)\b", t)
    if m:
        out["material"] = m.group(1).upper()
    return out


class WorldModel:
    def __init__(self, state_dir: str | Path, *, bus=None, llm=None):
        self.state_path = Path(state_dir) / "world.json"
        self._state = self._load()
        self._bus = bus              # F1 runtime EventBus (optional, guarded)
        self._llm = llm              # cloud extraction callable (optional)
        self._llm_enabled = (llm is not None
                             and os.environ.get(LLM_EXTRACT_ENV, "0") == "1")
        # Temporal graph store — runtime state, gitignored (*.db).
        self.db_path = Path(state_dir) / "world.db"
        self._db = _open_world_db(self.db_path)
        self.contradictions: list[dict] = []

    def _load(self) -> dict:
        try:
            if self.state_path.exists():
                d = json.loads(self.state_path.read_text())
                d.setdefault("routines", {})
                d.setdefault("projects", {})
                d.setdefault("deviations", [])
                return d
        except Exception:
            pass
        return {"routines": {}, "projects": {}, "deviations": []}

    def _save(self):
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self._state, indent=1))
        except Exception as e:
            print(f"[WORLD] save failed: {e}")

    # ── note ──────────────────────────────────────────────────────────
    def note(self, text: str) -> str:
        """Record routine defaults, projects, milestones, deadlines from prose."""
        text = (text or "").strip()
        if not text:
            return "Nothing to note."
        t = text.lower()
        # Project with deadline: "gearbox due Friday" / "sparrow deadline 2026-10-09"
        m = re.search(r"(gearbox|turbocharger|sparrow|falcon|[\w-]{3,20})\s+"
                      r"(due|deadline)\s+([\w\-/ ]{1,20})", t)
        if m and ("due" in t or "deadline" in t):
            name = m.group(1)
            date_s = m.group(3).strip()
            ts = self._parse_day(date_s)
            p = self._state["projects"].setdefault(name, {"notes": []})
            if ts:
                p["deadline_ts"] = ts
                p["deadline_s"] = date_s
            p["notes"].append(f"[{time.strftime('%Y-%m-%d')}] {text[:200]}")
            self._save()
            return f"Project '{name}' noted" + (f", deadline {date_s}." if ts else ".")
        # Milestone: "gearbox: 3 of 10 parts done" / "sparrow milestone: mounts printed"
        m = re.search(r"(\w[\w-]{2,20})\s*:\s*(\d+)\s*(?:of|/)\s*(\d+)", t)
        if m and any(w in t for w in ("part", "done", "complete", "milestone", "%")):
            name, done, total = m.group(1), int(m.group(2)), int(m.group(3))
            p = self._state["projects"].setdefault(name, {"notes": []})
            p["milestones_done"] = done
            p["milestones_total"] = total
            p["last_pace_ts"] = time.time()
            self._save()
            return f"Milestone noted: {name} {done}/{total}."
        # Routine defaults (temps, material, slicer, units)
        parsed = _parse_temp(text)
        for w in ("slicer", "orcasilcer", "orca", "prusaslicer", "bambu"):
            if w in t:
                parsed["slicer"] = "OrcaSlicer" if "orca" in t else w
        if "metric" in t or "millimetre" in t or "millimeter" in t:
            parsed["units"] = "metric"
        if parsed:
            self._state["routines"].update(parsed)
            self._save()
            kv = ", ".join(f"{k}={v}" for k, v in parsed.items())
            return f"Routine noted: {kv}."
        # Generic project note
        for name in list(self._state["projects"]):
            if name in t:
                self._state["projects"][name].setdefault("notes", []).append(
                    f"[{time.strftime('%Y-%m-%d')}] {text[:200]}")
                self._save()
                return f"Noted on project '{name}'."
        self._state.setdefault("notes", []).append(
            f"[{time.strftime('%Y-%m-%d')}] {text[:200]}")
        self._save()
        return "Noted."

    @staticmethod
    def _parse_day(s: str):
        s = s.strip().lower()
        now = time.time()
        if s in ("today",):
            return now
        if s in ("tomorrow",):
            return now + 86400
        # weekday name → next occurrence
        days = ["monday", "tuesday", "wednesday", "thursday",
                "friday", "saturday", "sunday"]
        import datetime as _dt
        for i, d in enumerate(days):
            if d in s:
                today = _dt.date.today().weekday()
                delta = (i - today) % 7 or 7
                return now + delta * 86400
        m = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
        if m:
            try:
                import datetime as _dt
                return _dt.datetime(int(m.group(1)), int(m.group(2)),
                                    int(m.group(3))).timestamp()
            except Exception:
                return None
        return None

    # ── check: deviation detection ────────────────────────────────────
    def check(self, action: str) -> str:
        """Compare a planned action against routines + pace. Honest ranges."""
        action = (action or "").strip()
        if not action:
            return "Nothing to check."
        flags = []
        routines = self._state.get("routines", {})
        got = _parse_temp(action)
        # Material / temp deviation (the PETG-at-200 prophecy)
        if got.get("material") and routines.get("material"):
            if got["material"] != routines["material"]:
                flags.append(
                    f"Material deviation: profile says {got['material']}, "
                    f"you usually print {routines['material']} — sure?")
        for k in ("nozzle_c", "bed_c"):
            if got.get(k) and routines.get(k):
                if abs(got[k] - routines[k]) > (15 if k == "nozzle_c" else 10):
                    flags.append(
                        f"Temperature deviation: {k.replace('_c', '')} "
                        f"{got[k]}C vs usual {routines[k]}C — sure?")
        if not routines:
            return ("I have no routines on file yet — tell me defaults "
                    "('I print PETG, nozzle 240, bed 80') and I'll watch them.")
        # Pace check for mentioned projects
        pace = self._pace_lines(action.lower())
        out = "\n".join(flags + pace)
        if not out:
            return "Matches your routines — nothing deviating."
        self._state["deviations"].append(
            {"ts": time.time(), "action": action[:200], "flags": flags + pace})
        self._state["deviations"] = self._state["deviations"][-30:]
        self._save()
        return out

    def _pace_lines(self, t: str) -> list[str]:
        lines = []
        for name, p in self._state.get("projects", {}).items():
            if name not in t:
                continue
            dl = p.get("deadline_ts")
            done = p.get("milestones_done")
            total = p.get("milestones_total")
            if dl and done is not None and total:
                left_days = max(0.0, (dl - time.time()) / 86400)
                rate = done / max(total, 1)
                # naive linear projection from first-seen pace timestamp
                start = p.get("last_pace_ts", time.time())
                elapsed = max(time.time() - start, 3600.0)
                proj_rate = done / (elapsed / 86400)  # per day
                need = (total - done) / max(left_days, 0.25)
                if proj_rate < need * 0.8:
                    lines.append(
                        f"Pace risk on '{name}': {done}/{total} with "
                        f"~{left_days:.1f} days left — current pace misses "
                        f"by ~{(need - proj_rate) * left_days:.0f} units.")
                else:
                    lines.append(f"Pace on '{name}': {done}/{total}, on track.")
            elif dl:
                left_days = (dl - time.time()) / 86400
                if 0 < left_days < 2 and not p.get("warned"):
                    p["warned"] = True
                    lines.append(f"'{name}' due in ~{left_days:.1f} days.")
        if lines:
            self._save()
        return lines

    # ── compare: two options from stored facts ────────────────────────
    def compare(self, option_a: str, option_b: str) -> str:
        a, b = (option_a or "").strip(), (option_b or "").strip()
        if not a or not b:
            return "Give me two options to compare."
        routines = self._state.get("routines", {})
        ga, gb = _parse_temp(a), _parse_temp(b)

        def score(g: dict) -> tuple[int, list[str]]:
            s, why = 0, []
            if g.get("material") == routines.get("material") and routines.get("material"):
                s += 2
                why.append(f"matches your {routines['material']} routine")
            elif g.get("material"):
                why.append(f"breaks {routines.get('material', 'no')} routine")
            for k in ("nozzle_c", "bed_c"):
                if g.get(k) == routines.get(k) and routines.get(k):
                    s += 1
                    why.append(f"{k.replace('_c', '')} matches usual {routines[k]}C")
            return s, why

        sa, wa = score(ga)
        sb, wb = score(gb)
        verdict = ("Option A fits your routines better." if sa > sb
                   else "Option B fits your routines better." if sb > sa
                   else "Both match (or neither touches) your routines — "
                        "no data to separate them, and I won't invent any.")
        return (f"A ({a[:80]}): {'; '.join(wa) or 'no routine overlap'}\n"
                f"B ({b[:80]}): {'; '.join(wb) or 'no routine overlap'}\n{verdict}")

    # ── watch: proactive deadline signals ─────────────────────────────
    def watch(self) -> list[tuple[str, str]]:
        """Deadline-risk signals for the proactive loop. Max 1 per project."""
        out = []
        for name, p in self._state.get("projects", {}).items():
            dl = p.get("deadline_ts")
            if not dl:
                continue
            left = (dl - time.time()) / 86400
            done = p.get("milestones_done")
            total = p.get("milestones_total")
            if done is not None and total and left > 0:
                frac = done / max(total, 1)
                # behind if less than half done with less than a third of time
                # — deliberately crude; stated as risk, never certainty
                if frac < 0.5 and left < 2 and not p.get("warned_pace"):
                    p["warned_pace"] = True
                    out.append((f"pace_{name}",
                                f"[emotion:worried] Sir — '{name}' is {done} of {total} "
                                f"with about {left:.0f} days left. At this pace it slips. "
                                f"Want to re-plan tonight?"))
            elif 0 < left < 1 and not p.get("warned_due"):
                p["warned_due"] = True
                out.append((f"due_{name}",
                            f"[emotion:thoughtful] Sir — '{name}' is due within a day."))
        if out:
            try:
                self._save()
            except Exception:
                pass
        return out

    def status(self) -> str:
        r = self._state.get("routines", {})
        ps = self._state.get("projects", {})
        bits = []
        if r:
            bits.append("routines: " + ", ".join(f"{k}={v}" for k, v in r.items()))
        if ps:
            names = []
            for n, p in ps.items():
                d, t = p.get("milestones_done"), p.get("milestones_total")
                names.append(f"{n} ({d}/{t})" if d is not None and t else n)
            bits.append("projects: " + ", ".join(names))
        return ("World model: " + (" | ".join(bits) if bits else
                "empty — teach me routines ('nozzle 240, bed 80, PETG') and projects ('gearbox due Friday')."))


# ══════════════════════════════════════════════════════════════════════
# F3 — Temporal personal world model (SQLite, stdlib only)
#
#   world.db (WAL, runtime state — gitignored):
#     entities(id, kind, name, attrs_json, first_seen, last_seen)
#     edges(id, a_id, relation, b_id, valid_from, valid_to,
#           source_episode, confidence, attrs_json)
#       valid_to = NULL  →  currently true
#     episodes(id, ts, text, source)
#     mentions(episode_id, entity_id) — entity↔episode links
#
# As-of query:  valid_from <= t AND (valid_to IS NULL OR valid_to > t)
# Contradictions retire the old edge (valid_to = now) — never deleted.
# LLM extraction is OFF by default (ADA_WORLD_LLM_EXTRACT=1 + llm=...).
# ══════════════════════════════════════════════════════════════════════

_SCHEMA_V = 1


def _open_world_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    with conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS entities ("
            "id INTEGER PRIMARY KEY, kind TEXT NOT NULL, name TEXT NOT NULL, "
            "attrs_json TEXT NOT NULL DEFAULT '{}', "
            "first_seen REAL NOT NULL, last_seen REAL NOT NULL)")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_name "
            "ON entities(lower(name))")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS edges ("
            "id INTEGER PRIMARY KEY, a_id INTEGER NOT NULL, "
            "relation TEXT NOT NULL, b_id INTEGER NOT NULL, "
            "valid_from REAL NOT NULL, valid_to REAL, "
            "source_episode TEXT, confidence REAL NOT NULL DEFAULT 1.0, "
            "attrs_json TEXT NOT NULL DEFAULT '{}', "
            "FOREIGN KEY(a_id) REFERENCES entities(id), "
            "FOREIGN KEY(b_id) REFERENCES entities(id))")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_abr "
            "ON edges(a_id, relation, b_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_valid "
            "ON edges(valid_from, valid_to)")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS episodes ("
            "id INTEGER PRIMARY KEY, ts REAL NOT NULL, "
            "text TEXT NOT NULL, source TEXT NOT NULL)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(ts)")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS mentions ("
            "episode_id INTEGER NOT NULL, entity_id INTEGER NOT NULL, "
            "PRIMARY KEY(episode_id, entity_id))")
    return conn


# ── entity mention extraction (stdlib, conservative, deterministic) ──

_CAP_PHRASE_RE = re.compile(
    r"\b([A-Z][a-zA-Z]{1,30}(?:\s+[A-Z][a-zA-Z]{1,30}){1,3})\b")
_QUOTED_RE = re.compile(r'"([^"\n]{2,60})"')
_ISA_RE = re.compile(
    r"\b([A-Z][a-zA-Z]{1,30}(?:\s+[A-Z][a-zA-Z]{1,30}){0,2})\s+is\s+"
    r"(?:my|the|a|an)\s+([a-z][a-z\-]{1,30})\b")
_SKIP_WORDS = frozenset({
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "january", "february", "march", "april",
    "may", "june", "july", "august", "september", "october",
    "november", "december", "sir", "okay", "hello", "thanks",
})


def _extract_mentions(text: str) -> list[tuple[str, str]]:
    """(name, kind) pairs. Conservative: ≥2-word capitalized phrases or
    quoted names only — sentence starts are mostly excluded by the
    multi-word requirement."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for m in _QUOTED_RE.finditer(text or ""):
        name = m.group(1).strip()
        if name and name.lower() not in seen:
            seen.add(name.lower())
            out.append((name, "name"))
    for m in _CAP_PHRASE_RE.finditer(text or ""):
        name = re.sub(r"\s+", " ", m.group(1)).strip()
        words = name.split()
        if (len(words) >= 2 and name.lower() not in seen
                and not all(w.lower() in _SKIP_WORDS for w in words)):
            seen.add(name.lower())
            out.append((name, "mention"))
    return out


def _extract_relations(text: str) -> list[tuple[str, str, str]]:
    """Tiny stdlib relation patterns, e.g. 'OrcaSlicer is my slicer'."""
    out = []
    for m in _ISA_RE.finditer(text or ""):
        a = re.sub(r"\s+", " ", m.group(1)).strip()
        b = m.group(2).strip().lower()
        if a and b:
            out.append((a, "is_a", b))
    return out


class _WorldTemporalMixin:
    """F3 temporal methods, mixed into WorldModel below (additive only —
    the original note()/check()/watch()/compare() stay untouched)."""
    # ── temporal store ──────────────────────────────────────────────
    def get_or_create_entity(self, name: str, kind: str = "thing",
                             attrs: dict | None = None,
                             ts: float | None = None) -> int:
        ts = ts if ts is not None else time.time()
        row = self._db.execute(
            "SELECT id FROM entities WHERE lower(name) = lower(?)",
            (name,)).fetchone()
        if row:
            with self._db:
                self._db.execute(
                    "UPDATE entities SET last_seen = ?, "
                    "attrs_json = ? WHERE id = ?",
                    (ts, json.dumps(attrs or {}), row["id"]))
            return row["id"]
        cur = self._db.execute(
            "INSERT INTO entities(kind, name, attrs_json, first_seen, last_seen)"
            " VALUES(?,?,?,?,?)",
            (kind, name, json.dumps(attrs or {}), ts, ts))
        self._db.commit()
        return cur.lastrowid

    def _entity_id(self, name: str) -> int | None:
        row = self._db.execute(
            "SELECT id FROM entities WHERE lower(name) = lower(?)",
            (name,)).fetchone()
        return row["id"] if row else None

    def observe_episode(self, text: str, source: str = "voice",
                         ts: float | None = None) -> int:
        """Record an episode; extract entity mentions (stdlib rules,
        deterministic; LLM batch path only when ADA_WORLD_LLM_EXTRACT=1)."""
        text = (text or "").strip()
        ts = ts if ts is not None else time.time()
        cur = self._db.execute(
            "INSERT INTO episodes(ts, text, source) VALUES(?,?,?)",
            (ts, text, source))
        ep_id = cur.lastrowid
        mentions = _extract_mentions(text)
        if self._llm_enabled and text:
            try:
                for name, kind in self._llm_batch_extract([text]):
                    if name and name.lower() not in {
                            n.lower() for n, _ in mentions}:
                        mentions.append((name, kind))
            except Exception:
                pass
        with self._db:
            for name, kind in mentions:
                eid = self.get_or_create_entity(name, kind=kind, ts=ts)
                self._db.execute(
                    "INSERT OR IGNORE INTO mentions(episode_id, entity_id)"
                    " VALUES(?,?)", (ep_id, eid))
            for a, relation, b in _extract_relations(text):
                aid = self.get_or_create_entity(a, kind="mention", ts=ts)
                bid = self.get_or_create_entity(b, kind="role", ts=ts)
                self._add_edge_raw(aid, relation, bid, ts,
                                   source_episode=str(ep_id),
                                   confidence=0.7)
        self._db.commit()
        return ep_id

    def _llm_batch_extract(self, texts: list[str]) -> list[tuple[str, str]]:
        """Cloud extraction — only ever called when the env flag is on.
        The stdlib path above always runs regardless."""
        if not self._llm_enabled or not texts:
            return []
        try:
            raw = self._llm(texts)
        except Exception:
            return []
        out = []
        for item in raw or []:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                name = str(item[0]).strip()
                kind = str(item[1]).strip() if len(item) > 1 else "thing"
                if name:
                    out.append((name, kind))
        return out

    # ── edges with validity windows + contradiction handling ─────────
    def _add_edge_raw(self, a_id: int, relation: str, b_id: int,
                      valid_from: float, source_episode: str | None,
                      confidence: float,
                      attrs: dict | None = None) -> int:
        cur = self._db.execute(
            "INSERT INTO edges(a_id, relation, b_id, valid_from, valid_to,"
            " source_episode, confidence, attrs_json)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (a_id, relation, b_id, valid_from, None, source_episode,
             confidence, json.dumps(attrs or {})))
        return cur.lastrowid

    def add_fact(self, a: str, relation: str, b: str, *,
                 source_episode: str | None = None,
                 confidence: float = 1.0, attrs: dict | None = None,
                 ts: float | None = None,
                 kind_a: str = "thing", kind_b: str = "thing") -> int:
        """Add (a, relation, b). A new b that contradicts the currently-true
        b for the same (a, relation) retires the old edge
        (valid_to = now — never deleted) and emits world.contradiction."""
        ts = ts if ts is not None else time.time()
        a_id = self.get_or_create_entity(a, kind=kind_a, ts=ts)
        b_id = self.get_or_create_entity(b, kind=kind_b, ts=ts)
        current = self._db.execute(
            "SELECT * FROM edges WHERE a_id = ? AND relation = ? "
            "AND valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)",
            (a_id, relation, ts, ts)).fetchall()
        same = [e for e in current if e["b_id"] == b_id]
        if same:
            # idempotent re-assertion — refresh, no contradiction
            with self._db:
                self._db.execute(
                    "UPDATE edges SET confidence = ?, attrs_json = ? "
                    "WHERE id = ?",
                    (max(same[0]["confidence"], confidence),
                     json.dumps(attrs or {}), same[0]["id"]))
            return same[0]["id"]
        retired = []
        with self._db:
            for e in current:  # contradicting edges → retire, keep rows
                self._db.execute(
                    "UPDATE edges SET valid_to = ? WHERE id = ?",
                    (ts, e["id"]))
                retired.append(e)
            new_id = self._add_edge_raw(a_id, relation, b_id, ts,
                                        source_episode, confidence, attrs)
        if retired:
            rec = {"a": a, "relation": relation,
                   "old_b": self._entity_name(retired[0]["b_id"]),
                   "new_b": b, "old_edge_ids": [e["id"] for e in retired],
                   "new_edge_id": new_id, "ts": ts}
            self.contradictions.append(rec)
            self._emit_bus_event("world.contradiction", rec)
        return new_id

    def _entity_name(self, eid: int) -> str:
        row = self._db.execute(
            "SELECT name FROM entities WHERE id = ?", (eid,)).fetchone()
        return row["name"] if row else f"#{eid}"

    def _emit_bus_event(self, event_type: str, payload: dict):
        """Best-effort publish of world events on the F1 runtime bus.
        Bus is optional — everything is guarded; silence on failure."""
        try:
            bus = self._bus
            if bus is None:
                for modname in ("runtime.bus", "backend.runtime.bus"):
                    try:
                        mod = __import__(modname, fromlist=["x"])
                        bus = (getattr(mod, "bus", None)
                               or getattr(mod, "get_bus", lambda: None)())
                        if bus is not None:
                            break
                    except Exception:
                        continue
            if bus is None:
                return
            pub = getattr(bus, "publish", None)
            if pub is None:
                return
            res = pub(event_type, source="world_model", payload=payload)
            if asyncio.iscoroutine(res):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    try:
                        asyncio.run(res)
                    except Exception:
                        pass
                else:
                    loop.create_task(res)
        except Exception:
            pass

    # ── as-of queries ───────────────────────────────────────────────
    def state_at(self, t: float) -> list[dict]:
        """All edges true at time t:
        valid_from <= t AND (valid_to IS NULL OR valid_to > t)."""
        rows = self._db.execute(
            "SELECT e.*, a.name AS a_name, b.name AS b_name FROM edges e "
            "JOIN entities a ON a.id = e.a_id "
            "JOIN entities b ON b.id = e.b_id "
            "WHERE e.valid_from <= ? AND (e.valid_to IS NULL OR e.valid_to > ?)"
            " ORDER BY e.valid_from ASC",
            (t, t)).fetchall()
        return [dict(r) for r in rows]

    def current_facts(self) -> list[dict]:
        return self.state_at(time.time())

    # ── mined routines ─────────────────────────────────────────────
    def upsert_mined_routine(self, activity: str, hour: float,
                             confidence: float, support: int,
                             ts: float | None = None) -> int:
        """Write one mined routine edge: user —usually_does_at→ activity.
        Dedupes on (activity, hour±0.5) so nightly re-runs update in place
        instead of contradicting — different hours are separate truths."""
        ts = ts if ts is not None else time.time()
        a_id = self.get_or_create_entity("user", kind="person", ts=ts)
        b_id = self.get_or_create_entity(activity, kind="activity", ts=ts)
        cur = self._db.execute(
            "SELECT * FROM edges WHERE a_id = ? AND relation = 'usually_does_at'"
            " AND b_id = ? AND valid_to IS NULL", (a_id, b_id)).fetchall()
        for e in cur:
            try:
                old_h = float(json.loads(e["attrs_json"]).get("hour", -999))
            except Exception:
                old_h = -999
            if abs(old_h - hour) <= 0.5:
                with self._db:
                    self._db.execute(
                        "UPDATE edges SET confidence = ?, attrs_json = ? "
                        "WHERE id = ?",
                        (confidence,
                         json.dumps({"hour": hour, "support": support,
                                     "confidence": confidence}), e["id"]))
                return e["id"]
        return self._add_edge_raw(
            a_id, "usually_does_at", b_id, ts, "miner", confidence,
            {"hour": hour, "support": support, "confidence": confidence})

    def predict(self, hour: float, weekday: int) -> tuple[str, float, int]:
        """What is he likely doing at (hour, weekday)? Returns
        (activity, confidence, support) from mined routines; honest
        fallback when there is no data."""
        a_id = self._entity_id("user")
        if a_id is None:
            return ("nothing reliable yet", 0.0, 0)
        rows = self._db.execute(
            "SELECT b.name AS activity, e.confidence, e.attrs_json FROM edges e "
            "JOIN entities b ON b.id = e.b_id "
            "WHERE e.a_id = ? AND e.relation = 'usually_does_at' "
            "AND e.valid_from <= ? AND (e.valid_to IS NULL OR e.valid_to > ?)",
            (a_id, time.time(), time.time())).fetchall()
        best = None
        for r in rows:
            try:
                attrs = json.loads(r["attrs_json"])
                h = float(attrs.get("hour", -999))
            except Exception:
                continue
            dh = abs(h - hour)
            dh = min(dh, 24 - dh)  # circular
            if dh <= 1.0:
                try:
                    support = int(attrs.get("support", 0))
                except Exception:
                    support = 0
                score = r["confidence"] * max(support, 1)
                if best is None or score > best[0]:
                    best = (score, r["activity"], r["confidence"], support)
        if best is None:
            return ("nothing reliable yet", 0.0, 0)
        return (best[1], float(best[2]), int(best[3]))

    # ── proactivity feed ───────────────────────────────────────────
    def world_risk_signals(self) -> list[dict]:
        """Structured deadline/deviation signals for the proactivity
        policy loop (a later feature consumes this; proactive.py untouched)."""
        sigs = []
        for key, text in self.watch():
            if key.startswith("pace_"):
                kind, proj = "pace", key[len("pace_"):]
            elif key.startswith("due_"):
                kind, proj = "due", key[len("due_"):]
            else:
                kind, proj = "info", key
            clean = re.sub(r"\[emotion:[^\]]*\]\s*", "", text).strip()
            sigs.append({"id": key, "kind": kind, "project": proj,
                         "severity": "warn", "text": clean,
                         "ts": time.time()})
        return sigs

    # ── prompt-injection context for the brain ─────────────────────
    def get_entity_context(self, name: str, max_edges: int = 12) -> str:
        """Pure function: current edges touching `name`, rendered as text
        the brain can inject into a prompt."""
        eid = self._entity_id(name)
        if eid is None:
            return ""
        t = time.time()
        rows = self._db.execute(
            "SELECT e.*, a.name AS a_name, b.name AS b_name FROM edges e "
            "JOIN entities a ON a.id = e.a_id "
            "JOIN entities b ON b.id = e.b_id "
            "WHERE (e.a_id = ? OR e.b_id = ?) "
            "AND e.valid_from <= ? AND (e.valid_to IS NULL OR e.valid_to > ?) "
            "ORDER BY e.confidence DESC LIMIT ?",
            (eid, eid, t, t, max_edges)).fetchall()
        lines = []
        for r in rows:
            extra = ""
            try:
                attrs = json.loads(r["attrs_json"])
                if attrs:
                    extra = " (" + ", ".join(
                        f"{k}={v}" for k, v in attrs.items()) + ")"
            except Exception:
                pass
            lines.append(f"- {r['a_name']} {r['relation']} {r['b_name']}{extra}")
        return "\n".join(lines)


# Attach the temporal methods to WorldModel (additive — the V1 verbs
# note()/check()/compare()/watch()/status() are defined above and kept).
for _m in [n for n in vars(_WorldTemporalMixin) if not n.startswith("__")]:
    setattr(WorldModel, _m, vars(_WorldTemporalMixin)[_m])
del _m
