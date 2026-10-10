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

import json
import re
import time
from pathlib import Path


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
    def __init__(self, state_dir: str | Path):
        self.state_path = Path(state_dir) / "world.json"
        self._state = self._load()

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
