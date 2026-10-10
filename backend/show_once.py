"""Show-once — teach by showing (Grok's best trick, done privately).

"Watch me do this" → Ada captures screen captions into a named procedure
→ "do the weekly report like I showed you" → she walks it as a checklist.

Honest scope, stated upfront: this records a PROCEDURE (ordered steps
with the app names and actions she saw), not executable code. Steps she
can drive (open app, type, click via desktop_autopilot) are marked
`drivable`; judgment steps stay `guide` (she narrates, Sir decides).
The Voyager skill loop can later codify drivable procedures into real
tools — capture first, code later.

Storage: ada_kernel/soul/procedures/<name>.json (git-tracked, editable,
deletable). Capture uses the existing ScreenRecall captions (task
one-liners, never keystrokes) at ~30s cadence while recording.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


class ShowOnce:
    def __init__(self, state_dir: str | Path, shot_fn=None):
        self.dir = Path(state_dir) / "soul" / "procedures"
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        self._rec: dict | None = None
        self._shot_fn = shot_fn  # async () -> caption str (wired by ada.py)

    def _path(self, name: str) -> Path:
        safe = "".join(c if (c.isalnum() or c in "-_") else "_"
                       for c in (name or "").strip().lower())[:40] or "untitled"
        return self.dir / f"{safe}.json"

    def start(self, name: str) -> str:
        if self._rec is not None:
            return f"Already recording '{self._rec['name']}' — stop it first."
        self._rec = {"name": (name or "untitled")[:60], "t0": time.time(),
                     "steps": []}
        return (f"Watching, Sir — do '{self._rec['name']}' once, naturally. "
                f"Say 'done showing' when finished.")

    async def capture_step(self, caption: str = "") -> str:
        """One observed step. Called by the watch loop or manually."""
        if self._rec is None:
            return "Not recording."
        cap = caption
        if not cap and self._shot_fn is not None:
            try:
                cap = await self._shot_fn()
            except Exception:
                cap = ""
        cap = (cap or "").strip()[:200]
        if not cap or (self._rec["steps"] and self._rec["steps"][-1]["cap"] == cap):
            return "No change."
        self._rec["steps"].append({"t": round(time.time() - self._rec["t0"]),
                                   "cap": cap, "kind": "guide"})
        return f"Step {len(self._rec['steps'])} noted."

    def stop(self) -> str:
        if self._rec is None:
            return "Not recording anything."
        rec, self._rec = self._rec, None
        if len(rec["steps"]) < 2:
            return ("Too short to be a procedure, Sir — show me at least two "
                    "distinct steps.")
        # mark drivable steps: app launches / navigations she can repeat
        for s in rec["steps"]:
            c = s["cap"].lower()
            if any(w in c for w in ("open", "launch", "browser", "slic", "type", "click")):
                s["kind"] = "drivable"
        try:
            self._path(rec["name"]).write_text(json.dumps(rec, indent=1))
        except Exception as e:
            return f"Could not save: {type(e).__name__}"
        return (f"Learned '{rec['name']}': {len(rec['steps'])} steps, "
                f"{sum(1 for s in rec['steps'] if s['kind'] == 'drivable')} drivable.")

    def list_procedures(self) -> str:
        try:
            names = sorted(p.stem for p in self.dir.glob("*.json"))
        except Exception:
            names = []
        return ("Procedures: " + ", ".join(names)) if names else "No procedures taught yet."

    def read(self, name: str) -> str:
        try:
            rec = json.loads(self._path(name).read_text())
        except Exception:
            return f"No procedure '{name}' — check the list."
        lines = [f"{i + 1}. [{s['kind']}] {s['cap']}"
                 for i, s in enumerate(rec.get("steps", []))]
        return f"Procedure '{rec.get('name')}' ({len(lines)} steps):\n" + "\n".join(lines)
