"""Fabricator — the Jarvis workshop loop (honest edition).

One job, seven stages, no hands between them except where physics or
safety demands a human:
  intent → build → slice → preflight → print → verify → learn

  intent    deterministic spec + questions (cad_intent, already live)
  build     existing CAD engine (handle_cad_request path, not duplicated)
  slice     OrcaSlicer/PrusaSlicer CLI if installed; else waits for
            Sir-sliced G-code (says so, never fakes a slice)
  preflight deterministic verdict (preflight.verdict — CLEAR/CAUTION/NO-GO)
  print     Moonraker upload + start, ONLY after Sir's spoken yes for
            this exact file (the voice layer owns approval, recorded here)
  verify    outcome note: clean / warped / failed + why (photo compare
            when a camera frame exists, else Sir's one-line verdict)
  learn     lesson → world routines + human LEARNED + scorecard log

Jobs persist as projects/fabricator/<id>.json so a restart never loses
the loop. Stages run in a background task with task-board narration;
every stage transition is a milestone, every job states its ETA upfront
(mission-ETA registry lives here too — see JOB_ETAS).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path


# Mission ETAs (JARVIS "rendering… 30 seconds"): stated upfront, narrated
# at milestones. Measured, not marketing — update when reality disagrees.
JOB_ETAS = {
    "cad_build": "about two minutes",
    "intent": "a few seconds",
    "dream": "about a minute",
    "council": "a few minutes",
    "dossier": "a few minutes",
    "slice": "about a minute",
    "preflight": "seconds",
    "print": "per the slicer estimate",
    "verify": "a minute",
}

STAGES = ["intent", "build", "slice", "preflight", "print", "verify", "learn"]


def find_slicer() -> str | None:
    for cand in ("orcaslicer", "prusa-slicer", "prusaslicer", "bambu-studio"):
        if shutil.which(cand):
            return cand
    for p in ("/Applications/OrcaSlicer.app/Contents/MacOS/OrcaSlicer",
              "/Applications/PrusaSlicer.app/Contents/MacOS/PrusaSlicer"):
        if Path(p).exists():
            return p
    return None


class Fabricator:
    def __init__(self, state_dir: str | Path, root_dir: str | Path):
        self.jobs_dir = Path(root_dir) / "projects" / "fabricator"
        self.state_dir = Path(state_dir)
        try:
            self.jobs_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def new_job(self, prompt: str, photo_caption: str = "") -> dict:
        jid = time.strftime("%Y%m%d-%H%M%S")
        job = {"id": jid, "prompt": (prompt or "")[:300],
               "photo_caption": (photo_caption or "")[:500],
               "stage": "intent", "stages": {}, "created": time.time(),
               "approved_print": False, "outcome": None}
        self._save(job)
        return job

    def _path(self, jid: str) -> Path:
        return self.jobs_dir / f"{jid}.json"

    def _save(self, job: dict):
        try:
            self._path(job["id"]).write_text(json.dumps(job, indent=1))
        except Exception as e:
            print(f"[FABRICATOR] save failed: {e}")

    def load(self, jid: str) -> dict | None:
        try:
            return json.loads(self._path(jid).read_text())
        except Exception:
            return None

    def latest(self) -> dict | None:
        try:
            files = sorted(self.jobs_dir.glob("*.json"))
            return json.loads(files[-1].read_text()) if files else None
        except Exception:
            return None

    def mark(self, jid: str, stage: str, info: str = "", ok: bool = True) -> dict:
        job = self.load(jid) or {}
        job["stages"][stage] = {"ok": ok, "info": (info or "")[:500],
                                "ts": time.time()}
        job["stage"] = stage
        self._save(job)
        return job

    def approve_print(self, jid: str) -> dict:
        job = self.load(jid) or {}
        job["approved_print"] = True
        job["stages"]["approved"] = {"ok": True, "ts": time.time(),
                                    "info": "Sir said yes for this exact file"}
        self._save(job)
        return job

    def record_outcome(self, jid: str, outcome: str, note: str = "") -> dict:
        """clean | warped | stringing | failed | ... + one-line why."""
        job = self.load(jid) or {}
        job["outcome"] = {"result": (outcome or "").strip().lower()[:40],
                          "note": (note or "")[:300], "ts": time.time()}
        job["stage"] = "learn"
        job["stages"]["verify"] = {"ok": job["outcome"]["result"] == "clean",
                                   "info": note[:300], "ts": time.time()}
        self._save(job)
        return job

    def status(self, jid: str = "") -> str:
        job = self.load(jid) if jid else self.latest()
        if not job:
            return "No fabrication jobs yet."
        done = [s for s, r in job.get("stages", {}).items() if r.get("ok")]
        out = job.get("outcome") or {}
        return (f"Job {job['id']} '{job.get('prompt', '')[:60]}' — "
                f"at stage {job.get('stage')}, {len(done)} stages green."
                + (f" Outcome: {out.get('result')} ({out.get('note', '')[:80]})"
                   if out else ""))

    @staticmethod
    def eta_for(kind: str) -> str:
        return JOB_ETAS.get(kind, "a little while")

    @staticmethod
    def slice_stl(stl_path: str, out_dir: str = "") -> dict:
        """Slice via installed slicer CLI. No slicer → honest stop, not a fake."""
        slicer = find_slicer()
        if not slicer:
            return {"ok": False,
                    "msg": ("No slicer installed (OrcaSlicer/PrusaSlicer). Slice it "
                            "once in the app and tell me the G-code path — I'll take "
                            "it from there.")}
        out = Path(out_dir or Path(stl_path).parent) / (Path(stl_path).stem + ".gcode")
        try:
            # Prusa-family CLIs share --slice/--load/--output vocabulary.
            r = subprocess.run(
                [slicer, "--slice", str(stl_path), "--output", str(out)],
                capture_output=True, text=True, timeout=600)
            if out.exists():
                return {"ok": True, "gcode": str(out),
                        "log": (r.stdout or "")[-500:]}
            return {"ok": False,
                    "msg": f"Slicer ran but no G-code appeared: {(r.stderr or '')[-200:]}"}
        except Exception as e:
            return {"ok": False, "msg": f"Slicer failed: {type(e).__name__}"}
