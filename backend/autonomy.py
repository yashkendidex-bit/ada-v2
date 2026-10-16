"""
Autonomy Levels — the KAREN principle (Spider-Man: Homecoming).

Power is earned, not given. Ada operates at one of four autonomy levels:

  0  TRAINING WHEELS  — every action gated behind explicit permission
  1  STANDARD         — read-only tools free, background tasks announced,
                        destructive steps always confirmed (default)
  2  TRUSTED          — non-destructive autopilot flows without asking;
                        destructive steps still confirmed
  3  INSTANT KILL     — everything auto-approved. Must be ARMED through the
                        spoken ritual and can be stood down at any moment.

The ritual matters (canon): "Ada, arm Instant Kill" -> she challenges back
("Arming full autonomy, Sir — confirm.") -> only an explicit confirm arms it.
"Stand down" disarms instantly, from any state. Arming expires after
ARM_EXPIRY_S (a session-long power, never forever).

These gates are REAL: `allows_background_tool` and
`allows_autonomous_destructive` are consulted by the tool dispatch
(`ada.py::_check_confirmation`) on every tool call. The full-auto
"directive" that used to force level 3 / auto-approve everything is gone.

CAD tools are deliberately OUTSIDE this ladder — untouched, always as they
were (that domain is off-limits by standing order).
"""
import time
import json
from pathlib import Path

LEVEL_NAMES = {0: "Training Wheels", 1: "Standard", 2: "Trusted", 3: "Instant Kill"}
ARM_EXPIRY_S = 8 * 3600          # armed state expires after 8 hours
CONFIRM_TIMEOUT_S = 90

# ── tool risk classes ─────────────────────────────────────────────────────────
# The ladder gates by (level x risk class), never by tool name alone.
RISK_READONLY = "read-only"                    # pure reads / display / reports
RISK_REVERSIBLE = "state-changing-reversible"  # writes that can be undone
RISK_DESTRUCTIVE = "destructive-irreversible"  # deletion, arbitrary code, computer control
RISK_EXTERNAL = "external-side-effect"         # email, purchases, web submits, physical world

# CAD/model-surgery tools live OUTSIDE the ladder by standing order — the
# ladder never gates them; their own domain rules apply, exactly as before.
CAD_DOMAIN_TOOLS = frozenset({
    "list_parts", "edit_part", "undo_edit",
    "save_version", "list_versions", "compare_versions",
    "generate_cad", "iterate_cad", "design_from_photo",
    "test_model", "render_scene", "build_scene",
    "cad_intent", "handle_cad_request", "handle_cad_iteration",
})

# The ladder's own controls can never be gated by the ladder — otherwise a
# voice-ask could deadlock the ritual that resolves it.
NEVER_GATED_TOOLS = frozenset({"autonomy_control"})

_REVERSIBLE_TOOLS = frozenset({
    # jailed file writes + project state
    "write_file", "create_project", "switch_project",
    # memory / learning writes (undoable via forget)
    "remember", "human_learn", "people_learn", "identity_update",
    "notebook_write", "clipboard",
    # scheduler / watchers / subscriptions (all cancellable)
    "create_reminder", "schedule_recurring", "cancel_scheduled",
    "create_plan", "watch_folder", "watch_me", "watch_rule",
    "screen_watch", "deeper_checkins",
    # launched research (information gathering; dangerous downstream
    # tools are gated on their own)
    "osint_dossier", "deep_sweep", "investigate_now", "dream_now",
    "overnight_research", "face_osint", "index_documents",
    "skill_practice", "quarantine_review",
    # display-domain writes
    "show_hud", "hud_demo", "show_widget", "update_widget",
    "close_widget", "build_widget_code",
})

_DESTRUCTIVE_TOOLS = frozenset({
    "forget", "forget_complete",      # irreversible memory deletion
    "create_custom_tool",             # forges code that runs in-process
    "run_custom_tool",                # runs arbitrary forged code in-process
    "desktop_autopilot",              # drives the real computer
})

_EXTERNAL_TOOLS = frozenset({
    "send_email",                     # external comms: never silent
    "smart_home",                     # physical world
    "launch_app",                     # launches real applications
    "run_web_agent",                  # browser agent: clicks, forms, submits
    "calendar_event",                 # touches other people's calendars
    "print_it", "print_outcome", "printer_set", "printer_pause",  # physical printer
    "fabricate",                      # spawns a physical fabrication job
})


def risk_class(tool: str) -> str:
    """Map a tool name to its risk class. Unknown tools fail CLOSED:
    treated as destructive-irreversible so they are voice-confirmed at
    every level below armed Instant Kill and never run silently."""
    name = str(tool or "")
    if name in _DESTRUCTIVE_TOOLS:
        return RISK_DESTRUCTIVE
    if name in _EXTERNAL_TOOLS:
        return RISK_EXTERNAL
    if name in _REVERSIBLE_TOOLS:
        return RISK_REVERSIBLE
    return RISK_READONLY if _looks_readonly(name) else RISK_DESTRUCTIVE


def _looks_readonly(name: str) -> bool:
    # Conservative allowlist for the read-only class: known pure-read
    # prefixes/infixes. Anything not recognised here falls through to
    # RISK_DESTRUCTIVE (fail closed) in risk_class().
    n = name.lower()
    return (
        n.startswith(("list_", "get_", "read_", "search_", "recall",
                      "observe_", "what_", "how_", "show_", "audit_",
                      "world_", "screen_", "system_", "telemetry",
                      "soul_", "people_", "notebook_read", "cost_",
                      "daily_", "dossier_", "investigator_", "scorecard",
                      "status", "preflight", "second_opinion"))
        or n in {"search_web", "read_file", "read_directory", "list_projects",
                 "identity_read", "soul_read", "autonomy_control",
                 "printer_status", "screenshot", "observe_status",
                 "observe_level"}
    )


# ── per-capability autonomy (Feature 9 trust console) ──────────────────────
# The trust console offers per-capability toggles — ask-first vs act — which
# refine the ladder per domain. A toggle sets the *effective level* for that
# capability: "ask-first" behaves as level 1 (every side-effect step asked),
# "act" behaves as level 2 (reversible flows run free). Two things a toggle
# can NEVER do: auto-approve destructive/external tools (those still ask
# below an armed level 3 — the ritual keeps its meaning), or touch
# ladder-exempt domains (CAD + the ladder's own controls stay frictionless).
CAPABILITY_MODES = ("ask-first", "act")

CAPABILITIES = {
    "email": ("send_email",),
    "web": ("run_web_agent", "search_web", "investigate_now", "deep_sweep",
            "osint_dossier", "face_osint"),
    "files": ("write_file", "read_file", "read_directory", "list_projects",
              "create_project", "switch_project"),
    "scheduling": ("create_reminder", "schedule_recurring", "cancel_scheduled",
                   "list_scheduled", "create_plan", "watch_folder", "watch_me",
                   "watch_rule", "screen_watch", "deeper_checkins"),
    "memory": ("remember", "remember_context", "remember_fact", "human_learn",
               "people_learn", "identity_update", "notebook_write",
               "clipboard", "forget", "forget_complete"),
    "smart_home": ("smart_home",),
    "printing": ("print_it", "print_outcome", "printer_set", "printer_pause",
                 "fabricate", "preflight"),
    "apps": ("launch_app", "desktop_autopilot"),
    "calendar": ("calendar_event",),
}


def capability_of(tool: str) -> str:
    """Map a tool to its trust-console capability, or 'other'."""
    name = str(tool or "")
    for cap, tools in CAPABILITIES.items():
        if name in tools:
            return cap
    return "other"


def evaluate_gate(tool: str, *, level: int, armed: bool = False,
                  exempt: bool = False, voice_confirm_names=()) -> dict:
    """Pure gate math, shared by ada.py::_check_confirmation and
    replay.redecide. Returns {risk, needs_voice, verdict, reason}.

    verdict is "auto-approved" (runs without asking) or "voice-ask" (a human
    is asked first — the F0 gate never hard-denies on its own; denial comes
    from the voice answer, which is recorded in the audit row's decision).
    """
    risk = risk_class(tool)
    if exempt:
        return {"risk": risk, "needs_voice": False, "verdict": "auto-approved",
                "reason": "ladder-exempt domain; frictionless"}
    if armed:
        return {"risk": risk, "needs_voice": False, "verdict": "auto-approved",
                "reason": f"armed Instant Kill auto-approve (risk={risk})"}
    needs_voice = (
        tool in (voice_confirm_names or ())
        or risk in (RISK_DESTRUCTIVE, RISK_EXTERNAL)
        or (risk == RISK_REVERSIBLE and level < 2)
        or level == 0
    )
    reason = (f"voice ask at L{level} (risk={risk})" if needs_voice
              else f"frictionless at L{level} (risk={risk})")
    return {"risk": risk, "needs_voice": needs_voice,
            "verdict": "voice-ask" if needs_voice else "auto-approved",
            "reason": reason}


def is_ladder_exempt(tool: str) -> bool:
    """True for tools the ladder must never gate (CAD domain + its own controls)."""
    name = str(tool or "")
    return name in CAD_DOMAIN_TOOLS or name in NEVER_GATED_TOOLS


class AutonomyLevels:
    # DEFAULT = 1 (Standard), NOT 0 (Training Wheels). Level 0 gates *every*
    # action — including read-only tools — behind explicit permission, which
    # would violate the standing 2026-10-01 directive that ordinary tools stay
    # frictionless. Level 1 is the safest level compatible with that directive:
    # read-only tools run free, but EVERY side-effect step is asked out loud
    # before execution. The ladder docstring above also marks L1 "(default)".
    DEFAULT_LEVEL = 1

    def __init__(self, state_dir, default_level=None, audit=None):
        self.state_path = Path(state_dir) / "autonomy_state.json"
        self.level = self.DEFAULT_LEVEL if default_level is None else max(0, min(3, int(default_level)))
        self._arm_pending_ts = None
        self._armed_ts = None
        self._audit = audit or (lambda *a, **k: None)
        # Per-capability autonomy toggles (Feature 9): capability ->
        # "ask-first" | "act". Absent = follow the ladder.
        self._capability_modes: dict[str, str] = {}
        # The constructor NEVER arms. An armed level 3 exists only after the
        # spoken ritual completed (confirm_arm) — a persisted armed_ts from a
        # previous session survives a restart only inside its expiry window.
        self._load()

    # ── persistence ───────────────────────────────────────────────────────────
    def _load(self):
        try:
            if not self.state_path.exists():
                return
            data = json.loads(self.state_path.read_text())
            lvl = max(0, min(3, int(data.get("level", self.DEFAULT_LEVEL))))
            armed_ts = data.get("armed_ts")
            if lvl == 3:
                # armed L3 survives a restart only within its expiry window AND
                # only if the state file was written by the real ladder (era
                # marker) — a level-3 record from the old full-auto era, with
                # or without an armed timestamp, is residue and drops to
                # Standard. The ritual is the only way to re-arm.
                fresh = armed_ts and time.time() - armed_ts <= ARM_EXPIRY_S
                if fresh and data.get("era") == "ladder-v2":
                    self._armed_ts = armed_ts
                else:
                    lvl = self.DEFAULT_LEVEL
            self.level = lvl
            if self.level != 3:
                self._armed_ts = None
            # Per-capability toggles survive restarts with the ladder state.
            modes = data.get("capability_modes") or {}
            self._capability_modes = {
                str(k): v for k, v in modes.items()
                if k in CAPABILITIES and v in CAPABILITY_MODES}
        except Exception:
            self.level = self.DEFAULT_LEVEL
            self._armed_ts = None
            self._capability_modes = {}

    def _save(self):
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(
                {"era": "ladder-v2",
                 "level": self.level,
                 "armed_ts": self._armed_ts if self.level == 3 else None,
                 "capability_modes": self._capability_modes}))
        except Exception:
            pass

    def _audit_log(self, action, target="", decision="", detail=""):
        try:
            self._audit("autonomy", action, target, decision, detail)
        except Exception:
            pass

    # ── risk-class API (used by the tool dispatch) ────────────────────────────
    def risk(self, tool: str) -> str:
        return risk_class(tool)

    def ladder_exempt(self, tool: str) -> bool:
        return is_ladder_exempt(tool)

    # ── the ritual ────────────────────────────────────────────────────────────
    def request_arm(self) -> str:
        """'Ada, arm Instant Kill' -> she challenges back; not armed yet."""
        if self.level == 3 and self.armed:
            return "Instant Kill is already armed, Sir. Say 'stand down' to disarm it."
        self._arm_pending_ts = time.time()
        self._audit_log("arm-request", "", "pending", "challenge issued, awaiting 'confirm'")
        return ("Understood. Arming FULL autonomy — every action will execute without "
                "asking. Say 'confirm' to arm it, or 'stand down' to cancel.")

    def confirm_arm(self) -> str:
        """'confirm' completes the ritual — but only if recently challenged."""
        if self._arm_pending_ts is None:
            return "Nothing to confirm, Sir. Say 'arm Instant Kill' first."
        if time.time() - self._arm_pending_ts > CONFIRM_TIMEOUT_S:
            self._arm_pending_ts = None
            self._audit_log("arm-confirm", "", "expired", "confirm arrived after the 90s window")
            return "The arm request expired — say 'arm Instant Kill' again."
        self.level = 3
        self._armed_ts = time.time()
        self._arm_pending_ts = None
        self._save()
        self._audit_log("arm-confirm", "", "armed", "spoken ritual completed; expires in 8h")
        return ("Instant Kill ARMED. I will act without asking. Say 'stand down' anytime "
                "— I will hear you.")

    def stand_down(self) -> str:
        """'stand down' — works from any state, no confirmation needed.

        Revokes for real: level drops to Standard AND the armed state is
        cleared, so allows_autonomous_destructive() returns False immediately.
        """
        was = self.level
        self.level = self.DEFAULT_LEVEL
        self._armed_ts = None
        self._arm_pending_ts = None
        self._save()
        self._audit_log("stand-down", "", "disarmed",
                        f"level {was} -> {self.level}; armed state cleared")
        if was == 3:
            return "Standing down. Full autonomy disengaged — back to Standard, Sir."
        return f"Standing down from {LEVEL_NAMES.get(was, was)}. Standard mode restored."

    def set_level(self, level: int) -> str:
        level = max(0, min(3, int(level)))
        if level == 3:
            return self.request_arm()
        prev = self.level
        self.level = level
        if level != 3:
            self._armed_ts = None
        self._save()
        self._audit_log("set-level", "", "ok", f"{LEVEL_NAMES.get(prev, prev)} -> {LEVEL_NAMES[level]}")
        return f"Autonomy {LEVEL_NAMES[prev]} -> {LEVEL_NAMES[level]}."

    # ── per-capability toggles (Feature 9 trust console) ────────────────────
    def capability_modes(self) -> dict:
        """Every known capability with its current mode ('ask-first' / 'act'
        / 'ladder' when unset)."""
        return {cap: self._capability_modes.get(cap, "ladder")
                for cap in CAPABILITIES}

    def capability_mode(self, capability: str) -> str:
        return self._capability_modes.get(str(capability), "ladder")

    def set_capability_mode(self, capability: str, mode: str) -> str:
        """Trust-console toggle: 'ask-first' (level-1 behaviour for this
        capability), 'act' (level-2 behaviour), or 'ladder' (clear the
        override). Voice-driven; the change itself is audit-logged."""
        cap = str(capability or "").strip().lower()
        mode = str(mode or "").strip().lower()
        if cap not in CAPABILITIES:
            return f"Unknown capability '{cap}'. Known: {', '.join(sorted(CAPABILITIES))}."
        if mode not in CAPABILITY_MODES and mode != "ladder":
            return f"Mode must be ask-first, act, or ladder — not '{mode}'."
        prev = self._capability_modes.get(cap, "ladder")
        if mode == "ladder":
            self._capability_modes.pop(cap, None)
        else:
            self._capability_modes[cap] = mode
        self._save()
        self._audit_log("capability-toggle", cap, "ok",
                        f"{prev} -> {mode or 'ladder'}")
        return (f"{cap}: {prev} -> {mode}. "
                f"{'I will ask first for every step here.' if mode == 'ask-first' else 'I will act freely here (destructive steps still asked).' if mode == 'act' else 'Back to the ladder, Sir.'}")

    def effective_level(self, tool: str) -> int:
        """The ladder level that actually gates this tool, after the
        per-capability toggle. 'ask-first' -> 1, 'act' -> 2, else the ladder."""
        mode = self._capability_modes.get(capability_of(tool))
        if mode == "ask-first":
            return 1
        if mode == "act":
            return 2
        return self.level

    def status(self) -> str:
        extra = ""
        if self.level == 3 and self._armed_ts:
            left = int(ARM_EXPIRY_S - (time.time() - self._armed_ts))
            extra = f" (auto-expires in {left // 3600}h {(left % 3600) // 60}m)"
        return f"Autonomy level {self.level} — {LEVEL_NAMES[self.level]}{extra}."

    # ── gating API ────────────────────────────────────────────────────────────
    @property
    def armed(self) -> bool:
        """True only while level 3 is armed via the spoken ritual AND the arm
        has not expired. Expiry revokes immediately: level drops back and the
        armed timestamp is cleared."""
        if self.level != 3:
            return False
        ts = self._armed_ts
        if not ts or time.time() - ts > ARM_EXPIRY_S:   # expiry check on every use
            self.level = self.DEFAULT_LEVEL
            self._armed_ts = None
            self._save()
            self._audit_log("arm-expiry", "", "revoked", "8h window elapsed; back to Standard")
            return False
        return True

    def allows_autonomous_destructive(self) -> bool:
        """True ONLY at level 3 AND armed through the spoken arm ritual AND
        the arm has not expired. Everywhere else: False, no exceptions.

        Because the constructor never arms and _armed_ts is set solely by
        confirm_arm() (which requires the challenge + 'confirm' inside the
        90s window), a True here proves the ritual happened."""
        return self.armed

    def allows_background_tool(self, tool: str) -> bool:
        """Background contexts (scheduler, proactive engine) have no voice
        channel to ask through, so a tool is allowed only when the level
        permits it to run silently:

          L0 Training Wheels : nothing runs ungated — every action needs a yes
          L1 Standard         : read-only only
          L2 Trusted          : read-only + reversible
          L3 Instant Kill     : everything, but ONLY while armed via the ritual
        Destructive and external-side-effect tools never run in the
        background below armed L3.
        """
        if is_ladder_exempt(tool):
            return True
        if self.armed:
            return True
        if self.level == 0:
            return False
        risk = risk_class(tool)
        if risk == RISK_READONLY:
            return self.level >= 1
        if risk == RISK_REVERSIBLE:
            return self.level >= 2
        return False
