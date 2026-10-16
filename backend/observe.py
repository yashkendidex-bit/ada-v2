"""Observe tiers — cooperative observation (Organizm pattern, honest edition).

Watching is binary today: on or off. Trust isn't binary. Three tiers:

  0 OFF      — nothing captured, nothing watched. Full privacy.
  1 CAPTIONS — task one-liners only ("Sparrow mount open in slicer").
               The current screen_recall behavior. Default.
  2 DETAIL   — window titles + selected text, ONLY on Yash's explicit ask
               per moment ("read this window"). Never silent, never stored
               beyond the ring.

Movement is EARNED, never taken:
  +1 tier after 3 consecutive useful votes (never past 1 automatically —
    tier 2 ALWAYS needs Yash's spoken word, no exceptions)
  -1 tier after 2 consecutive useless votes ("stop watching so closely")
  votes via `observe_vote` ("that helped" / "not helpful") or voice

Every suggestion carries its PROOF on its face:
  "saw: OrcaSlicer open 20 min + PETG profile → suggest: slice now?"
No proof, no suggestion. That's the whole trust contract.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


class ObserveTiers:
    def __init__(self, state_dir: str | Path):
        self.state_path = Path(state_dir) / "soul" / "observe_state.json"
        self._state = self._load()

    def _load(self) -> dict:
        try:
            if self.state_path.exists():
                d = json.loads(self.state_path.read_text())
                d.setdefault("tier", 1)
                d.setdefault("useful_streak", 0)
                d.setdefault("useless_streak", 0)
                d.setdefault("votes", 0)
                return d
        except Exception:
            pass
        return {"tier": 1, "useful_streak": 0, "useless_streak": 0, "votes": 0,
                "history": []}

    def _save(self):
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self._state, indent=1))
        except Exception as e:
            print(f"[OBSERVE] save failed: {e}")

    @property
    def tier(self) -> int:
        return int(self._state.get("tier", 1))

    def set_tier(self, tier: int, by: str = "voice") -> str:
        tier = max(0, min(2, int(tier)))
        self._state["tier"] = tier
        self._state["useful_streak"] = 0
        self._state["useless_streak"] = 0
        self._save()
        # Timeline (Feature 4): retention follows the tier. Tier 0 stops DB
        # writes entirely (ring only); safe no-op when uninitialized.
        try:
            from perception import set_retention_tier
            set_retention_tier(tier)
        except Exception:
            pass
        names = {0: "OFF — blind by choice", 1: "CAPTIONS — task one-liners only",
                 2: "DETAIL — window titles + selected text on your explicit ask"}
        return f"Observation {names[tier]} (set by {by})."

    def vote(self, useful: bool, skill: str | None = None) -> str:
        """Record whether her last watched suggestion helped.

        Pass skill=<name> to also record a per-skill quality vote — those
        streaks feed the canary auto-rollback signal (Feature 6): sustained
        useless votes on a canary skill demote it back to shadow.
        """
        reply = self._tier_vote(useful)
        if skill:
            skill_reply = self.skill_vote(skill, useful)
            reply = f"{reply} {skill_reply}".strip()
        return reply

    def _tier_vote(self, useful: bool) -> str:
        self._state["votes"] = int(self._state.get("votes", 0)) + 1
        if useful:
            self._state["useful_streak"] = int(self._state.get("useful_streak", 0)) + 1
            self._state["useless_streak"] = 0
            if self._state["useful_streak"] >= 3 and self.tier == 0:
                self._state["tier"] = 1
                self._state["useful_streak"] = 0
                self._save()
                try:
                    from perception import set_retention_tier
                    set_retention_tier(1)
                except Exception:
                    pass
                return ("Three helps in a row — stepping back up to CAPTIONS. "
                        "Say 'watch less' any time.")
        else:
            self._state["useless_streak"] = int(self._state.get("useless_streak", 0)) + 1
            self._state["useful_streak"] = 0
            if self._state["useless_streak"] >= 2 and self.tier > 0:
                self._state["tier"] = self.tier - 1
                self._state["useless_streak"] = 0
                self._save()
                try:
                    from perception import set_retention_tier
                    set_retention_tier(self.tier)
                except Exception:
                    pass
                return ("Heard — watching less closely now. Say 'watch more' "
                        "when you want me closer.")
        self._save()
        return "Noted."

    # ── per-skill quality (Feature 6: canary auto-rollback signal) ────────────
    USELESS_DEMOTE_AFTER = 3

    def _skill_row(self, name: str) -> dict:
        q = self._state.setdefault("skill_quality", {})
        row = q.setdefault(name, {"useful_streak": 0, "useless_streak": 0,
                                  "votes": 0, "demote_flag": False})
        return row

    def skill_vote(self, name: str, useful: bool) -> str:
        """Per-skill quality vote. 3 straight useless votes arm the demote
        signal that the eval gate's canary health check consumes."""
        row = self._skill_row(name)
        row["votes"] = int(row.get("votes", 0)) + 1
        if useful:
            row["useful_streak"] = int(row.get("useful_streak", 0)) + 1
            row["useless_streak"] = 0
            row["demote_flag"] = False
            self._save()
            return f"Noted — '{name}' helped."
        row["useless_streak"] = int(row.get("useless_streak", 0)) + 1
        row["useful_streak"] = 0
        self._save()
        if row["useless_streak"] >= self.USELESS_DEMOTE_AFTER:
            row["demote_flag"] = True
            self._save()
            return (f"Noted — '{name}' hasn't helped {row['useless_streak']} "
                    f"times running. I'll pull it back from canary.")
        return f"Noted — '{name}' wasn't useful this time."

    def skill_signal(self, name: str) -> str:
        """'demote' when sustained useless votes warrant pulling the skill
        back; 'ok' otherwise. Consumed by SkillEvalGate.maybe_promote_canary."""
        row = self._skill_row(name)
        return "demote" if row.get("demote_flag") else "ok"

    def skill_signal_ack(self, name: str):
        """Clear the demote flag after the gate has acted on it."""
        row = self._skill_row(name)
        row["demote_flag"] = False
        row["useless_streak"] = 0
        self._save()

    def skill_quality_status(self, name: str) -> str:
        row = self._skill_row(name)
        return (f"'{name}': {row['votes']} votes, "
                f"useful x{row['useful_streak']}, useless x{row['useless_streak']}, "
                f"signal={self.skill_signal(name)}")

    @staticmethod
    def proof(saw: str, suggest: str) -> str:
        """One-line proof face for every watched suggestion."""
        return f"saw: {saw[:120]} → suggest: {suggest[:120]}"

    def status(self) -> str:
        names = {0: "OFF", 1: "CAPTIONS", 2: "DETAIL"}
        return (f"Observation tier {self.tier} ({names.get(self.tier, '?')}) — "
                f"{int(self._state.get('votes', 0))} votes so far. "
                f"'watch more' / 'watch less' / 'that helped' / 'not helpful'.")
