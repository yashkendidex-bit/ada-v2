"""Soul — Ada's file-backed self (Letta MemFS pattern, Genesis tiers).

Split enforced by FILE LOCATION, not by prompt plea:

  persona.md  — who SHE is. Authored by Yash. Ada READS every turn,
                NEVER writes directly. Edits only via proposal file
                + Yash's spoken yes.
  human.md    — who YASH is. CORE (top, Yash-locked, Ada never writes)
                + LEARNED (bottom, Ada auto-appends timestamped
                learnings with source). Yash prunes anytime.
  humans/     — the human database. One file per person she learns
                about (humans/<Name>.md). Ada auto-appends with
                provenance; Yash owns deletion. Yash himself also gets
                humans/Yash.md as the canonical record over time.
  now.md      — what's live this week. Authored by ADA (dream +
                investigator loops). High churn, pruned weekly.
  diary/      — append-only first-person continuity thread.
                YYYY-MM-DD.md. Never a promotion source directly;
                only grounded facts promote into memory.

Budgets (always-in-context, so small by design):
  persona ~2000 chars, human ~4000, now ~2000, people index ~500.
  Diary never injected whole — only latest entry. Person files load
  on demand via people_read, not every turn.

This is continuity of self, not retrieval. Retrieval answers
"what happened?". Soul answers "who am I, who is he, who are they,
where are we?"
"""
from __future__ import annotations

import time
from pathlib import Path


PERSONA_BUDGET = 2000
HUMAN_BUDGET = 4000
NOW_BUDGET = 2000
PEOPLE_INDEX_BUDGET = 500
PERSON_BUDGET = 1500
LEARNED_CAP = 30
PERSON_CAP = 40
# Below this total, selection buys nothing — inject everything.
FULL_BELOW_CHARS = 6000

# Task-conditioned section routing (Atomic User Model pattern, keyword edition:
# no LLM call, deterministic, falls back to full on any doubt).
_IDENTITY_HINTS = ("who are you", "your name", "about yourself", "soul")
_PEOPLE_HINTS = ("who do you know", "what do you know about", "tell me about",
                 "friend", "client", "people")
_BUILD_HINTS = ("design", "build", "cad", "print", "slice", "gear", "part",
                "model", "prototype", "printer", "filament", "bracket",
                "turbocharger", "sparrow")


def _safe_name(name: str) -> str:
    keep = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in (name or "").strip())
    return keep[:40] or "unknown"


def _near_dup(new: str, existing: list[str], thresh: float = 0.6) -> bool:
    """Word-overlap guard: same fact via two paths (tool + bridge) lands once."""
    nw = set(new.lower().split())
    if not nw:
        return True
    for e in existing:
        ew = set(e.lower().split())
        if not ew:
            continue
        if len(nw & ew) / max(len(nw), len(ew)) >= thresh:
            return True
    return False
    keep = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in (name or "").strip())
    return keep[:40] or "unknown"


class SoulManager:
    def __init__(self, state_dir: str | Path):
        # state_dir = ada_kernel/  → soul lives at ada_kernel/soul/
        self.root = Path(state_dir) / "soul"
        self.persona_path = self.root / "persona.md"
        self.human_path = self.root / "human.md"
        self.now_path = self.root / "now.md"
        self.diary_dir = self.root / "diary"
        self.humans_dir = self.root / "humans"
        self.proposals_path = self.root / "proposals_pending.json"
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self.diary_dir.mkdir(parents=True, exist_ok=True)
            self.humans_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    # ── reads (hot path — never raise, never block) ─────────────────────
    def _read(self, path: Path, budget: int) -> str:
        try:
            if not path.exists():
                return ""
            text = path.read_text().strip()
            # Strip comment header lines starting with # for prompt hygiene,
            # but keep the content lines.
            lines = [l for l in text.splitlines() if not l.strip().startswith("#")]
            text = "\n".join(lines).strip()
            return text[:budget]
        except Exception:
            return ""

    def persona(self) -> str:
        return self._read(self.persona_path, PERSONA_BUDGET)

    def human(self) -> str:
        return self._read(self.human_path, HUMAN_BUDGET)

    def now(self) -> str:
        return self._read(self.now_path, NOW_BUDGET)

    def latest_diary(self) -> str:
        try:
            files = sorted(self.diary_dir.glob("*.md"))
            if not files:
                return ""
            return files[-1].read_text().strip()[-800:]
        except Exception:
            return ""

    def context_block(self) -> str:
        """Assembled soul block for _system_prompt_base(). Small, always."""
        parts = []
        p = self.persona()
        if p:
            parts.append("SOUL — WHO YOU ARE (Yash-authored, read-only for you):\n" + p)
        h = self.human()
        if h:
            parts.append("SOUL — WHO YASH IS (CORE locked; LEARNED is your own notes):\n" + h)
        idx = self.people_index()
        if idx:
            parts.append("SOUL — HUMANS YOU KNOW (full file via people_read):\n" + idx)
        n = self.now()
        if n:
            parts.append("SOUL — WHAT'S NOW (you maintain this):\n" + n)
        d = self.latest_diary()
        if d:
            parts.append("SOUL — LAST DIARY (continuity thread, yesterday's you):\n" + d)
        return "\n\n".join(parts)

    def context_block_for(self, hint: str = "") -> str:
        """Budgeted soul block: sections selected by task, not injected wholesale.

        Mirrors the Atomic User Model finding (8 fields ≈ full-model fidelity
        at ~23% context): identity questions load persona, people questions
        load human+index, build questions load now+diary. Anything ambiguous
        or short loads EVERYTHING — a wrong omission costs more than tokens.
        Below FULL_BELOW_CHARS selection is skipped entirely.
        """
        full = self.context_block()
        if len(full) < FULL_BELOW_CHARS:
            return full
        t = (hint or "").lower()
        if not t or len(t.split()) < 2:
            return full
        want_persona = any(h in t for h in _IDENTITY_HINTS)
        want_people = any(h in t for h in _PEOPLE_HINTS)
        want_build = any(h in t for h in _BUILD_HINTS)
        if not (want_persona or want_people or want_build):
            return full  # doubt → full, always
        parts = []
        if want_persona:
            p = self.persona()
            if p:
                parts.append("SOUL — WHO YOU ARE:\n" + p)
        if want_people:
            h = self.human()
            if h:
                parts.append("SOUL — WHO YASH IS:\n" + h)
            idx = self.people_index()
            if idx:
                parts.append("SOUL — HUMANS YOU KNOW:\n" + idx)
        if want_build:
            n = self.now()
            if n:
                parts.append("SOUL — WHAT'S NOW:\n" + n)
            d = self.latest_diary()
            if d:
                parts.append("SOUL — LAST DIARY:\n" + d)
        # Always keep the partner line so tone never degrades to default-bot.
        if not want_persona:
            parts.insert(0, "SOUL — YOU ARE: Ada, built by Yash, witty partner "
                            "and real friend (Jarvis-for-Tony), never a servant.")
        return "\n\n".join(parts) or full

    # ── human database ──────────────────────────────────────────────
    def list_people(self) -> list[str]:
        try:
            if not self.humans_dir.exists():
                return []
            return sorted(p.stem for p in self.humans_dir.glob("*.md"))
        except Exception:
            return []

    def people_index(self) -> str:
        names = self.list_people()
        if not names:
            return ""
        return ("known people: " + ", ".join(names))[:PEOPLE_INDEX_BUDGET]

    def read_person(self, name: str) -> str:
        try:
            path = self.humans_dir / f"{_safe_name(name)}.md"
            if not path.exists():
                return f"No record for {name} yet — say what you learned and I'll start one."
            lines = [l for l in path.read_text().splitlines()
                     if not l.strip().startswith("#")]
            return ("\n".join(lines).strip() or "(empty record)")[:PERSON_BUDGET]
        except Exception as e:
            return f"people read failed: {type(e).__name__}"

    def learn_about_person(self, name: str, line: str, source: str = "chat") -> str:
        """Append one timestamped learning to humans/<Name>.md. Ada-writable."""
        line = (line or "").strip()[:300]
        name = _safe_name(name)
        if not line or name.lower() == "unknown":
            return "empty learning skipped."
        try:
            self.humans_dir.mkdir(parents=True, exist_ok=True)
            path = self.humans_dir / f"{name}.md"
            if not path.exists():
                path.write_text(
                    f"# {name}\n# Auto-built by Ada from conversation. "
                    f"Yash owns deletion.\n# Format: - [date source] fact\n\n")
            lines = path.read_text().splitlines()
            stamp = time.strftime("%Y-%m-%d")
            if _near_dup(line, [l for l in lines if l.startswith("- ")]):
                return f"Already knew that about {name} — skipped duplicate."
            lines.append(f"- [{stamp} {source}] {line}")
            # Cap: keep header + newest PERSON_CAP learnings
            header = [l for l in lines if l.startswith("#")][:5]
            learnings = [l for l in lines if l.startswith("- ")]
            path.write_text("\n".join(header + learnings[-PERSON_CAP:]) + "\n")
            return f"Learned about {name}."
        except Exception as e:
            return f"people learn failed: {type(e).__name__}"

    def learn_about_sir(self, line: str, source: str = "chat") -> str:
        """Append one learning to human.md LEARNED zone. Never touches CORE."""
        line = (line or "").strip()[:300]
        if not line:
            return "empty learning skipped."
        try:
            if not self.human_path.exists():
                return "human.md missing."
            text = self.human_path.read_text()
            marker = "## LEARNED"
            stamp = time.strftime("%Y-%m-%d")
            entry = f"- [{stamp} {source}] {line}"
            if marker in text:
                head, tail = text.split(marker, 1)
                learnings = [l for l in tail.splitlines() if l.startswith("- ")]
                if _near_dup(entry, learnings):
                    return "Already knew that about Yash — skipped duplicate."
                if entry not in learnings:
                    learnings.append(entry)
                learnings = learnings[-LEARNED_CAP:]
                # Rebuild: head + marker + format-hint lines + learnings
                hints = [l for l in tail.splitlines()
                         if l.startswith("#")][:3]
                text = (head.rstrip() + f"\n\n{marker} — Ada auto-appends below "
                        f"(one line each, newest last)\n"
                        + "\n".join(hints) + "\n" + "\n".join(learnings) + "\n")
            else:
                text = (text.rstrip() + f"\n\n{marker} — Ada auto-appends below "
                        f"(one line each, newest last)\n{entry}\n")
            self.human_path.write_text(text)
            return "human.md LEARNED updated."
        except Exception as e:
            return f"human learn failed: {type(e).__name__}"

    # ── writes: ONLY now.md + diary/ are self-writable ───────────────────
    # persona.md / human.md have NO writer here by design. Any edit must go
    # through propose_identity_edit() → Sir's spoken approval → applied by
    # the kernel tool path, never by the model writing a file.

    def update_now(self, text: str) -> str:
        """Replace now.md content (Ada-authored). Keeps header comment."""
        try:
            header = ("# WHAT'S NOW — Ada maintains this herself\n"
                      "# Author: ADA. Sir can read/edit anytime.\n\n")
            body = (text or "").strip()[:NOW_BUDGET]
            self.root.mkdir(parents=True, exist_ok=True)
            self.now_path.write_text(header + body + "\n")
            return "now.md updated."
        except Exception as e:
            return f"now.md update failed: {type(e).__name__}"

    def append_diary(self, line: str) -> str:
        """Append one entry to today's diary file. Append-only — never rewrite."""
        line = (line or "").strip()[:500]
        if not line:
            return "empty diary line skipped."
        try:
            self.diary_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y-%m-%d")
            path = self.diary_dir / f"{stamp}.md"
            ts = time.strftime("%H:%M")
            with open(path, "a") as f:
                f.write(f"- [{ts}] {line}\n")
            return f"diary {stamp} appended."
        except Exception as e:
            return f"diary append failed: {type(e).__name__}"

    # ── proposals: the ONLY path to persona/human change ─────────────────
    def propose_identity_edit(self, target: str, key: str, value: str) -> str:
        """Stage a persona/human edit for Sir's voice approval. Never applies."""
        import json
        target = (target or "").lower()
        if target not in ("self", "sir", "persona", "human"):
            return "target must be self/persona or sir/human."
        try:
            pending = []
            if self.proposals_path.exists():
                pending = json.loads(self.proposals_path.read_text() or "[]")
            pending.append({
                "ts": time.time(),
                "target": target,
                "key": (key or "").strip()[:80],
                "value": (value or "").strip()[:500],
                "status": "pending",
            })
            self.proposals_path.write_text(json.dumps(pending[-20:], indent=1))
            return (f"Proposed {target}/{key} staged for Sir's spoken approval. "
                    f"Say it out loud and wait for yes — never apply silently.")
        except Exception as e:
            return f"proposal staging failed: {type(e).__name__}"

    def pending_proposals(self) -> list:
        import json
        try:
            if self.proposals_path.exists():
                return json.loads(self.proposals_path.read_text() or "[]")
        except Exception:
            pass
        return []

    def status(self) -> str:
        p = "yes" if self.persona_path.exists() else "missing"
        h = "yes" if self.human_path.exists() else "missing"
        try:
            n_diary = len(list(self.diary_dir.glob("*.md"))) if self.diary_dir.exists() else 0
        except Exception:
            n_diary = 0
        try:
            n_people = len(self.list_people())
        except Exception:
            n_people = 0
        n_prop = len(self.pending_proposals())
        return (f"Soul: persona {p}, human {h}, people {n_people}, "
                f"diary days {n_diary}, proposals pending {n_prop}.")
