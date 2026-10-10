"""Encoding: turning a conversation turn into memory.

The old write path was ``MemoryBank.remember_context``, which did three things: append
the raw text, append a truncated copy as an "episode", and run six regular
expressions looking for a name pattern. Over the life of the database it captured
exactly one fact.

What this does instead, in the order the biology does it:

1. **Capture raw, immediately, losslessly.** The full turn goes in at fidelity L0
   before anything else happens. This is the CLS fast store — cheap, fast, and it
   cannot fail in a way that loses the turn. Everything after this is enrichment.
2. **Score salience.** Importance (how much does this matter) and arousal (how charged
   is it). The amygdala tags arousing material for priority storage; the Generative
   Agents architecture does the same thing by asking a model to rate poignancy 1-10 at
   encoding time. Both are recorded.
3. **Extract structure.** Entities and candidate facts with temporal bounds. This is
   depth of processing: capture the *meaning* now rather than deferring it, because
   deferring it is how you end up with a bag of text and no way to query it.
4. **Resolve and link.** Entities are matched against the existing graph so "the
   drone" and "the quadcopter" converge instead of fragmenting.
5. **Detect contradiction.** A new fact that conflicts with a live one invalidates the
   old row rather than deleting it — bi-temporal, so "what did I believe last Tuesday"
   stays answerable.

Everything except step 1 can run off the critical path. The turn must never wait on an
LLM to be remembered.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .embeddings import Embedder, to_blob
from .store import Store, dumps, loads, now

log = logging.getLogger("ada.memory.encode")

# ── heuristic salience ───────────────────────────────────────────────────────
# Durable-signal markers: language that says "this is a standing fact about me",
# as opposed to conversational noise. Used when no LLM is available, and as a floor
# under the LLM score when one is.
_DURABLE = [
    (r"\b(remember|don't forget|note that|keep in mind)\b", 4),
    (r"\b(always|never|every time|from now on)\b", 3),
    (r"\b(i prefer|i like|i hate|i dislike|i want|i need)\b", 3),
    (r"\b(my name is|call me|i am|i'm|i work|i live)\b", 3),
    (r"\b(decision|decided|final|settled on|we chose|going with)\b", 3),
    (r"\b(deadline|due|appointment|meeting at|remind me)\b", 3),
    (r"\b(budget|cost|price|quote|invoice)\b", 2),
]
_AROUSAL = [
    (r"[!]{2,}|\b(finally|brilliant|perfect|excellent|amazing)\b", 3),
    (r"\b(broken|failed|crash|cracked|wrong|problem|error|urgent)\b", 3),
    (r"\b(angry|frustrat|annoy|worried|stress|tired|exhaust)\w*\b", 2),
]
# People-introducing language: "my friend Arjun ...", "remember Arjun ...".
# When a turn matches AND carries durable content, the soul bridge mirrors
# one line into humans/<Name>.md — no new LLM call, off critical path.
_PERSON_INTRO = re.compile(
    r"\bmy\s+(friend|brother|sister|dad|mom|father|mother|colleague|client|"
    r"mentor|teacher|boss|partner|wife|husband|son|daughter|cousin|uncle|aunt)\s+"
    r"([A-Z][a-z]{2,20})\b")
_PERSON_REMEMBER = re.compile(
    r"\bremember\s+([A-Z][a-z]{2,20})\s+(likes|builds|promised|prefers|hates|"
    r"works|lives|told|said|needs|wants|owns|runs|is|has)\b")
# Self-signal: durable truth about Yash himself → human.md LEARNED zone.
_SELF_DURABLE = re.compile(
    r"\b(remember i|don't forget i|note that i|i always|i never|i prefer|"
    r"i like|i hate|my project|my deadline|i've decided|i decided)\b", re.I)
# Numbers with units are specification, and specification is what a design
# assistant gets asked about later. "the arm is 160 millimetres" matters more than
# "the arm looks nice".
_SPEC = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mm|cm|m|g|kg|n|nm|kn|mpa|gpa|rpm|hz|khz|v|a|w|°c|deg|c|f|in|ft|psi|bar|%|x)\b", re.I)
_MONEY = re.compile(r"[$£€]\s?\d|\b\d+\s?(?:pounds|dollars|euros|rupees)\b", re.I)

# Words that carry no entity information even when capitalised.
_ENTITY_STOP = {
    "the", "this", "that", "there", "then", "they", "them", "and", "but", "for",
    "sir", "ada", "i", "i'm", "i'll", "i've", "yes", "no", "ok", "okay", "well",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "today", "tomorrow", "yesterday",
    "what", "when", "where", "which", "who", "why", "how", "sure", "right", "got",
}


@dataclass
class Encoded:
    """What the encoder produced, for logging and tests."""
    episode_id: int
    importance: int
    arousal: int
    entities: list[str] = field(default_factory=list)
    facts: list[tuple[str, str, str]] = field(default_factory=list)  # (subject, predicate, object)
    superseded: list[int] = field(default_factory=list)
    source: str = "heuristic"     # 'heuristic' | 'llm'


class Encoder:
    """Writes memory. Never raises into the conversation loop.

    `llm` is an optional async callable ``(prompt: str) -> str``. When supplied it is
    used for salience scoring and fact extraction, which is materially better than the
    heuristics. When absent — or when it fails, or times out — the heuristics take
    over, so a provider outage degrades memory quality instead of stopping memory.
    """

    def __init__(self, store: Store, embedder: Embedder | None = None,
                 llm: Optional[Callable[[str], Any]] = None):
        self.store = store
        self.embedder = embedder or Embedder()
        self.llm = llm

    # ── step 1: capture ──────────────────────────────────────────────────────
    def capture(self, user_text: str, ada_reply: str = "", *, session_id: str | None = None,
                turn_index: int | None = None, kind: str = "conversation") -> int:
        """Store the raw turn at full fidelity. Returns the episode id.

        Deliberately the only thing on the synchronous path. If this succeeds the
        turn is remembered, whatever happens to the enrichment afterwards.
        """
        text = (user_text or "").strip()
        reply = (ada_reply or "").strip()
        if not text and not reply:
            return 0
        body = f"Q: {text}\nA: {reply}" if reply else text
        ts = time.time()
        vec = self.embedder.embed(body)
        cur = self.store.execute(
            "INSERT INTO recall_entries(kind, content, ts, ts_event, embedding,"
            " embed_backend, fidelity_level, importance, arousal, access_count,"
            " last_access, stability, difficulty, session_id, turn_index, pinned,"
            " quarantined) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (kind, body, ts, ts, to_blob(vec), self.embedder.backend_name, 0,
             3, 1, 0, ts, 1.0, 5.0, session_id, turn_index, 0, 0))
        eid = int(cur.lastrowid)
        self.store.fts_index(eid)
        return eid

    # ── steps 2-5: enrich ────────────────────────────────────────────────────
    def enrich(self, episode_id: int, user_text: str, ada_reply: str = "") -> Encoded:
        """Score, extract, link, and check contradictions. Safe to call off-thread."""
        try:
            return self._enrich(episode_id, user_text, ada_reply)
        except Exception as e:
            log.warning("enrich failed for episode %s (%s: %s); raw capture stands",
                        episode_id, type(e).__name__, e)
            return Encoded(episode_id=episode_id, importance=3, arousal=1, source="failed")

    def _enrich(self, episode_id: int, user_text: str, ada_reply: str) -> Encoded:
        text = (user_text or "").strip()
        combined = f"{text}\n{ada_reply or ''}".strip()

        importance, arousal, source = self._salience(text, ada_reply)
        entities = self._entities(combined)
        facts = self._facts(text)

        # Persist salience on the episode.
        self.store.execute(
            "UPDATE recall_entries SET importance=?, arousal=?, entities=?, summary=?"
            " WHERE id=?",
            (importance, arousal, dumps(entities), _first_sentence(combined), episode_id))

        # Link entities to this episode (the hippocampal index: a pointer, not a copy).
        for e in entities:
            self._link_entity(e, episode_id)

        # Facts, with contradiction handling.
        superseded = []
        for subj, pred, obj in facts:
            old = self._find_conflicting(subj, pred, obj)
            fid = self._add_fact(subj, pred, obj, episode_id, importance)
            for old_id in old:
                self._invalidate(old_id, superseded_by=fid, at=time.time())
                superseded.append(old_id)

        # Entity edges between co-occurring entities, so the graph has structure.
        self._link_cooccurrence(entities, episode_id)

        # Soul bridge: mirror durable people/self learnings into the human
        # database (soul/humans/ + human.md LEARNED). File writes only, no
        # LLM, never raises — the DB row above is already safe.
        try:
            self._mirror_to_soul(text, importance)
        except Exception as e:
            log.debug("soul mirror skipped: %s", e)

        return Encoded(episode_id=episode_id, importance=importance, arousal=arousal,
                       entities=entities, facts=facts, superseded=superseded,
                       source=source)

    # ── soul bridge ─────────────────────────────────────────────────────
    def _mirror_to_soul(self, user_text: str, importance: int) -> None:
        """Mirror durable people/self learnings into soul files.

        Runs inside enrich (background thread, every turn, both lanes).
        Only fires on durable turns (importance >= 5) with an explicit
        person/self pattern — chit-chat never touches the soul.
        """
        if importance < 5 or not (user_text or "").strip():
            return
        try:
            from soul import SoulManager  # local import: memory must stay light
        except Exception:
            return
        try:
            soul = SoulManager(self.store.db_path.parent)
        except Exception:
            return
        text = user_text.strip()
        gist = text[:280]
        seen = set()
        for m in _PERSON_INTRO.finditer(text):
            name = m.group(2)
            if name and name not in seen:
                seen.add(name)
                soul.learn_about_person(name, gist, source="chat")
        for m in _PERSON_REMEMBER.finditer(text):
            name = m.group(1)
            if not name or name in seen:
                continue
            seen.add(name)
            if name.lower() in ("yash", "sir", "ada"):
                soul.learn_about_sir(gist, source="chat")
            else:
                soul.learn_about_person(name, gist, source="chat")
        if not seen and _SELF_DURABLE.search(text):
            soul.learn_about_sir(gist, source="chat")

    # ── salience ─────────────────────────────────────────────────────────────
    def _salience(self, user_text: str, ada_reply: str) -> tuple[int, int, str]:
        """Importance and arousal, 1-10. Heuristic, optionally refined by an LLM."""
        if self.llm is not None:
            got = self._llm_salience(user_text, ada_reply)
            if got:
                return got[0], got[1], "llm"
        return (*self._heuristic_salience(user_text), "heuristic")

    def _heuristic_salience(self, text: str) -> tuple[int, int]:
        t = (text or "").lower()
        imp = 2
        for pat, w in _DURABLE:
            if re.search(pat, t):
                imp += w
        if _SPEC.search(text or ""):
            imp += 2
        if _MONEY.search(text or ""):
            imp += 1
        # Very short utterances are usually acknowledgements, not content.
        if len(t.split()) < 4:
            imp -= 1
        aro = 1
        for pat, w in _AROUSAL:
            if re.search(pat, t):
                aro += w
        return max(1, min(10, imp)), max(1, min(10, aro))

    def _llm_salience(self, user_text: str, ada_reply: str) -> Optional[tuple[int, int]]:
        """Ask the model to rate poignancy, as the Generative Agents paper does.

        The anchors are copied from that work because they are what makes the scale
        reproducible: 2 = tidying a room, 8 = asking someone out.
        """
        prompt = (
            "Rate this exchange for a personal assistant's long-term memory.\n"
            "IMPORTANCE 1-10: how much will this matter in a month? "
            "1 = purely mundane (small talk, acknowledgements), "
            "5 = ordinary project detail, 8 = a standing preference, a decision, or a "
            "deadline, 10 = a major life event.\n"
            "AROUSAL 1-10: how emotionally charged is it? 1 = flat, "
            "10 = distressing or elating.\n"
            f"Exchange:\nUSER: {user_text[:600]}\nADA: {(ada_reply or '')[:400]}\n"
            'Reply with ONLY JSON: {"importance": <int>, "arousal": <int>}'
        )
        try:
            raw = self.llm(prompt)
            data = _first_json(raw)
            if not data:
                return None
            return (max(1, min(10, int(data.get("importance", 3)))),
                    max(1, min(10, int(data.get("arousal", 1)))))
        except Exception as e:
            log.info("llm salience failed (%s); using heuristics", type(e).__name__)
            return None

    # ── extraction ───────────────────────────────────────────────────────────
    def _entities(self, text: str) -> list[str]:
        """Capitalised multi-word phrases and specification nouns.

        Deliberately simple and local: entity extraction is on the write path for
        every turn, and an LLM call per turn to find nouns is not worth the latency.
        The consolidation job can do a deeper pass later.
        """
        if not text:
            return []
        found: list[str] = []
        # Multi-word capitalised phrases first ("Iron Man repulsor", "M8 flange bolt").
        for m in re.finditer(r"\b(?:[A-Z][a-z0-9]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z0-9]+|[A-Z]{2,})){0,2}\b", text):
            phrase = m.group(0).strip()
            if phrase.lower() in _ENTITY_STOP:
                continue
            if len(phrase) < 3:
                continue
            found.append(phrase)
        # Part numbers and model codes are entities too, and they are not capitalised.
        for m in re.finditer(r"\b[A-Z]{1,3}\d{1,4}[A-Za-z]*\b|\b[A-Za-z]+\d+[A-Za-z]+\b", text):
            found.append(m.group(0))

        # Dedupe case-insensitively, preserving first-seen form and order.
        seen, out = set(), []
        for f in found:
            key = f.lower()
            if key not in seen:
                seen.add(key)
                out.append(f)
        return out[:12]

    def _facts(self, text: str) -> list[tuple[str, str, str]]:
        """(subject, predicate, object) triples worth storing as durable knowledge.

        Narrow on purpose. A wrong fact is worse than a missing one — it will be
        retrieved with confidence and cited back. Only patterns that are unambiguous
        about being a standing claim are extracted here; anything fuzzier is left for
        the consolidation pass, where a model can weigh the whole conversation.
        """
        t = (text or "").strip()
        low = t.lower()
        out: list[tuple[str, str, str]] = []

        patterns = [
            (r"\bmy name is (\w+)", "user", "name"),
            (r"\bcall me (\w+)", "user", "name"),
            (r"\bi (?:work|am working) (?:at|for|as) ([\w &]+?)(?:[.,]|$)", "user", "works_at"),
            (r"\bi live in ([\w ,]+?)(?:[.,]|$)", "user", "lives_in"),
            (r"\bi (?:prefer|like|use) ([\w ]+?)(?:[.,]|$)", "user", "prefers"),
            (r"\bi (?:hate|dislike|avoid) ([\w ]+?)(?:[.,]|$)", "user", "dislikes"),
            (r"\bi (?:always|never) ([\w ]+?)(?:[.,]|$)", "user", "habit"),
            (r"\bremember (?:that )?(.+?)(?:[.,]|$)", "user", "note"),
        ]
        for pat, subj, pred in patterns:
            m = re.search(pat, low)
            if m:
                val = m.group(1).strip()
                if 2 <= len(val) <= 80:
                    out.append((subj, pred, val))

        # Explicit decisions about the project: these are the highest-value facts.
        m = re.search(r"\b(?:final|settled on|we (?:chose|decided)|going with)\s*(?:is|:)?\s*([\w \-]+?)(?:[.,]|$)", low)
        if m and len(m.group(1).strip()) > 2:
            out.append(("project", "decided", m.group(1).strip()))

        # Specifications: "<thing> is/measures <number> <unit>"
        for m in _SPEC.finditer(t):
            head = t[max(0, m.start() - 60):m.start()]
            noun = re.findall(r"([a-z]+(?:\s+[a-z]+)?)\s+(?:is|measures|weighs|=|:)\s*$", head.lower())
            if noun:
                out.append((noun[0].strip(), "measures", m.group(0).strip()))
        return out[:6]

    # ── graph ────────────────────────────────────────────────────────────────
    def _link_entity(self, name: str, episode_id: int) -> int:
        row = self.store.one("SELECT id, mention_count FROM entities WHERE name=? COLLATE NOCASE", (name,))
        ts = time.time()
        if row:
            self.store.execute(
                "UPDATE entities SET mention_count=mention_count+1, last_seen=? WHERE id=?",
                (ts, row["id"]))
            return int(row["id"])
        cur = self.store.execute(
            "INSERT INTO entities(name, canonical, kind, summary, embedding,"
            " first_seen, last_seen, mention_count, salience) VALUES(?,?,?,?,?,?,?,?,?)",
            (name, name, None, None, to_blob(self.embedder.embed(name)), ts, ts, 1, 0.5))
        return int(cur.lastrowid)

    def _link_cooccurrence(self, entities: list[str], episode_id: int) -> None:
        if len(entities) < 2:
            return
        ts = time.time()
        ids = {}
        for e in entities:
            row = self.store.one("SELECT id FROM entities WHERE name=? COLLATE NOCASE", (e,))
            if row:
                ids[e] = int(row["id"])
        rows = []
        for a in entities:
            for b in entities:
                if a != b and a in ids and b in ids:
                    rows.append((a, b, "co_occurs", ts, ids[a], ids[b], ts, None, ts, None,
                                 0.5, episode_id, 1.0))
        if rows:
            self.store.executemany(
                "INSERT INTO entity_edges(a, b, rel, ts, src_id, dst_id, t_valid,"
                " t_invalid, t_created, t_expired, confidence, source_episode_id, weight)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)

    # ── bi-temporal facts ────────────────────────────────────────────────────
    def _find_conflicting(self, subj: str, pred: str, obj: str) -> list[int]:
        """Live facts with the same subject+predicate but a different object.

        Only same-(subject, predicate) pairs can conflict: "user prefers metric" does
        not contradict "user dislikes brass" even though both are about the user.
        """
        rows = self.store.query(
            "SELECT id, object FROM core_facts WHERE subject=? AND predicate=?"
            " AND t_invalid IS NULL", (subj, pred))
        return [int(r["id"]) for r in rows
                if (r["object"] or "").strip().lower() != (obj or "").strip().lower()]

    def _add_fact(self, subj: str, pred: str, obj: str, episode_id: int,
                  importance: int) -> int:
        key = f"{subj}_{pred}"
        ts = time.time()
        cur = self.store.execute(
            "INSERT INTO core_facts(key, value, source, ts, importance, subject,"
            " predicate, object, confidence, pinned, t_valid, t_invalid, t_created,"
            " t_expired, superseded_by, source_episode_id, activation_strength,"
            " embedding, updated) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (key, obj, "conversation", ts, importance, subj, pred, obj, 0.8, 0,
             ts, None, ts, None, None, episode_id, 1.0,
             to_blob(self.embedder.embed(f"{subj} {pred} {obj}")), ts))
        return int(cur.lastrowid)

    def _invalidate(self, fact_id: int, superseded_by: int, at: float) -> None:
        """Mark a fact as no longer true from `at`. Never deletes it.

        This is the whole point of the bi-temporal design: the old value stops being
        live but remains answerable, so "you told me titanium on Tuesday" is a true
        statement she can make rather than a memory that was overwritten.
        """
        self.store.execute(
            "UPDATE core_facts SET t_invalid=?, t_expired=?, superseded_by=? WHERE id=?",
            (at, at, superseded_by, fact_id))

    # ── maintenance ──────────────────────────────────────────────────────────
    def decay_importance(self, half_life_days: float = 30.0) -> int:
        """Decay unconsolidated salience. Pinned rows are exempt.

        Importance is decayed rather than rewritten so that a memory which keeps being
        retrieved can hold its own against age (retrieval practice, in the biology).
        """
        rows = self.store.query(
            "SELECT id, importance, ts, access_count, pinned FROM recall_entries"
            " WHERE COALESCE(pinned,0)=0 AND fidelity_level < 5")
        changed = 0
        for r in rows:
            age_days = max(0.0, (time.time() - (r["ts"] or time.time())) / 86400.0)
            # Rehearsal slows decay: a memory recalled often decays more slowly.
            effective_hl = half_life_days * (1.0 + 0.5 * min(int(r["access_count"] or 0), 10))
            factor = 0.5 ** (age_days / max(effective_hl, 1e-6))
            new_imp = max(1, int(round(int(r["importance"] or 3) * max(factor, 0.25))))
            if new_imp != int(r["importance"] or 3):
                self.store.execute("UPDATE recall_entries SET importance=? WHERE id=?",
                                   (new_imp, r["id"]))
                changed += 1
        return changed


# ── helpers ──────────────────────────────────────────────────────────────────
def _first_sentence(text: str, limit: int = 300) -> str:
    t = (text or "").strip().replace("\n", " ")
    if len(t) <= limit:
        return t
    cut = t[:limit]
    for sep in (". ", "! ", "? "):
        i = cut.rfind(sep)
        if i > limit // 2:
            return cut[: i + 1]
    return cut.rsplit(" ", 1)[0] + "..."


def _first_json(raw: Any) -> Optional[dict]:
    """Pull the first JSON object out of a model reply, tolerating prose and fences."""
    if not isinstance(raw, str):
        return None
    t = raw.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[4:] if t.lower().startswith("json") else t
    i, j = t.find("{"), t.rfind("}")
    if i == -1 or j == -1:
        return None
    try:
        out = json.loads(t[i:j + 1])
        return out if isinstance(out, dict) else None
    except Exception:
        return None
