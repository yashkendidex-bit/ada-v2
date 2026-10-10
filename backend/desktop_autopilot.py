"""
Desktop Autopilot — the Iron Man 1 principle (the suit flies Tony home).

Extends the browser web-agent idea to the whole OS: screenshot -> the vision
model decides one action -> pyautogui executes it -> repeat until done.
Destructive steps are gated behind a spoken confirmation, and the failsafe
(slamming the mouse into a screen corner) hard-aborts everything.

The model may only answer with ONE json action:
  {"action": "click"|"double_click"|"right_click"|"type"|"key"|"scroll"|"drag"|"done"|"cannot",
   "x": int, "y": int, "text": str, "note": str, "destructive": bool}

Safety doctrine (fail-closed):
  - A step marked destructive is executed ONLY after an explicit True from
    ask_user_fn. If no confirmation channel exists, the step is REFUSED, never
    executed. Silence is not consent.
  - Unknown/empty actions are skipped, never guessed at.
"""
import io
import json
import base64
import asyncio

MAX_STEPS = 15
# Identical action signatures in a row before the run is declared stuck.
# A vision loop repeating itself is not progress — stop and report
# INCONCLUSIVE with the evidence instead of burning the step budget.
STUCK_REPEAT_LIMIT = 3
Screenshot = None  # lazy


def _arm_failsafe():
    """Turn on pyautogui's corner failsafe before we drive the mouse.

    With FAILSAFE on, jerking the pointer into any screen corner raises
    FailSafeException and the run aborts immediately. pyautogui ships it enabled,
    but it is a plain module attribute that anything can flip, and the abort only
    means something if the caller catches it — so we set it explicitly here and
    `run_task` treats the exception as a hard stop. This is the physical kill
    switch the module docstring has always promised.
    """
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05


def _is_failsafe(exc: BaseException) -> bool:
    """Match FailSafeException without importing pyautogui at module load.

    Name-matching keeps this module importable on machines with no display, which
    matters because ADA's backend starts even when screen control is unusable.
    """
    return type(exc).__name__ == "FailSafeException"


ABORTED_BY_FAILSAFE = (
    "Aborted — failsafe triggered. You moved the mouse into a screen corner, "
    "so I stopped and left everything as it was."
)


class DesktopAutopilot:
    VALID_ACTIONS = frozenset({
        "click", "double_click", "right_click", "type", "key",
        "scroll", "drag", "done", "cannot",
    })

    def __init__(self, vision_fn, ask_user_fn=None, update_fn=None, audit_fn=None):
        """
        vision_fn:   async(messages_openai_format) -> str  (vision model chain)
        ask_user_fn: async(question) -> bool               (spoken confirmation gate)
        update_fn:   async(event_dict) -> None             (frontend progress feed)
        audit_fn:    sync/async(actor, action, target, decision, detail) -> None
                     (security audit trail; optional, failures never break a run)
        """
        self.vision_fn = vision_fn
        self.ask_user_fn = ask_user_fn
        self.update_fn = update_fn
        self.audit_fn = audit_fn

    # ── helpers ───────────────────────────────────────────────────────────────
    def _screenshot_b64(self, max_side=1280):
        import pyautogui
        from PIL import Image
        _arm_failsafe()
        shot = pyautogui.screenshot()
        w, h = shot.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            shot = shot.resize((int(w * scale), int(h * scale)))
        buf = io.BytesIO()
        shot.convert("RGB").save(buf, format="JPEG", quality=72)
        return base64.b64encode(buf.getvalue()).decode(), scale

    def _system_rules(self) -> str:
        return (
            "You are ADA's hands, operating the user's desktop to complete their goal. "
            "You receive a screenshot each turn. Reply with ONLY one JSON action, no prose:\n"
            '{"action": "click|double_click|right_click|type|key|scroll|drag|done|cannot", '
            '"x": <pixel x in the image you see>, "y": <pixel y>, "text": "<for type/key>", '
            '"note": "<one short sentence: what you see/did or why done/cannot>", '
            '"destructive": <true if this step could delete, send, submit, pay, or be hard to undo>}\n'
            "Rules: coordinates are in the coordinate space of the image you see. "
            "Prefer precise clicks on visible UI. Type into fields before pressing enter. "
            "For drag, put the START in x/y and the DESTINATION in text as 'x2,y2'. "
            "Mark destructive=true for send/submit/delete/pay/purchase steps. "
            "Never repeat an action that just failed — try a different control. "
            "Use done when the goal is visibly achieved; cannot if blocked (captcha, "
            "missing app, permission wall) — explain in note.")

    async def _decide(self, goal: str, history: list, img_b64: str):
        content = [{"type": "text", "text": goal + "\n\nHistory so far:\n" + "\n".join(history)}]
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})
        messages = [{"role": "system", "content": self._system_rules()},
                    {"role": "user", "content": content}]
        return await self.vision_fn(messages)

    @staticmethod
    def _parse_action(raw: str):
        if not raw:
            return None
        t = raw.strip()
        if t.startswith("```"):
            t = t.strip("`").lstrip("json").strip()
        i, j = t.find("{"), t.rfind("}")
        if i == -1 or j == -1:
            return None
        try:
            act = json.loads(t[i:j + 1])
        except Exception:
            return None
        if not isinstance(act, dict) or "action" not in act:
            return None
        return act

    @classmethod
    def _validate_action(cls, act: dict) -> str | None:
        """Returns an error string for an unusable action, else None.

        Validation is strict about SHAPE (known action, numeric coords,
        sane text) and permissive about INTENT (the vision model owns that).
        An invalid action is skipped with a history note, never executed."""
        a = act.get("action")
        if a not in cls.VALID_ACTIONS:
            return f"unknown action '{a}'"
        if a in ("click", "double_click", "right_click", "drag"):
            try:
                x, y = float(act.get("x", "nan")), float(act.get("y", "nan"))
            except (TypeError, ValueError):
                return f"{a} needs numeric x/y"
            if not (0 <= x <= 10000 and 0 <= y <= 10000):
                return f"{a} coordinates out of range"
            if a == "drag":
                import re as _re
                if not _re.match(r"^\s*\d+(\.\d+)?\s*,\s*\d+(\.\d+)?\s*$",
                                 str(act.get("text", ""))):
                    return "drag needs text as 'x2,y2' destination"
        if a in ("type", "key") and not str(act.get("text", "")).strip():
            return f"{a} needs text"
        return None

    async def _audit(self, action, target="", decision="allow", detail=""):
        if not self.audit_fn:
            return
        try:
            r = self.audit_fn("autopilot", action, target, decision, detail)
            if asyncio.iscoroutine(r):
                await r
        except Exception:
            pass

    @staticmethod
    def _action_signature(act: dict) -> str:
        return f"{act.get('action')}|{act.get('x')}|{act.get('y')}|{str(act.get('text', ''))[:40]}"

    async def _execute(self, act: dict, scale: float):
        import pyautogui
        inv = 1.0 / max(scale, 1e-6)
        a = act.get("action")

        def _run():
            _arm_failsafe()
            if a in ("click", "double_click", "right_click"):
                x = int(float(act.get("x", 0)) * inv)
                y = int(float(act.get("y", 0)) * inv)
                if a == "click":
                    pyautogui.click(x, y)
                elif a == "double_click":
                    pyautogui.doubleClick(x, y)
                else:
                    pyautogui.rightClick(x, y)
                return f"{a} at ({x}, {y})"
            if a == "type":
                pyautogui.write(str(act.get("text", ""))[:500], interval=0.02)
                return f"typed: {str(act.get('text',''))[:40]}"
            if a == "key":
                pyautogui.hotkey(*[k.strip() for k in str(act.get("text", "escape")).split("+")][:3])
                return f"key: {act.get('text')}"
            if a == "scroll":
                direction = str(act.get("text", "down")).lower()
                pyautogui.scroll(500 if direction.startswith("up") else -500)
                return "scrolled"
            if a == "drag":
                import re as _re
                m = _re.match(r"^\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*$",
                              str(act.get("text", "")))
                x1 = int(float(act.get("x", 0)) * inv)
                y1 = int(float(act.get("y", 0)) * inv)
                x2 = int(float(m.group(1)) * inv)
                y2 = int(float(m.group(2)) * inv)
                pyautogui.moveTo(x1, y1)
                pyautogui.dragTo(x2, y2, duration=0.4)
                return f"drag from ({x1}, {y1}) to ({x2}, {y2})"
            return f"unknown action {a}"

        return await asyncio.to_thread(_run)

    # ── main loop ─────────────────────────────────────────────────────────────
    async def run_task(self, goal: str) -> str:
        history = []
        repeat_signatures: list[str] = []
        for step in range(1, MAX_STEPS + 1):
            if self.update_fn:
                await self._safe_update({"step": step, "max": MAX_STEPS,
                                         "log": f"Looking at the screen ({step}/{MAX_STEPS})..."})
            # Screenshot + JPEG encode is blocking work (tens of ms, more on a retina
            # display). Running it inline stalled the entire asyncio loop every step,
            # which froze ADA's voice while the autopilot was working.
            try:
                img_b64, scale = await asyncio.to_thread(self._screenshot_b64)
            except Exception as e:
                if _is_failsafe(e):
                    return ABORTED_BY_FAILSAFE
                return f"Autopilot could not read the screen: {str(e)[:120]}"
            try:
                raw = await self._decide(goal, history[-6:], img_b64)
            except Exception as e:
                return f"Autopilot stalled: {str(e)[:120]}"
            act = self._parse_action(raw)
            if act is None:
                history.append(f"step {step}: model gave no valid action; pressing on")
                continue
            bad = self._validate_action(act)
            if bad is not None:
                history.append(f"step {step}: invalid action skipped ({bad}); pressing on")
                if self.update_fn:
                    await self._safe_update({"step": step, "max": MAX_STEPS,
                                             "log": f"Skipped invalid action ({bad})."})
                continue
            # Stuck-loop detection: the same action three times running is not a
            # plan, it is a loop. Stop with the evidence, not the step budget.
            sig = self._action_signature(act)
            repeat_signatures.append(sig)
            if (len(repeat_signatures) >= STUCK_REPEAT_LIMIT
                    and len(set(repeat_signatures[-STUCK_REPEAT_LIMIT:])) == 1):
                await self._audit("stuck-loop", goal[:120], "abort",
                                  f"same action {STUCK_REPEAT_LIMIT}x: {sig[:120]}")
                return (f"INCONCLUSIVE — I repeated the same step "
                        f"({act.get('action')}) {STUCK_REPEAT_LIMIT} times without progress. "
                        f"Last state: {history[-1][:160] if history else 'no actions'}. "
                        f"Try a more specific goal or a different app state.")
            note = str(act.get("note", ""))[:160]
            a = act.get("action")

            if a == "done":
                if self.update_fn:
                    await self._safe_update({"step": step, "max": MAX_STEPS,
                                             "log": f"Done: {note}"})
                await self._audit("run", goal[:120], "done", note[:200])
                return f"Task complete. {note}"
            if a == "cannot":
                await self._audit("run", goal[:120], "cannot", note[:200])
                return f"I couldn't finish — {note}"

            if act.get("destructive"):
                # FAIL-CLOSED: without a confirmation channel the step is
                # refused, never executed. Silence is not consent.
                if not self.ask_user_fn:
                    await self._audit("destructive-refused", goal[:120], "deny",
                                      note[:200])
                    return ("Stopped — that step could be hard to undo "
                            f"({note}), and there is no confirmation channel open, "
                            "so I refused to act rather than risk it.")
                ok = await self.ask_user_fn(
                    f"This step could be hard to undo: {note}. Shall I go ahead?")
                await self._audit("destructive", note[:200],
                                  "approved" if ok else "denied", goal[:120])
                if not ok:
                    return "Stopped — you declined that step, so I left everything untouched."

            try:
                result = await self._execute(act, scale)
            except Exception as e:
                if _is_failsafe(e):
                    return ABORTED_BY_FAILSAFE
                result = f"action failed: {str(e)[:80]}"
            history.append(f"step {step}: {a} -> {result} | screen said: {note}")
            if self.update_fn:
                await self._safe_update({"step": step, "max": MAX_STEPS,
                                         "log": f"{a}: {note}"})
        return f"Reached the {MAX_STEPS}-step safety limit. Last state: {history[-1][:120] if history else 'no actions'}"

    async def _safe_update(self, event):
        try:
            r = self.update_fn(event)
            if asyncio.iscoroutine(r):
                await r
        except Exception:
            pass
