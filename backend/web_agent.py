"""Web agent — browser automation on Qwen 3.8 27B (Groq, OpenAI tools).

Each turn: screenshot -> Qwen picks ONE browser tool (OpenAI function-calling,
verified live 2026-09-30: correct type_text_at in 1.4s) -> Playwright executes
-> repeat until done(summary). Same 0-1000 coordinate space as before.
Override back via ADA_WEB_PROVIDER / ADA_WEB_MODEL (e.g. the old
nvidia/meta/muse-glimmer-30b).

Safety: submit-style steps (pressing Enter to send/submit, Enter hotkeys) are
asked out loud FIRST when a confirm channel is wired (ada wires the spoken
gate). Fail-closed: with no channel, or a declined/timed-out answer, the
submit does NOT happen — silence is never consent.
"""
import os
import asyncio
import base64
from dotenv import load_dotenv
from playwright.async_api import async_playwright

import providers

# 1. Load config (all optional — providers.available() gates at call time)
load_dotenv(override=True)

# 2. Configuration
SCREEN_WIDTH = 1440
SCREEN_HEIGHT = 900
WEB_PROVIDER = os.getenv("ADA_WEB_PROVIDER", "groq")
WEB_MODEL = os.getenv("ADA_WEB_MODEL", "qwen/qwen3.8-27b")
MAX_TURNS = 20

SYSTEM_RULES = (
    "You are ADA's browser hands, operating a Chromium window to complete the user's goal. "
    "You receive a screenshot each turn. Reply with tool calls ONLY — no prose, no markdown. "
    "Coordinates are 0-1000, origin top-left of the screenshot you see. "
    "Rules: prefer precise clicks on visible controls; type into a field (click it first) before pressing Enter; "
    "never repeat a tool call that just failed — try a different control; "
    "call done(summary) with a one-paragraph result the moment the goal is visibly achieved, "
    "or done(summary) explaining the blockage (captcha, login wall, missing page) if it cannot be done."
)

WEB_TOOLS = [
    {"type": "function", "function": {
        "name": "navigate",
        "description": "Go to a URL in the address bar.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "description": "Full https URL."}}, "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "go_back", "description": "Browser back button.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "go_forward", "description": "Browser forward button.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "click_at", "description": "Left-click a point.",
        "parameters": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"}}, "required": ["x", "y"]}}},
    {"type": "function", "function": {
        "name": "type_text_at",
        "description": "Click a field, clear it, type text, optionally press Enter to submit.",
        "parameters": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"},
            "text": {"type": "string"},
            "press_enter": {"type": "boolean", "description": "Submit the field. Default false."},
            "clear_before_typing": {"type": "boolean", "description": "Default true."}},
            "required": ["x", "y", "text"]}}},
    {"type": "function", "function": {
        "name": "hover_at", "description": "Move the mouse without clicking.",
        "parameters": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"}}, "required": ["x", "y"]}}},
    {"type": "function", "function": {
        "name": "drag_and_drop", "description": "Drag from a point to a destination.",
        "parameters": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"},
            "destination_x": {"type": "number"}, "destination_y": {"type": "number"}},
            "required": ["x", "y", "destination_x", "destination_y"]}}},
    {"type": "function", "function": {
        "name": "key_combination", "description": "Press keys, e.g. 'Enter', 'Escape', 'Control+t'.",
        "parameters": {"type": "object", "properties": {
            "keys": {"type": "string"}}, "required": ["keys"]}}},
    {"type": "function", "function": {
        "name": "scroll_document",
        "description": "Scroll the page.",
        "parameters": {"type": "object", "properties": {
            "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
            "magnitude": {"type": "number", "description": "Pixels. Default 800."}},
            "required": ["direction"]}}},
    {"type": "function", "function": {
        "name": "done",
        "description": "Finish: report the result summary, or the blockage if unachievable.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string"}}, "required": ["summary"]}}},
]

# Tool calls that submit/send — confirmed out loud when a channel is wired.
_SUBMIT_TOOLS = {"type_text_at": "press_enter", "key_combination": "keys"}


class WebAgent:
    def __init__(self, ask_user_fn=None, audit_fn=None):
        """
        ask_user_fn: async(question) -> bool  (spoken confirmation gate for submits)
        audit_fn:    sync/async(actor, action, target, decision, detail) -> None
        """
        self.ask_user_fn = ask_user_fn
        self.audit_fn = audit_fn
        self.provider = WEB_PROVIDER
        self.model = WEB_MODEL
        self.browser = None
        self.context = None
        self.page = None

    def denormalize_x(self, x: int, width: int) -> int:
        return int((x / 1000) * width)

    def denormalize_y(self, y: int, height: int) -> int:
        return int((y / 1000) * height)

    async def _audit(self, action, target="", decision="allow", detail=""):
        if not self.audit_fn:
            return
        try:
            r = self.audit_fn("web_agent", action, target, decision, detail)
            if asyncio.iscoroutine(r):
                await r
        except Exception:
            pass

    async def _maybe_confirm_submit(self, fn_name, args) -> bool | None:
        """True = proceed, False = declined, None = not a submit step.

        Submit-style steps are asked out loud through the wired confirm
        channel (voice-direct, no popup). Fail-closed: with no channel, or a
        declined/timed-out answer, the submit does NOT happen. Silence is
        never consent."""
        flag = _SUBMIT_TOOLS.get(fn_name)
        if not flag:
            return None
        val = args.get(flag)
        is_submit = (val is True) if isinstance(val, bool) else (
            isinstance(val, str) and "enter" in val.lower())
        if not is_submit:
            return None
        if not self.ask_user_fn:
            await self._audit(fn_name, str(args)[:200], "deny",
                              "no confirm channel; fail-closed")
            return False
        ok = await self.ask_user_fn(
            f"The browser is about to submit ({fn_name}). Shall I go ahead?")
        await self._audit(fn_name, str(args)[:200], "approved" if ok else "denied", "")
        return bool(ok)

    async def execute_function_calls(self, function_calls):
        """Executes OpenAI-style tool calls against the live page.

        Accepts dicts {id, function:{name, arguments(str|dict)}} — the shape
        providers.chat_message returns — and maps them onto Playwright.
        Returns [(call_id, name, result_dict)]. Unknown functions and invalid
        coordinates are refused as error results, never guessed at."""
        import json as _json

        results = []
        for call in function_calls:
            if isinstance(call, dict):
                call_id = call.get("id")
                fn = call.get("function", {}) or {}
                fn_name, raw_args = fn.get("name", ""), fn.get("arguments", {})
            else:  # duck-typed SDK object, just in case
                call_id = getattr(call, "id", None)
                fn = getattr(call, "function", None) or {}
                fn_name = getattr(fn, "name", "") if not isinstance(fn, dict) else fn.get("name", "")
                raw_args = getattr(fn, "arguments", {}) if not isinstance(fn, dict) else fn.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    args = _json.loads(raw_args or "{}")
                except Exception:
                    results.append((call_id, fn_name, {"error": "refused: arguments were not valid JSON"}))
                    continue
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                args = {}
            print(f"[ACTION] Action: {fn_name} {str(args)[:160]}")

            verdict = await self._maybe_confirm_submit(fn_name, args)
            if verdict is False:
                results.append((call_id, fn_name,
                                {"error": "stopped: you declined that submit step, so nothing was sent."}))
                continue

            result_data = {}
            try:
                # --- NAVIGATION ---
                if fn_name == "navigate":
                    await self.page.goto(args["url"])
                elif fn_name == "go_back":
                    await self.page.go_back()
                elif fn_name == "go_forward":
                    await self.page.go_forward()
                # --- MOUSE CLICKS & TYPING ---
                elif fn_name == "click_at":
                    x = self.denormalize_x(args["x"], SCREEN_WIDTH)
                    y = self.denormalize_y(args["y"], SCREEN_HEIGHT)
                    await self.page.mouse.click(x, y)
                elif fn_name == "type_text_at":
                    x = self.denormalize_x(args["x"], SCREEN_WIDTH)
                    y = self.denormalize_y(args["y"], SCREEN_HEIGHT)
                    text = args["text"]
                    press_enter = args.get("press_enter", False)
                    clear_before = args.get("clear_before_typing", True)
                    await self.page.mouse.click(x, y)
                    if clear_before:
                        await self.page.keyboard.press("Control+A")
                        await self.page.keyboard.press("Backspace")
                    await self.page.keyboard.type(text)
                    if press_enter:
                        await self.page.keyboard.press("Enter")
                # --- MOUSE MOVEMENT / HOVER ---
                elif fn_name == "hover_at":
                    x = self.denormalize_x(args["x"], SCREEN_WIDTH)
                    y = self.denormalize_y(args["y"], SCREEN_HEIGHT)
                    await self.page.mouse.move(x, y)
                elif fn_name == "drag_and_drop":
                    start_x = self.denormalize_x(args["x"], SCREEN_WIDTH)
                    start_y = self.denormalize_y(args["y"], SCREEN_HEIGHT)
                    end_x = self.denormalize_x(args["destination_x"], SCREEN_WIDTH)
                    end_y = self.denormalize_y(args["destination_y"], SCREEN_HEIGHT)
                    await self.page.mouse.move(start_x, start_y)
                    await self.page.mouse.down()
                    await self.page.mouse.move(end_x, end_y)
                    await self.page.mouse.up()
                # --- KEYBOARD ---
                elif fn_name == "key_combination":
                    await self.page.keyboard.press(args.get("keys"))
                # --- SCROLLING ---
                elif fn_name == "scroll_document":
                    magnitude = args.get("magnitude", 800)
                    direction = args.get("direction", "down")
                    dx, dy = 0, 0
                    if direction == "down": dy = magnitude
                    elif direction == "up": dy = -magnitude
                    elif direction == "right": dx = magnitude
                    elif direction == "left": dx = -magnitude
                    await self.page.mouse.wheel(dx, dy)
                elif fn_name == "done":
                    result_data = {"done": True, "summary": args.get("summary", "")}
                else:
                    print(f"[WARN] Warning: Model requested unimplemented function {fn_name}")
                    result_data = {"error": f"refused: unknown function {fn_name}"}
                await asyncio.sleep(1)  # let the UI settle
            except Exception as e:
                print(f"[ERR] Error executing {fn_name}: {e}")
                result_data = {"error": str(e)}

            results.append((call_id, fn_name, result_data))
        return results

    async def search(self, query, max_results=5):
        """
        Fast keyless web search. Uses the DuckDuckGo API (ddgs library), falling
        back to a headless-browser Yandex search if that fails.
        Returns a formatted text summary of the top results.
        """
        print(f"[WEB SEARCH] Query: '{query}'")
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=max_results))
            results = [(r.get("title", ""), r.get("href", ""), r.get("body", "")) for r in raw]
            results = [r for r in results if r[0] and r[1]]
            if results:
                lines = []
                for i, (title, url, snippet) in enumerate(results[:max_results]):
                    lines.append(f"{i+1}. {title}\n   URL: {url}")
                    if snippet:
                        lines.append(f"   {snippet}")
                return "Web search results:\n" + "\n".join(lines)
            print("[WEB SEARCH] DDGS returned no results, falling back to Yandex...")
        except Exception as e:
            print(f"[WEB SEARCH] DDGS failed ({e}), falling back to Yandex...")
        return await self._search_yandex(query, max_results)

    async def _search_yandex(self, query, max_results=5):
        """Fallback search via Yandex in a headless browser."""
        import re as _re

        html = ""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
            )
            try:
                await page.goto(f"https://yandex.com/search/?text={query.replace(' ', '+')}", timeout=15000)
                await page.wait_for_load_state("networkidle")
                html = await page.content()
            except Exception as e:
                print(f"[WEB SEARCH] Error: {e}")
            finally:
                await browser.close()

        if not html:
            return "Search failed. The search engine may be blocking automated requests."

        results = []
        for m in _re.finditer(r'<li class="[^"]*serp-item[^"]*".*?</li>', html, _re.S):
            block = m.group(0)
            title_m = _re.search(r"<h2[^>]*>(.*?)</h2>", block, _re.S)
            title = _re.sub(r"\s+", " ", _re.sub(r"<[^>]+>", "", title_m.group(1))).strip() if title_m else ""
            url_m = _re.search(r'href="(https?://[^"]+)"', block)
            url = url_m.group(1) if url_m else ""
            if url and ("yandex.com" in url or "company.yandex" in url):
                url = ""
            if not title or not url:
                continue
            text = _re.sub(r"\s+", " ", _re.sub(r"<[^>]+>", "", block)).strip()
            snippet = text.replace(title, "", 1).strip()[:250]
            results.append((title, url, snippet))
            if len(results) >= max_results:
                break

        if not results:
            return "No results found for the query."
        lines = []
        for i, (title, url, snippet) in enumerate(results):
            lines.append(f"{i+1}. {title}\n   URL: {url}")
            if snippet:
                lines.append(f"   {snippet}")
        return "Web search results:\n" + "\n".join(lines)

    async def _new_browser(self, p):
        """(Re)launches Chromium and returns (browser, context, page). Split out
        so a dead browser can be replaced mid-run without losing the goal."""
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": SCREEN_WIDTH, "height": SCREEN_HEIGHT},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        return browser, context, page

    async def _ensure_alive(self, p):
        """Replaces a dead browser (idle-reaped/OOM) and restores the last URL.
        Returns True if the page is usable afterwards. This is what makes a
        slow thinking model survivable: the run continues where it left off."""
        try:
            await self.page.title()
            return True
        except Exception:
            pass
        print("[WEB] browser died mid-run — relaunching and restoring page.")
        try:
            await self.browser.close()
        except Exception:
            pass
        try:
            self.browser, self.context, self.page = await self._new_browser(p)
            await self.page.goto(getattr(self, "_last_url", "https://www.google.com"),
                                  timeout=30000)
            return True
        except Exception as e:
            print(f"[WEB] relaunch failed: {type(e).__name__}: {str(e)[:120]}")
            return False

    async def _grab_frame(self, p):
        """Screenshot with auto-recovery. Returns (png_bytes | None)."""
        if not await self._ensure_alive(p):
            return None
        try:
            shot = await self.page.screenshot(type="png")
            self._last_url = self.page.url
            return shot
        except Exception:
            if not await self._ensure_alive(p):
                return None
            try:
                shot = await self.page.screenshot(type="png")
                self._last_url = self.page.url
                return shot
            except Exception as e:
                print(f"[WEB] screenshot failed after relaunch: {e}")
                return None

    def _prune_images(self, messages):
        """Keeps only the LATEST screenshot in the history. Older frames are
        replaced by a placeholder so a 20-turn run cannot blow the context —
        the model navigates from the current frame plus the text trail."""
        pruned, seen_image = [], True
        for msg in reversed(messages):
            content = msg.get("content")
            if isinstance(content, list) and any(
                    isinstance(p, dict) and p.get("type") == "image_url" for p in content):
                if seen_image:
                    seen_image = False
                    pruned.append(msg)
                else:
                    pruned.append({"role": msg.get("role", "user"), "content": [
                        {"type": "text", "text": "(earlier screenshot omitted — see the latest frame)"}]})
            else:
                pruned.append(msg)
        pruned.reverse()
        return pruned

    async def run_task(self, prompt, update_callback=None):
        """
        Runs the agent with the given prompt on Glimmer (OpenAI tool loop).
        update_callback: async function(screenshot_b64: str, logs: str)
        Returns the final response from the agent.
        """
        print(f"[START] WebAgent started ({self.provider}/{self.model}). Goal: {prompt}")
        final_response = "Agent finished without a final summary."

        if not providers.available(self.provider):
            return f"Web agent unavailable: provider '{self.provider}' has no API key configured."

        async with async_playwright() as p:
            self.browser, self.context, self.page = await self._new_browser(p)
            await self.page.goto("https://www.google.com", timeout=30000)
            self._last_url = self.page.url

            shot = await self._grab_frame(p)
            if shot is None:
                return "Web agent stalled: the browser could not start."
            if update_callback:
                await update_callback(base64.b64encode(shot).decode("utf-8"), "Web Agent Initialized")

            messages = [
                {"role": "system", "content": SYSTEM_RULES},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Goal: {prompt}\nCurrent URL: {self.page.url}"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64.b64encode(shot).decode('utf-8')}"}},
                ]},
            ]

            for turn in range(MAX_TURNS):
                print(f"\n--- Turn {turn + 1} ---")
                try:
                    msg = await providers.chat_message(
                        self.provider, self.model, self._prune_images(messages),
                        tools=WEB_TOOLS, temperature=0.2,
                        max_tokens=1024, timeout=120.0)
                except Exception as e:
                    print(f"[CRITICAL] Model call failed: {e}")
                    if update_callback:
                        await update_callback(None, f"Error: {str(e)[:120]}")
                    final_response = f"Web agent stalled: {str(e)[:150]}"
                    break

                text = (msg.get("content") or "").strip()
                tool_calls = msg.get("tool_calls") or []
                if text:
                    print(f"[AGENT] {text[:160]}")
                    final_response = text
                messages.append({"role": "assistant", "content": text or "",
                                 "tool_calls": tool_calls or None})

                if not tool_calls:
                    print("[DONE] No tool calls — task finished.")
                    if update_callback:
                        await update_callback(None, "Task Finished")
                    break

                # done(summary) ends the run with the model's own verdict.
                done_call = next((tc for tc in tool_calls
                                  if (tc.get("function", {}) or {}).get("name") == "done"), None)

                results = await self.execute_function_calls(tool_calls)
                actions_log = ", ".join(r[1] for r in results)
                shot = await self._grab_frame(p)
                if shot is None:
                    final_response = "Web agent stalled: the browser died and could not be restored."
                    break
                if update_callback:
                    await update_callback(base64.b64encode(shot).decode("utf-8"),
                                          f"Executed: {actions_log}")
                tool_msgs = []
                for call_id, name, result in results:
                    summary = result.get("summary") or result.get("error") or "ok"
                    # Plain user message, NOT role=tool: some OpenAI-compatible
                    # endpoints (NIM) mishandle the tool role and lose context.
                    tool_msgs.append({"role": "user", "content":
                                      f"Result of {name}: {str(summary)[:500]}"})
                tool_msgs.append({"role": "user", "content": [
                    {"type": "text", "text":
                     f"Goal (restated every turn so it is never lost): {prompt}\n"
                     f"Current URL: {self.page.url}. Continue the goal or call done(summary)."},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64.b64encode(shot).decode('utf-8')}"}},
                ]})
                messages.extend(tool_msgs)

                if done_call is not None:
                    try:
                        import json as _json

                        fn = done_call.get("function", {}) or {}
                        fn_args = fn.get("arguments", {})
                        summary = (_json.loads(fn_args)
                                   if isinstance(fn_args, str) else fn_args).get("summary", "")
                        if summary:
                            final_response = summary
                    except Exception:
                        pass
                    print("[DONE] Model called done.")
                    break
            else:
                final_response = (f"Reached the {MAX_TURNS}-turn safety limit. "
                                  f"Last state: {final_response[:160]}")

            try:
                await self.browser.close()
            except Exception:
                pass
            print("[CLOSE] Browser closed.")
            await self._audit("run", prompt[:120], "done", final_response[:200])
            return final_response

if __name__ == "__main__":
    agent = WebAgent()
    asyncio.run(agent.run_task("Go to example.com and report the page title."))
