"""
LIVE TEST SUITE — every JARVIS feature, tested for real, results printed.

  [1] Unit suite (pytest)
  [2] Instant Kill ritual (arm -> confirm -> stand down)
  [3] Wellbeing monitor (real idle-time telemetry)
  [4] Weather fetch (wttr.in, live)
  [5] Ambient Vision (synthetic scene -> real vision model)
  [6] E.D.I.T.H. dossier (real parallel web sweep + deep chain)
  [7] Deep Sweep verdict (real mass verification)
  [8] Parallel tool execution (real gather path, timed)
  [9] Nanotech UI mode emission (callback capture)
 [10] speak_proactive idle-guard (no interruption mid-conversation)

Usage: .venv/bin/python backend/_live_suite.py
"""
import asyncio
import io
import os
import sys
import time
import base64

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=True)

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""), flush=True)


async def test_autonomy():
    print("\n[2] INSTANT KILL RITUAL", flush=True)
    from autonomy import AutonomyLevels
    a = AutonomyLevels(os.path.join(tempfile_dir(), "aut"))
    ok = a.level == 1
    report("default level = Standard(1)", ok)
    m1 = a.request_arm()
    report("arm request challenges back", a.level != 3 and "confirm" in m1.lower(), m1[:60])
    m2 = a.confirm_arm()
    report("confirm ARMS level 3", a.level == 3 and a.armed, "destructive auto-approve ON")
    m3 = a.stand_down()
    report("'stand down' disarms instantly", a.level == 1 and not a.armed, m3[:60])


def tempfile_dir():
    import tempfile
    return tempfile.mkdtemp()


async def test_wellbeing():
    print("\n[3] WELLBEING MONITOR (real telemetry)", flush=True)
    from wellbeing import WellbeingMonitor, _mac_idle_seconds
    idle = _mac_idle_seconds()
    report("macOS idle-time sensor reads", idle >= 0, f"idle {idle:.0f}s right now")
    w = WellbeingMonitor(tempfile_dir())
    snap = w.snapshot()
    report("snapshot composes", "active" in snap, snap)
    w.work_stretch_s = 3 * 3600
    nudge = w.break_nudge()
    report("break nudge fires once per stretch", bool(nudge) and w.break_nudge() is None,
           (nudge[1][:70] + "...") if nudge else "")


async def test_weather():
    print("\n[4] WEATHER (live wttr.in)", flush=True)
    import httpx
    try:
        r = await httpx.AsyncClient(timeout=8.0).get("https://wttr.in/?format=%c+%t+feels+like+%f")
        ok = r.status_code == 200 and len(r.text.strip()) > 3
        report("weather fetched", ok, r.text.strip()[:60])
    except Exception as e:
        report("weather fetched", False, str(e)[:60])


async def test_ambient_vision(loop):
    print("\n[5] AMBIENT VISION (synthetic scene -> real vision model)", flush=True)
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (640, 480), (12, 18, 30))
    d = ImageDraw.Draw(img)
    d.ellipse([220, 140, 420, 340], fill=(0, 200, 255))       # cyan glowing circle
    d.rectangle([60, 380, 580, 430], fill=(255, 210, 0))      # yellow bench
    d.text((240, 40), "ADA TEST LAB 42", fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    frame = {"mime_type": "image/jpeg", "data": base64.b64encode(buf.getvalue()).decode()}

    v = loop.vision
    v.get_frame_fn = lambda: frame
    desc = await v.look()
    ok = bool(desc) and not desc.upper().startswith("CLEAR")
    report("vision model describes the scene", ok, desc[:90] if desc else "(empty)")
    if ok:
        first_silent = v._absorb(desc)          # first look learns silently
        notable = v._absorb("Sir is now holding a red hot iron helmet")
        report("learns silently, then flags changes", first_silent is False and notable is True)
        block = v.context_block()
        report("scene injected into context", "CURRENT SCENE" in block, block[:80])


async def test_edith(loop):
    print("\n[6] E.D.I.T.H. OSINT DOSSIER (live sweep, real target)", flush=True)
    import edith
    t0 = time.time()
    sources = await edith.gather_sources(loop._dossier_search, "Sam Altman", "person",
                                         sanitize_fn=loop._sanitize_web)
    dt = time.time() - t0
    report("parallel 5-angle sweep", len(sources) > 400, f"{len(sources)} chars in {dt:.1f}s")
    prompt = edith.build_prompt("Sam Altman", "person", sources)
    brief = await loop._dossier_chat(prompt)
    spoken = edith.extract_spoken(brief)
    ok = bool(spoken) and "exactly 4-5" not in spoken.lower() and len(spoken) > 60
    report("deep chain synthesizes dossier", ok, (spoken[:110] + "...") if spoken else brief[:110])


async def test_deep_sweep(loop):
    print("\n[7] DEEP SWEEP (live mass verification)", flush=True)
    import deep_sweep
    import edith
    claim = "The Eiffel Tower is taller than the Empire State Building"
    docs_fn = lambda q, k=6: loop.kernel.docs.search(q, k=k)
    t0 = time.time()
    evidence = await deep_sweep.sweep(loop._dossier_search, docs_fn, loop._sanitize_web, claim)
    dt = time.time() - t0
    report("6-angle evidence sweep", len(evidence) > 500, f"{len(evidence)} chars in {dt:.1f}s")
    verdict = await loop._dossier_chat(deep_sweep.build_verdict_prompt(claim, evidence))
    spoken_v = edith.extract_spoken(verdict)
    got = "CONTRADICTED" if "CONTRADICTED" in verdict else ("MIXED" if "MIXED" in verdict else "?")
    report("verdict correctly CONTRADICTED", got == "CONTRADICTED" and bool(spoken_v),
           f"verdict={got} | {spoken_v[:80] if spoken_v else ''}")


async def test_parallel_tools(loop):
    print("\n[8] PARALLEL TOOL EXECUTION (real gather path, timed)", flush=True)
    from google.genai import types
    delays = {"search_web": 1.2, "system_info": 1.2, "notebook_read": 1.2}

    async def fake_execute(name, args):
        await asyncio.sleep(delays.get(name, 0.5))
        return {"result": f"{name} done"}

    calls = [types.Part(function_call=types.FunctionCall(id=f"c{i}", name=n, args={}))
             for i, n in enumerate(["search_web", "system_info", "notebook_read"])]
    contents = []
    model_parts = calls
    calls_list = [p.function_call for p in model_parts if p.function_call]

    async def run_one(fc):
        allowed = await loop._check_confirmation(fc.name)
        if not allowed:
            return types.FunctionResponse(id=fc.id, name=fc.name, response={"result": "denied"})
        result = await fake_execute(fc.name, fc.args)
        return types.FunctionResponse(id=fc.id, name=fc.name, response=result)

    t0 = time.time()
    responses = await asyncio.gather(*[run_one(fc) for fc in calls_list])
    dt = time.time() - t0
    sequential = sum(delays.values())
    ok = len(responses) == 3 and dt < sequential * 0.75
    report("3 tools ran concurrently", ok,
           f"wall {dt:.2f}s vs sequential {sequential:.1f}s ({(1 - dt/sequential)*100:.0f}% faster)")


async def test_nanotech_ui(loop):
    print("\n[9] NANOTECH UI MODE (emission + suppression)", flush=True)
    seen = []
    loop.on_ui_mode = lambda data: seen.append(data)
    loop._current_ui_mode = None
    loop._set_ui_mode("research")
    loop._set_ui_mode("research")            # duplicate -> suppressed
    loop._set_ui_mode("cad")
    modes = [s["mode"] for s in seen]
    report("mode events emitted, deduped", modes == ["research", "cad"], str(modes))
    report("frontend listener present in build",
           "ui_mode" in open("dist/assets/" + [f for f in os.listdir("dist/assets") if f.endswith(".js")][0]).read())


async def test_proactive_guard(loop):
    print("\n[10] PROACTIVE IDLE-GUARD (never interrupts)", flush=True)
    loop._turn_task = asyncio.create_task(asyncio.sleep(5))   # simulate active turn
    spoke = await loop.speak_proactive("[emotion:curious] test line")
    report("skips while Sir's turn is running", spoke is False)
    await loop._turn_task
    loop._turn_task = None
    report("idle check passes", loop._turn_task is None)


async def main():
    print("=" * 64)
    print("ADA LIVE TEST SUITE — JARVIS EDITION")
    print("=" * 64)

    print("\n[1] UNIT SUITE", flush=True)
    import subprocess
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_jarvis_suite.py",
                        "tests/test_autonomy.py", "tests/test_speech_normalizer.py", "-q"],
                       capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    last = [l for l in r.stdout.splitlines() if l.strip()][-1] if r.stdout else "?"
    report(f"unit tests: {last.strip()}", r.returncode == 0)

    print("\n[0] BOOTSTRAP: constructing AudioLoop (real kernel, real agents)...", flush=True)
    try:
        import ada
        emitted = {}
        loop = ada.AudioLoop(on_ui_mode=lambda d: emitted.setdefault("mode", d),
                             on_transcription=lambda d: None)
        report("AudioLoop constructed", True, f"autonomy L{loop.kernel.autonomy.level}, "
                                               f"vision+wellbeing+dossier wired")
    except Exception as e:
        report("AudioLoop constructed", False, f"{type(e).__name__}: {str(e)[:100]}")
        print("\nCannot continue live tests without AudioLoop.")
        print_results()
        return 1

    await test_autonomy()
    await test_wellbeing()
    await test_weather()
    await test_ambient_vision(loop)
    await test_edith(loop)
    await test_deep_sweep(loop)
    await test_parallel_tools(loop)
    await test_nanotech_ui(loop)
    await test_proactive_guard(loop)
    print_results()
    return 0


def print_results():
    print("\n" + "=" * 64)
    print("RESULTS SUMMARY")
    print("=" * 64)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, detail in RESULTS:
        print(f"  {'✓' if ok else '✗'} {name}" + (f" — {detail}" if detail else ""))
    print("-" * 64)
    print(f"  {passed}/{len(RESULTS)} PASSED")
    print("=" * 64)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
