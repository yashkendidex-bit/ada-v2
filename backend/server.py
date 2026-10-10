import sys
import asyncio

# Fix for asyncio subprocess support on Windows
# MUST BE SET BEFORE OTHER IMPORTS
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import socketio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse
import asyncio
import threading
import sys
import os
import json
from datetime import datetime
from pathlib import Path



# Ensure we can import ada
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import ada
try:
    # MediaPipe is ~400MB+ of resident RAM and is only needed when face auth
    # is enabled — import failure (or absence) must never kill the backend.
    from authenticator import FaceAuthenticator
except Exception as e:
    print(f"[SERVER] Face authenticator unavailable ({e}); face auth disabled.")
    FaceAuthenticator = None

# Create a Socket.IO server
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')


async def _emit_speech_audio(data):
    """Sends a synthesized TTS chunk to the frontend for realtime playback."""
    try:
        await sio.emit('tts_audio', data)
    except Exception as e:
        print(f"[TTS] Emit failed: {str(e)[:100]}")

async def _emit_ui_mode(data):
    """Nanotech suit adaptation: tells the frontend which mode Ada is in."""
    try:
        await sio.emit('ui_mode', data)
    except Exception as e:
        print(f"[UI] Emit failed: {str(e)[:100]}")

async def _emit_autonomy(data):
    """Autonomy level changes for the HUD badge."""
    try:
        await sio.emit('autonomy_level', data)
    except Exception as e:
        print(f"[UI] Autonomy emit failed: {str(e)[:100]}")
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://127.0.0.1:5174", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app_socketio = socketio.ASGIApp(sio, app)

import providers


@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    """Transcribes an uploaded audio clip with Groq Whisper (large-v3-turbo)."""
    try:
        data = await file.read()
        if not data:
            return JSONResponse(status_code=400, content={"error": "empty audio"})
        text = await providers.transcribe_audio(data, filename=file.filename or "audio.webm")
        return {"text": text}
    except Exception as e:
        print(f"[STT] Transcription failed: {type(e).__name__}: {str(e)[:150]}")
        return JSONResponse(status_code=500, content={"error": str(e)[:200]})

import signal

# --- SHUTDOWN HANDLER ---
def signal_handler(sig, frame):
    print(f"\n[SERVER] Caught signal {sig}. Exiting gracefully...")
    # Clean up audio loop
    if audio_loop:
        try:
            print("[SERVER] Stopping Audio Loop...")
            audio_loop.stop() 
        except:
            pass
    # Force kill
    print("[SERVER] Force exiting...")
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Global state
audio_loop = None
loop_task = None
authenticator = None
SETTINGS_FILE = "settings.json"

def _declared_tool_names():
    """Every tool ADA can actually call, read from her live declarations.

    The permission map below used to be a hand-written list of the eleven original
    tools. Everything added since — desktop_autopilot, the part-surgery tools, and
    all 35 kernel tools — was missing from it, so those tools could not be switched
    off in Settings: they fell through to `check_permission`'s permissive default.
    Reading the declarations keeps the two in sync automatically. If `ada` cannot be
    imported (no API key, missing dep) this returns [] and the permissive default
    still applies, so the failure mode is unchanged rather than newly locked-out.
    """
    names = set()
    try:
        for group in getattr(ada, "TOOLS", []) or []:
            for decl in group.get("function_declarations", []) or []:
                n = decl.get("name")
                if n:
                    names.add(n)
    except Exception as e:
        print(f"[SETTINGS] could not read tool declarations: {e}")
    return sorted(names)


DEFAULT_SETTINGS = {
    "face_auth_enabled": False, # Default OFF as requested
    "tool_permissions": {name: True for name in _declared_tool_names()},
    "camera_flipped": False # Invert cursor horizontal direction
}

# Deep-ish copy. A plain .copy() left SETTINGS["tool_permissions"] aliased to the
# nested dict inside DEFAULT_SETTINGS, and load_settings() then called .update() on
# it — so reading settings.json permanently mutated the in-memory defaults.
SETTINGS = {**DEFAULT_SETTINGS,
            "tool_permissions": dict(DEFAULT_SETTINGS["tool_permissions"])}

def load_settings():
    global SETTINGS
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                loaded = json.load(f)
                # Merge with defaults to ensure new keys exist
                # Deep merge for tool_permissions would be better but shallow merge of top keys + tool_permissions check is okay for now
                for k, v in loaded.items():
                    if k == "tool_permissions" and isinstance(v, dict):
                         SETTINGS["tool_permissions"].update(v)
                    else:
                        SETTINGS[k] = v
            print(f"Loaded settings: {SETTINGS}")
        except Exception as e:
            print(f"Error loading settings: {e}")

def save_settings():
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(SETTINGS, f, indent=4)
        print("Settings saved.")
    except Exception as e:
        print(f"Error saving settings: {e}")

# Load on startup
load_settings()

authenticator = None
# tool_permissions is now SETTINGS["tool_permissions"]

@app.on_event("startup")
async def startup_event():
    import sys
    print(f"[SERVER DEBUG] Startup Event Triggered")
    print(f"[SERVER DEBUG] Python Version: {sys.version}")
    try:
        loop = asyncio.get_running_loop()
        print(f"[SERVER DEBUG] Running Loop: {type(loop)}")
        policy = asyncio.get_event_loop_policy()
        print(f"[SERVER DEBUG] Current Policy: {type(policy)}")
    except Exception as e:
        print(f"[SERVER DEBUG] Error checking loop: {e}")

@app.get("/status")
async def status():
    out = {"status": "running", "service": "A.D.A Backend"}
    try:
        from tts_agent import tts_health
        out["tts"] = tts_health()
    except Exception:
        pass
    try:
        if audio_loop is not None and getattr(audio_loop, "kernel", None) is not None:
            out["autonomy"] = audio_loop.kernel.autonomy.status()
    except Exception:
        pass
    return out

@app.post("/chat")
async def chat(data: dict):
    """Thin-client endpoint (phone app, scripts, anything HTTP).

    {"text": "hey ada"} -> {"reply": "Yes, Sir?", "route": "qwen"}.
    Same brain, same memory, no audio — shares the turn lock with voice so
    the phone and the desktop never talk over her.
    """
    text = (data or {}).get("text", "")
    if not text or not str(text).strip():
        return JSONResponse(status_code=400, content={"error": "no text provided"})
    if not audio_loop:
        return JSONResponse(status_code=503, content={"error": "ADA not ready"})
    import time as _t
    t0 = _t.time()
    try:
        reply = await audio_loop.ask_text(str(text))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)[:200]})
    return {"reply": reply, "route": getattr(audio_loop, "_last_route", "brain"),
            "dur_ms": int((_t.time() - t0) * 1000)}

@app.get("/demo/briefing")
async def demo_briefing():
    if not audio_loop or not audio_loop.kernel:
        return JSONResponse(status_code=503, content={"error": "ADA not ready"})
    res = await audio_loop.kernel.handle_tool("daily_briefing", {}, permissions={"daily_briefing": True})
    # Also have ADA say it naturally — fire and forget so HTTP returns instantly
    try:
        asyncio.create_task(audio_loop.process_message("Good morning ADA, give me my daily briefing Sir", echo=False))
    except Exception as e:
        print(f"[DEMO] briefing speech failed: {e}")
    return res

@app.post("/simulate")
async def simulate(data: dict):
    """Real FEA: takes {stl_b64 or stl_path, material, force_n} and returns heatmap + safety."""
    import base64, tempfile, os, pathlib
    try:
        from cad_v3.fea_solver import compute_von_mises_stress_approx
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"FEA not available: {e}"})
    stl_b64 = data.get("stl_b64")
    stl_path = data.get("stl_path")
    material = data.get("material", "steel")
    force_n = float(data.get("force_n", 500))
    tmp_path = None
    try:
        if stl_b64:
            import base64 as b64
            raw = b64.b64decode(stl_b64)
            fd, tmp_path = tempfile.mkstemp(suffix=".stl")
            os.write(fd, raw); os.close(fd)
            stl_path = tmp_path
        if not stl_path or not os.path.exists(stl_path):
            return JSONResponse(status_code=400, content={"error": "no STL provided"})
        res = compute_von_mises_stress_approx(stl_path=stl_path, force_n=force_n, material=material)
        # If we created tmp, include heatmap as base64 of colored STL if exists
        heatmap_b64 = None
        if res.get("colored_mesh_path") and os.path.exists(res["colored_mesh_path"]):
            try:
                import base64 as b64
                heatmap_b64 = b64.b64encode(open(res["colored_mesh_path"], "rb").read()).decode()
            except: pass
        res["heatmap_b64"] = heatmap_b64
        return res
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)[:300]})
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass

@app.post("/overnight")
async def overnight(data: dict):
    """JARVIS overnight evolution: mutate part, FEA-test variants, propose best."""
    import os
    try:
        from cad_overnight import evolve_overnight
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"overnight unavailable: {e}"})
    stl_path = data.get("stl_path")
    material = data.get("material", "steel")
    force_n = float(data.get("force_n", 500))
    if not stl_path or not os.path.exists(stl_path):
        return JSONResponse(status_code=400, content={"error": "no STL path provided"})
    try:
        res = evolve_overnight(stl_path, material=material, force_n=force_n)
        # Have ADA announce it naturally (fire and forget)
        try:
            msg = f"System notification: Overnight evolution finished. {res.get('message','')}"
            if audio_loop:
                asyncio.create_task(audio_loop.process_message(msg, echo=False))
        except Exception:
            pass
        return res
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)[:300]})

@app.post("/twin")
async def twin(data: dict):
    """Digital Twin hook: sensor CSV -> live stress overlay on the model."""
    import os, csv, io
    stl_path = data.get("stl_path")
    sensor_csv = data.get("sensor_csv")  # raw CSV text: timestamp,value OR value per line
    material = data.get("material", "steel")
    if not stl_path or not os.path.exists(stl_path):
        return JSONResponse(status_code=400, content={"error": "no STL path"})
    if not sensor_csv:
        return JSONResponse(status_code=400, content={"error": "no sensor_csv"})
    try:
        values = []
        for line in sensor_csv.strip().splitlines():
            parts_ = line.split(",")
            try:
                values.append(float(parts_[-1]))
            except Exception:
                continue
        if not values:
            return JSONResponse(status_code=400, content={"error": "no numeric sensor values"})
        avg_load = sum(values) / len(values)
        peak_load = max(values)
        from cad_v3.fea_solver import compute_von_mises_stress_approx
        res_avg = compute_von_mises_stress_approx(stl_path, force_n=avg_load, material=material)
        res_peak = compute_von_mises_stress_approx(stl_path, force_n=peak_load, material=material)
        warning = None
        if res_peak.get("safety_factor", 99) < 1.5:
            warning = f"PEAK load {peak_load}N gives safety factor {res_peak['safety_factor']} — inspect part soon."
        elif res_avg.get("safety_factor", 99) < 2.0:
            warning = f"Average load is close to limit (SF {res_avg['safety_factor']}). Consider redesign."
        return {
            "ok": True,
            "samples": len(values),
            "avg_load_n": round(avg_load, 1),
            "peak_load_n": round(peak_load, 1),
            "avg_safety_factor": res_avg.get("safety_factor"),
            "peak_safety_factor": res_peak.get("safety_factor"),
            "status": res_peak.get("status"),
            "warning": warning,
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)[:300]})

@app.get("/demo/gear")
async def demo_gear():
    import base64, pathlib, glob
    exp_root = pathlib.Path(__file__).parent.parent / "projects" / "cad_exports"
    if not exp_root.exists():
        return JSONResponse(status_code=404, content={"error": "no exports"})
    cands = sorted([p for p in exp_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        return JSONResponse(status_code=404, content={"error": "no cad exports"})
    # Prefer the brass gear assembly if exists
    target = None
    for c in cands:
        if "brass" in c.name.lower() and "gear" in c.name.lower():
            target = c
            break
    target = target or cands[0]
    stls = sorted(target.glob("output*.stl"))
    if not stls:
        return JSONResponse(status_code=404, content={"error": "no STLs"})
    parts = []
    for i, p in enumerate(stls):
        parts.append({"name": f"part_{i+1}", "data": base64.b64encode(p.read_bytes()).decode()})
    payload = {"format": "stl", "data": parts[0]["data"], "parts": parts, "material": "brass", "name": "brass gear and shaft", "exports": {"dir": str(target)}}
    await sio.emit('cad_data', payload)
    await sio.emit('cad_status', {'status': 'generated', 'attempt': 1, 'max_attempts': 3})
    return {"status": "emitted", "parts": len(parts), "dir": str(target)}

@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")
    await sio.emit('status', {'msg': 'Connected to A.D.A Backend'}, room=sid)

    global authenticator
    
    # Callback for Auth Status
    async def on_auth_status(is_auth):
        print(f"[SERVER] Auth status change: {is_auth}")
        await sio.emit('auth_status', {'authenticated': is_auth})

    # Callback for Auth Camera Frames
    async def on_auth_frame(frame_b64):
        await sio.emit('auth_frame', {'image': frame_b64})

    # Initialize Authenticator LAZILY and only when face auth is enabled:
    # constructing it loads MediaPipe (~400MB+) plus the landmarker model, so
    # building it on every connect — even with auth disabled — was the single
    # biggest avoidable memory cost in the backend.
    if authenticator is None and FaceAuthenticator is not None \
            and SETTINGS.get("face_auth_enabled", False):
        authenticator = FaceAuthenticator(
            reference_image_path="reference.jpg",
            on_status_change=on_auth_status,
            on_frame=on_auth_frame
        )

    # Check if already authenticated or needs to start
    if authenticator is not None and authenticator.authenticated:
        await sio.emit('auth_status', {'authenticated': True})
    else:
        # Check Settings for Auth
        if SETTINGS.get("face_auth_enabled", False):
            if authenticator is None:
                await sio.emit('error', {'msg': 'Face auth enabled but unavailable (MediaPipe missing).'})
                await sio.emit('auth_status', {'authenticated': True})
            else:
                await sio.emit('auth_status', {'authenticated': False})
                # Start the auth loop in background
                asyncio.create_task(authenticator.start_authentication_loop())
        else:
            # Bypass Auth
            print("Face Auth Disabled. Auto-authenticating.")
            # We don't change authenticator state to true to avoid confusion if re-enabled? 
            # Or we should just tell client it's auth'd.
            await sio.emit('auth_status', {'authenticated': True})

    # Bulletproof: if the AudioLoop isn't running yet (frontend's start_audio emit
    # can get lost across reconnects), start it now so ADA is always alive.
    if audio_loop is None:
        print("[SERVER DEBUG] Auto-starting AudioLoop (no start_audio received).")
        asyncio.create_task(start_audio(sid, None))

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")

@sio.event
async def start_audio(sid, data=None):
    global audio_loop, loop_task
    
    # Optional: Block if not authenticated
    # Only block if auth is ENABLED and not authenticated
    if SETTINGS.get("face_auth_enabled", False):
        if authenticator and not authenticator.authenticated:
            print("Blocked start_audio: Not authenticated.")
            await sio.emit('error', {'msg': 'Authentication Required'})
            return

    print("Starting Audio Loop...")
    
    device_index = None
    device_name = None
    if data:
        if 'device_index' in data:
            device_index = data['device_index']
        if 'device_name' in data:
            device_name = data['device_name']
            
    print(f"Using input device: Name='{device_name}', Index={device_index}")
    
    if audio_loop:
        if loop_task and (loop_task.done() or loop_task.cancelled()):
             print("Audio loop task appeared finished/cancelled. Clearing and restarting...")
             audio_loop = None
             loop_task = None
        else:
             print("Audio loop already running. Re-connecting client to session.")
             await sio.emit('status', {'msg': 'A.D.A Already Running'})
             return


    # Callback to send audio data to frontend
    def on_audio_data(data_bytes):
        # We need to schedule this on the event loop
        # This is high frequency, so we might want to downsample or batch if it's too much
        asyncio.create_task(sio.emit('audio_data', {'data': list(data_bytes)}))

    # Callback to send CAL data to frontend
    def on_cad_data(data):
        info = f"{len(data.get('vertices', []))} vertices" if 'vertices' in data else f"{len(data.get('data', ''))} bytes (STL)"
        print(f"Sending CAD data to frontend: {info}")
        asyncio.create_task(sio.emit('cad_data', data))

    # Callback to send Browser data to frontend
    def on_web_data(data):
        print(f"Sending Browser data to frontend: {len(data.get('log', ''))} chars logs")
        asyncio.create_task(sio.emit('browser_frame', data))

    # Callback for generative HUD events (hud:start/chunk/data/final)
    def on_hud(data):
        try:
            print(f"Sending HUD event: {data.get('t')} {data.get('hudId', '')}")
        except Exception:
            pass
        asyncio.create_task(sio.emit('hud', data))

    @sio.event
    async def hud_action(sid, data):
        """HUD button press → allowlisted backend action. Nothing else runs."""
        from hud_spec import ACTIONS
        action = (data or {}).get("action", "")
        if action not in ACTIONS:
            print(f"[HUD] blocked action: {action}")
            return
        print(f"[HUD] action: {action}")
        if audio_loop is None:
            return
        try:
            if action == "dismiss_hud":
                return  # frontend already removed it; nothing to do
            if action == "pause_print":
                prt = getattr(audio_loop.kernel, "printer", None)
                res = prt.pause() if prt else "Printer not wired."
                await audio_loop._notify_model(f"HUD action. Speak briefly: {res}")
            elif action == "print_progress":
                prt = getattr(audio_loop.kernel, "printer", None)
                res = prt.progress() if prt else "Printer not wired."
                await audio_loop._notify_model(f"HUD action. Speak briefly: {res}")
            elif action == "run_preflight":
                res = await audio_loop._execute_tool(
                    "preflight", {"stl_path": audio_loop._latest_stl() or ""})
                await audio_loop._notify_model(
                    f"HUD action. Speak briefly: {(res or {}).get('result', '')[:300]}")
            elif action == "show_scorecard":
                res = await audio_loop._execute_tool("scorecard", {})
                await audio_loop._notify_model(
                    f"HUD action. Speak briefly: {(res or {}).get('result', '')[:300]}")
            elif action == "show_ledger":
                res = await audio_loop._execute_tool("cost_ledger", {})
                await audio_loop._notify_model(
                    f"HUD action. Speak briefly: {(res or {}).get('result', '')[:300]}")
            elif action == "approve_spec":
                await audio_loop._notify_model(
                    "HUD action. Speak briefly: intent approved, Sir — say 'print it' when ready for plastic.")
            elif action == "refresh_data":
                await audio_loop._notify_model(
                    "HUD action. Speak briefly: refreshed, Sir.")
        except Exception as e:
            print(f"[HUD] action failed: {type(e).__name__}")
        
    # Callback to send Transcription data to frontend
    def on_transcription(data):
        # data = {"sender": "User"|"ADA", "text": "..."}
        asyncio.create_task(sio.emit('transcription', data))

    # Callback to send Confirmation Request to frontend
    def on_tool_confirmation(data):
        # data = {"id": "uuid", "tool": "tool_name", "args": {...}}
        print(f"Requesting confirmation for tool: {data.get('tool')}")
        asyncio.create_task(sio.emit('tool_confirmation_request', data))

    # Callback for a Level 3 clarifying question. Not a gate — the card carries the
    # question and its options, and the answer comes back on `answer_question` (or by
    # voice, which never reaches the UI at all).
    def on_ada_question(data):
        print(f"Asking Sir: {data.get('question', '')[:100]}")
        asyncio.create_task(sio.emit('ada_question', data))

    # Callback to send CAD status to frontend
    def on_cad_status(status):
        # status can be: 
        # - a string like "generating" (from ada.py handle_cad_request)
        # - a dict with {status, attempt, max_attempts, error} (from CadAgent)
        if isinstance(status, dict):
            print(f"Sending CAD Status: {status.get('status')} (attempt {status.get('attempt')}/{status.get('max_attempts')})")
            asyncio.create_task(sio.emit('cad_status', status))
        else:
            # Legacy: simple string
            print(f"Sending CAD Status: {status}")
            asyncio.create_task(sio.emit('cad_status', {'status': status}))

    # Callback to send CAD thoughts to frontend (streaming)
    def on_cad_thought(thought_text):
        asyncio.create_task(sio.emit('cad_thought', {'text': thought_text}))

    # Callback to send Project Update to frontend
    def on_project_update(project_name):
        print(f"Sending Project Update: {project_name}")
        asyncio.create_task(sio.emit('project_update', {'project': project_name}))

    # Callback to send Error to frontend
    def on_error(msg):
        print(f"Sending Error to frontend: {msg}")
        asyncio.create_task(sio.emit('error', {'msg': msg}))

    # Callback for live model surgery edits (delete/hide/isolate/recolor)
    def on_model_edit(event):
        print(f"Sending model edit: {event.get('action')} -> {event.get('summary','')[:60]}")
        asyncio.create_task(sio.emit('model_edit', event))

    @sio.event
    async def part_pick(sid, data):
        """User pointed at (clicked/pinched) a part in the 3D view -> becomes 'this'."""
        if not audio_loop:
            return
        name = data.get('name', '')
        try:
            from model_surgery import registry
            res = registry.set_selection(name)
            if res:
                await sio.emit('part_selected', res)  # broadcast glow to viewer(s)
                print(f"[SERVER] part_pick resolved: {res['selected']}")
        except Exception as e:
            print(f"[SERVER] part_pick failed: {e}")

    # Initialize ADA
    try:
        print(f"Initializing AudioLoop with device_index={device_index}")
        audio_loop = ada.AudioLoop(
            video_mode="none", 
            on_audio_data=on_audio_data,
            on_cad_data=on_cad_data,
            on_web_data=on_web_data,
            on_hud_data=on_hud,
            on_transcription=on_transcription,
            on_tool_confirmation=on_tool_confirmation,
            on_ada_question=on_ada_question,
            on_cad_status=on_cad_status,
            on_cad_thought=on_cad_thought,
            on_project_update=on_project_update,
            on_error=on_error,
            on_speech_audio=_emit_speech_audio,
            on_ui_mode=_emit_ui_mode,
            on_autonomy=_emit_autonomy,
            on_model_edit=on_model_edit,

            input_device_index=device_index,
            input_device_name=device_name,
        )
        print("AudioLoop initialized successfully.")

        # Apply current permissions
        audio_loop.update_permissions(SETTINGS["tool_permissions"])

        # Sync autonomy badge state to the HUD
        try:
            await sio.emit('autonomy_level', {'level': audio_loop.kernel.autonomy.level,
                                              'name': {0: 'Training Wheels', 1: 'Standard',
                                                       2: 'Trusted', 3: 'INSTANT KILL'}[
                                                           audio_loop.kernel.autonomy.level]})
        except Exception:
            pass
        
        # NOTE: the mic toggle no longer pauses the brain - voice input is a
        # frontend-only concern, so ADA always replies (typed or spoken).

        print("Creating asyncio task for AudioLoop.run()")
        loop_task = asyncio.create_task(audio_loop.run())
        
        # Add a done callback to catch silent failures in the loop
        def handle_loop_exit(task):
            try:
                task.result()
            except asyncio.CancelledError:
                print("Audio Loop Cancelled")
            except Exception as e:
                print(f"Audio Loop Crashed: {e}")
                # You could emit 'error' here if you have context
        
        loop_task.add_done_callback(handle_loop_exit)
        
        print("Emitting 'A.D.A Started'")
        await sio.emit('status', {'msg': 'A.D.A Started'})
        
    except Exception as e:
        print(f"CRITICAL ERROR STARTING ADA: {e}")
        import traceback
        traceback.print_exc()
        await sio.emit('error', {'msg': f"Failed to start: {str(e)}"})
        audio_loop = None # Ensure we can try again


@sio.event
async def stop_audio(sid):
    global audio_loop
    if audio_loop:
        audio_loop.stop() 
        print("Stopping Audio Loop")
        audio_loop = None
        await sio.emit('status', {'msg': 'A.D.A Stopped'})

@sio.event
async def pause_audio(sid):
    global audio_loop
    if audio_loop:
        audio_loop.set_paused(True)
        print("Pausing Audio")
        await sio.emit('status', {'msg': 'Audio Paused'})

@sio.event
async def resume_audio(sid):
    global audio_loop
    if audio_loop:
        audio_loop.set_paused(False)
        print("Resuming Audio")
        await sio.emit('status', {'msg': 'Audio Resumed'})

@sio.event
async def confirm_tool(sid, data):
    # data: { "id": "...", "confirmed": True/False }
    request_id = data.get('id')
    confirmed = data.get('confirmed', False)
    
    print(f"[SERVER DEBUG] Received confirmation response for {request_id}: {confirmed}")
    
    if audio_loop:
        audio_loop.resolve_tool_confirmation(request_id, confirmed)
    else:
        print("Audio loop not active, cannot resolve confirmation.")

@sio.event
async def answer_question(sid, data):
    """Level 3 clarifier: the operator answered Ada's question on screen."""
    request_id = data.get('id')
    answer = data.get('answer', '')
    print(f"[SERVER DEBUG] Clarifying answer for {request_id}: {str(answer)[:120]}")
    if audio_loop:
        ok = audio_loop.resolve_question(request_id, answer)
        if not ok:
            print(f"[SERVER DEBUG] No open question matched {request_id}.")
    else:
        print("Audio loop not active, cannot resolve question.")

@sio.event
async def shutdown(sid, data=None):
    """Gracefully shutdown the server when the application closes."""
    global audio_loop, loop_task, authenticator
    
    print("[SERVER] ========================================")
    print("[SERVER] SHUTDOWN SIGNAL RECEIVED FROM FRONTEND")
    print("[SERVER] ========================================")
    
    # Stop audio loop
    if audio_loop:
        print("[SERVER] Stopping Audio Loop...")
        audio_loop.stop()
        audio_loop = None
    
    # Cancel the loop task if running
    if loop_task and not loop_task.done():
        print("[SERVER] Cancelling loop task...")
        loop_task.cancel()
        loop_task = None
    
    # Stop authenticator if running
    if authenticator:
        print("[SERVER] Stopping Authenticator...")
        authenticator.stop()
    
    print("[SERVER] Graceful shutdown complete. Terminating process...")
    
    # Force exit immediately - os._exit bypasses cleanup but ensures termination
    os._exit(0)

@sio.event
async def tts_played(sid, data):
    """Playback anchoring (PACE-lite): frontend reports per-utterance played
    fractions on barge-in — {utt: idx, frac: 0.0-1.0}. Preferred over the
    backend time-estimate when truncating interrupted history."""
    try:
        if audio_loop is not None and getattr(audio_loop, "speech", None) is not None:
            audio_loop.speech.note_played((data or {}).get("utt"),
                                          (data or {}).get("frac", 0))
    except Exception as e:
        print(f"[SERVER] tts_played ignored: {type(e).__name__}")

@sio.event
async def user_input(sid, data):
    text = data.get('text')
    print(f"[SERVER DEBUG] User input received: '{text}'")
    
    if not audio_loop:
        # Bulletproof: never drop a message because the loop isn't up yet.
        print("[SERVER DEBUG] Audio loop missing - auto-starting.")
        await start_audio(sid, None)

    if text and audio_loop:
        print(f"[SERVER DEBUG] Sending message to model: '{text}'")
        # echo=False: the frontend already displays the user's own message locally
        await audio_loop.process_message(text, echo=False)
        print(f"[SERVER DEBUG] Message sent to model successfully.")

import json
from datetime import datetime
from pathlib import Path

# ... (imports)

@sio.event
async def video_frame(sid, data):
    # data should contain 'image' which is binary (blob) or base64 encoded
    image_data = data.get('image')
    if image_data and audio_loop:
        # We don't await this because we don't want to block the socket handler
        # But send_frame is async, so we create a task
        asyncio.create_task(audio_loop.send_frame(image_data))

@sio.event
async def save_memory(sid, data):
    try:
        messages = data.get('messages', [])
        if not messages:
            print("No messages to save.")
            return

        # Ensure directory exists
        memory_dir = Path("long_term_memory")
        memory_dir.mkdir(exist_ok=True)

        # Generate filename
        # Use provided filename if available, else timestamp
        provided_name = data.get('filename')
        
        if provided_name:
            # Simple sanitization
            if not provided_name.endswith('.txt'):
                provided_name += '.txt'
            # Prevent directory traversal
            filename = memory_dir / Path(provided_name).name 
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = memory_dir / f"memory_{timestamp}.txt"

        # Write to file. The loop body used to only bind `sender`/`text` and never
        # write them, so every "save memory" produced a 0-byte file.
        written = 0
        with open(filename, 'w', encoding='utf-8') as f:
            for msg in messages:
                sender = msg.get('sender', 'Unknown')
                text = msg.get('text', '')
                if not text:
                    continue
                f.write(f"[{sender}] {text}\n")
                written += 1
        if written == 0:
            # Nothing had a `text` field — do not leave an empty file behind.
            try:
                filename.unlink()
            except OSError:
                pass
            await sio.emit('error', {'msg': 'Nothing to save: no message text found.'})
            return
        print(f"Conversation saved to {filename} ({written} messages)")
        await sio.emit('status', {'msg': 'Memory Saved Successfully'})

    except Exception as e:
        print(f"Error saving memory: {e}")
        await sio.emit('error', {'msg': f"Failed to save memory: {str(e)}"})

@sio.event
async def upload_memory(sid, data):
    print(f"Received memory upload request")
    try:
        memory_text = data.get('memory', '')
        if not memory_text:
            print("No memory data provided.")
            return

        if not audio_loop:
             print("[SERVER DEBUG] [Error] Audio loop is None. Cannot load memory.")
             await sio.emit('error', {'msg': "System not ready (Audio Loop inactive)"})
             return

        # Send to model
        print("Sending memory context to model...")
        context_msg = f"System Notification: The user has uploaded a long-term memory file. Please load the following context into your understanding. The format is a text log of previous conversations:\n\n{memory_text}"
        
        await audio_loop.process_message(context_msg)
        print("Memory context sent successfully.")
        await sio.emit('status', {'msg': 'Memory Loaded into Context'})

    except Exception as e:
        print(f"Error uploading memory: {e}")
        await sio.emit('error', {'msg': f"Failed to upload memory: {str(e)}"})

@sio.event
async def iterate_cad(sid, data):
    # data: { prompt: "make it bigger" }
    prompt = data.get('prompt')
    print(f"Received iterate_cad request: '{prompt}'")
    
    if not audio_loop or not audio_loop.cad_agent:
        await sio.emit('error', {'msg': "CAD Agent not available"})
        return

    try:
        # Notify user work has started
        await sio.emit('status', {'msg': 'Iterating design...'})
        await sio.emit('cad_status', {'status': 'generating'})
        
        # Call the agent with project path
        cad_output_dir = str(audio_loop.project_manager.get_current_project_path() / "cad")
        result = await audio_loop.cad_agent.iterate_prototype(prompt, output_dir=cad_output_dir)
        
        if result:
            info = f"{len(result.get('data', ''))} bytes (STL)"
            print(f"Sending updated CAD data: {info}")
            await sio.emit('cad_data', result)
            # Save to Project
            if 'file_path' in result:
                saved_path = audio_loop.project_manager.save_cad_artifact(result['file_path'], prompt)
                if saved_path:
                    print(f"[SERVER] Saved iterated CAD to {saved_path}")

            await sio.emit('status', {'msg': 'Design updated'})
        else:
            await sio.emit('error', {'msg': 'Failed to update design'})
            
    except Exception as e:
        print(f"Error iterating CAD: {e}")
        await sio.emit('error', {'msg': f"Iteration Error: {str(e)}"})

@sio.event
async def generate_cad(sid, data):
    # data: { prompt: "make a cube" }
    prompt = data.get('prompt')
    print(f"Received generate_cad request: '{prompt}'")
    
    if not audio_loop or not audio_loop.cad_agent:
        await sio.emit('error', {'msg': "CAD Agent not available"})
        return

    try:
        await sio.emit('status', {'msg': 'Generating new design...'})
        await sio.emit('cad_status', {'status': 'generating'})
        
        # Use generate_prototype based on prompt with project path
        cad_output_dir = str(audio_loop.project_manager.get_current_project_path() / "cad")
        result = await audio_loop.cad_agent.generate_prototype(prompt, output_dir=cad_output_dir)
        
        if result:
            info = f"{len(result.get('data', ''))} bytes (STL)"
            print(f"Sending newly generated CAD data: {info}")
            await sio.emit('cad_data', result)


            # Save to Project
            if 'file_path' in result:
                saved_path = audio_loop.project_manager.save_cad_artifact(result['file_path'], prompt)
                if saved_path:
                    print(f"[SERVER] Saved generated CAD to {saved_path}")

            await sio.emit('status', {'msg': 'Design generated'})
        else:
            await sio.emit('error', {'msg': 'Failed to generate design'})
            
    except Exception as e:
        print(f"Error generating CAD: {e}")
        await sio.emit('error', {'msg': f"Generation Error: {str(e)}"})

@sio.event
async def prompt_web_agent(sid, data):
    # data: { prompt: "find xyz" }
    prompt = data.get('prompt')
    print(f"Received web agent prompt: '{prompt}'")
    
    if not audio_loop or not audio_loop.web_agent:
        await sio.emit('error', {'msg': "Web Agent not available"})
        return

    try:
        await sio.emit('status', {'msg': 'Web Agent running...'})

        # Route through AudioLoop.handle_web_agent_request, which is the same path the
        # brain's `run_web_agent` tool takes: it wires the screenshot/log callback that
        # feeds `browser_frame`, sanitises the result for prompt injection, and reports
        # back to the model. This handler used to call `web_agent.run(prompt)` — a
        # method that does not exist on WebAgent, so every call raised AttributeError.
        await audio_loop.handle_web_agent_request(prompt)

        await sio.emit('status', {'msg': 'Web Agent finished'})

    except Exception as e:
        print(f"Error running Web Agent: {e}")
        await sio.emit('error', {'msg': f"Web Agent Error: {str(e)}"})

@sio.event
async def get_settings(sid):
    await sio.emit('settings', SETTINGS)

@sio.event
async def update_settings(sid, data):
    # Generic update
    print(f"Updating settings: {data}")
    
    # Handle specific keys if needed
    if "tool_permissions" in data:
        SETTINGS["tool_permissions"].update(data["tool_permissions"])
        if audio_loop:
            audio_loop.update_permissions(SETTINGS["tool_permissions"])
            
    if "face_auth_enabled" in data:
        SETTINGS["face_auth_enabled"] = data["face_auth_enabled"]
        # If turned OFF, maybe emit auth status true?
        if not data["face_auth_enabled"]:
             await sio.emit('auth_status', {'authenticated': True})
             # Stop auth loop if running?
             if authenticator:
                 authenticator.stop() 

    if "camera_flipped" in data:
        SETTINGS["camera_flipped"] = data["camera_flipped"]
        print(f"[SERVER] Camera flip set to: {data['camera_flipped']}")

    save_settings()
    # Broadcast new full settings
    await sio.emit('settings', SETTINGS)


# Deprecated/Mapped for compatibility if frontend still uses specific events
@sio.event
async def get_tool_permissions(sid):
    await sio.emit('tool_permissions', SETTINGS["tool_permissions"])

@sio.event
async def update_tool_permissions(sid, data):
    print(f"Updating permissions (legacy event): {data}")
    SETTINGS["tool_permissions"].update(data)
    save_settings()
    
    if audio_loop:
        audio_loop.update_permissions(SETTINGS["tool_permissions"])
    # Broadcast update to all
    await sio.emit('tool_permissions', SETTINGS["tool_permissions"])

if __name__ == "__main__":
    uvicorn.run(
        "server:app_socketio", 
        host="127.0.0.1", 
        port=8000, 
        reload=False, # Reload enabled causes spawn of worker which might miss the event loop policy patch
        loop="asyncio",
        reload_excludes=["temp_cad_gen.py", "output.stl", "*.stl"]
    )
