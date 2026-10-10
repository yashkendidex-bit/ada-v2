# ADA — The Jarvis Upgrade Plan
### Written for Naz (no developer-speak — just what we'll build, why, and what you'll see)

Two missions in this plan:
1. **CAD Mission** — take 3D modeling from "flat blue shape on screen" to Iron Man movie level.
2. **Speed Mission** — make ADA answer and speak in under 2 seconds, not 10.

Everything here is based on real 2025–2026 research (papers, tools, benchmarks) — nothing imaginary. Effort ratings are honest: ⭐ = days, ⭐⭐ = 1–2 weeks, ⭐⭐⭐ = 2–6 weeks.

---

# MISSION 1: JARVIS-LEVEL 3D MODELING

## What Jarvis had (from the movies) vs what ADA has today

| Jarvis in the movies | ADA today |
|---|---|
| Glowing holographic model floating in the air | A flat blue-gray 3D shape on screen |
| Realistic materials — titanium, carbon fiber, glass | One cyan metal color for everything |
| Parts fly apart and snap together (exploded views) | One solid piece, spins slowly |
| Instantly redesigns from a sentence | Takes 1–3 minutes of code-trying |
| Precise engineering — threads, tolerances, assemblies | Good for simple parts (gears, brackets) |
| Feels alive — lighting, motion, physics | Static scene, basic shadows |

**The honest truth:** no assistant on Earth has true movie-Jarvis yet — but every single ingredient exists in 2026, and ADA already owns the two hardest parts (voice + an AI that writes real CAD code). We're building the rest on top.

## The 5-Level Roadmap

### LEVEL 1 — "Movie Looks" (visuals upgrade) ⭐
*The fastest, biggest visual wow — no AI changes, all in the viewer.*

**What we'll do:**
- **Real materials library**: metal, titanium, brass, carbon fiber, glass, rubber — each part can get a real material with correct reflections (this is "PBR" — the same tech game engines use). ADA will decide materials from your prompt ("a titanium gear" → looks like titanium).
- **Studio lighting**: swap the current generic environment for a pro studio HDR lighting rig — soft shadows, floor reflections, cinematic background.
- **WebGPU raytracing**: 2026 browsers can do real-time ray-traced reflections (like movie renders, at 60fps). One-line-ish switch in the viewer engine.
- **Hologram mode** (the Iron Man look): transparent glowing model + wireframe overlay + scanning light lines + floating particles + your name/measurements floating next to it. Toggle: [REALISTIC] / [HOLOGRAM].
- **Measurement labels**: dimensions drawn on the model (width 30mm, hole Ø10mm).

**What you'll see:** the same gear now looks like a product shot from a studio — or like it just came out of Tony Stark's lab.

---

### LEVEL 2 — "Real Engineering" (the geometry engine) ⭐⭐
*From "one part" to "a machine with parts that fit."*

**What we'll do:**
- **Assemblies**: multiple parts in one model that connect with constraints — gear + shaft + housing, with bolts that actually go into holes. This is what enterprise CAD (Fusion 360, SolidWorks) does, and it's achievable with ADA's engine (OpenCASCADE — the same kernel real CAD software is built on).
- **True mechanical features**: threads, precise fillets/chamfers, tolerances (so parts fit in the real world, not just on screen).
- **STEP export** (the universal engineering file format) — open any result in Fusion/SolidWorks/FreeCAD. Plus OBJ/GLTF for games and web.
- **Physics awareness**: ADA will know a part must be watertight, printable, within material limits — and verify before showing you.

**What you'll see:** "design a bottle opener that fits in my pocket" → a complete, manufacturable part with correct holes and edges, plus the file you could send to a machine shop.

---

### LEVEL 3 — "AI That Plans Like an Engineer" (the brain upgrade) ⭐⭐⭐
*This is the biggest leap — built from the newest research (2025–2026 CAD-AI papers).*

**Why it matters:** today ADA's process is "one AI writes code, if it crashes try again." Research proves a **team of AI agents** is dramatically better:

```
YOU: "A desk lamp with an adjustable arm"

        ┌─ CLARIFIER agent  ── asks you 2-3 smart questions
        │                    ("tabletop or floor? LED or bulb?")
        │                    (research: this alone cuts errors by ~80%)
        │
        ├─ PLANNER agent  ──── turns the answer into an engineering plan
        │                      with real dimensions
        │
        ├─ DESIGNER agent ──── writes the actual 3D code
        │
        ├─ INSPECTOR agent ─── measures the result like quality control:
        │                      volume, wall thickness, hole sizes,
        │                      "does the arm actually bend?"
        │
        └─ REFINER agent ────── fixes whatever the inspector flagged
                                and sends it back for re-checking
        │
        └── YOU get a verified model + a plain-English explanation
```

**Also included:**
- **Ambiguity detection**: ADA tells you when your request needs more detail (instead of guessing wrong).
- **Learning loop**: every part you make is stored (we already built the memory) — next time you ask for a similar part, ADA reuses what worked.
- **Follow-up voice iteration**: "make the handle thicker" → the inspector verifies the change, not just blind regeneration.

**What you'll see:** complex, working designs on the first or second try — and a voice conversation where ADA reasons out loud like Jarvis explaining the design.

---

### LEVEL 4 — "AI Imagination" (neural 3D for organic shapes) ⭐⭐⭐
*For shapes that code can't express.*

**What we'll do:**
- **Text-to-3D models** (Microsoft TRELLIS 2, open-source; or Meshy/Tripo APIs): "a sleek robot shoulder armor" — shapes a CAD engine can't build, generated by AI that learned from millions of 3D objects. Same tech as Autodesk's "neural CAD" (the tech they're putting into Fusion).
- **Photo-to-3D**: show a photo → a 3D model of it (proven trick: generate a perfect 2D picture first, convert to 3D — much better quality).
- **Hybrid design**: the neural AI makes the artistic shell, the CAD engine adds the precise holes/threads/mounts, we assemble them.

**What you'll see:** "design me a sci-fi weapon handle that mounts onto my gear" → organic sculpted shape with real engineering where it matters.

---

### LEVEL 5 — "Jarvis in the Room" (the experience) ⭐⭐⭐
*Everything above, wrapped in the Iron Man feel.*

**What we'll do:**
- **Voice-driven redesign loop**: say "make it sleeker" → the model updates live while ADA narrates (Level 3's agents + real-time pipeline).
- **Exploded-view animation**: parts fly apart in slow motion and reassemble — like the Iron Man suit scenes.
- **Physics playground**: drop the part, it lands; snap pieces together; gravity works.
- **Holographic HUD**: part names floating in space, wireframe ghosts, rotating scanner rings, measurement readouts (web browser can do all this at 60fps today).
- **AR-ready** (WebXR): later, hold your phone up and see the model on your desk.

---

### Recommended order
**Level 1 first** (2–4 days, instant wow) → **Level 3** (the engineering brain — the real power) → **Level 2** (assemblies + STEP) → **Level 4** (organic shapes) → **Level 5** (the movie experience). Level 5 is mostly "dressing" on top of 1–4.

---

# MISSION 2: THE 2-SECOND BRAIN

## Why "hey ada" took 10 seconds (plain English)

1. **Your brain model is a "thinker."** Dots3 is a reasoning model — like a professor who silently thinks before answering. Even for "hey ada," it generates 5–15 seconds of hidden thoughts first. Research: any wait over ~1 second breaks a conversation's flow — and we're paying thinking-time for every trivial question.
2. **Free tier = slow lane.** OpenRouter's free models queue behind paying users and run on slower hardware.
3. **The voice broke.** The speech synthesizer (TTS) engine we used was updated and rejected ADA's voice setting — so speech falls back to a slower engine, and in some cases nothing comes out at all. This also blocks "speak-while-thinking" (see below).

## The Fix — 4 levers, biggest first

### Lever 1 — Brain Routing (the receptionist) ⭐
*Like a great office: simple calls go to the assistant, complex ones to the expert.*

- **INSTANT lane** → tiny local model running on YOUR Mac (Ollama, ~0.2s reply) for greetings, confirmations, chit-chat, "what time is it"-style questions. Works even offline, $0.
- **FAST lane** → gpt-oss-120b (already ADA's fallback — it replies in ~0.3s first word) for normal conversation.
- **DEEP lane** → Nemotron (already built) for research/coding — the slow thinking stays where it's worth it.
- Research says routing simple queries to fast models = **32–38% faster responses** — and with the local lane, greetings become instant.

### Lever 2 — Thinking Budgets ⭐
*Tell the model how much to think per question.* "hey ada" → zero thinking, reply now. "design a robot arm" → think fully. (This is the standard "token-budget-aware reasoning" technique — up to 67% less wasted thinking.)

### Lever 3 — Fix the Voice (TTS) ⭐⭐
*The speech engine, rebuilt to be instant and reliable:*
- **Kokoro** — a small, free, local voice AI that runs on your Mac (82M params, Apache license, natural voice, first sound in ~90ms). No network, no cost, never breaks.
- **Instant speech**: the moment the brain emits the first words, ADA starts talking — she speaks while still thinking (this architecture exists; the broken voice is what's blocking it).
- **Fallback chain**: Kokoro (local) → edge-tts → cloud — if one engine fails, the next takes over silently.
- Your cloned Voxtral voice stays available as an option (voice cloning engine = XTTS, used only when you want it — it's slower, so it's not the default).

### Lever 4 — Context Slimming + Caching ⭐
- Keep the conversation history lean (we added a memory system — use it instead of replaying everything).
- Cache the fixed part of the prompt so the model doesn't re-read it every turn (standard provider trick, saves hundreds of ms).

## Target: the experience
```
You say:        "Hey Ada"
ADA says:       "Yes, sir?"        ← in under 1 second, her voice, no lag
You ask:        "Design a gear for a 3D printer"
ADA says:       "On it."           ← instantly, then plans/designs in the background
                (2 min later)      ← model appears, Jarvis-style, with narration
```

---

# WHAT I NEED FROM YOU (to start)
1. **A fresh Gemini API key** (Google AI Studio) — the current one is rejected; it's used by the web agent and as CAD's primary brain. (After Level 3, Nemotron becomes the CAD brain anyway, and Gemini is a backup.)
2. **Pick the starting level**: my recommendation is **Level 1 (movie looks) + Lever 1–3 (speed + voice)** in the first sprint — you'll see and hear the difference within days, then we build the engineer brain.

*Sources: 2026 text-to-3D comparisons (Meshy/Tripo/Rodin/TRELLIS.2), CAD-AI research (CADDesigner, ProCAD, CADSmith, agentcad — multi-agent + clarify-then-build + geometric inspection), Autodesk neural CAD / Project Bernini, WebGPU/three.js 2026 (raytraced reflections, 95% browser coverage), latency benchmarks (TTFT routing, hybrid routing 32–38% gains), TTS 2026 benchmarks (Kokoro ~90ms first audio, edge-tts, RealtimeTTS streaming).*