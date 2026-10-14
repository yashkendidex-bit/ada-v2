# Notice: GUI files taken from JARVIS

The A.D.A screen is the **JARVIS** interface, copied from
https://github.com/ethanplusai/jarvis (Copyright (c) 2026 Ethan Rogers). The
JARVIS licence is in [`licenses/JARVIS-LICENSE.txt`](licenses/JARVIS-LICENSE.txt),
unchanged.

## Which files it covers

| File | Origin | Change from the original |
|------|--------|--------------------------|
| `src/style.css` | `frontend/src/style.css` | Verbatim. |
| `src/voice.ts` | `frontend/src/voice.ts` | Verbatim. A.D.A uses its audio queue and mic meter from it. Its Web Speech input is not used; see `src/ada-voice.ts`. |
| `index.html` | `frontend/index.html` | Page title and on-screen label changed from JARVIS to A.D.A. Favicon line added. |
| `src/main.ts` | `frontend/src/main.ts` | Speech input imported from `ada-voice.ts`. The microphone level is passed to the orb, and `orb-theme.css` is imported. Settings and Restart menu actions replaced with notices. First-run settings check removed. |
| `src/ws.ts` | `frontend/src/ws.ts` (replaced) | Same interface, so `main.ts` is unchanged in its use of it. Underneath it speaks A.D.A's Socket.IO events instead of JARVIS's websocket protocol. |

The rest of the GUI is A.D.A's own code under the MIT licence in
[`LICENSE`](LICENSE): `src/ada-voice.ts`, `src/ada-state.ts`, `src/mic.ts`,
`src/env.d.ts`, `src/orb-motion.ts` (timing and voice-envelope helpers), `src/orb-theme.css` (teal control colours) and `src/orb.ts`, and everything in `src/widgets/` (the widget layer: manager, dock, dials, panels, scene renderer, 3D viewer, HUD renderer, sandbox).

`src/orb.ts` is new A.D.A code, no longer the JARVIS orb. It is a GPU point-sprite renderer written from scratch and modelled on the look of a reference video (Marvel's Iron Man 3 second-screen JARVIS trailer), measured for ring radii, colour and how the ring distorts with the voice. No code, frames or other assets from that video or from JARVIS are in it. The JARVIS and Marvel names and the film's look belong to their owners; this is a personal, non-commercial homage, so keep it non-commercial.

## What this means

- The JARVIS licence allows **personal, educational and non-commercial use**,
  including modifying and running the software.
- It **does not allow commercial use** of the software or any derivative work
  without a commercial licence from the JARVIS author. That covers selling it,
  using it in a paid product, generating revenue with it, and offering it as a
  hosted service.
- The files in the table, and any copy or modification of them, therefore carry
  the JARVIS non-commercial terms, not the MIT terms of the rest of this
  repository.
- Any copy of this project that includes those files must keep this notice and
  the JARVIS copyright line.

For commercial use, contact the JARVIS author via https://ethanplus.ai, or
remove the files in the table above and build a different screen.
