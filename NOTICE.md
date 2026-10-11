# Notice: GUI files derived from JARVIS

The A.D.A interface is partly derived from **JARVIS Voice AI Assistant**
(https://github.com/ethanplusai/jarvis), Copyright (c) 2026 Ethan Rogers.
The JARVIS licence is in [`licenses/JARVIS-LICENSE.txt`](licenses/JARVIS-LICENSE.txt),
unchanged.

## Which files it covers

| File | Derived from | What was taken or adapted |
|------|--------------|---------------------------|
| `src/orb.ts` | `frontend/src/orb.ts` | Particle set-up, colour palette, state targets, electrons and camera. Motion and link drawing were rewritten (fixed nearest-neighbour graph, frame-rate-independent smoothing). |
| `src/style.css` | `frontend/src/style.css` and the JARVIS palette | Accent colour, letter-spaced labels, reduced-motion handling. The layout is A.D.A's own. |

The remaining GUI files (`src/main.ts`, `src/player.ts`, `src/mic.ts`,
`src/socket.ts`, `index.html`) are A.D.A's own code and fall under the
MIT licence in [`LICENSE`](LICENSE).

## What this means

- The JARVIS licence allows **personal, educational and non-commercial use**,
  including modifying and running the software.
- It **does not allow commercial use** of the software or any derivative work
  without a commercial licence from the JARVIS author. That covers selling it,
  using it in a paid product, generating revenue with it, and offering it as a
  hosted service.
- Those listed files, and any copy or modification of them, therefore carry the
  JARVIS non-commercial terms, not the MIT terms of the rest of this repository.
- Any copy of this project that includes those files must keep this notice and
  the JARVIS copyright line.

For commercial use, contact the JARVIS author via https://ethanplus.ai, or
remove the derived files and rebuild the orb from scratch.
