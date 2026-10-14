import type { KindSpec } from "../types";
import { DIALS } from "./dial";
import { PANELS } from "./panels";
import { SCENE } from "./scene";
import { MODEL } from "./model3d";
import { HUD } from "./hud";
import { CODE } from "./code";

/** Every widget kind the screen can draw. Anything else she asks for is ignored. */
export const KINDS: Record<string, KindSpec> = { ...DIALS, ...PANELS, ...SCENE, ...MODEL, ...HUD, ...CODE };

/** Kinds that open inside the orb unless she gives a slot. Same list as backend/widgets.py. */
export const INNER_KINDS = new Set([
  "clock", "timer", "gauge", "progress", "power", "tile", "connect", "radar",
  "waveform", "model3d", "notification", "scene",
]);
