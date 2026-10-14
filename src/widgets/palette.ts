import type { Palette } from "./types";

/** The orb's own teal first, then the family the widgets may use. */
const BASE: Record<string, string> = {
  teal: "#3fe0d4",
  cyan: "#4fc8ff",
  blue: "#4d8dff",
  amber: "#ffb43f",
  red: "#ff5a5a",
  green: "#4be28c",
  violet: "#a98bff",
  white: "#e8f8f6",
};

export const PALETTE_NAMES = Object.keys(BASE);

const HEX = /^#[0-9a-f]{6}$/i;

function parse(hex: string): [number, number, number] {
  return [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)];
}

function toHex(r: number, g: number, b: number): string {
  const f = (v: number) => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, "0");
  return `#${f(r)}${f(g)}${f(b)}`;
}

/** A name from the family, or "#rrggbb". Anything else is the orb's teal. */
export function resolvePalette(name: unknown): Palette {
  let hex = BASE.teal;
  if (typeof name === "string") {
    const n = name.trim().toLowerCase();
    if (BASE[n]) hex = BASE[n];
    else if (HEX.test(n)) hex = n;
  }
  const [r, g, b] = parse(hex);
  // Bright: pulled toward white. Dim: pulled toward black.
  const c2 = toHex(r + (255 - r) * 0.55, g + (255 - g) * 0.55, b + (255 - b) * 0.55);
  const cd = toHex(r * 0.42, g * 0.42, b * 0.42);
  return { c: hex, c2, cd, rgb: `${r}, ${g}, ${b}` };
}

/** Copy a palette's values into another, so canvas widgets holding the first see the change. */
export function assignPalette(into: Palette, from: Palette): void {
  into.c = from.c; into.c2 = from.c2; into.cd = from.cd; into.rgb = from.rgb;
}

export function paletteVars(el: HTMLElement, p: Palette): void {
  el.style.setProperty("--c", p.c);
  el.style.setProperty("--c2", p.c2);
  el.style.setProperty("--cd", p.cd);
  el.style.setProperty("--rgb", p.rgb);
}
