/**
 * Widgets: what A.D.A can put on screen, inside and around the orb.
 * The backend validates everything (backend/widgets.py); the screen checks
 * again, because it never trusts what arrives.
 */

export type Props = Record<string, unknown>;

export type Slot = "auto" | "inner" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w" | "nw";

/** One colour family. Every widget draws from these four and nothing else. */
export interface Palette {
  /** Main colour. */
  c: string;
  /** Bright shade: text, highlights. */
  c2: string;
  /** Dim shade: guides, grids. */
  cd: string;
  /** "r, g, b" of the main colour, for rgba(). */
  rgb: string;
}

export interface WidgetEnv {
  /** Live palette. Canvas widgets read it every frame, so a recolour needs no rebuild. */
  pal: Palette;
  /** The user asked the system for less motion. */
  calm: boolean;
  /** A button inside a widget was pressed. Only allow-listed actions go anywhere. */
  action(name: string): void;
  /** Ask to be closed (the widget decided it is finished). */
  close(): void;
}

export interface WidgetInstance {
  el: HTMLElement;
  /** New props, already merged. Animate to them; do not rebuild. */
  update(props: Props): void;
  /** Called every frame while the widget is on screen. t = seconds, dt = seconds since the last call. */
  tick?(t: number, dt: number): void;
  /** True while this widget wants the orb to churn (a task is running). */
  busy?(): boolean;
  destroy(): void;
}

export interface KindSpec {
  /** Natural size in CSS px. The manager scales the whole widget to fit. */
  design: [number, number];
  /** Diameter inside the orb, in orb radii. Only meaningful for kinds that sit inside. */
  inner?: number;
  /** Preferred size around the orb, in orb radii (width, height). */
  around: [number, number];
  /** Frame it like a panel (corner brackets, faint fill). Dials and rings draw their own shape. */
  panel: boolean;
  make(props: Props, env: WidgetEnv): WidgetInstance;
}

export interface OpenMsg {
  t: "widget:open";
  id: string;
  kind: string;
  props: Props;
  slot?: string;
  color?: string;
  ttl?: number;
}
