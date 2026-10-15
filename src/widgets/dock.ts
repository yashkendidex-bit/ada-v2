/**
 * The dock: a few buttons at the bottom of the screen for opening widgets
 * without asking. Clock and timer toggle; "Showcase" opens a few at once so the
 * whole look can be seen; "Clear" closes everything.
 */
import type { WidgetManager } from "./manager";

const ICONS: Record<string, string> = {
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  timer: '<circle cx="12" cy="13" r="8"/><path d="M12 9v4M9 2h6M12 2v3"/>',
  showcase: '<path d="M12 3l2.4 5.6L20 11l-5.6 2.4L12 19l-2.4-5.6L4 11l5.6-2.4z"/>',
  clear: '<path d="M6 6l12 12M18 6L6 18"/>',
};

export function createDock(mgr: WidgetManager): void {
  const dock = document.createElement("div");
  dock.id = "widget-dock";
  const buttons: Record<string, HTMLButtonElement> = {};

  const add = (key: string, label: string, onClick: () => void) => {
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("aria-label", label);
    b.innerHTML = `<svg viewBox="0 0 24 24">${ICONS[key]}</svg><span class="tip"></span>`;
    (b.querySelector(".tip") as HTMLElement).textContent = label;
    b.addEventListener("click", (e) => { e.stopPropagation(); onClick(); });
    dock.appendChild(b);
    buttons[key] = b;
  };

  const toggle = (id: string, open: () => void) => { if (mgr.isOpen(id)) mgr.close(id); else open(); };

  add("clock", "Clock", () => toggle("clock-local", () => mgr.openLocal("clock", { seconds: true })));
  add("timer", "Timer", () => toggle("timer-local", () => mgr.openLocal("timer", { seconds: 300, label: "timer" })));
  add("showcase", "Showcase", () => {
    mgr.closeAll();
    mgr.openLocal("power", { label: "power level", value: 100, sub: "all systems" }, "inner", "teal", "demo-power", true);
    mgr.openLocal("status", {
      title: "Systems",
      items: [
        { label: "Voice", state: "ok", value: "online" },
        { label: "Memory", state: "ok", value: "synced" },
        { label: "Printer", state: "warn", value: "idle" },
        { label: "Vision", state: "busy", value: "scanning" },
      ],
    }, "e", "teal", "demo-status", true);
    mgr.openLocal("chart", {
      title: "Activity", type: "area", unit: "%",
      labels: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
      series: [{ name: "load", points: [22, 35, 31, 48, 62, 44, 57] }, { name: "memory", points: [40, 42, 47, 45, 52, 58, 55] }],
    }, "w", "cyan", "demo-chart", true);
  });
  add("clear", "Clear all", () => mgr.closeAll());

  document.body.appendChild(dock);

  const sync = () => {
    buttons.clock.classList.toggle("on", mgr.isOpen("clock-local"));
    buttons.timer.classList.toggle("on", mgr.isOpen("timer-local"));
  };
  mgr.onChange(sync);

  // Awake while the pointer moves, quiet otherwise.
  let timer = 0;
  const wake = () => {
    dock.classList.add("awake");
    clearTimeout(timer);
    timer = window.setTimeout(() => dock.classList.remove("awake"), 4000);
  };
  window.addEventListener("pointermove", wake, { passive: true });
  window.addEventListener("pointerdown", wake, { passive: true });
  wake();
}
