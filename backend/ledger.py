"""Ledger — trust with receipts (spoken cost + latency truth).

"Today: 41 turns, $0.38, instant lane won."

Reads the telemetry turns table (brain lane, latency, tool calls, ok) and
reports the day honestly:
  turns, success rate, avg latency, tool calls, lane split (which brain
  answered what share), slowest turn.

Money, stated exactly: every provider here runs on free tiers, so actual
spend is $0.00 — reported as $0.00, never estimated. The ledger ALSO shows
the commodity-floor equivalent (Oct 2026: ~$2 input / $10 output per M
tokens) against a conservative token estimate CLEARLY LABELLED as an
estimate, so Sir sees what this week would cost anywhere else.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

# Conservative blended estimate for a voice-assistant turn incl. context.
TOKENS_PER_TURN = 2500
FLOOR_IN_PER_M = 2.0
FLOOR_OUT_PER_M = 10.0


def daily(root_dir: str | Path, hours: int = 24) -> str:
    db = Path(root_dir) / "ada_kernel" / "ada.db"
    if not db.exists():
        return "No ledger yet — talk to me first, Sir."
    try:
        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM turns WHERE ts >= ? ORDER BY ts DESC LIMIT 2000",
            (time.time() - hours * 3600,)).fetchall()
        conn.close()
    except Exception as e:
        return f"Ledger unreadable: {type(e).__name__}"
    if not rows:
        return "Quiet books — no turns in range."
    n = len(rows)
    ok = sum(1 for r in rows if r["ok"])
    lat = [r["latency_ms"] for r in rows if r["latency_ms"]]
    avg = sum(lat) / len(lat) if lat else 0
    tools = sum(r["tools"] or 0 for r in rows)
    lanes: dict[str, int] = {}
    for r in rows:
        lanes[r["brain"] or "?"] = lanes.get(r["brain"] or "?", 0) + 1
    top = sorted(lanes.items(), key=lambda kv: -kv[1])[:3]
    lane_s = ", ".join(f"{k} {v}" for k, v in top)
    est_tokens = n * TOKENS_PER_TURN
    # 70/30 in/out blend at commodity floor — labelled estimate, not a bill.
    floor = est_tokens / 1e6 * (0.7 * FLOOR_IN_PER_M + 0.3 * FLOOR_OUT_PER_M)
    slow = max(lat) if lat else 0
    return (f"{n} turns, {ok} clean, average {avg / 1000:.1f}s a turn, "
            f"{tools} tool calls. Lanes: {lane_s}. Slowest turn {slow / 1000:.0f}s. "
            f"Spent $0.00 — everything runs on free tiers. "
            f"(Elsewhere-equivalent ≈ ${floor:.2f} at commodity rates, estimated.)")
