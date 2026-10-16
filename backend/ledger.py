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


def weekly_trust(root_dir: str | Path, days: int = 7) -> str:
    """Spoken weekly trust summary (Feature 9): decisions made, confirmations
    asked, suggestions suppressed, actions undone. Reads the audit log +
    the F8 policy.decision timeline — plain numbers, no estimates."""
    from replay import restraint_summary
    kdir = Path(root_dir) / "ada_kernel"
    adb = kdir / "ada.db"
    if not adb.exists():
        return "No trust ledger yet — talk to me first, Sir."
    cutoff = time.time() - days * 86400
    try:
        conn = sqlite3.connect(f"file:{adb}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT decision, gate_json, undone FROM audit "
            "WHERE action='tool-gate' AND ts >= ?", (cutoff,)).fetchall()
        undone = conn.execute(
            "SELECT COUNT(*) AS n FROM audit WHERE undone=1 AND ts >= ?",
            (cutoff,)).fetchone()["n"]
        conn.close()
    except Exception as e:
        return f"Trust ledger unreadable: {type(e).__name__}"
    decisions = len(rows)
    asked = denied = 0
    for r in rows:
        if r["decision"] == "denied":
            denied += 1
        try:
            import json as _j
            g = _j.loads(r["gate_json"]) if r["gate_json"] else {}
            if g.get("verdict") == "voice-ask" or g.get("needs_voice"):
                asked += 1
        except Exception:
            pass
    rest = restraint_summary(kdir / "ada_events.db", hours=days * 24)
    supp = rest.get("suppressed", 0) if not rest.get("unavailable") else 0
    bits = [f"{decisions} decisions"]
    if asked:
        bits.append(f"{asked} confirmations asked")
    if denied:
        bits.append(f"{denied} denied by you")
    if supp:
        bits.append(f"{supp} suggestions held back")
    if undone:
        bits.append(f"{undone} undone")
    return ("This week: " + ", ".join(bits) + ". "
            "Every one of them is in the audit log, Sir — replay any of them "
            "from the trust console.")


def _cli():
    import sys as _s
    root = _s.argv[1] if len(_s.argv) > 1 else "."
    if len(_s.argv) > 2 and _s.argv[2] == "trust":
        print(weekly_trust(root))
    else:
        print(daily(root))


if __name__ == "__main__":
    _cli()
