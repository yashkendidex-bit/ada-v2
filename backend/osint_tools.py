"""
Ada's OSINT Arsenal — the unified runner over the best open-source tools.

Installed arsenal:
  maigret      username -> 3000+ sites, profile parsing, recursive pivots
  sherlock     username -> 400+ sites (fast sweep)
  holehe       email    -> registered on 120+ services (reset-flow probe)
  blackbird    username/email -> 600+ platforms (WhatsMyName DB)
  phoneinfoga  phone    -> carrier, line type, VoIP, reputation
  native       domain   -> crt.sh certs, Wayback history, HIBP breaches

Every runner: async subprocess, hard timeout, robust parsing -> one evidence
block for the deep chain. Availability is detected once at startup; missing
tools degrade gracefully instead of failing the dossier.
"""
import asyncio
import json
import os
import shutil
from pathlib import Path

BACKEND_DIR = Path(__file__).parent
VENV_BIN = BACKEND_DIR.parent / ".venv" / "bin"
TOOLS_BIN = BACKEND_DIR / "tools_bin"
REPORTS_DIR = TOOLS_BIN / "osint_reports"

TOOL_TIMEOUT_S = 240          # hard cap per tool
QUICK_SITES = 120             # maigret fast subset (deep mode uses 500)


def _bin(name: str) -> str:
    """Path to a venv CLI if present, else PATH lookup."""
    p = VENV_BIN / name
    if p.exists():
        return str(p)
    return shutil.which(name) or ""


async def _run(cmd: list, timeout: float = TOOL_TIMEOUT_S, cwd: str = None) -> str:
    """Runs a CLI tool, returns combined stdout+stderr (truncated)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            cwd=cwd)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return f"(tool timed out after {timeout:.0f}s — partial results only)"
        return out.decode("utf-8", "ignore")
    except FileNotFoundError:
        return "(tool not installed)"
    except Exception as e:
        return f"(tool error: {str(e)[:100]})"


# ── availability ─────────────────────────────────────────────────────────────
def available() -> dict:
    bb = TOOLS_BIN / "blackbird" / "blackbird.py"
    return {
        "maigret": bool(_bin("maigret")),
        "sherlock": bool(_bin("sherlock")),
        "holehe": bool(_bin("holehe")),
        "blackbird": bb.exists(),
        "phoneinfoga": (TOOLS_BIN / "phoneinfoga").exists(),
    }


# ── stdout parsers (robust against tool formatting) ─────────────────────────
_JUNK = ("Searching |", "\x1b[?25", "DB auto-update", "Support Maigret",
         "Donate on Patreon", "MAIGRET - collect", "Using sites database",
         "Starting a search", "run search by full list", "Checking username")


def _clean_stdout(text: str) -> list:
    """Meaningful lines from tool stdout (drops banners, progress bars, ANSI)."""
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or any(j in line for j in _JUNK):
            continue
        if "▁" in line or "▄" in line or "█" in line:      # progress bars
            continue
        out.append(line)
    return out


def _hits_from_stdout(text: str) -> str:
    """Extracts found-account lines: '[+] Site: url' style + profile fields."""
    hits, current = [], None
    for line in _clean_stdout(text):
        if line.startswith("[+]") and "http" in line:
            current = line[3:].strip()
            hits.append(current)
        elif line.startswith("├─") or line.startswith("└─"):
            if hits is not None and current is not None:
                hits.append("    " + line[2:].strip())   # profile detail under last hit
        elif line.lower().startswith("found") or " email found" in line.lower():
            hits.append(line)
    return "\n".join(hits[:100])


def _parse_holehe(text: str) -> str:
    """Holehe: keep genuine found lines, drop the promo banner."""
    banner_markers = ("@palenath", "BTC", "Donation", "github.com/megadose",
                      "Twitter :", "Holehe V", "For BTC")
    out = []
    for line in _clean_stdout(text):
        low = line.lower()
        if any(m.lower() in low for m in banner_markers):
            continue
        if "found" in low and "not found" not in low:
            out.append(line)
    return "\n".join(out[:80])
def _parse_maigret_json(report_path) -> str:
    try:
        data = json.loads(Path(report_path).read_text())
    except Exception:
        return ""
    found = []
    for entry in data if isinstance(data, list) else []:
        status = entry.get("status", {})
        if isinstance(status, dict) and status.get("found"):
            blocks = [entry.get("site", "?")]
            url = entry.get("url", "")
            if url:
                blocks.append(url)
            for key in ("status",):
                pass
            found.append(" - ".join(blocks))
    return "\n".join(found[:80])


def _parse_sherlock_json(report_path) -> str:
    try:
        data = json.loads(Path(report_path).read_text())
    except Exception:
        return ""
    lines = []
    for site, info in data.items():
        # sherlock json: {site: [{"status": "claimed"/..., "url": ...}]}
        entries = info if isinstance(info, list) else [info]
        for e in entries:
            if isinstance(e, dict) and e.get("status") in ("claimed", "found"):
                lines.append(f"{site} - {e.get('url','')}")
    return "\n".join(lines[:60])


def _parse_blackbird_json(report_path) -> str:
    try:
        data = json.loads(Path(report_path).read_text())
    except Exception:
        return ""
    lines = []
    for e in data if isinstance(data, list) else []:
        site = e.get("site", "?")
        url = e.get("url", "")
        status = e.get("found", e.get("status", ""))
        if url and (status in (True, "found", "claimed") or e.get("found") is True):
            lines.append(f"{site} - {url}")
    return "\n".join(lines[:80])


# ── runners ──────────────────────────────────────────────────────────────────
async def run_username(username: str, deep: bool = False) -> str:
    """Maigret + Sherlock + Blackbird in parallel. Returns merged evidence.
    (Maigret's rich stdout carries parsed profiles; JSON report flags differ
    per version, so stdout is the canonical channel.)"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(c for c in username if c.isalnum() or c in "_-.")[:40] or "target"

    maigret_bin = _bin("maigret")
    maigret_cmd = [maigret_bin, username, "--timeout", "12", "--no-color",
                   "--top-sites", "500" if deep else str(QUICK_SITES)] if maigret_bin else None

    sherlock_bin = _bin("sherlock")
    sherlock_report = REPORTS_DIR / f"sherlock_{safe}.json"
    sherlock_cmd = [sherlock_bin, username, "--json", str(sherlock_report),
                    "--timeout", "15", "--print-found", "--no-color"] if sherlock_bin else None

    bb_py = TOOLS_BIN / "blackbird" / "blackbird.py"
    bb_report = REPORTS_DIR / f"blackbird_{safe}.json"
    bb_cmd = [sys_python(), str(bb_py), "-u", username, "--json",
              "--no-update", "--timeout", "20"] if bb_py.exists() else None

    outs = await asyncio.gather(
        _run(maigret_cmd) if maigret_cmd else _none("maigret not installed"),
        _run(sherlock_cmd) if sherlock_cmd else _none("sherlock not installed"),
        _run(bb_cmd, cwd=str(TOOLS_BIN / "blackbird")) if bb_cmd else _none("blackbird not installed"),
    )

    m = _hits_from_stdout(outs[0])
    s = _hits_from_stdout(outs[1]) or (_parse_sherlock_json(sherlock_report) if sherlock_report.exists() else "")
    b = _hits_from_stdout(outs[2])
    if not b and bb_report.exists():
        b = _parse_blackbird_json(bb_report)
    # cap maigret's verbose profile fields so the deep chain isn't flooded
    m_lines = m.splitlines()
    if len(m_lines) > 90:
        m = "\n".join(m_lines[:90]) + f"\n(...+{len(m_lines) - 90} more lines)"
    blocks = [
        f"MAIGRET (3000+ site engine, parsed profiles):\n{m or '(no hits)'}",
        f"SHERLOCK (400-site sweep):\n{s or '(no hits)'}",
        f"BLACKBIRD (WhatsMyName 600+):\n{b or '(no hits)'}",
    ]
    return "\n\n".join(blocks)


async def run_email(email: str) -> str:
    """Holehe (reset-flow probe) + Blackbird email mode in parallel."""
    bb_py = TOOLS_BIN / "blackbird" / "blackbird.py"
    holehe_bin = _bin("holehe")

    outs = await asyncio.gather(
        _run([holehe_bin, email, "--only-used", "--no-color"]) if holehe_bin else _none("holehe not installed"),
        _run([sys_python(), str(bb_py), "-e", email, "--json", "--no-update", "--timeout", "20"],
             cwd=str(TOOLS_BIN / "blackbird")) if bb_py.exists() else _none("blackbird not installed"),
    )
    found = _parse_holehe(outs[0])
    bb_lines = _hits_from_stdout(outs[1])
    return (f"HOLEHE (email registered on 120+ services):\n{found or '(no registrations found)'}\n\n"
            f"BLACKBIRD EMAIL SWEEP:\n{bb_lines or '(no additional hits)'}")


async def run_phone(phone: str) -> str:
    """PhoneInfoga scan."""
    binp = TOOLS_BIN / "phoneinfoga"
    if not binp.exists():
        return "(phoneinfoga not installed)"
    out = await _run([str(binp), "scan", "-n", phone], timeout=90)
    lines = [l for l in out.splitlines() if l.strip()][:60]
    return "PHONEINFOGA (carrier/line-type/reputation scan):\n" + "\n".join(lines)


async def run_domain(domain: str, hibp_key: str = None) -> str:
    """Native no-key sweep: crt.sh certs + Wayback history + optional HIBP."""
    import httpx
    blocks = []
    async with httpx.AsyncClient(timeout=25.0, follow_redirects=True,
                                 headers={"User-Agent": "Mozilla/5.0"}) as c:
        try:
            r = await c.get(f"https://crt.sh/?q=%.{domain}&output=json")
            try:
                entries = r.json()
            except Exception:
                entries = []
            if not entries:      # crt.sh often rate-limits -> one patient retry
                await asyncio.sleep(4)
                r = await c.get(f"https://crt.sh/?q=%.{domain}&output=json")
                try:
                    entries = r.json()
                except Exception:
                    entries = []
            subs = sorted({e.get("name_value", "").replace("*.", "").strip()
                           for e in entries if e.get("name_value")})
            subs = [s for s in subs if s]
            blocks.append(f"CRT.SH SUBDOMAINS ({len(subs)} unique):\n" +
                          ("\n".join(subs[:60]) or "(none returned)"))
        except Exception as e:
            blocks.append(f"crt.sh: ({str(e)[:60]})")
        try:
            r = await c.get(
                "https://web.archive.org/cdx/search/cdx",
                params={"url": f"{domain}/*", "output": "json", "limit": "40"},
            )
            rows = r.json()[1:] if isinstance(r.json(), list) and len(r.json()) > 1 else []
            hist = [f"{row[1]} @ {row[2][:10]}" for row in rows if len(row) > 2]
            blocks.append("WAYBACK HISTORY (surviving pages):\n" +
                          ("\n".join(hist[:25]) or "(no archived pages)"))
        except Exception as e:
            blocks.append(f"wayback: ({str(e)[:60]})")
        if hibp_key:
            try:
                r = await c.get(f"https://haveibeenpwned.com/api/v3/breaches/{domain}",
                                headers={"hibp-api-key": hibp_key})
                if r.status_code == 200:
                    breaches = [f"{b.get('Name')} ({b.get('BreachDate')})" for b in r.json()]
                    blocks.append("HIBP BREACHES:\n" + "\n".join(breaches[:20]))
            except Exception:
                pass
    return "DOMAIN SWEEP:\n" + "\n\n".join(blocks)


async def _none(msg: str) -> str:
    return msg


def sys_python() -> str:
    return str(VENV_BIN / "python")
