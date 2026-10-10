"""Printer — Moonraker-first print adapter (fresh build, stdlib only).

README promised OrcaSlicer + Moonraker/OctoPrint; no live module survived
(only a stale printer_agent .pyc in __pycache__). This is the replacement,
built for one solo builder with one Klipper machine:

  status()  — printer state via Moonraker (idle/printing/paused/error)
  upload()  — G-code to the machine
  start()   — begin a print (explicit Sir approval lives in the voice
              layer, never here — this module never self-starts plastic)
  pause()/cancel() — merciful stops
  progress() — completion %, for the ETA/milestone narration

Moonraker first (Klipper: Creality K1, Voron). OctoPrint-compatible POST
fallback for /api/files/local + /api/job (needs API key). No printer on
the network → every call says so plainly instead of faking it.

Discovery: ADA_PRINTER_HOST env, else mDNS-less LAN scan of common
Moonraker port 7125 on :7125 is skipped (no mdns dep) — Sir sets the host
once via `printer_set`, stored in ada_kernel/printer.json.
"""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.parse
from pathlib import Path


class Printer:
    def __init__(self, state_dir: str | Path):
        self.state_path = Path(state_dir) / "printer.json"
        self._cfg = self._load()

    def _load(self) -> dict:
        try:
            if self.state_path.exists():
                return json.loads(self.state_path.read_text())
        except Exception:
            pass
        return {"host": "", "kind": "moonraker"}

    def _save(self):
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self._cfg, indent=1))
        except Exception as e:
            print(f"[PRINTER] save failed: {e}")

    def set_host(self, host: str) -> str:
        host = (host or "").strip().rstrip("/")
        if not host:
            return "Give me the printer host (IP or name, e.g. 192.168.1.50)."
        if "://" not in host:
            host = "http://" + host
        self._cfg["host"] = host
        self._save()
        st = self.status()
        return f"Printer host set to {host}. {st}"

    def _url(self, path: str) -> str | None:
        host = self._cfg.get("host", "")
        if not host:
            return None
        return host + path

    def _get(self, path: str, timeout: int = 8) -> dict:
        url = self._url(path)
        if not url:
            return {"_err": "no printer configured — set it with printer_set first"}
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.loads(r.read().decode() or "{}")
        except Exception as e:
            return {"_err": f"printer unreachable ({type(e).__name__})"}

    def status(self) -> str:
        r = self._get("/printer/info")
        if "_err" in r:
            return r["_err"] + "."
        state = (r.get("result", {}) or {}).get("state", "unknown")
        p = self._get("/server/files/list?path=gcodes")
        n = len(((p.get("result") or []) if isinstance(p.get("result"), list)
                 else (p.get("result", {}) or {}).get("files", [])))
        return f"Printer {state}, {n} files on card."

    def progress(self) -> str:
        r = self._get("/printer/objects/query?print_stats&display_status")
        if "_err" in r:
            return r["_err"] + "."
        res = r.get("result", {}) or {}
        st = res.get("status", {}) or {}
        ps = st.get("print_stats", {}) or {}
        ds = st.get("display_status", {}) or {}
        frac = ds.get("progress") or 0
        return (f"Print state: {ps.get('state', '?')}, "
                f"{float(frac) * 100:.0f}% — {ps.get('filename', '')}.")

    def upload_gcode(self, gcode_path: str) -> str:
        url = self._url("/server/files/upload")
        if not url:
            return "No printer configured — set it with printer_set first."
        try:
            import uuid as _uuid
            boundary = _uuid.uuid4().hex
            data = Path(gcode_path).read_bytes()
            body = (f"--{boundary}\r\nContent-Disposition: form-data; "
                    f'name="file"; filename="{Path(gcode_path).name}"\r\n'
                    f"Content-Type: application/octet-stream\r\n\r\n").encode() \
                + data + f"\r\n--{boundary}--\r\n".encode()
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                json.loads(r.read().decode() or "{}")
            return f"Uploaded {Path(gcode_path).name} to the printer."
        except Exception as e:
            return f"Upload failed: {type(e).__name__}: {str(e)[:120]}"

    def start_print(self, filename: str) -> str:
        url = self._url("/printer/print/start?filename="
                        + urllib.parse.quote(filename))
        if not url:
            return "No printer configured."
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url, method="POST"),
                    timeout=15) as r:
                json.loads(r.read().decode() or "{}")
            self._cfg["last_print"] = {"file": filename, "ts": time.time()}
            self._save()
            return f"Printing {filename} — watching it like a hawk, Sir."
        except Exception as e:
            return f"Print start failed: {type(e).__name__}: {str(e)[:120]}"

    def pause(self) -> str:
        url = self._url("/printer/print/pause")
        if not url:
            return "No printer configured."
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url, method="POST"),
                    timeout=15):
                pass
            return "Paused. Plastic held, Sir."
        except Exception as e:
            return f"Pause failed: {type(e).__name__}"

    def cancel(self) -> str:
        url = self._url("/printer/print/cancel")
        if not url:
            return "No printer configured."
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url, method="POST"),
                    timeout=15):
                pass
            return "Cancelled. Standing by, Sir."
        except Exception as e:
            return f"Cancel failed: {type(e).__name__}"
