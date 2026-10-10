"""Conversational Model Surgery — the JARVIS 'remove this, this and this' system.

Tracks every part of the CURRENT model in a registry so ADA can speak verbs at it:
  list_parts()                          -> what's in the model right now
  edit_part(target, action, ...)        -> delete / hide / isolate / recolor / rescale
  undo()                                -> restore last deleted part (trash stack)
  snapshot() / versions()               -> silent version history (Stark timeline)
  compare(v1, v2)                       -> ghost diff between two versions

Also runs SILENT FEA on every structural edit and warns when safety drops —
physics-on-every-edit, like JARVIS checking the suit while Tony modifies it.

Edits apply live to the frontend via socket events AND to the on-disk STLs
(delete archives the part's STL, rescale rewrites its vertices), so exports
and downloads always match what Sir sees. Undo restores both.
"""
from __future__ import annotations
import os
import re
import json
import time
import base64
import threading

BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))
VERSIONS_DIR = os.path.join(os.path.dirname(BACKEND_ROOT), "workspace", "versions")
# Archived STLs from deleted parts (undo moves them back).
TRASH_DIR = os.path.join(os.path.dirname(BACKEND_ROOT), "workspace", "surgery_trash")


class ModelRegistry:
    """In-memory registry of the current model's parts + trash stack."""

    def __init__(self):
        self._lock = threading.Lock()
        self._parts = []       # [{id, name, material, size, visible, color_override}]
        self._model_name = ""
        self._trash = []       # deleted parts (restorable)
        self._updated_at = 0.0
        self.last_pick = None  # part id the user physically pointed at ("this")
        self._versions = []    # [{id, ts, label, visible_ids}]
        self._stl_paths = {}   # part id -> stl path (for FEA)
        self._last_safety = None

    def set_stl_paths(self, paths: dict):
        """part id -> on-disk STL path, used for silent physics checks."""
        with self._lock:
            self._stl_paths.update(paths or {})

    # ── Point & Speak ────────────────────────────────────────────────────────
    def set_selection(self, target: str) -> dict | None:
        """User clicked/pinched a part in the 3D view -> remember it as 'this'."""
        with self._lock:
            p = self._resolve(target)
            if not p:
                return None
            self.last_pick = p["id"]
            print(f"[SURGERY] Selection: '{p['id']}' is now THIS")
            return {"selected": p["id"], "index": p["index"], "material": p.get("material", "")}

    def get_selection(self) -> str | None:
        return self.last_pick

    # ── Registration ─────────────────────────────────────────────────────────
    def set_model(self, name: str, parts: list[dict]):
        """Called whenever a new model arrives from cad_agent."""
        with self._lock:
            self._model_name = name or "model"
            self._parts = []
            self._trash.clear()
            for i, p in enumerate(parts):
                self._parts.append({
                    "id": p.get("name", f"part_{i+1}"),
                    "index": i,
                    "material": p.get("material", ""),
                    "size": p.get("size") or (p.get("dims") or []),
                    "visible": True,
                    "color_override": None,
                })
            self._updated_at = time.time()

    def has_model(self) -> bool:
        return len(self._parts) > 0

    def model_name(self) -> str:
        return self._model_name

    # ── Queries ──────────────────────────────────────────────────────────────
    def list_parts(self) -> str:
        """Human-readable part list for the brain."""
        with self._lock:
            if not self._parts:
                return "No model is loaded right now."
            lines = [f"Current model '{self._model_name}' has {len(self._parts)} parts:"]
            for p in self._parts:
                state = "" if p["visible"] else " (hidden)"
                trash = " (DELETED)" if p.get("deleted") else ""
                sz = p.get("size") or []
                dims = f" {sz[0]:.0f}x{sz[1]:.0f}x{sz[2]:.0f}mm" if len(sz) == 3 else ""
                mat = f" {p['material']}" if p["material"] else ""
                col = f" color={p['color_override']}" if p["color_override"] else ""
                lines.append(f"  - {p['id']} |{mat}{dims}{col}{state}{trash}")
            return "\n".join(lines)

    def _resolve(self, target: str):
        t = (target or "").strip().lower().replace("_", " ").replace("-", " ")
        # PRONOUN RESOLUTION: 'this'/'that'/'it' = the part the user pointed at
        if t in ("this", "that", "it", "the one", "selected", "that one", "this one"):
            if self.last_pick:
                for p in self._parts:
                    if p["id"] == self.last_pick and not p.get("deleted"):
                        return p
            return None
        # exact id, normalized contains, or index ("part 2")
        for p in self._parts:
            pid = p["id"].lower().replace("_", " ")
            if t == pid or t in pid or pid in t:
                return p
        import re as _re
        m = _re.search(r"(?:part\s*)?#?(\d+)", t)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(self._parts):
                return self._parts[idx]
        return None

    # ── Actions ──────────────────────────────────────────────────────────────
    # -- STL truthfulness -------------------------------------------------
    # Surgery used to be view-layer only: the frontend flipped flags while the
    # on-disk STLs stayed stale. Deletes now archive the part's STL and
    # rescales rewrite its vertices, so exports stay truthful. Undo restores.

    def _trash_stl(self, part_id):
        """Move a part's STL (plus any .bak files) aside. Returns trash path or None."""
        import shutil
        import glob
        src = self._stl_paths.get(part_id)
        if not src or not os.path.exists(src):
            return None
        try:
            os.makedirs(TRASH_DIR, exist_ok=True)
            dst = os.path.join(TRASH_DIR, f"{part_id}_{int(time.time() * 1000)}.stl")
            shutil.move(src, dst)
            for bak in glob.glob(src + ".bak*"):
                try:
                    os.remove(bak)
                except OSError:
                    pass
            return dst
        except Exception as e:
            print(f"[SURGERY] stl trash failed for {part_id}: {e}")
            return None

    def _restore_stl(self, part_id, trash_path):
        """Move an archived STL back to its live path."""
        import shutil
        dst = self._stl_paths.get(part_id)
        if not trash_path or not dst or not os.path.exists(trash_path):
            return False
        try:
            parent = os.path.dirname(dst)
            if parent:
                os.makedirs(parent, exist_ok=True)
            shutil.move(trash_path, dst)
            return True
        except Exception as e:
            print(f"[SURGERY] stl restore failed for {part_id}: {e}")
            return False

    def _scale_stl(self, path, factor):
        """Scale every vertex in an STL by factor (binary or ASCII).

        Backs the original up to path.bak_<ms> first; returns the backup path.
        On failure the original is restored and the error re-raised.
        """
        import shutil
        import struct
        backup = f"{path}.bak_{int(time.time() * 1000)}"
        shutil.copy2(path, backup)
        try:
            with open(path, "rb") as f:
                head = f.read(84)
            count = struct.unpack("<I", head[80:84])[0]
            if os.path.getsize(path) == 84 + count * 50:
                self._scale_stl_binary(path, factor, count)
            else:
                self._scale_stl_ascii(path, factor)
            return backup
        except Exception:
            try:
                shutil.move(backup, path)
            except Exception:
                pass
            raise

    @staticmethod
    def _scale_stl_binary(path, factor, count):
        """Uniform scale of a binary STL, in place. Normals are untouched
        (a uniform scale never changes a normal's direction)."""
        try:
            import numpy as np
            dt = np.dtype([("n", "<f4", 3), ("v", "<f4", 9), ("a", "u2")])
            facets = np.memmap(path, dtype=dt, mode="r+", offset=84, shape=(count,))
            facets["v"] *= factor
            facets.flush()
            del facets
            return
        except ImportError:
            pass
        import struct
        with open(path, "r+b") as f:
            f.seek(84)
            for _ in range(count):
                raw = f.read(50)
                if len(raw) < 50:
                    break
                floats = struct.unpack("<12f", raw[:48])
                attr = raw[48:]
                out = list(floats[:3])
                out.extend(v * factor for v in floats[3:])
                f.seek(-50, os.SEEK_CUR)
                f.write(struct.pack("<12f", *out) + attr)

    @staticmethod
    def _scale_stl_ascii(path, factor):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        out = []
        for ln in lines:
            s = ln.strip()
            if s.startswith("vertex"):
                parts = s.split()
                try:
                    x, y, z = (float(parts[1]) * factor,
                               float(parts[2]) * factor,
                               float(parts[3]) * factor)
                    out.append(f"      vertex {x:.6f} {y:.6f} {z:.6f}\n")
                    continue
                except (IndexError, ValueError):
                    pass
            out.append(ln)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(out)

    def edit_part(self, target: str, action: str, value=None) -> dict:
        """action: delete | hide | show | isolate | recolor | rescale"""
        with self._lock:
            if not self._parts:
                return {"ok": False, "error": "No model loaded."}
            if target in (None, "", "all", "everything"):
                matches = [p for p in self._parts if not p.get("deleted")]
            elif "," in (target or "") or " and " in (target or "").lower():
                tokens = [t.strip() for t in re.split(r",| and ", target, flags=re.I) if t.strip()] if (target := (target or "")) else []
                matches = []
                for tk in tokens:
                    r = self._resolve(tk)
                    if r and r not in matches:
                        matches.append(r)
            else:
                r = self._resolve(target)
                matches = [r] if r else []

            matches = [m for m in matches if m and not m.get("deleted")]
            if not matches:
                return {"ok": False, "error": f"Part '{target}' not found.", "parts": self.list_parts()}

            results = []
            for p in matches:
                if action == "delete":
                    p["deleted"] = True
                    p["visible"] = False
                    trash_stl = self._trash_stl(p["id"])
                    if trash_stl:
                        p["_stl_trash"] = trash_stl
                        results.append(f"{p['id']} removed (STL archived)")
                    else:
                        results.append(f"{p['id']} removed")
                    self._trash.append(p)
                elif action == "hide":
                    p["visible"] = False
                    results.append(f"{p['id']} hidden")
                elif action == "show":
                    p["visible"] = True
                    p.pop("deleted", None)
                    results.append(f"{p['id']} shown")
                elif action == "isolate":
                    for other in self._parts:
                        other["visible"] = other is p
                    results.append(f"{p['id']} isolated")
                elif action == "recolor":
                    p["color_override"] = str(value or "#22d3ee")
                    results.append(f"{p['id']} recolored {p['color_override']}")
                elif action == "rescale":
                    try:
                        new_scale = float(value)
                    except (TypeError, ValueError):
                        return {"ok": False, "error": "rescale needs a number"}
                    prev = float(p.get("scale", 1) or 1)
                    delta = new_scale / prev if prev else new_scale
                    stl = self._stl_paths.get(p["id"])
                    if stl and os.path.exists(stl):
                        try:
                            backup = self._scale_stl(stl, delta)
                        except Exception as e:
                            return {"ok": False,
                                    "error": f"STL rescale failed: {e}"}
                        p["_stl_backup"] = backup
                        self._trash.append({"_rescale_undo": True,
                                            "id": p["id"],
                                            "backup": backup,
                                            "prev_scale": prev})
                        results.append(f"{p['id']} scaled x{new_scale} (STL rewritten)")
                    else:
                        results.append(f"{p['id']} scaled x{new_scale}")
                    p["scale"] = new_scale
                else:
                    return {"ok": False, "error": f"Unknown action '{action}'"}

            self._updated_at = time.time()
            edit_event = {
                "ts": self._updated_at,
                "action": action,
                "targets": [p["id"] for p in matches],
                "value": value,
                "summary": ", ".join(results),
                # full visibility map so frontend syncs exactly
                "visibility": [{"id": p["id"], "visible": p["visible"],
                                 "color": p.get("color_override"),
                                 "scale": p.get("scale", 1)} for p in self._parts],
            }
            return {"ok": True, **edit_event}

    # ── Silent physics on structural edits ───────────────────────────────────
    def physics_check(self, force_n: float = 500.0) -> dict | None:
        """After a delete/isolate, check the remaining PRIMARY part still passes FEA.
        Returns warning text or None. Never blocks the edit."""
        try:
            with self._lock:
                alive = [p for p in self._parts if not p.get("deleted") and p["visible"]]
                if not alive:
                    return None
                # Check the biggest remaining part (the load-bearing one)
                def _vol(p):
                    s = p.get("size") or []
                    return (s[0] * s[1] * s[2]) if len(s) == 3 else 0
                primary = max(alive, key=_vol)
                stl = self._stl_paths.get(primary["id"])
                if not stl or not os.path.exists(stl):
                    return None
                material = primary.get("material") or "steel"
            from cad_v3.fea_solver import compute_von_mises_stress_approx
            res = compute_von_mises_stress_approx(stl, force_n=force_n, material=material)
            sf = res.get("safety_factor", 0)
            status = res.get("status", "?")
            self._last_safety = sf
            if status == "FAIL":
                return (f"Physics check: removing that dropped safety factor to {sf} — "
                        f"the {primary['id']} would FAIL under {force_n}N. Consider undo.")
            if status == "WARN":
                return f"Physics note: safety factor is now {sf} — workable but close to the limit."
            return f"Physics check: still sound — safety factor {sf} ({material}, {force_n}N)."
        except Exception as e:
            print(f"[SURGERY] physics_check skipped: {e}")
            return None

    # ── Version timeline (Stark snapshots) ───────────────────────────────────
    def snapshot(self, label: str = "") -> dict:
        with self._lock:
            vid = f"v{len(self._versions) + 1}"
            entry = {
                "id": vid,
                "ts": time.time(),
                "label": label or f"{self._model_name} — {len([p for p in self._parts if not p.get('deleted')])} parts",
                "model_name": self._model_name,
                "visible_ids": [p["id"] for p in self._parts if not p.get("deleted") and p["visible"]],
                "colors": {p["id"]: p.get("color_override") for p in self._parts if p.get("color_override")},
            }
            self._versions.append(entry)
            os.makedirs(VERSIONS_DIR, exist_ok=True)
            with open(os.path.join(VERSIONS_DIR, f"{vid}.json"), "w") as f:
                json.dump(entry, f, indent=1)
            return entry

    def versions(self) -> str:
        with self._lock:
            if not self._versions:
                return "No versions yet — say 'save this version' to start a timeline."
            lines = ["Version timeline:"]
            for v in self._versions:
                t = time.strftime("%H:%M:%S", time.localtime(v["ts"]))
                lines.append(f"  {v['id']} [{t}] {v['label']} ({len(v['visible_ids'])} parts)")
            return "\n".join(lines)

    def compare(self, v1: str = "", v2: str = "") -> dict:
        """Ghost diff between two versions: what was added/removed."""
        with self._lock:
            def find(vid):
                vid = vid.strip().lower()
                for v in self._versions:
                    if v["id"].lower() == vid:
                        return v
                return None
            a = find(v1) or (self._versions[-2] if len(self._versions) >= 2 else None)
            b = find(v2) or (self._versions[-1] if self._versions else None)
            if not a or not b or a is b:
                return {"ok": False, "error": "Need two saved versions to compare. Say 'save version' first."}
            added = [pid for pid in b["visible_ids"] if pid not in a["visible_ids"]]
            removed = [pid for pid in a["visible_ids"] if pid not in b["visible_ids"]]
            color_changed = {k: v for k, v in b.get("colors", {}).items()
                             if a.get("colors", {}).get(k) != v}
            return {
                "ok": True,
                "from": a["id"], "to": b["id"],
                "added": added, "removed": removed,
                "recolors": color_changed,
                "summary": (f"{a['id']} → {b['id']}: "
                           + (f"added {', '.join(added)}; " if added else "")
                           + (f"removed {', '.join(removed)}; " if removed else "")
                           + (f"recolored {', '.join(color_changed)}" if color_changed else "")
                           or "identical"),
            }

    def undo(self) -> dict:
        with self._lock:
            if not self._trash:
                return {"ok": False, "message": "Nothing to restore — trash is empty."}
            item = self._trash.pop()
            if isinstance(item, dict) and item.get("_rescale_undo"):
                # Revert an STL rescale: put the pre-scale vertices back.
                part_id = item["id"]
                stl = self._stl_paths.get(part_id)
                backup = item.get("backup")
                if backup and stl and os.path.exists(backup):
                    import shutil
                    try:
                        shutil.copy2(backup, stl)
                    except Exception as e:
                        print(f"[SURGERY] stl rescale-undo failed: {e}")
                for p in self._parts:
                    if p["id"] == part_id:
                        p["scale"] = item.get("prev_scale", 1)
                        p.pop("_stl_backup", None)
                self._updated_at = time.time()
                return {"ok": True, "restored": part_id,
                        "summary": f"{part_id} scale reverted to x{item.get('prev_scale', 1)}",
                        "visibility": [{"id": q["id"], "visible": q["visible"],
                                         "color": q.get("color_override"),
                                         "scale": q.get("scale", 1)} for q in self._parts]}
            p = item
            p.pop("deleted", None)
            p["visible"] = True
            trash_stl = p.pop("_stl_trash", None)
            if trash_stl:
                self._restore_stl(p["id"], trash_stl)
            self._updated_at = time.time()
            return {"ok": True, "restored": p["id"],
                    "summary": f"{p['id']} brought back",
                    "visibility": [{"id": q["id"], "visible": q["visible"],
                                     "color": q.get("color_override"),
                                     "scale": q.get("scale", 1)} for q in self._parts]}


# Singleton shared by ada.py + server.py
registry = ModelRegistry()
