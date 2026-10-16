"""Access to the canonical ``mcp`` PyPI SDK without a name-shadowing trap.

``backend/`` sits at ``sys.path[0]`` (tests insert it; the server runs from it),
so this package — ``backend/mcp`` — wins the top-level ``mcp`` name. The
canonical SDK (``mcp`` on PyPI, here v2.x with ``mcp.server.mcpserver``)
cannot be imported with a plain ``import mcp`` from inside this package:
it would resolve to *us*.

This module loads the real SDK from site-packages exactly once, lets its
internal absolute imports (``from mcp.server... import ...``) resolve to
itself during initialisation, then re-keys every SDK module in
``sys.modules`` from ``mcp*`` to ``_mcp_sdk*`` and restores this package as
``sys.modules["mcp"]``.

All SDK-internal imports are top-level (the only runtime-lazy absolute
import is the optional auth-routes one, which ADA never configures), so the
re-keyed modules keep working through their already-bound references.

Usage inside this package::

    from ._sdk import SDK
    MCPServer = SDK.server.mcpserver.MCPServer
"""

import importlib.util
import os
import site
import sys


def _find_sdk_init():
    """Locate the site-packages copy of the canonical SDK (never this package)."""
    here = os.path.abspath(os.path.join(os.path.dirname(__file__), "__init__.py"))
    search = list(site.getsitepackages())
    try:
        search.append(site.getusersitepackages())
    except Exception:
        pass
    for sp in search:
        init = os.path.join(sp, "mcp", "__init__.py")
        if os.path.isfile(init) and os.path.abspath(init) != here:
            return sp, init
    raise ImportError(
        "canonical `mcp` SDK not found in site-packages "
        "(expected the PyPI `mcp` package; not the standalone `fastmcp` one)"
    )


def _load():
    if "_mcp_sdk" in sys.modules:
        return sys.modules["_mcp_sdk"]
    _sp, init = _find_sdk_init()
    pkg_dir = os.path.dirname(init)
    ours = sys.modules.pop("mcp", None)  # this package, partially initialised
    real = None
    try:
        spec = importlib.util.spec_from_file_location(
            "mcp", init, submodule_search_locations=[pkg_dir]
        )
        real = importlib.util.module_from_spec(spec)
        sys.modules["mcp"] = real  # SDK absolute imports self-resolve while exec'ing
        spec.loader.exec_module(real)
    finally:
        # Re-key SDK modules so they never collide with this package's
        # own ``mcp.*`` submodule names (e.g. our ``mcp.server``).
        renamed: dict[str, object] = {}
        for key in [k for k in sys.modules if k == "mcp" or k.startswith("mcp.")]:
            mod = sys.modules[key]
            if mod is real or (getattr(mod, "__file__", "") or "").startswith(
                pkg_dir + os.sep
            ):
                renamed[key] = mod
        for key, mod in renamed.items():
            new_key = "_mcp_sdk" + key[3:]  # mcp -> _mcp_sdk, mcp.x -> _mcp_sdk.x
            sys.modules[new_key] = mod
            del sys.modules[key]
        # Pydantic resolves forward references (e.g. Settings -> MCPServer)
        # through ``sys.modules[cls.__module__]``; point every SDK-defined
        # class/function at its re-keyed module or those lookups fail.
        for mod in renamed.values():
            for obj in list(vars(mod).values()):
                om = getattr(obj, "__module__", None)
                if isinstance(om, str) and (om == "mcp" or om.startswith("mcp.")):
                    try:
                        obj.__module__ = "_mcp_sdk" + om[3:]
                    except Exception:
                        pass
        if ours is not None:
            sys.modules["mcp"] = ours  # restore this package as top-level `mcp`
    sys.modules["_mcp_sdk"] = real
    return real


SDK = _load()
