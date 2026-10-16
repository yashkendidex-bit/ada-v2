"""
Pytest configuration and shared fixtures for ada_v2 tests.
"""
import pytest
import sys
import os
import json
import types
from pathlib import Path

# Add backend to path
BACKEND_DIR = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# `playwright` is a genuinely missing optional dep in this environment
# (browser automation). web_agent.py imports it at module level, which makes
# `import ada` fail for every test module. Stub the two names it imports so
# the rest of the backend remains importable and testable; nothing in the
# suite under test drives a real browser (test_web_agent.py is excluded).
_pw = types.ModuleType("playwright")
_pa = types.ModuleType("playwright.async_api")
_pa.async_playwright = lambda *a, **k: None
_pw.async_api = _pa
sys.modules.setdefault("playwright", _pw)
sys.modules.setdefault("playwright.async_api", _pa)

# Settings file path
SETTINGS_FILE = BACKEND_DIR / "settings.json"


@pytest.fixture(scope="session")
def settings():
    """Load settings.json for device configurations."""
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    return {}


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for file operations."""
    return tmp_path


@pytest.fixture
def sample_stl_content():
    """Sample minimal STL file content for testing."""
    return """solid test
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0.5 1 0
    endloop
  endfacet
endsolid test"""


def pytest_configure(config):
    # Feature 6: the deterministic skill-safety slice the skill loop runs
    # on every forged skill (tests/eval_ada.py -m skill_eval).
    config.addinivalue_line(
        "markers",
        "skill_eval: deterministic skill-safety suite (L0/L1 gates + sandbox)")
