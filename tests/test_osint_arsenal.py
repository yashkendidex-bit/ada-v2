"""Tests for the OSINT arsenal: target routing, parsers, availability."""
import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

from osint_tools import (available, _parse_maigret_json, _parse_sherlock_json,
                         _parse_blackbird_json)
from face_osint import reverse_search_links, describe_hunt_plan


def test_availability_detection():
    avail = available()
    assert isinstance(avail, dict)
    assert avail.get("maigret") is True      # we installed it
    assert avail.get("holehe") is True
    assert avail.get("sherlock") is True
    assert avail.get("blackbird") is True
    assert avail.get("phoneinfoga") is True


# ── parsers ──────────────────────────────────────────────────────────────────
def test_parse_maigret_json():
    p = os.path.join(tempfile.mkdtemp(), "m.json")
    with open(p, "w") as f:
        json.dump([
            {"site": "GitHub", "url": "https://github.com/jack", "status": {"found": True}},
            {"site": "MySpace", "url": "https://myspace.com/jack", "status": {"found": False}},
        ], f)
    out = _parse_maigret_json(p)
    assert "GitHub" in out and "https://github.com/jack" in out
    assert "MySpace" not in out


def test_parse_sherlock_json():
    p = os.path.join(tempfile.mkdtemp(), "s.json")
    with open(p, "w") as f:
        json.dump({
            "Twitter": [{"status": "claimed", "url": "https://twitter.com/jack"}],
            "Reddit": [{"status": "illegal", "url": "https://reddit.com/jack"}],
        }, f)
    out = _parse_sherlock_json(p)
    assert "Twitter" in out
    assert "Reddit" not in out


def test_parse_blackbird_json():
    p = os.path.join(tempfile.mkdtemp(), "b.json")
    with open(p, "w") as f:
        json.dump([
            {"site": "Instagram", "url": "https://insta.com/x", "found": True},
            {"site": "Facebook", "url": "https://fb.com/x", "found": False},
        ], f)
    out = _parse_blackbird_json(p)
    assert "Instagram" in out
    assert "Facebook" not in out


# ── face osint ───────────────────────────────────────────────────────────────
def test_reverse_links_empty_without_url():
    assert reverse_search_links("") == []


def test_reverse_links_generated():
    links = reverse_search_links("https://img.example.com/face.jpg")
    names = [n for n, _ in links]
    assert any("Yandex" in n for n in names)
    assert any("Lens" in n for n in names)
    for _, url in links:
        assert "face.jpg" in url


def test_hunt_plan_honest():
    plan = describe_hunt_plan(has_api=False)
    assert "Yandex" in plan and "Search4Faces" not in plan
    plan2 = describe_hunt_plan(has_api=True)
    assert "Search4Faces" in plan2
