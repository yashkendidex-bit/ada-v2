"""Widgets: what A.D.A may put on screen is validated before it leaves the backend."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import widgets as w  # noqa: E402


def test_open_preset_defaults():
    msg, why = w.build_open("gauge", {"label": "Nozzle", "value": 240, "max": 300})
    assert why == ""
    assert msg["t"] == "widget:open" and msg["kind"] == "gauge"
    assert msg["slot"] == "auto" and msg["color"] == "teal" and msg["id"].startswith("gauge-")


def test_props_may_arrive_as_json_text_or_fenced_block():
    msg, _ = w.build_open("tile", '```json\n{"label": "Revenue", "value": 12}\n```')
    assert msg["props"]["value"] == 12
    msg, _ = w.build_open("tile", '{"label": "x", "value": 1}}}garbage')  # one repair
    assert msg and msg["props"]["value"] == 1


def test_unknown_kind_and_missing_props_are_refused():
    assert w.build_open("rocket", {})[0] is None
    msg, why = w.build_open("chart", {"title": "x"})
    assert msg is None and "series" in why
    assert w.build_open("code", {"html": "<p/>"})[0] is None      # code has its own tool
    assert w.build_open("scene", {"elements": []})[0] is None      # so do scenes


def test_colour_is_a_name_or_a_hex_and_nothing_else():
    assert w.clean_color("AMBER") == "amber"
    assert w.clean_color("#00FFaa") == "#00ffaa"
    assert w.clean_color("red; background:url(x)") == "teal"
    assert w.clean_color("#12345") == "teal"
    assert w.clean_color(None) == "teal"


def test_values_are_clamped_and_handlers_dropped():
    raw = {"value": float("inf"), "text": "x" * 5000, "onclick": "alert(1)",
           "innerHTML": "<b>", "__proto__": {"a": 1}, "ok": 5,
           "deep": {"a": {"b": {"c": {"d": {"e": 1}}}}}, "big": list(range(1000))}
    c = w.clean_value(raw)
    assert c["value"] == 0.0 and len(c["text"]) == w.MAX_STR
    assert "onclick" not in c and "innerHTML" not in c and "__proto__" not in c
    assert c["ok"] == 5 and len(c["big"]) == w.MAX_LIST
    assert c["deep"]["a"]["b"]["c"] is None            # depth limit


def test_slot_and_id_are_sanitised():
    msg, _ = w.build_open("clock", {}, slot="NE", wid="my id<script>")
    assert msg["slot"] == "ne" and msg["id"] == "myidscript"
    msg, _ = w.build_open("clock", {}, slot="../../etc")
    assert msg["slot"] == "auto"


def test_update_and_close():
    msg, _ = w.build_update("g1", '{"value": 3}')
    assert msg["t"] == "widget:update" and msg["id"] == "g1" and msg["props"] == {"value": 3}
    assert msg["src"]["verified"] is False          # new numbers with no source are not vouched for
    assert w.build_update("", {"a": 1})[0] is None
    assert w.build_close("all")["id"] == "all"
    assert w.build_close("")["id"] == "all"
    assert w.build_close("a b")["id"] == "ab"


def test_hud_kind_runs_through_the_hud_sanitiser():
    nodes = {"r": {"type": "HudRoot", "props": {"title": "T"}, "children": ["g"]},
             "g": {"type": "Gauge", "props": {"value": 3}, "children": []},
             "x": {"type": "Evil", "props": {}, "children": []}}
    msg, why = w.build_open("hud", {"nodes": nodes})
    assert why == "" and set(msg["props"]["nodes"]) == {"r", "g"}
    assert msg["props"]["root"] == "r"


def test_scene_keeps_only_known_shapes_and_clean_fields():
    els = [{"shape": "ring", "x": 50, "y": 50, "r": 40, "spin": 0.3, "color": "amber"},
           {"shape": "script", "x": 1},
           {"shape": "text", "text": "hi", "color": "url(javascript:x)", "x": 1e99},
           {"shape": "polyline", "points": [[0, 0], [10, 10], "bad"]}]
    msg, why = w.build_scene("T", json.dumps(els), color="#ff8800", width=5000)
    assert why == "" and msg["kind"] == "scene" and msg["color"] == "#ff8800"
    shapes = [e["shape"] for e in msg["props"]["elements"]]
    assert shapes == ["ring", "text", "polyline"]
    assert msg["props"]["elements"][1]["color"] == "main"
    assert msg["props"]["elements"][1]["x"] == 1e9
    assert msg["props"]["width"] == 1000
    assert w.build_scene("T", "[]")[0] is None


def test_code_is_filtered_before_it_leaves_the_backend():
    ok = "<canvas id=c></canvas><script>requestAnimationFrame(function f(){});</script>"
    msg, why = w.build_code("Spinner", ok)
    assert why == "" and msg["kind"] == "code" and msg["props"]["html"] == ok
    for bad in ("<script>fetch('http://x')</script>",
                "<script>new WebSocket('ws://x')</script>",
                "<script>window.parent.postMessage(1,'*')</script>",
                "<script>document.cookie</script>",
                "<iframe src=x></iframe>",
                "<img src='https://evil/x.png'>",
                "<style>@import url(https://evil/x.css)</style>",
                "<script>eval('1')</script>"):
        assert w.build_code("x", bad)[0] is None, bad
    assert w.build_code("x", "")[0] is None
    assert w.build_code("x", "a" * (w.MAX_CODE_BYTES + 1))[0] is None


def test_catalog_prompt_lists_every_preset():
    p = w.catalog_prompt()
    for k in w.KINDS:
        if k not in ("scene", "code"):
            assert k in p, k


# ── real data only ────────────────────────────────────────────────────────
def test_source_must_match_a_tool_that_actually_returned():
    now = 1000.0
    recent = {"get_weather": 970.0, "system_stats": 995.0}
    src = w.make_src("get_weather", recent, now=now)
    assert src == {"name": "get_weather", "age": 30.0, "verified": True}
    assert w.make_src("GET_WEATHER", recent, now=now)["verified"] is True
    assert w.make_src("weather", recent, now=now)["verified"] is True       # a clear part of the tool name
    assert w.make_src("made_up_tool", recent, now=now)["verified"] is False
    assert w.make_src("", recent, now=now) == {"name": "none given", "age": 0.0, "verified": False}
    assert w.make_src("get_weather", {}, now=now)["verified"] is False       # nothing ran this session
    assert w.make_src("user", {}, now=now)["verified"] is True               # Sir said it himself


def test_data_widgets_carry_a_source_and_others_do_not():
    msg, _ = w.build_open("gauge", {"value": 5}, source="system_stats", recent={"system_stats": __import__("time").time() - 2})
    assert msg["src"]["verified"] is True and 1 <= msg["src"]["age"] < 10
    msg, _ = w.build_open("gauge", {"value": 5})
    assert msg["src"]["verified"] is False and msg["src"]["name"] == "none given"
    msg, _ = w.build_open("clock", {})
    assert "src" not in msg                                                  # a clock measures nothing


def test_stale_after_is_clamped_and_optional():
    assert "stale_after" not in w.make_src("user", {})
    assert w.make_src("user", {}, stale_after=99999999)["stale_after"] == 86400
    assert w.make_src("user", {}, stale_after=-5).get("stale_after") is None


def test_a_failed_source_opens_as_an_error_not_as_numbers():
    msg, why = w.build_open("weather", {"error": "weather service timed out"}, source="get_weather")
    assert why == "" and msg["props"]["error"] == "weather service timed out"
    msg, why = w.build_open("weather", {"place": "Pune"})          # no temp and no error: still refused
    assert msg is None and "temp" in why
