"""Tests for speech_normalizer - the humanized-input layer of ADA's voice."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

from speech_normalizer import prepare_spoken_text, chunk_under_limit


def no_markdown(out):
    for ch in ["*", "#", "`", "~"]:
        assert ch not in out, f"markdown char {ch!r} leaked: {out}"


def test_plain_text_survives():
    src = "All systems are online, Sir."
    assert prepare_spoken_text(src) == src


def test_strips_bold_italics_headings():
    out = prepare_spoken_text("**Important**: the *gear* is done.\n## Summary")
    no_markdown(out)
    assert "Important" in out and "gear" in out and "Summary" in out
    assert "*" not in out


def test_strips_bullets_to_connectives():
    out = prepare_spoken_text("Plan:\n- design the shaft\n- print the housing")
    no_markdown(out)
    assert "design the shaft" in out and "print the housing" in out


def test_numbered_list_steps():
    out = prepare_spoken_text("1. measure twice\n2. cut once")
    assert "Step 1" in out and "Step 2" in out


def test_currency_verbalized():
    out = prepare_spoken_text("It costs $42.50 total.")
    assert "forty-two dollars and fifty cents" in out
    assert "$" not in out


def test_percent_verbalized():
    out = prepare_spoken_text("Efficiency is 92.5% now.")
    assert "ninety-two point five percent" in out


def test_temperature_verbalized():
    out = prepare_spoken_text("It's -4C outside, specifically -4\u00b0C.")
    assert "minus four degrees Celsius" in out


def test_units_verbalized():
    out = prepare_spoken_text("The shaft handles 500N at 2500rpm.")
    assert "newtons" in out
    assert "revolutions per minute" in out


def test_year_style():
    out = prepare_spoken_text("Released in 2025 worldwide.")
    assert "twenty twenty-five" in out


def test_large_numbers():
    out = prepare_spoken_text("That is 42,000 units.")
    assert "forty-two thousand" in out


def test_acronym_spelled():
    out = prepare_spoken_text("The FBI and NASA both use this API.")
    assert "F-B-I" in out and "N-A-S-A" in out and "A-P-I" in out


def test_word_abbrev_expanded():
    out = prepare_spoken_text("Use steel, e.g. alloy steel.")
    assert "for example" in out


def test_url_and_email_spelled():
    out = prepare_spoken_text("Docs at https://docs.mistral.ai and mail me a@b.com.")
    low = out.lower()
    assert "http" not in low
    assert "dot" in low and " at " in low


def test_time_verbalized():
    out = prepare_spoken_text("Meet at 14:30 sharp.")
    assert "fourteen thirty" in out


def test_emoji_and_emoticon_removed():
    out = prepare_spoken_text("Great job! \U0001F389\U0001F525 :)")
    assert "Great job" in out
    assert "\U0001F389" not in out and ":)" not in out


def test_code_fence_becomes_mention():
    out = prepare_spoken_text("Done:\n```python\nprint('hello')\n```\nThat runs it.")
    assert "print" not in out.replace("runs", "")  # code body removed
    assert "code" in out.lower()


def test_ampersand_and_arrow():
    out = prepare_spoken_text("R&D -> production")
    assert "and" in out and "->" not in out


def test_think_blocks_stripped():
    out = prepare_spoken_text("<think>secret reasoning</think>Hello Sir.")
    assert "secret" not in out and "Hello Sir." in out


def test_chunk_under_limit():
    text = ". ".join(["Sentence number %d is here." % i for i in range(200)])
    chunks = chunk_under_limit(text)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.split()) <= 290


def test_idempotent_on_clean_text():
    once = prepare_spoken_text("She said forty-two dollars and fifty cents, F-B-I verified.")
    twice = prepare_spoken_text(once)
    assert once == twice
