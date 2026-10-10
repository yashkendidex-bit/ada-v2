"""
Speech text preparation for ADA's voice pipeline — the "humanized input" layer.

Voxtral (and every modern TTS) sounds robotic when it is fed LLM-formatted
text: markdown asterisks read aloud, "42.5%" mangled, emojis ignored, code
fences recited. Mistral's own guidelines require verbalizable form: numbers
and symbols as words, no rich formatting, abbreviations spelled out, under
300 words per prompt.

This module guarantees that whatever the brain writes, only clean, speakable,
expressive prose reaches her mouth.

    prepare_spoken_text(text) -> pure conversational prose, safe for TTS.

Everything here is deterministic and fast (no model calls) so it can run on
every streamed utterance without adding latency.
"""
import re

# ─── Structural removal ──────────────────────────────────────────────────────
_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_EMOTION_TAG_RE = re.compile(r"\[emotion:\s*([a-z\- ]{1,24})\]", re.I)
_FENCE_RE = re.compile(r"```[^\n]*\n?.*?```", re.S)
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:\-|]+\|?\s*$", re.M)
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.M)
_LINK_RE = re.compile(r"\[([^\]]{1,120})\]\([^)]{0,400}\)")
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_BULLET_RE = re.compile(r"^\s*[-*•·–—]\s+", re.M)
_NUM_ITEM_RE = re.compile(r"^\s*(\d{1,2})[.)]\s+", re.M)
_BLOCKQUOTE_RE = re.compile(r"^\s*>\s?", re.M)
_INLINE_CODE_RE = re.compile(r"`([^`\n]{1,200})`")
_BOLD_RE = re.compile(r"\*\*([^*\n]+)\*\*|\*([^*\n]+)\*|__([^_\n]+)__|_([^_\n]+)_")
_STRIKE_RE = re.compile(r"~~?([^~\n]+)~~?")
_TABLE_PIPE_RE = re.compile(r"\s*\|\s*")

# Emojis / pictographs / dingbats / variation selectors / zero-width joiners
_EMOJI_RE = re.compile(
    "[" "\U0001F000-\U0001FAFF" "\U00002600-\U000027BF" "\U0001F1E6-\U0001F1FF"
    "\U00002B00-\U00002BFF" "\U0000FE00-\U0000FE0F" "\U0000200D" "\U00002190-\U000021FF"
    "\U00002700-\U000027BF" "\U0001F900-\U0001F9FF" "\U00002764" "\U00002728"
    "]+",
)
_EMOTICON_RE = re.compile(r"(?:^|[\s(])(?::\)|:\(|:D|;\)|:P|xD|XD|:<3|<3)(?=[\s.,!?]|$)", re.I)

# Symbols that must become words (or vanish)
_SYMBOLS = [
    (re.compile(r"\s*&\s*"), " and "),
    (re.compile(r"\s*%\s*$"), " percent "),      # fallback; % handled contextually first
    (re.compile(r"\s*°\s*(?=[^CF\d]|$)"), " degrees "),   # stray degree signs
    (re.compile(r"\s+—\s+"), ", "),
    (re.compile(r"\s+--\s+"), ", "),
    (re.compile(r"\s+->\s+"), " to "),
    (re.compile(r"\s+=>\s+"), " gives "),
    (re.compile(r"<3\b"), " love"),
    (re.compile(r"[ \t]*[~^][ \t]*"), " "),
]

# ─── Abbreviations → spoken form ─────────────────────────────────────────────
_LETTER_ABBREVS = {  # hyphenated letter-by-letter (Mistral guideline style)
    "FBI", "CIA", "NASA", "API", "CPU", "GPU", "RAM", "ROM", "USB", "HTML",
    "CSS", "JSON", "SQL", "PDF", "GPS", "CAD", "STL", "STEP", "FEA", "GLB",
    "URL", "HTTP", "HTTPS", "SMTP", "IMAP", "SSH", "VPN", "DNS", "IP", "AI",
    "LLM", "TTS", "STT", "ASR", "NLP", "VR", "AR", "UI", "UX", "SDK", "CLI",
    "GUI", "OS", "PS", "FAQ", "DNA", "OK", "PS5", "Xbox",
}
_WORD_ABBREVS = {
    "e.g.": "for example", "i.e.": "that is", "etc.": "et cetera", "vs.": "versus",
    "approx.": "approximately", "min.": "minutes", "max.": "maximum",
    "w/": "with", "w/o": "without", "b/c": "because", "btw": "by the way",
    "idk": "I don't know", "fyi": "just so you know", "asap": "as soon as possible",
}

# unit -> spoken expansion (applied only when glued to a number)
_UNIT_WORDS = {
    "mm": "millimeters", "cm": "centimeters", "km": "kilometers", "m": "meters",
    "in": "inches", "ft": "feet", "yd": "yards", "mi": "miles",
    "mg": "milligrams", "g": "grams", "kg": "kilograms", "lb": "pounds",
    "lbs": "pounds", "oz": "ounces", "t": "tonnes",
    "ml": "milliliters", "l": "liters", "kl": "kiloliters",
    "ms": "milliseconds", "s": "seconds", "sec": "seconds", "min": "minutes",
    "hr": "hours", "hrs": "hours", "h": "hours", "d": "days",
    "kb": "kilobytes", "mb": "megabytes", "gb": "gigabytes", "tb": "terabytes",
    "kbit": "kilobits", "mbit": "megabits", "gbit": "gigabits",
    "kw": "kilowatts", "mw": "megawatts", "w": "watts",
    "hz": "hertz", "khz": "kilohertz", "mhz": "megahertz", "ghz": "gigahertz",
    "rpm": "revolutions per minute", "n": "newtons", "kn": "kilonewtons",
    "pa": "pascals", "kpa": "kilopascals", "mpa": "megapascals", "gpa": "gigapascals",
    "psi": "pounds per square inch", "bar": "bar",
    "km/h": "kilometers per hour", "kmh": "kilometers per hour",
    "mph": "miles per hour", "m/s": "meters per second", "mps": "meters per second",
}

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]
_SCALE = ["", " thousand", " million", " billion", " trillion"]


def _int_to_words(n: int) -> str:
    if n < 0:
        return "minus " + _int_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("-" + _ONES[n % 10] if n % 10 else "")
    if n < 1000:
        return _ONES[n // 100] + " hundred" + (" " + _int_to_words(n % 100) if n % 100 else "")
    chunk = 0
    while n >= 1000 ** (chunk + 1):
        chunk += 1
    head = n // (1000 ** chunk)
    tail = n % (1000 ** chunk)
    out = _int_to_words(head) + _SCALE[chunk]
    if tail:
        if tail < 100:
            out += " and " + _int_to_words(tail)
        else:
            out += " " + _int_to_words(tail)
    return out


def _year_style(n: int) -> str:
    """2025 -> 'twenty twenty-five', 1999 -> 'nineteen ninety-nine'."""
    hi, lo = divmod(n, 100)
    if lo == 0:
        return f"{_int_to_words(hi)} hundred"
    if lo < 10:
        return f"{_int_to_words(hi)} oh {_ONES[lo]}"
    return f"{_int_to_words(hi)} {_int_to_words(lo)}"


def _decimal_to_words(num_str: str) -> str:
    if "." in num_str:
        whole, frac = num_str.split(".", 1)
        frac_words = " ".join(_ONES[int(d)] for d in frac if d.isdigit())
        return f"{_int_to_words(int(whole))} point {frac_words}" if whole else f"point {frac_words}"
    return _int_to_words(int(num_str))


_CURRENCY_NAMES = {"$": ("dollar", "dollars"), "€": ("euro", "euros"),
                   "£": ("pound", "pounds"), "¥": ("yen", "yen")}
_CENT_SYMBOLS = {"$": "cents", "€": "cents", "£": "pence", "¥": None}


def _verbalize_currency(m) -> str:
    sym, amount, cent_word = m.group(1), m.group(2), m.group(3)
    sing, plur = _CURRENCY_NAMES.get(sym, ("dollar", "dollars"))
    cent_name = _CENT_SYMBOLS.get(sym) or "cents"
    whole, _, frac = amount.partition(".")
    w = int(whole.replace(",", "")) if whole else 0
    f = int(frac) if frac else 0
    if len(frac) == 1:
        f *= 10
    parts = []
    if w:
        parts.append(f"{_int_to_words(w)} {plur if w != 1 else sing}")
    if f:
        parts.append(f"{_int_to_words(f)} {cent_name}")
    if not parts:
        parts.append(f"zero {plur}")
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return parts[0]


_CURRENCY_RE = re.compile(
    r"([$€£¥])\s?(\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)(?:(?<=\d)\s?(cents?))?",
)

# temperature: -4°C / 98.6°F / 300K (K only when preceded by digits & space-less)
_TEMP_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s?°?\s?([CF])\b(?:\.)?", re.I)


def _verbalize_temp(m) -> str:
    val, unit = m.group(1), m.group(2).upper()
    words = _decimal_to_words(val) if "." in val else _int_to_words(int(val))
    name = "degrees Celsius" if unit == "C" else "degrees Fahrenheit"
    return f"{words} {name}"


_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s?%(?!\w)")

_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\s?([AaPp]\.?[Mm]\.?)?")


def _verbalize_time(m) -> str:
    hh, mm, ampm = int(m.group(1)), m.group(2), m.group(3)
    mm_v = _int_to_words(int(mm)) if int(mm) else "o'clock"
    base = f"{_int_to_words(hh)} {mm_v}"
    if ampm:
        ap = ampm.replace(".", "").lower()
        base += " " + ap
    return base


_PHONE_RE = re.compile(r"(?<![\w.])(\+?\d[\d\- ]{7,13}\d)(?![\w.])")


def _verbalize_phone(m) -> str:
    raw = m.group(1).strip()
    digits = [d for d in raw if d.isdigit()]
    plus = raw.startswith("+")
    spoken = " ".join(_ONES[int(d)] for d in digits)
    return ("plus " if plus else "") + spoken


_URL_RE = re.compile(r"https?://\S+|www\.\S+|\b[\w.-]+\.(?:com|org|net|io|ai|dev|gov|edu|co|app)\b/?\S*", re.I)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")


def _spell_domain(url: str) -> str:
    u = url.rstrip("/.,")
    u = re.sub(r"^https?://", "", u, flags=re.I)
    u = re.sub(r"^www\.", "", u, flags=re.I)
    return u.replace(".", " dot ").replace("-", " dash ").replace("_", " underscore")


# number (incl commas & decimals) NOT already consumed by richer patterns above
_NUMBER_RE = re.compile(r"(?<![\w.])(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+\.\d+|-?\d+)(?![\w:]|_\w)")


def _verbalize_number(m) -> str:
    s = m.group(1)
    neg = s.startswith("-")
    s = s.lstrip("+-")
    if neg:
        return "minus " + _number_body(s)
    return _number_body(s)


def _number_body(s: str) -> str:
    plain = s.replace(",", "")
    # Year-like 4-digit numbers get natural year phrasing
    if re.fullmatch(r"\d{4}", plain) and 1100 <= int(plain) <= 2099:
        return _year_style(int(plain))
    return _decimal_to_words(plain)


# ─── Stage 1: structural cleanup ─────────────────────────────────────────────
def _strip_structure(text: str) -> str:
    text = _THINK_RE.sub(" ", text)
    text = _EMOTION_TAG_RE.sub(" ", text)   # never let control tags reach the voice
    # fenced code blocks: keep a short spoken mention of their presence
    def _fence_sub(m):
        first = m.group(0).splitlines()[1] if len(m.group(0).splitlines()) > 1 else ""
        first = first.strip().rstrip(":")[:60]
        return f" Here is the code, shown on screen. " if first else " "
    text = _FENCE_RE.sub(_fence_sub, text)
    text = _TABLE_SEP_RE.sub(" ", text)
    # tables: pipes between cells become spoken separators
    text = _TABLE_PIPE_RE.sub(", ", text)
    text = _IMAGE_RE.sub(" ", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _HEADING_RE.sub("", text)
    # lists become spoken connectives
    text = _BULLET_RE.sub("Also, ", text)
    text = _NUM_ITEM_RE.sub(lambda m: f"Step {m.group(1)}: ", text)
    text = _BLOCKQUOTE_RE.sub("", text)
    text = _INLINE_CODE_RE.sub(r" \1 ", text)
    text = _BOLD_RE.sub(lambda m: next(g for g in m.groups() if g), text)
    text = _STRIKE_RE.sub(r"\1", text)
    return text


# ─── Stage 2: symbol & emoji cleanup ─────────────────────────────────────────
def _strip_symbols_and_emoji(text: str) -> str:
    text = _EMOJI_RE.sub(" ", text)
    text = _EMOTICON_RE.sub(" ", text)
    for rx, repl in _SYMBOLS:
        text = rx.sub(repl, text)
    return text


# ─── Stage 3: verbalization ──────────────────────────────────────────────────
def _verbalize(text: str) -> str:
    text = _CURRENCY_RE.sub(_verbalize_currency, text)
    text = _TEMP_RE.sub(_verbalize_temp, text)
    text = _PERCENT_RE.sub(lambda m: f"{_decimal_to_words(m.group(1))} percent", text)
    text = _EMAIL_RE.sub(lambda m: m.group(0).replace("@", " at ").replace(".", " dot ")
                         .replace("_", " underscore "), text)
    text = _URL_RE.sub(lambda m: _spell_domain(m.group(0)), text)
    text = _TIME_RE.sub(_verbalize_time, text)

    # units glued to numbers BEFORE generic numbers (so 'kg' etc. survive)
    units_alt = "|".join(re.escape(k) for k in sorted(_UNIT_WORDS, key=len, reverse=True))
    unit_re = re.compile(
        r"(?<![\w.])(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+\.\d+|-?\d+)\s?(" + units_alt + r")(?=\W|$)",
        re.I,
    )

    def _verbalize_unit(m):
        num, unit = m.group(1), m.group(2)
        year = _maybe_year(num.lstrip("+-"))
        words = year or _number_body(num)
        return f"{words} {_UNIT_WORDS[unit.lower()]}"

    text = unit_re.sub(_verbalize_unit, text)

    text = _PHONE_RE.sub(_verbalize_phone, text)
    text = _NUMBER_RE.sub(_verbalize_number, text)

    # word abbreviations, then letter-by-letter acronyms
    for k, v in _WORD_ABBREVS.items():
        text = re.sub(re.escape(k) + r"(?=[\s.,;:!?)\]]|$)", v, text, flags=re.I)
    def _letters(m):
        w = m.group(0)
        return "-".join(w)
    pattern = re.compile(r"\b(" + "|".join(sorted(_LETTER_ABBREVS, key=len, reverse=True)) + r")\b")
    text = pattern.sub(_letters, text)
    return text


def _maybe_year(digits: str):
    """Returns year-style words for 4-digit years, else None."""
    if re.fullmatch(r"(19|20)\d\d", digits):
        return _year_style(int(digits))
    return None


# ─── Stage 4: delivery polish ────────────────────────────────────────────────
_MULTI_NEWLINE = re.compile(r"\n{2,}")
_NEWLINE = re.compile(r"\n")
_SPACES_RE = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
_DUP_PUNCT = re.compile(r"([,.;:])\s*\1+")
_RUNS = re.compile(r"([!?]){4,}")
_SENTENCE_SPACE_FIX = re.compile(r"([.!?])([A-Za-z])")


def _delivery_polish(text: str) -> str:
    text = _MULTI_NEWLINE.sub(". ", text)
    text = _NEWLINE.sub(", ", text)
    text = _SPACES_RE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _DUP_PUNCT.sub(r"\1", text)
    text = _RUNS.sub(r"\1\1\1", text)
    text = _SENTENCE_SPACE_FIX.sub(r"\1 \2", text)
    text = text.replace("*", " ").replace("#", " ").replace("`", " ")
    text = re.sub(r"\s+", " ", text).strip()
    # never leave a bare comma/dash ending (TTS trails off awkwardly)
    text = re.sub(r"[\s,;:\-–—]+$", ".", text)
    if text and text[-1] not in ".!?…":
        text += "."
    return text


def prepare_spoken_text(text: str) -> str:
    """Full pipeline: any LLM output -> clean expressive speech prose.

    Idempotent: running it on already-clean text returns near-identical text.
    """
    if not text or not text.strip():
        return ""
    t = text.strip()
    t = _strip_structure(t)
    t = _strip_symbols_and_emoji(t)
    t = _verbalize(t)
    t = _delivery_polish(t)
    return t


def chunk_under_limit(text: str, max_words: int = 290) -> list:
    """Splits long prose into Voxtral-friendly (<300 word) chunks at sentence
    boundaries. Returns a list of strings."""
    words = text.split()
    if len(words) <= max_words:
        return [text] if text else []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, buf = [], []
    count = 0
    for s in sentences:
        sc = len(s.split())
        if buf and count + sc > max_words:
            chunks.append(" ".join(buf))
            buf, count = [], 0
        buf.append(s)
        count += sc
    if buf:
        chunks.append(" ".join(buf))
    return chunks
