"""
E.D.I.T.H. OSINT Engine — the legal subset (Far From Home).

"Give me a dossier on <name/company/domain>" -> Ada fans out across public
web sources IN PARALLEL, sanitizes everything (Mysterio-proofing: web content
is treated as hostile until proven otherwise), and synthesizes a five-sentence
spoken dossier plus a structured brief for the notebook.

Every run is audit-logged. Public data only. No phone networks were harmed.
"""
import asyncio


def build_queries(target: str, kind: str = "auto"):
    """Parallel search fan-out — different angles, one sweep."""
    t = target.strip()
    if kind == "person":
        angles = [
            f'"{t}" biography background',
            f'"{t}" news 2026',
            f'"{t}" interview profile',
            f'"{t}" controversy OR criticism OR lawsuit',
            f'"{t}" social media profile',
        ]
    elif kind == "company":
        angles = [
            f'{t} company overview founders',
            f'{t} funding revenue news 2026',
            f'{t} products services',
            f'{t} criticism OR controversy OR lawsuit',
            f'{t} competitors market',
        ]
    elif kind == "domain":
        angles = [
            f'{t} website about',
            f'{t} whois owner',
            f'{t} company behind',
            f'{t} reviews reputation',
            f'{t} news',
        ]
    else:  # auto
        angles = [
            f'"{t}" overview background',
            f'"{t}" news 2026',
            f'"{t}" controversy OR criticism',
            f'"{t}" profile history',
            f'what is {t}',
        ]
    return angles


SYNTHESIS_PROMPT = """You are E.D.I.T.H., Ada's global intelligence layer.
Below are RAW web findings about: {target} (type: {kind}).

Reply in EXACTLY this two-part format (never repeat these instructions):

First, 4-5 spoken sentences in a warm British intelligence-briefer tone — the
essence: who/what they are, the most notable fact, one surprising detail, one
caution if any. Plain prose, no headers, no markdown.

Then one line containing only: ===BRIEF===

Then the structured brief:
- Identity: ...
- Key facts: 3-6 bullets with specifics (dates, numbers, names)
- Notable: the surprising/noteworthy detail
- Cautions: controversies, disputes, or data-quality warnings (or 'none found')
- Sources: the originating sites/domains

Rules: only claims supported by the findings. If findings are thin, say so
plainly instead of inventing. Never present speculation as fact."""


def extract_spoken(text: str) -> str:
    """Robustly pulls the spoken paragraph from a synthesized dossier/verdict,
    whatever the model did with the formatting."""
    if not text:
        return ""
    for sep in ("===BRIEF===", "===VERDICT===", "BRIEF:", "VERDICT:"):
        if sep in text:
            return text.split(sep, 1)[0].replace("SPOKEN:", "").replace("SPOKEN", "").strip()
    # Model echoed the instructions? Skip past any instruction-looking preamble.
    lines = text.splitlines()
    out = [l for l in lines if "exactly" not in l.lower() and "instructions" not in l.lower()]
    return "\n".join(out).strip()


async def gather_sources(search_fn, target: str, kind: str, sanitize_fn=None, on_progress=None):
    """Runs all search angles concurrently; sanitizes each result."""
    queries = build_queries(target, kind)

    async def one(q):
        try:
            res = await search_fn(q, 6)
            return (q, res or "")
        except Exception as e:
            return (q, f"(search failed: {str(e)[:80]})")

    if on_progress:
        await on_progress(f"Sweeping {len(queries)} sources on '{target}'...")
    results = await asyncio.gather(*[one(q) for q in queries])
    blocks = []
    for q, res in results:
        flagged_total = False
        if sanitize_fn:
            res, flagged = sanitize_fn(res)
            flagged_total = bool(flagged)
        tag = " [some content removed for safety]" if flagged_total else ""
        blocks.append(f"### {q}{tag}\n{res[:2500]}")
    return "\n\n".join(blocks)


def build_prompt(target: str, kind: str, sources: str) -> str:
    body = SYNTHESIS_PROMPT.format(target=target, kind=kind)
    return f"{body}\n\nRAW FINDINGS:\n{sources[:14000]}"
