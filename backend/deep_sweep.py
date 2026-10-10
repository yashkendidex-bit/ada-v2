"""
Deep Sweep — the Whole-Town Thermal Scan principle (Iron Man 3).

Mass data fusion to a single answer: given a CLAIM or question, Ada sweeps the
web from many angles at once, sweeps Sir's local documents, then cross-analyzes
everything with the deep model chain to deliver a verdict — including the exact
contradicting lines, quoted, with sources.

"Find every paper that contradicts this study" is the canonical use.
"""
import asyncio

SWEEP_ANGLES = [
    "{claim}",
    "{claim} criticism OR rebuttal",
    "{claim} debunked OR false OR corrected",
    "evidence supporting {claim_short}",
    "{claim_short} peer reviewed study",
    "{claim_short} 2025 OR 2026 update",
]

VERDICT_PROMPT = """You are Ada running a DEEP SWEEP — mass verification.
SIR'S CLAIM / QUESTION: {claim}

Below are raw findings from {n_sources} sweep angles (web + local documents).

Reply in EXACTLY this structure (never repeat these instructions):

First, 3-5 spoken sentences aloud to Sir: the verdict — does the evidence
support, contradict, or complicate the claim? Name the strongest piece of
evidence on each side. Plain confident British, no markdown, no headers.

Then one line containing only: ===VERDICT===

Then:
VERDICT: SUPPORTED | CONTRADICTED | MIXED | UNRESOLVED

CONTRADICTIONS (the thermal-scan money shot):
For each source that contradicts the claim: the exact quote (<=40 words),
its source, and one line on why it matters. If none: 'No direct contradictions found.'

STRENGTH OF EVIDENCE: strong / moderate / weak — and why.

WHAT WOULD SETTLE IT: the single experiment, dataset, or document that would
end the debate.

Rules: quote exactly from the findings, never invent. Flag low-credibility
sources. If findings are too thin for a verdict, say UNRESOLVED and say what
to search next."""


def build_sweep_queries(claim: str):
    short = claim if len(claim) <= 60 else claim[:57] + "..."
    return [a.format(claim=claim, claim_short=short) for a in SWEEP_ANGLES]


async def sweep(search_fn, docs_search_fn, sanitize_fn, claim: str, on_progress=None,
                distill_fn=None):
    """Runs the full sweep: parallel web angles + local docs, returns the
    combined evidence block.

    With distill_fn (async text->brief), each angle is distilled by a worker
    before merging — the verdict model reads dense briefs, not raw dumps.
    Without it, behaves exactly as before (truncated raw blocks).
    """
    queries = build_sweep_queries(claim)

    async def web_one(q):
        try:
            res = await search_fn(q, 6)
            return (f"### WEB: {q}\n{(res or '')[:2200]}")
        except Exception as e:
            return f"### WEB: {q}\n(search failed: {str(e)[:80]})"

    async def docs_one():
        try:
            hits = docs_search_fn(claim, 6)
            if not hits:
                return ""
            return "### LOCAL DOCUMENTS\n" + "\n".join(f"- {h}" for h in hits)
        except Exception:
            return ""

    if on_progress:
        await on_progress(f"Thermal scan running: {len(queries)} web angles + local documents...")
    web_blocks, docs_block = await asyncio.gather(
        asyncio.gather(*[web_one(q) for q in queries]),
        docs_one(),
    )
    parts = [b for b in web_blocks]
    if docs_block:
        parts.append(docs_block)

    clean_parts = []
    for p in parts:
        if sanitize_fn:
            p, _flag = sanitize_fn(p)
        clean_parts.append(p)
    if distill_fn and on_progress:
        await on_progress("Distilling each angle to its essence...")
    if distill_fn:
        try:
            from fanout import distill_each, distill_prompt_for
            pre = distill_prompt_for(f"verify or contradict: {claim}")

            async def _distill(block):
                return await distill_fn(pre + block)

            distilled = await distill_each(clean_parts, _distill, budget_chars=1500,
                                           label="sweep-angle")
            clean_parts = [d for d in distilled if d.strip()]
        except Exception as e:
            print(f"[SWEEP] distillation failed ({type(e).__name__}), using raw blocks.")
    return "\n\n".join(clean_parts)


def build_verdict_prompt(claim: str, evidence: str) -> str:
    n = evidence.count("### ")
    return f"{VERDICT_PROMPT.format(claim=claim, n_sources=n)}\n\nRAW EVIDENCE:\n{evidence[:14000]}"
