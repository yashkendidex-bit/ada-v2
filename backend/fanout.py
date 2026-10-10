"""Subagent fan-out (the Anthropic multi-agent pattern).

One big context doing everything is the failure mode: tens of thousands of
exploration tokens pollute the lead model's window. Instead N workers explore
in parallel and each returns a DISTILLED brief (≤budget dense chars). The lead
model only ever sees the distillations — separation of concerns IS memory
management.

Workers are async callables here (search fns, chat fns), not processes: same
ratio, none of the orchestration tax. A failed worker degrades to truncated
raw text, never kills the merge.
"""
import asyncio


async def distill_each(items, distill_fn, budget_chars=1500, timeout=120.0,
                       max_parallel=4, label="angle"):
    """Run distill_fn(item) over all items concurrently; return distilled list.

    distill_fn: async (item_text) -> brief text. Exceptions and timeouts fall
    back to the raw item truncated to budget_chars.
    """
    sem = asyncio.Semaphore(max(1, max_parallel))

    async def one(i, item):
        try:
            async with sem:
                brief = await asyncio.wait_for(distill_fn(item), timeout=timeout)
                brief = (brief or "").strip()
                if brief:
                    return brief[:budget_chars]
        except Exception as e:
            print(f"[FANOUT] {label} {i} distill failed ({type(e).__name__}), using raw.")
        return (item or "")[:budget_chars]

    return await asyncio.gather(*[one(i, it) for i, it in enumerate(items)])


def distill_prompt_for(goal: str, budget_chars: int = 1500) -> str:
    """Wraps raw findings with the distillation contract for a chat_fn."""
    return (f"You are a subagent distiller. Goal: {goal}\n"
            f"Distill the findings below into at most {budget_chars} characters of "
            f"DENSE brief: verdict-relevant facts, exact quotes, numbers, source "
            f"names. Drop filler, drop duplicates, keep contradictions. No preamble, "
            f"no advice — just the concentrated evidence.\n\nFINDINGS:\n")
