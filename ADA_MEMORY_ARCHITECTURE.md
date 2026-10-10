# ADA Memory Architecture — A Biological Design

**Written for Naz.** Two questions drive this: *how does human memory actually work*, and *how do we
give ADA that — but without the forgetting*.

The short answer is that "without the forgetting" is the wrong target, and the research is unusually
clear about why. Total recall is not a superpower. It is a pathology with a name — hyperthymesia —
and people who have it describe it as exhausting and disabling, not enviable. What you actually want
is **selective permanence**: she forgets the mundane so she stays sharp on what matters, and nothing
that matters is ever lost. That distinction shapes every design decision below.

> **Companion document.** `ADA_MEMORY_ENGRAM_PLAN.md` in this same folder covers the same problem from
> a different angle, and the two are meant to be read together rather than as alternatives. That one
> goes deeper on the cellular neurobiology — synaptic tagging and capture, engram allocation, sparsity
> — and gives an **ACT-R base-level activation** retrieval formula (`B_i = ln(Σ t_k^-d)`, d = 0.5)
> that is arguably better than the one used here for a system with real access history. This document
> goes deeper on the *evidence*: what is actually broken in the current code, measured, and which of
> the proposed mechanisms have been ablated and which have not. **Merge them before building.**

---

## Build status (2026-09-15)

**Phases 0–3 are built and tested.** `backend/memory/` is ~3,400 lines across 9 modules;
`agent_kernel.MemoryBank` is now a one-line shim over it. 28 tests pass.

What the build changed versus this plan, all on measurement:

| Plan said | Build found | Decision |
|---|---|---|
| Fix the embedder | `fastembed` (BGE-small, 384-d) lifted realistic recall from 0% to 67% | Then **Mistral `mistral-embed` became the preferred tier** — 1024-d, no local RAM cost, which matters on an 8 GB machine. Local `fastembed` stays as the offline fallback |
| Cross-encoder rerank, on by default | Both ms-marco and bge-reranker-base left realistic recall unchanged, dropped adversarial recall 25%→0%, and broke the confidence gate | **Off by default.** Switchable via `ADA_MEMORY_RERANK=1` |
| Dedup threshold = P99 of pairwise similarity | Percentiles are **not idempotent** — the threshold moves as the set shrinks, so each pass eats more of the store | **Fixed measured threshold 0.965** (exact dup 1.000, punctuation dup 0.979, distinct-by-one-word 0.948) |
| Generative Agents triad, α=1 each, min-max normalised | Min-max **amplifies noise**: with everything recent, a ~1% recency spread stretched to 0..1 and decided the ranking | Relevance decides; recency and importance are bounded tie-breakers (0.12 / 0.08) |
| Confidence floor on the blended score | Every memory in a young store clears a blended floor on recency alone | Confidence from raw channel evidence, plus a **margin against the query's own similarity distribution** so it transfers between embedding models without a second calibration |

**Still open:** maturation and reconsolidation (maths in `decay.py`, not applied by the job — neither
was ablated in the source study, so they stay last), and an ANN index once the store passes a few
thousand rows.

---

## Part 1 — What ADA has today

I read the memory code and then measured the live database rather than trusting the docstrings. The
gap between the two is large.

### The architecture on paper

`MemoryBank` in `backend/agent_kernel.py` (line 104) describes itself as *"layered memory: core
(always in context), recall (searchable history), archival (vector store), episodic (dated summaries
+ entity edges), notebook."* It owns five of the **nine** tables in `ada_kernel/ada.db`:

| Table | Owner | Purpose | Live rows |
|---|---|---|---|
| `core_facts` | MemoryBank | Durable key→value user facts, always injected | **1** |
| `recall_entries` | MemoryBank | Conversation turns + document chunks, with an `embedding` BLOB | 27 |
| `episodes` | MemoryBank | Dated one-line summaries + extracted entities | 27 |
| `entity_edges` | MemoryBank | Co-occurrence graph between entities | **0** |
| `notebook` | MemoryBank | Named freeform sections | 5 |
| `reflections` | Telemetry | Abstracted insights | **0** |
| `scheduled` | Scheduler | Reminders and recurring tasks | 0 |
| `turns` | AudioLoop | Turn log | 36 |
| `audit` | AuditLog | Tool-permission audit trail | 24 |

That is a reasonable *shape*. The problem is what is actually inside it — and note that
`reflections` has never held a single row, because `Telemetry.add_reflection` is defined and never
called. There is no consolidation job anywhere in the codebase.

### The architecture in practice

The row counts above are the whole story: a database that has been in use for weeks holds **one**
fact, **zero** graph edges, and **zero** reflections. Four findings explain why she doesn't feel like
she remembers anything.

**Finding 1 — the embedding model was never installed.** `sentence-transformers` is not in the venv,
so `_Embedder` (line 51) silently falls through to `_tfidf_hash()`: an MD5 hash of each token,
bucketed into 384 dimensions. That is not a semantic embedding; it is a bag of words with no
notion of meaning. I measured what it does:

```
SEMANTIC MATCH (should be HIGH)               similarity = +0.185
  "my drone arm keeps cracking at the motor mount"
  "the quadcopter bracket is failing near where the engine bolts on"

SAME WORDS, OPPOSITE MEANING (should be LOW)  similarity = +0.857
  "the titanium part passed every stress test"
  "the titanium part failed every stress test"

UNRELATED (should be LOW)                     similarity = +0.000
PARAPHRASE (should be HIGH)                   similarity = +0.183
  "remember I always work in metric"  /  "note that I use millimetres"
```

The ranking is **inverted**. A memory that *contradicts* the query scores 0.857 while a memory that
*paraphrases* it scores 0.185. If you ask her whether a part passed its stress test, the broken
retriever's top hit is the one saying it failed. This single defect invalidates the entire retrieval
layer.

**Finding 2 — retrieval is never query-aware.** `build_context(user_text="")` at `agent_kernel.py:930`
takes a query, but `_system_prompt()` at `ada.py:995` calls it with **no arguments**. So every turn
the retrieval runs with an empty query. Look at `recall_episodes`:

```python
if query:                                     # empty string is falsy
    rows = sorted(rows, key=lambda r: -similarity(qvec, ...))
```

With an empty query the similarity sort is skipped entirely and it returns the two most *recent*
episodes. Those are then labelled `"RECENT RELATED MEMORIES"` in the prompt. They are not related to
anything — they are just the newest two.

**Finding 3 — the real vector search is dead code.** `search_recall()` (line 167) is the only method
that does an actual similarity search over the archive. `build_context` never calls it. It is
reachable only if the model *chooses* to call the `recall` tool. The same is true of
`related_entities()` (the graph) and `notebook_read()` — neither appears in `build_context`. She has
a vector store, a knowledge graph, and a notebook, and the context builder reads none of them.

**Finding 4 — the write path is regex-shaped.** `_maybe_extract_facts` (line 265) is six regular
expressions (`my name is (\w+)`, `i like ([\w ]+)`, …). It produced one fact in the entire life of
the database. Entity extraction is `re.findall(r"[A-Z][a-z]{3,}", user_text)` — any capitalised word
of four letters or more — which is why `entity_edges` is empty: a typical turn yields zero or one
"entity", and edges require at least two.

### Verdict

**She has the schema of memory and almost none of the substance.** Facts are captured by regex,
embeddings are meaningless, retrieval ignores the query, the graph and notebook are write-only, and
nothing ever consolidates, decays, or contradicts. The 27 rows in `recall_entries` are written every
turn and read back only as "the last six by timestamp."

This is good news in one specific sense: there is no working system to preserve. The substrate
(SQLite, a clean five-table layout, a write hook on every turn at `ada.py:1172`) is sound, and the
missing pieces are additive.

---

## Part 2 — How human memory actually works

Ten mechanisms, each with the biology and the numbers. These are the things worth copying.

### 2.1 The taxonomy — memory is not one thing

Memory is a set of systems with different lifetimes, different contents, and different rules.

| System | Contents | Lifetime | Conscious? |
|---|---|---|---|
| **Sensory** | Raw percepts (iconic ~250 ms, echoic ~3 s) | Milliseconds–seconds | No |
| **Working** | ~4±1 chunks, actively held | Seconds | Yes |
| **Episodic** (declarative) | Specific events, "what happened to me, when" | Days–decades | Yes |
| **Semantic** (declarative) | Facts divorced from context, "what I know" | Years–permanent | Yes |
| **Procedural** (non-declarative) | Skills, "how to do it" | Permanent, hard to extinguish | No |
| **Priming** (non-declarative) | Prior exposure speeding later processing | Minutes–months | No |
| **Conditioning** (non-declarative) | Learned associations | Long | No |

The critical structural fact: **episodic and semantic memory are separate systems that interact**,
and procedural memory is a different thing again. A design that stores one undifferentiated blob of
"memory" is not brain-like, and — as Part 3 shows — it is also the thing that breaks retrieval.

### 2.2 Encoding — attention is the gate, and depth is the multiplier

Nothing gets stored that isn't attended to. The **von Restorff effect** (distinctive items are
remembered better) and the **serial position curve** (primacy and recency advantages) both fall out
of how attention and rehearsal allocate resources.

**Depth of processing** (Craik & Lockhart, 1972): retention depends on *how* material is processed,
not how long. Shallow processing (typography, rhyme) leaves weak traces; deep processing (meaning,
connection to prior knowledge) leaves strong ones. This is why "elaborative rehearsal" beats rote
repetition.

**Emotional tagging.** The amygdala modulates hippocampal encoding under arousal via noradrenaline
and cortisol. Arousing events get priority storage — this is the mechanism behind "flashbulb"
memories. Critically, arousal is a *salience* signal: it decides what gets stored, and it also
distorts what is stored. The first half is worth copying; the second is not.

**The design lesson:** importance must be computed *at encoding time*, from the content and the
context, and it must be a real signal — not a constant.

### 2.3 The hippocampal trick — separation and completion

The hippocampus does two apparently opposite things:

- **Pattern separation** (dentate gyrus): similar experiences are pushed into *distinct*
  representations, so "the bracket cracked on Tuesday" and "the bracket cracked on Friday" do not
  blur into one memory. Achieved by sparse, high-dimensional expansion coding.
- **Pattern completion** (CA3): a partial or degraded cue reconstructs the whole memory. Show the
  hippocampus a fragment and it returns the entire episode.

These are in tension — one wants memories kept apart, the other wants them merged — and the brain
runs both because they serve different jobs. Separation prevents interference at write time;
completion enables recall from partial cues at read time.

The **hippocampal index** theory (Teyler & DiScenna) adds the crucial piece: the hippocampus does not
store the memory itself. It stores a *pointer* — an index — to the neocortical pattern that
represents the experience. That is why a small structure can index a lifetime: it stores keys, not
contents.

**The design lesson:** separate the *index* from the *content*, keep similar episodes distinct at
write time, and make partial cues work at read time.

### 2.4 Complementary Learning Systems — why you need two speeds

This is the most important theory in the whole document, and the one ADA most needs.

**The problem it solves: catastrophic interference.** A neural network that learns rapidly will
overwrite old representations when it learns something new. Learn task B, forget task A. The brain
obviously doesn't work this way, and CLS (McClelland, McNaughton & O'Reilly, 1995) explains why:

| System | Structure | Learns | Stores | Character |
|---|---|---|---|---|
| **Fast** | Hippocampus | One exposure | Specific episodes | Sparse, pattern-separating, high capacity |
| **Slow** | Neocortex | Thousands of exposures | Statistical regularities | Distributed, overlapping, generalising |

Fast learning is *confined to the hippocampus* so it cannot corrupt the neocortex. The hippocampus
then **replays** episodes to the neocortex over time — a **teacher–student** loop — letting the slow
system extract regularities gradually, interleaved with existing knowledge so nothing is overwritten.

A 2025 replication (Jun et al., arXiv 2507.11393) built this as a VAE (neocortex analog) plus a
Modern Hopfield Network (hippocampus analog) and confirmed the dissociation empirically: the Hopfield
network drove pattern separation, the VAE drove pattern completion, significant in all 30 comparisons
(*p* < 0.001). On Split-MNIST:

| Model | Paradigm | Accuracy |
|---|---|---|
| VAE upper baseline | Offline (all classes at once) | 95.55% |
| VAE control | Sequential, **no replay** | 67.75% |
| **VAE + MHN** | Class-incremental **with replay** | **89.71%** |

Replay recovered ~22 points of the ~28 lost to catastrophic forgetting, landing within 6% of the
offline ceiling. Replay ratio mattered quadratically — both too little and too much hurt, with the
optimum near **1:1**.

**The design lesson:** two stores, two speeds. Write everything to a fast, lossless, append-only
episodic layer immediately. Let a slow, consolidating semantic layer form gradually from *replayed*
episodes. Never let the fast layer write directly into the semantic layer.

### 2.5 Consolidation — and what sleep is actually for

Two stages:

**Synaptic consolidation** (minutes–hours): the cellular stabilisation of a fresh trace, via protein
synthesis and structural change at the synapse. The memory is fragile before this and permanent after.

**Systems consolidation** (days–years): the gradual transfer of dependency from hippocampus to
neocortex. Early memories need the hippocampus; old ones don't. This is why hippocampal damage
spares childhood memories but destroys recent ones.

**Sleep does the work.** During slow-wave sleep, the hippocampus emits **sharp-wave ripples** —
high-frequency bursts in which recently encoded ensembles are *reactivated* in compressed time. A
2025 *Neuron* study showed that large-amplitude ripples specifically promote hippocampo-cortical
memory reactivation. Replay is not random: a 2025 *Nature* paper found replay of recent experience
dominated during ripples in particular pupil-linked micro-states — sleep microstructure *organises*
which memories get replayed. **Targeted memory reactivation** experiments confirm causality: cueing a
memory with an odour or sound during SWS measurably improves its later recall.

REM does a different job — emotional processing and procedural consolidation, with high
acetylcholine and low noradrenaline, which is a chemically distinct mode from waking.

**The design lesson:** consolidation is a *scheduled, offline, batched* process that replays recent
experience into a slower store. It is not something that happens inline during a conversation. ADA
already has the right slot for this: the 02:00 overnight job.

### 2.6 Retrieval — cues, not queries

**Encoding specificity** (Tulving): what you can recall depends on the overlap between the cues
present at retrieval and the context encoded at storage. A memory stored in context A is best
retrieved in context A.

**Context- and state-dependency:** scuba divers who learned words underwater recalled them better
underwater; memories formed while happy are more accessible while happy.

**Retrieval practice (the testing effect):** actively retrieving a memory strengthens it far more
than re-reading it. Roediger & Karpicke (2006): *a single test outperforms repeated study over a
week's delay.* Retrieval is not a read operation — **it is a write operation.**

**The design lesson:** retrieval must be multi-cue (semantic + lexical + temporal + graph), and every
retrieval should update the memory's strength. A read that doesn't write is wasted.

### 2.7 Reconsolidation — memory rewrites itself on recall

Nader et al. (2000): when a consolidated memory is *retrieved*, it re-enters a **labile** state for a
window of roughly minutes to hours, during which it can be modified before re-stabilising. Anisomycin
(a protein-synthesis inhibitor) injected after recall abolished the memory; injected without recall
it did nothing. The memory was only vulnerable *because it had been recalled*.

This is how memories update: each recall is an opportunity to blend in new information. It is also
why human memory drifts — every retrieval is a rewrite, and errors compound.

**The design lesson:** retrieval should be able to update a memory (this is how "knowledge update"
works), but ADA should keep the original so drift is reversible. Humans can't; she can.

### 2.8 Forgetting — the part everyone gets wrong

**The shape.** Ebbinghaus's own data, measured by *savings* (how much faster relearning is):

| Interval | Savings |
|---|---|
| ~20 minutes | ~58% |
| 1 hour | ~44% |
| 1 day | ~34% |
| 1 month | ~21% |

The curve is **not exponential**. Ebbinghaus fit a logarithmic function; across 200+ retention
datasets, power and logarithmic forms consistently beat exponential. The reason is that forgetting
**decelerates** — old memories resist loss. Exponential decay implies a constant proportion lost per
unit time, which the data contradict. Best current fit for individual memories is an
**exponential-power** function (Averell & Heathcote, 2011).

The **permastore** (Bahrick, 1984) is the endpoint: after a few years of loss, retention of
school-learned Spanish *stabilised* and barely changed over subsequent decades regardless of use.

**The mechanisms.** Two distinct processes, not one:

- **Decay** — time-based trace weakening.
- **Interference** — memories competing. *Proactive*: old memories disrupt new learning. *Retroactive*:
  new learning disrupts old memories. Retroactive interference is the stronger of the two, which is
  why the day after studying is worse than the day before.
- **Retrieval-induced forgetting** (Anderson, 2003): recalling one memory actively *suppresses* its
  competitors. Practising some material makes you worse at the unpractised material.

**Forgetting is adaptive.** Schacter's *seven sins* frames every failure as a feature: transience
(frees resources), absent-mindedness (filters noise), blocking (temporary), misattribution,
suggestibility, bias (schemas speed interpretation), persistence (danger signals keep us safe). A
system that never forgets drowns in interference — which is exactly what hyperthymesia looks like
from the inside.

**The design lesson:** implement forgetting as a *deliberate, tunable* process with two independent
components (decay + interference), and a **fidelity ladder** rather than deletion — a memory can be
demoted to a summary, then a gist, then a tombstone that records only *that it existed*.

### 2.9 Spacing and stability — the algorithm that already exists

The practical implementation of all of the above is **FSRS** (Free Spaced Repetition Scheduler),
built on the **DSR** model:

- **Difficulty (D)** — how inherently hard this item is, 1–10.
- **Stability (S)** — memory strength in days; how long until recall becomes hard.
- **Retrievability (R)** — current probability of successful recall.

The forgetting curve is power-law:

```
R = (1 + FACTOR × elapsed_days / stability) ^ DECAY
      DECAY  = -0.5
      FACTOR = 19/81  ≈ 0.2346     (derived from 0.9^(1/DECAY) − 1)
```

Inverted, it gives the next review interval:

```
next_interval = stability / FACTOR × (request_retention ^ (1/DECAY) − 1)
```

**The key property:** a successful recall *increases stability*, which lengthens the half-life. Each
review doesn't just reset the clock — it slows the clock. That is exactly the "sawtooth that flattens"
in the spacing literature, and it's why two spaced reviews across 30 days yield ~56% retention versus
~0.7% from a single unreviewed session.

FSRS ships 19 trainable weights (defaults begin `0.4072, 1.1829, 3.1262, 15.4722, …`) and is
optimisable from review history. We can start with the defaults.

**The design lesson:** this is a solved problem with a public algorithm. ADA's memory should track
stability per memory and let successful recalls lengthen its half-life. Nothing needs to be invented.

### 2.10 Metamemory — knowing what you know

Humans have a **feeling of knowing**: a real, measurable signal that distinguishes "I don't know"
from "I know but can't retrieve it right now" (tip-of-the-tongue states). **Source monitoring** is
the related ability to track *where* a memory came from — and its failure mode, **confabulation**, is
what happens when the brain fills a gap with a plausible fabrication and presents it as recall.

Confabulation is the single most dangerous failure mode for an assistant with memory. A human who
misremembers is unreliable; an assistant who invents a fact *and attributes it to you* is
untrustworthy in a way that poisons everything else.

**The design lesson:** every memory needs a confidence and a source, and "I don't have that" must be
a first-class, preferred output when retrieval is weak.

---

## Part 3 — The mapping: biology → ADA

Each biological mechanism becomes a named component. The right-hand column is the concrete change.

| # | Biological mechanism | Function | ADA component | Concrete implementation |
|---|---|---|---|---|
| 1 | Sensory buffer | Transient raw input | Turn buffer | Keep the raw turn in RAM for the session; promote on write |
| 2 | Working memory (~4 chunks) | Active context | Hot cache | Session-scoped table, sub-second, no embedding needed |
| 3 | Pattern separation | Keep similar episodes distinct | Lossless episode log | Append-only `episodes` with full content + event time. Never merge at write |
| 4 | Pattern completion | Reconstruct from partial cues | Hybrid retrieval | Vector + FTS5/BM25 + graph BFS, then rerank |
| 5 | Hippocampal index | Pointer, not content | Episode→entity→fact edges | `episode_refs` join table; facts cite the episodes that produced them |
| 6 | CLS fast store | Rapid, safe capture | Episodic store | Write every turn immediately, full fidelity |
| 7 | CLS slow store | Gradual generalisation | Semantic graph | Built *only* by the consolidation job, never inline |
| 8 | Sharp-wave replay | Offline rehearsal | Consolidation job | Extend the existing 02:00 `overnight_dossier` slot |
| 9 | Synaptic consolidation | Stabilise a trace | Promotion pass | Score-based: promote top 20%, retain 60%, prune 20% |
| 10 | Depth of processing | Meaning-based encoding | Encoding pipeline | Extract entities + relations + intent at write time, not just text |
| 11 | Amygdala tagging | Salience weighting | Importance scorer | LLM rates 1–10 at encoding (Generative Agents), stored on the row |
| 12 | Power-law decay | Time-based loss | Decay engine | FSRS retrievability, `R = (1 + 0.2346·t/S)^-0.5` |
| 13 | Interference | Competitive loss | Interference pass | Retroactive weight 0.6, proactive 0.4 over similar memories |
| 14 | Fidelity ladder | Graceful degradation | Tiered demotion | L0 full → L2 summary → L3 gist → L5 tombstone. Never hard-delete |
| 15 | Retrieval practice | Recall strengthens | Access log | Every retrieval updates `stability` and `last_access` |
| 16 | Reconsolidation | Recall rewrites | Update-on-read | Labile window; blend new info; keep the prior version |
| 17 | Engram maturation | Slow stabilisation | Activation strength | Sigmoid over ~2 weeks; sub-threshold memories still prime retrieval |
| 18 | Encoding specificity | Cue-dependent recall | Query-aware context | **Fix `build_context` to receive the user's text** |
| 19 | Spreading activation | Associative recall | Graph traversal | 1–2 hop BFS from seed entities |
| 20 | Source monitoring | Provenance | `source_episode_id` | Every fact and edge carries its origin |
| 21 | Feeling of knowing | Calibrated uncertainty | Confidence gate | Below threshold, say "I don't have that" rather than confabulate |
| 22 | Schemas | Fast interpretation | Community summaries | Cluster entities; summarise clusters; retrieve the summary not the members |

---

## Part 4 — The design

### 4.1 Schema

Extend the existing five tables rather than replacing them — the substrate is fine.

```sql
-- Episodic: the lossless, append-only substrate. Pattern separation lives here.
-- Nothing is ever overwritten; a memory degrades in fidelity, it does not vanish.
episodes (
  id, content,                  -- full fidelity text (L0)
  summary, gist,                -- degraded forms (L2, L3)
  fidelity_level,               -- 0..5, see the ladder
  ts_event, ts_ingested,        -- when it happened vs when we learned it
  session_id, turn_index,
  entities,                     -- JSON
  embedding,                    -- real model, 384-dim or larger
  importance, arousal,          -- 1..10, scored at encoding
  access_count, last_access,
  stability, difficulty,        -- FSRS DSR state
  source, quarantined
)

-- Semantic: bi-temporal facts. The old value is never lost, it is invalidated.
semantic_facts (
  id, subject, predicate, object,
  t_valid, t_invalid,           -- event time: when the fact was true
  t_created, t_expired,         -- transaction time: when we believed it
  confidence, importance,
  source_episode_id,            -- provenance
  activation_strength,          -- maturation
  embedding, superseded_by
)

-- Entities with resolution and salience.
entities (
  id, name, canonical_name, type, summary,
  embedding, first_seen, last_seen, mention_count, salience
)

-- Relationships, also bi-temporal.
edges (
  id, src_id, dst_id, rel,
  t_valid, t_invalid, t_created, t_expired,
  confidence, source_episode_id
)

-- Community summaries (the "schema" layer).
communities (id, name, summary, member_entities, embedding, updated)

-- Reflections: the tree. `cites` points down to the evidence.
reflections (
  id, content, level,           -- depth in the reflection tree
  cites,                        -- JSON list of episode/reflection ids
  ts, embedding, importance
)

-- Every retrieval, so stability can be updated (retrieval practice).
access_log (id, memory_id, memory_kind, ts, query, was_useful)

-- Anomalies held back before they poison the store.
quarantine (id, payload, reason, ts, ttl)
```

Three things are load-bearing and easy to get wrong:

1. **`ts_event` vs `ts_ingested`.** Two clocks. Without this you cannot answer "what did I believe
   last Tuesday" or detect a late-arriving contradiction.
2. **`source_episode_id` on facts and edges.** Provenance is the antidote to confabulation.
3. **`fidelity_level`, not a delete flag.** Memory degrades; it does not disappear.

### 4.2 Write path — encoding (runs on every turn)

Currently a turn does: `add_recall` + `add_episode` + six regexes (`ada.py:1172`). Replace with:

1. **Capture raw** — full turn text, `ts_event`, session, into `episodes` at L0. Cheap, synchronous,
   lossless. This is the CLS fast store.
2. **Score** — one cheap-model call returning importance 1–10 and arousal 1–10. Generative Agents
   uses exactly this prompt shape ("where 1 is purely mundane, e.g. brushing teeth; 10 is extremely
   poignant, e.g. a breakup"). Calibration anchors matter: 2 = tidying a room, 8 = asking someone out.
3. **Extract** — entities, relations, and *candidate facts with temporal bounds*, in the same call
   where possible. This is depth of processing: the meaning is captured at write time, not deferred.
4. **Resolve** — match extracted entities against `entities` by embedding + string similarity, then
   an LLM resolution pass on close calls. Merge duplicates, update the summary.
5. **Embed** — real embeddings, one model, consistently.
6. **Contradiction check** — for each new fact, find semantically close existing facts and ask whether
   they conflict. On conflict: set the old `t_invalid = new.t_valid` and `t_expired = now`, mark
   `superseded_by`. **The old fact stays.** This is the bi-temporal invalidation that makes
   "knowledge update" work.
7. **Quarantine anomalies** — out-of-order arrivals, causal inversions, near-duplicates. Hold with a
   short TTL instead of writing garbage.

Keep steps 2–6 off the critical path. The turn must not wait for them.

### 4.3 Consolidation — the "sleep" job

Runs on the existing 02:00 slot (and optionally every 6 hours, which is the interval the Microsoft
paper found optimal). **This is the only writer to the semantic layer.**

The single most important finding in the research, and the one that should shape this entire phase:

> **Deduplicate, do not summarise.** Aggressive consolidation *reduced* LongMemEval accuracy to
> **48.4%**, versus **76.8%** for dedup-only. Over-consolidation is worse than no consolidation.

The measured pipeline:

1. **Temporal validation** — detect duplicates, out-of-order arrival, causal inversion → quarantine.
2. **Promotion scoring** — `S(e) = Σ wᵢ·fᵢ(e)` over recency, frequency, surprise, entity salience,
   outcome. Classify **promote (top 20%) / retain (middle 60%) / prune (bottom 20%)**.
3. **Dedup** — collapse near-duplicates only. Percentile thresholds transfer across domains without
   retuning: near-duplicate ≈ P99 of all-pairs similarity; interference ≈ P90.
4. **Graph integration** — create entities and bi-temporal edges. New semantic nodes start **silent**
   (`activation_strength = 0.0`).
5. **Reflection generation** — when summed importance since the last pass exceeds a threshold
   (Generative Agents uses 150), ask for the 3 most salient questions over recent records, retrieve
   evidence for each, extract insights *with citations*, and store as reflections that point at their
   evidence. Reflections can cite other reflections, forming a tree.
6. **Community refresh** — recluster entities, regenerate summaries. These become retrievable
   "schemas".
7. **Decay pass** — update retrievability for everything via the FSRS curve.
8. **Interference pass** — for high-interference, low-value memories, step down the fidelity ladder.

The Microsoft result worth aiming at: **97.2% retention precision with 58% store reduction**, with
the store self-regulating at **300–500 events** regardless of input volume. That is the shape of a
store that stays sharp.

### 4.4 Retrieval — hybrid, multi-cue, query-aware

The current `build_context` must be rewritten. Its three defects are the empty query, the missing
search, and the recency-only "related" block.

**Step 0 — fix the plumbing.** `_system_prompt()` must pass the user's text through to
`build_context(user_text)`. One-line change, largest single quality gain available.

**Step 1 — score.** Generative Agents' formula, with recency upgraded from a flat exponential to the
FSRS retrievability so that well-rehearsed memories stay accessible:

```
score = α_recency · R(FSRS retrievability)   α_recency   = 1
      + α_importance · importance            α_importance = 1
      + α_relevance · cosine_sim(query)      α_relevance  = 1
```
All three min–max normalised, as in the original.

**Step 2 — hybrid search, three channels.**
- **Semantic** — vector similarity (the fix for Finding 1)
- **Lexical** — SQLite FTS5/BM25. Catches exact names, part numbers, and identifiers that embeddings
  are bad at. Zep found this channel essential.
- **Graph** — 1–2 hop BFS from entities in the query. Spreading activation.

**Step 3 — rerank.** Reciprocal Rank Fusion or MMR for diversity, then an optional cross-encoder.
Prefer MMR: it avoids returning six near-identical memories.

**Step 4 — tier priority.** Hot (this session) → warm (recent episodic) → cold (mature semantic).
Always prefer episodic for recent questions so a not-yet-matured semantic layer never adds latency.

**Step 5 — inject temporal context.** Every retrieved memory carries its timestamp, and the prompt
carries the current date. In the Microsoft study this was worth **+10 percentage points**, consistently,
across every benchmark — the single best ratio of effort to gain in the whole design.

**Step 6 — log the access.** Write to `access_log` and update `stability`. Retrieval is a write.

**Step 7 — gate on confidence.** If the top score is below threshold, she says she doesn't have it.
This is the metamemory layer, and it is what keeps her trustworthy.

### 4.5 Forgetting — deliberate, tunable, non-destructive

Two independent engines, as in the biology:

**Decay:** `I(t) = I₀ · e^(−λt)`. The Microsoft paper's optimal λ = **0.001**, a half-life of ~29 days —
and they note this is *longer than human biology*, because the right horizon follows the rhythm of the
domain, not the circadian cycle. For a design assistant working on projects over months, 29 days is a
sensible starting point.

**Interference:** `I_interference = Σⱼ wⱼ · sim(mᵢ, mⱼ)`, with **retroactive = 0.6, proactive = 0.4**,
reflecting that new learning disrupts old memory more than the reverse.

**The ladder, not the bin:**

| Level | Content | Approx. size |
|---|---|---|
| L0 | Full episodic record | 100% |
| L2 | Summary | 50% |
| L3 | Gist | 25% |
| L5 | Tombstone — records only that a memory existed | 0% |

Degradation is triggered by **age combined with score, never by storage economics**. Disk is cheap;
a lost fact is not.

**What must never degrade:** core facts about the person, their preferences, their projects,
corrections they have given her, and anything they explicitly asked her to remember. Mark these
`importance = 10, pinned = true` and exempt them from every pass.

### 4.6 Maturation — the silent layer

When an episode is consolidated, the full-fidelity record stays immediately retrievable while a
*summarised semantic version* is created with `activation_strength = 0.0`. Strength follows a sigmoid:

```
A(t) = 1 / (1 + e^(−(t − t½)/k))     t½ = 168 h (1 week),  k = 48
```

So a new semantic memory is silent (A ≈ 0.03), crosses the retrieval threshold (A = 0.5) at about a
week, and is fully mature (A > 0.9) at two weeks. Below threshold it still exerts **priming** — it can
influence the relevance scoring of other memories without being surfaced itself, mirroring the
implicit/explicit distinction.

**Be honest about the evidence here.** The Microsoft paper implemented this but could *not* ablate it —
their benchmark gave no memory repeated retrieval over weeks, so every experiment ran with A = 1.0.
This is a design-rationale claim, not a measured win. Build it last, and measure it against your own
usage before trusting it.

### 4.7 Reconsolidation — recall rewrites, reversibly

On retrieval, a memory enters a labile window (default 60 minutes). If new information arrives —
an explicit correction, a contradiction, or elaborative detail — blend it in, weighted by confidence
and recency, and **keep the prior version** with a `superseded_by` pointer.

This is the mechanism behind "knowledge update" questions, and it is where ADA can decisively beat
human memory: humans rewrite and *lose* the original. She rewrites and keeps it. If you tell her on
Friday that the bracket is aluminium after saying titanium on Tuesday, she should say:

> "Noted — aluminium now. You'd told me titanium on Tuesday; I've kept that on record in case you
> want to compare."

**Evidence caveat again:** reconsolidation measured *marginal* in the Microsoft study, because their
benchmark deliberately contained no cross-session contradictions. It matters in proportion to how much
your facts actually change — which, for a design assistant iterating on hardware, is a lot. Expect it
to earn its keep here more than it did in their benchmark.

---

## Part 5 — Where ADA should be better than us

This is the part you asked for, and it is the most defensible part of the design, because each item
below is a place where human memory has a *known, named* failure mode and a machine does not.

| Human limitation | The failure mode | ADA's advantage |
|---|---|---|
| **Reconstructive recall** | Every retrieval rewrites the trace; errors compound across recalls | Keep an immutable L0 substrate *under* the mutable derived layer. She can always return to what was actually said |
| **Consolidation-dependent loss** | Unreplayed memories are lost during sleep | Consolidation demotes, never deletes. Nothing is lost to a bad night |
| **Misattribution / confabulation** | Source monitoring fails; the brain invents and attributes | `source_episode_id` on every fact. She can always show her working |
| **No contradiction history** | Conflicting beliefs coexist or silently overwrite | Bi-temporal invalidation: the old fact is marked false-from-T, not erased |
| **Emotional distortion** | Arousal warps content; mood-congruent recall biases what surfaces | Use arousal as a *salience* weight only. Content stays uncorrupted |
| **Retrieval-induced forgetting** | Recalling X suppresses its competitors — practice makes you worse at the rest | Off by default. Recall never degrades recall |
| **Serial, slow, effortful recall** | One cue at a time, seconds per item | ANN search over the whole life in milliseconds, across all channels in parallel |
| **Co-occurrence-dependent association** | Links form only through repeated pairing | Link at ingest, at zero marginal cost |
| **No introspection** | You cannot inspect or edit your own memory | The store is a SQLite file. Read it, correct it, export it, delete a fact on request |
| **Permastore takes years** | Stability builds only through years of spaced rehearsal | Deliberate rehearsal: she can consolidate on demand, so stability builds in days |
| **Uncalibrated confidence** | Strong feelings of knowing that are wrong | Calibrated confidence per memory, with "I don't know" as a preferred output |

**The honest counterweight — what we should copy rather than beat.** Forgetting is not a bug to
engineer away. It is load-bearing:

- **Interference grows with store size.** More memories means more competition at retrieval. A store
  that never forgets gets *worse* at recall as it fills.
- **Perfect recall is socially wrong.** A friend who never forgets a throwaway comment is not
  attentive, they are unsettling. An assistant that surfaces everything you said six months ago is
  not helpful, it is surveillance.
- **A tombstone beats a void.** Knowing *that* you discussed something, even when the detail is gone,
  is more useful than not knowing it happened.

So the target is not total recall. It is: **she forgets the mundane on a principled schedule, and
nothing that matters is ever lost.** Every forgetting mechanism in Part 4 is built to be inspectable,
tunable, and reversible — and pinned facts are exempt from all of it.

---

## Part 6 — Implementation plan

Six phases. Phase 0 is not optional: the retrieval layer is currently inverted, and nothing built on
top of it will work.

### Phase 0 — Fix the foundation ⭐

*Nothing else is worth building until this is done.*

| Change | File | Why |
|---|---|---|
| Install a real embedding model, and **fail loudly** if absent | `requirements.txt`, `agent_kernel.py:51` | The MD5 fallback is why retrieval is inverted. A silent fallback to a meaningless embedder is worse than a crash |
| Pass the user's text into `build_context` | `ada.py:995` | One line. Turns retrieval from recency-only into query-aware |
| Add an ANN index (sqlite-vec or FAISS) | `agent_kernel.py:167` | The O(n) full scan is the scaling wall |
| Add FTS5 + BM25 alongside the vector search | new | Exact names and part numbers; embeddings are bad at them |
| Stop labelling recency results as "related" | `agent_kernel.py:944` | She is currently told two arbitrary memories are relevant to the question |

**Acceptance:** the four probe pairs from Finding 1 return sane orderings — paraphrase above
contradiction, unrelated near zero.

### Phase 1 — Encoding pipeline ⭐⭐

Replace the six regexes with real extraction: importance + arousal scoring, entity and relation
extraction, temporal bounding, entity resolution, and contradiction detection with bi-temporal
invalidation. Run it off the critical path.

**Acceptance:** after 50 real conversations, `core_facts` holds tens of accurate facts (not one),
`entity_edges` is populated, and every fact has a `source_episode_id`.

### Phase 2 — Consolidation job ⭐⭐

Extend the existing 02:00 slot into the full pipeline: temporal validation, promotion scoring,
**dedup-not-summarise**, graph integration, reflection generation with citations, community refresh.

**Acceptance:** store size stays bounded while retention precision stays high; reflections cite their
evidence; running it twice in a row changes nothing (idempotence).

### Phase 3 — Forgetting engine ⭐⭐

Decay, interference, and the fidelity ladder. Ship it with the pinned-fact exemption from day one —
never retro-fit that.

**Acceptance:** low-value memories visibly degrade through the ladder; pinned facts are provably
untouched; the store stops growing without bound.

### Phase 4 — Maturation and reconsolidation ⭐⭐⭐

The sigmoid activation strength and the labile update window. **Build these last and measure them
honestly** — the published evidence for both is weak (no ablation), and they add real complexity.

**Acceptance:** a controlled test showing maturation improves precision *or* an honest decision to
remove it. Both are acceptable outcomes.

### Phase 5 — Metacognition and reflection ⭐⭐

Confidence-gated responses ("I don't have that"), provenance in answers ("you told me on 12 August"),
and the reflection tree feeding back into retrieval. Also wire up `add_reflection`, which currently
writes reflections that nothing ever reads.

**Acceptance:** she declines to answer when retrieval is weak instead of confabulating, and she cites
dates and sources unprompted.

---

## Part 7 — Risks, and what not to build

1. **Over-consolidation is destructive.** Measured: aggressive consolidation scored 48.4% versus 76.8%
   for conservative dedup. **Deduplicate, do not summarise.** When in doubt, keep more.
2. **Don't let her invent memories.** Every fact needs a `source_episode_id`. A memory layer without
   provenance is a confabulation engine with a straight face.
3. **Don't make recall degrade recall.** Leave retrieval-induced forgetting off unless there's a
   measured reason to enable it.
4. **Cost discipline.** Scoring and extraction are LLM calls. Use the cheapest model that works, batch
   them, and keep them off the conversation path. The Microsoft paper's lifecycle mechanisms are
   deliberately *deterministic algorithms* — no LLM calls in decay, interference, or promotion.
5. **Privacy is a design constraint, not an afterthought.** A perfect memory of everything is a
   liability. Ship `forget` as a real, complete deletion (including derived facts and reflections that
   cite it), and give the user a way to read and edit the store. A memory you cannot inspect is one
   you cannot trust.
6. **Don't chase hyperthymesia.** The goal is a colleague who remembers what matters, not a
   transcript with a voice.
7. **Measure against your own usage, not a benchmark.** LongMemEval is a single linear conversation
   with no repeated actions and no evolving facts — an adversarial case for this design. Your real
   workload is the opposite: recurring projects, changing specs, a person whose preferences shift.
   Build the eval from your own transcripts.

---

## Part 8 — How to know it's working

Six checks, in order of how much they'd tell you:

1. **The probe set.** The four pairs from Finding 1, run as a regression test. Currently failing.
2. **Contradiction handling.** Tell her a fact, change it a week later, ask about it. She should report
   the new value *and* know the old one existed.
3. **Continuity across sessions.** Reference something from three sessions ago, casually, mid-sentence.
   This is the felt experience of memory and the whole point of the exercise.
4. **Provenance.** Ask "how do you know that?" She should name the date and the conversation.
5. **Honest gaps.** Ask about something never discussed. She should say she doesn't have it — not
   produce something plausible.
6. **Store discipline.** After a month of heavy use: bounded size, high retention precision, pinned
   facts intact, no unbounded growth.

---

## Sources

**Human memory**
- Craik & Lockhart (1972), depth of processing
- Tulving & Thomson (1973), encoding specificity
- Teyler & DiScenna (1986), the hippocampal index theory
- McClelland, McNaughton & O'Reilly (1995), Complementary Learning Systems
- Nader et al. (2000), reconsolidation and the labile window
- Anderson (2003), retrieval-induced forgetting
- Bahrick (1984), permastore
- Roediger & Karpicke (2006), the testing effect
- Schacter (2001), *The Seven Sins of Memory*
- Averell & Heathcote (2011), exponential-power forgetting curves
- Murre & Dros (2015), replication of the Ebbinghaus curve
- Jun, Marupudi, Shah & Varma (2025), CLS as VAE + Modern Hopfield Network (arXiv 2507.11393)
- *Neuron* (2025), large sharp-wave ripples and hippocampo-cortical reactivation
- *Nature* (2025), sleep microstructure organises memory replay

**Agent memory**
- Park et al. (2023), Generative Agents — the recency/importance/relevance retrieval formula
- Packer et al. (2023), MemGPT / Letta — memory paging
- Rasmussen et al. (2025), Zep / Graphiti — bi-temporal knowledge graphs (arXiv 2501.13956)
- Kerestecioglu, Robsky, Vasters, Sharma & Kesselman (2026), *Human-Inspired Memory Architecture for
  LLM Agents* (Microsoft, arXiv 2605.08538) — the six-mechanism architecture, dedup-not-summarise,
  and the calibration thresholds
- Mem0, LangMem, Zep comparison (LongMemEval, DMR)
- *Agentic Context Engineering* (ACE) — Generator → Reflector → Curator
- FSRS / DSR model, open-spaced-repetition

**This codebase**
- `backend/agent_kernel.py:51` (`_Embedder`), `:104` (`MemoryBank`), `:167` (`search_recall`),
  `:265` (`_maybe_extract_facts`), `:930` (`build_context`)
- `backend/ada.py:995` (`_system_prompt`), `:1172` (`remember_context` call site)
- `ada_kernel/ada.db` — measured live state
