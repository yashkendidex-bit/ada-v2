# ENGRAM — Replicating Human Memory in ADA, and Beating It

### A hyper-deep plan for Naz
**Goal:** give Ada a memory that works *like* human memory — same architecture, same stages, same trade-offs — while removing every failure mode that makes human memory unreliable. We forget. She must not.

---

# PART I — HUMAN MEMORY: THE ACTUAL BIOLOGY

Everything below is real, named, and measurable. No hand-waving.

## 0. The one-paragraph model

Human memory is not a filing cabinet. It is a **reconstructive, multi-system, sleep-dependent, interference-managed prediction engine**. Experience activates a **sparse set of neurons**; a subset is *allocated* as an **engram** by molecular tagging; synapses among them strengthen (Hebbian LTP); over hours the trace stabilises (cellular consolidation); over nights and weeks it is **replayed** and its dependence shifts from hippocampus to cortex (systems consolidation); remembering it later makes it briefly **labile again** so it can be updated (reconsolidation); and recall is not playback but **pattern completion** from partial cues, which is why it is fast, associative, and *wrong surprisingly often*.

That is the machine we are copying.

---

## 1. The timescale ladder

| Store | Duration | Capacity | Substrate | Fails when |
|---|---|---|---|---|
| **Iconic (visual)** | ~250–500 ms | large, fragile | visual cortex | never attended |
| **Echoic (auditory)** | ~2–4 s | large | auditory cortex | never attended |
| **Short-term / working** | ~15–30 s unrehearsed | **~4 chunks** (Cowan), not 7±2 | prefrontal + parietal, basal ganglia gating | displacement, interference |
| **Long-term, declarative** | years–life | enormous | hippocampus → neocortex | retrieval failure, interference |
| **Long-term, procedural** | years–life | enormous | basal ganglia, cerebellum, motor cortex | rarely; very slow to change |

**Design consequence:** Ada needs *all five tiers*. Today she has exactly one — a flat SQLite table scanned by brute-force cosine.

---

## 2. Sensory buffers (the 3-second window)

Sperling's partial-report work established a high-capacity **iconic store** that vanishes in under half a second; the **echoic store** holds a few seconds. Their only job is to give attention time to arrive. Everything not attended in that window was never encoded — it was never *there*, cognitively.

**Ada analogue:** rolling ring buffers — last 30 s of mic audio, last 60 s of screen OCR, last 5 camera frames. Cheap, in RAM, mostly discarded. But when you say *"what did I just…"* or *"undo that"*, the buffer is still warm and she can answer instantly instead of guessing. **This is the tier she is completely missing and costs almost nothing to build.**

---

## 3. Working memory — the global workspace

Baddeley's multi-component model, still the standard:

- **Central executive** — limited-capacity attention controller (prefrontal)
- **Phonological loop** — verbal/auditory scratchpad (~2 s of speech)
- **Visuospatial sketchpad** — visual/spatial scratchpad
- **Episodic buffer** (added 2000) — binds the above into a single multimodal episode with time-ordering, and is the interface to long-term memory

Cowan's 2001 review cut the capacity number from Miller's 7±2 to **about 4 independent chunks**. Chunking is the escape hatch: experts hold more because their chunks are bigger.

**Ada analogue:** a bounded **global workspace** (~8 slots) that is the *only* thing in the model's context window. Everything else must be retrieved. This is what stops context bloat and forces the memory system to actually work. Her current `recent_exchanges(k=8)` is a crude version of this — keep it, formalise it.

---

## 4. Encoding — the gate that decides what gets to exist

This is where humans lose the most information, and where Ada can beat us outright.

### 4.1 Attention is the gate
No attention, no encoding. Maintenance rehearsal (repeating) is weak; **elaborative rehearsal** (relating to what you know) is strong — Craik & Lockhart's *levels of processing*.

### 4.2 Neuromodulators decide what matters
- **Novelty / prediction error** → VTA dopamine + **locus coeruleus norepinephrine** fire, marking "this is new, write it down"
- **Arousal / emotion** → **amygdala** + noradrenergic signalling strengthens consolidation; also narrows attention (weapon focus — you remember the gun, not the face)
- **Reward / surprise** → dopamine gates hippocampal encoding

### 4.3 Synaptic tagging and capture
A weak experience sets a *tag*; a nearby strong experience triggers *plasticity-related proteins* that get **captured** by the tag. Result: a trivial event adjacent to a salient one gets consolidated too. This is why context matters.

### 4.4 Engram allocation — who gets recruited
Not all active neurons become the memory. Allocation is governed by:
- **Intrinsic excitability** — the more excitable neurons at the moment of learning win the slot
- **Immediate-early-gene tagging** — **c-fos**, **Arc** mark recently active cells; researchers hijack these promoters to drive ChR2 and later *optogenetically reactivate a specific memory* (Ramirez & Liu; and they implanted a **false** fear memory this way)
- **CREB** — transcription factor; high-CREB neurons are preferentially allocated
- **Inhibitory balance** — mIPSC increase balances mEPSC increase, keeping ensembles specific and preventing runaway excitation
- **Timescale:** synaptic input increases within hours; **intrinsic excitability rises progressively over the first day**

### 4.5 Sparsity — and the counter-intuitive part
The engram is sparse. But recent hippocampal autoencoder modelling finds **optimal capacity when sparsity is *weakly* enforced, not maximised**. Over-sparsify and you lose; under-sparsify and memories interfere.

**Ada analogue:** an explicit **salience score** at write time, computed from: prediction error (did this contradict what she expected?), user emphasis ("remember this"), repetition, goal-relevance, emotional load, and recency of related tags. Everything gets *stored* — nothing is dropped at the door — but salience sets the **initial activation**, which determines how easily it surfaces later. Humans lose unattended input forever; Ada archives it at low activation and can still surface it with the right cue. **Guarantee #1.**

---

## 5. Consolidation — three distinct processes people conflate

### 5.1 Cellular / synaptic consolidation (hours → first day)
- **LTP**: NMDAR calcium influx → **AMPA receptor** upregulation and trafficking → bigger postsynaptic response
- **Structural**: increased **dendritic spine density**; **PSD-95** scaffolding is required for *long-term stability* of engram connectivity (PSD-95 knockout → more LTP but impaired spatial learning)
- **Refinement**: unused neurons are **pruned**, relevant connections strengthened; the engram becomes **sparser and more robust over time**, and **fewer cells are needed at recall** as it matures

### 5.2 Systems consolidation (nights, weeks, years)
- Dependence shifts from **hippocampus → medial prefrontal cortex / neocortex**
- Mechanism: offline **replay** of ensemble activity, organised by **hippocampal sharp-wave ripples (SWRs)** coordinated with **cortical sleep spindles** and **slow oscillations**. A 2025 *Neuron* paper identified a specific subset of **large SWRs** linked to reactivation in both hippocampus *and* prefrontal cortex — i.e. we can now see the transfer happening.
- **Targeted memory reactivation (TMR)**: replaying a cue during slow-wave sleep measurably strengthens that memory. Sleep is not rest; it is a **batch job**.

### 5.3 Reconsolidation (minutes, every time you remember)
Retrieval **destabilises** the trace. It must be **restabilised** (protein synthesis again) or it degrades. This is a *feature*: it is the brain's only mechanism for **updating** an existing memory with new information. It is also why eyewitness testimony is contaminable — every retelling rewrites it slightly.

**Ada analogue:** the **nightly consolidation job** (see Part III) plus an explicit **reconsolidation path**: when a retrieved memory is contradicted, mark it labile, write the correction *as a new version linked to the old*, restabilise, and keep the old version retrievable. Humans overwrite; Ada versions. **Guarantee #2.**

---

## 6. The long-term stores — it is not one thing

Squire's taxonomy, and the dissociations that prove it:

| System | What it holds | Substrate | Proof of independence |
|---|---|---|---|
| **Episodic** | events, with time/place/self | hippocampus (index) | H.M., Clive Wearing: cannot form new episodes |
| **Semantic** | facts, meaning, concepts | neocortex (temporal) | Patient K.C.: semantic without episodic; semantic dementia: reverse |
| **Procedural** | skills, habits, sequences | basal ganglia, cerebellum | H.M. improved at mirror-tracing with **no memory of practising** |
| **Priming / perceptual** | facilitated processing | neocortex | intact in amnesia |
| **Conditioning, emotional** | associations, fear | amygdala, cerebellum | intact in amnesia |
| **Prospective** | *intentions* — to do X later | prefrontal + hippocampal | the one assistants most need and never build |

Two lessons that matter enormously for Ada:

1. **Procedural memory is separate and far more durable.** You never forget how to ride a bike. Ada has the *seed* of this already: `cad_v3/vocabulary/` contains 12 hand-written and 10 LLM-synthesised **skills** that passed a geometry gate. That is a procedural store. It should be formalised and extended to daily work — "how you like a brief written", "how you run a standup".
2. **Episodic and semantic are genuinely different and need different tables.** Today `episodes` and `recall_entries` are barely distinguishable, and `core_facts` (semantic) has **one row**.

---

## 7. Retrieval — pattern completion, not playback

### 7.1 The mechanism
- **Ecphory**: a partial cue reactivates the whole engram. **CA3's** recurrent collaterals do **pattern completion**; the **dentate gyrus** does **pattern separation** (keeping similar memories from colliding). Human high-resolution fMRI has confirmed both computations in CA3/DG.
- **Encoding specificity** (Tulving): retrieval works best when the cue matches the *encoding context*, not the "meaning". Context — including internal state — is part of the memory.
- **Spreading activation**: activating one node primes its neighbours for a few hundred ms. This is free associative recall and it is why the entity graph matters.
- **Reconstruction**: remembering is *rebuilding from fragments*, then filling gaps with plausible defaults. That is the origin of confabulation and of Loftus's misinformation effect.

### 7.2 The math (ACT-R) — use this verbatim
ACT-R's declarative memory gives us a working quantitative model:

```
Base-level activation:   B_i = ln( Σ_k  t_k^(−d) )
```
- `t_k` = time since the k-th presentation/use of chunk i
- `d` = decay, default **0.5**
- `e^(B_i)` is the **odds that chunk i is needed right now**
- Retrieval succeeds if activation ≥ threshold τ; **recall latency ≈ F · e^(−B_i)**

This single equation reproduces: the **power law of forgetting**, the **spacing effect** (spaced rehearsals beat cramming), and the fact that *each successful retrieval counts as a new presentation* — remembering something strengthens it.

**Ada analogue:** every memory row carries `activation` computed from this formula, updated on every retrieval. Retrieval ranking = activation + semantic match + BM25 + entity match + context match. Memories below threshold are not deleted — they are **dormant** (the biological equivalent of a *silent engram*, which exists physically and can be reactivated artificially even though no natural cue reaches it). Ada can always "optogenetically" reactivate a dormant memory with the right query. **Guarantee #3.**

### 7.3 Metamemory — knowing what you don't know
Humans have **feeling-of-knowing** and **tip-of-the-tongue** states: they can tell retrieval has *failed* without knowing the answer, and can report confidence. This is a monitoring system over the memory system, and it is what stops us from stating garbage as fact.

**Ada analogue:** every recall returns a **confidence** and a **provenance** (who said it, when, source). Below a confidence floor she says *"I don't have that, Sir"* instead of inventing. Humans confabulate under uncertainty; Ada must refuse. **Guarantee #4.**

---

## 8. Forgetting — mostly a feature

- **Decay** is real but is the smaller effect; **interference** is the larger one (proactive: old blocks new; retroactive: new blocks old)
- **Retrieval-induced forgetting**: retrieving one item *actively suppresses* competitors. *Nature* (2014) framed this as **adaptive forgetting** — the brain is optimising retrieval, not failing
- **Extinction is not erasure**: the original engram is weakened and a **new inhibitory "safety" engram** is formed. The old trace is still there
- **Spacing and testing** flatten the forgetting curve far more than raw repetition

**Ada analogue:** she gets the *benefits* of forgetting (fast retrieval, no interference, no context bloat) by **decaying activation**, while keeping the *costs* at zero by never deleting the underlying record. Humans must trade off; Ada gets both sides. **Guarantee #5.**

---

## 9. Sleep — the nightly batch job

Slow-wave sleep → hippocampal replay coordinated with cortical spindles → transfer to cortex. TMR during SWS enhances consolidation. REM appears to contribute differently (representational change, emotional integration).

**Ada analogue: the 02:00 consolidation job.** She already has `overnight_dossier.py` with `RUN_HOUR = 2`. Extend it into a full NREM/REM cycle:

- **NREM phase (replay + transfer):** re-score the day's episodes, extract facts and entities, merge duplicates, strengthen what was retrieved, decay what wasn't
- **REM phase (integration + imagination):** build schemas, find contradictions, generate tomorrow's briefing, propose actions
- **Dream phase (optional):** run `cad_overnight.evolve_overnight` on open designs

---

## 10. Why memory exists at all

The **constructive episodic simulation hypothesis** (Schacter & Addis): episodic memory's real job is not nostalgia — it is to **flexibly recombine past fragments to simulate future scenarios**. Memory exists to *predict*.

**Ada analogue:** memory must feed **planning**, not just chat. Every retrieval should be usable by the planner. If her memory only helps her answer questions, we built a diary, not a Jarvis.

---

# PART II — THE TEN FAILURES OF HUMAN MEMORY (Ada's spec)

| # | Human failure | Biological cause | Ada's guarantee |
|---|---|---|---|
| 1 | Never encoded it | attention gate | Sensory buffers + always-on capture; salience scored, never dropped |
| 2 | Capacity ceiling | ~4 chunks | Unbounded storage, activation-ranked retrieval |
| 3 | Decay | power-law forgetting | Activation decays; **records never do**. Any dormant memory is retrievable |
| 4 | Interference | proactive/retroactive | Pattern separation (dedupe + versioning), never overwrite |
| 5 | Tip-of-the-tongue | retrieval failure, storage intact | Multi-signal retrieval; if all signals fail, indexed full-text search |
| 6 | Confabulation | reconstructive gap-filling | Confidence floor + provenance; she says "I don't know" |
| 7 | Updating destroys history | reconsolidation overwrites | Every update is a **new version**; old ones stay linked |
| 8 | Sleep loss wrecks consolidation | no replay | Deterministic nightly job; never skipped, never degraded |
| 9 | Emotion distorts | amygdala narrowing | Salience stored *as metadata*, separated from content accuracy |
| 10 | Death / no backup / no transmission | biology | Exportable, diff-able, encrypted, portable |

---

# PART III — THE DESIGN: **ENGRAM**

> **E**ncoding → **N**ightly consolidation → **G**raph → **R**etrieval → **A**ctivation → **M**etamemory

## 3.1 Substrate map

| Biology | Ada component | Status |
|---|---|---|
| Sensory buffers | `SensoryRing` — 30 s audio / 60 s screen / 5 frames | **build** |
| Central executive + episodic buffer | `Workspace` — 8 slots, the only context | **formalise** (`recent_exchanges`) |
| Engram allocation / IEG tagging | `EncodingGate` salience scorer | **build** |
| Hippocampal index | `episodes_v2` — events + pointers + context | **upgrade** (exists, 1 field) |
| Neocortex (semantic) | `facts_v2` — versioned, confidence, provenance | **upgrade** (`core_facts` has 1 row, overwrites) |
| DG pattern separation | `Deduper` — near-duplicate detection on write | **build** |
| CA3 pattern completion | `PatternComplete` — spreading activation over entity graph | **build** (`entity_edges` has **0 rows**) |
| Basal ganglia (procedural) | `skills` — workflows & preferences | **formalise** (`cad_v3/vocabulary` is the prototype) |
| Amygdala (salience) | `affect` column — emotion/importance as metadata | **build** |
| Sleep / SWR / spindles | `Consolidator` at 02:00 | **extend** (`overnight_dossier` exists, single-purpose) |
| Reconsolidation | `Reconsolidate` — version on contradiction | **build** |
| Metamemory / FOK | `Confidence` + provenance on every recall | **build** |
| Silent engrams | `archive` — dormant but reactivatable | **build** |

## 3.2 The activation equation, Ada-flavoured

```python
def activation(m, now):
    # ACT-R base level: power-law decay over every past retrieval
    base = math.log(sum((now - t) ** -DECAY for t in m.retrieval_times) or 1e-9)
    # Ada-only terms biology can't do:
    salience  = m.salience                     # set at encode time
    assoc     = entity_overlap(m, workspace)   # spreading activation
    return base + W_SAL * salience + W_ASSOC * assoc

DECAY = 0.5          # ACT-R default; tune on your own data
THRESHOLD = -2.0     # below this: dormant, not deleted
```

Retrieval score = `α·activation + β·semantic_cosine + γ·bm25 + δ·entity_match + ε·context_match + ζ·recency`, then **rerank** the top ~50 with a cross-encoder. Writes are **async** so they never add latency to her voice.

## 3.3 Schema (extends the existing 9 tables, does not replace them)

```sql
-- Hippocampal index: what happened, with full context
CREATE TABLE episodes_v2 (
  id INTEGER PRIMARY KEY, t REAL, kind TEXT, summary TEXT,
  actors TEXT, location TEXT, affect REAL, salience REAL,
  source TEXT, context_json TEXT, embedding BLOB,
  activation REAL DEFAULT 0, last_retrieved REAL
);

-- Neocortex: versioned facts. NEVER overwritten.
CREATE TABLE facts_v2 (
  id INTEGER PRIMARY KEY, subject TEXT, predicate TEXT, object TEXT,
  value TEXT, confidence REAL, source TEXT, first_seen REAL,
  valid_from REAL, valid_to REAL, supersedes INTEGER, embedding BLOB,
  activation REAL DEFAULT 0, retrieval_count INTEGER DEFAULT 0
);

-- The missing organ: entity graph (currently 0 rows)
CREATE TABLE entities (
  id INTEGER PRIMARY KEY, name TEXT UNIQUE, type TEXT,
  salience REAL, created REAL, last_mentioned REAL
);
CREATE TABLE entity_edges_v2 (
  a INTEGER, b INTEGER, rel TEXT, weight REAL,
  first_seen REAL, last_seen REAL, count INTEGER,
  PRIMARY KEY (a, b, rel)
);

-- Procedural: how you work
CREATE TABLE skills (
  name TEXT PRIMARY KEY, trigger TEXT, steps_json TEXT,
  uses INTEGER DEFAULT 0, success_rate REAL, updated REAL
);

-- Sleep log: prove consolidation actually ran
CREATE TABLE consolidation_runs (
  id INTEGER PRIMARY KEY, started REAL, finished REAL,
  episodes_in INTEGER, facts_extracted INTEGER, entities_linked INTEGER,
  contradictions INTEGER, pruned INTEGER, archived INTEGER
);
```

## 3.4 The nightly job — NREM/REM at 02:00

```
02:00  NREM/REPLAY
       ├─ replay day's episodes in order
       ├─ extract facts with confidence + provenance      → facts_v2
       ├─ extract entities + typed relations              → entities, entity_edges_v2
       ├─ dedupe / merge near-duplicates (pattern separation)
       ├─ strengthen what was retrieved today (spacing effect)
       └─ decay everything else by the power law

02:20  REM/INTEGRATION
       ├─ contradiction scan → reconsolidation queue
       ├─ schema update: recurring patterns → skills
       ├─ generate tomorrow's briefing
       └─ write reflection to `reflections` (table exists, 0 rows)

02:40  DREAM (optional)
       └─ cad_overnight.evolve_overnight on open designs

       → write consolidation_runs row. If it isn't logged, it didn't happen.
```

## 3.5 Retrieval pipeline

1. **Workspace** (8 slots) is always in context — free, zero latency
2. **Sensory ring** — sub-second, for "what did I just say"
3. **Associative spread** — 2-hop activation from workspace entities over the graph
4. **Multi-signal search** — semantic + BM25 + entity + recency + activation
5. **Rerank** top 50 with cross-encoder
6. **Metamemory gate** — confidence low → *"I don't have that, Sir. Shall I look?"*
7. **On success** — append `now` to `retrieval_times`. **Remembering strengthens.** That's the spacing effect, for free.

## 3.6 Forgetting policy

Nothing is ever deleted by the system. Deactivation is a **decay of activation**; the record remains in `archive`, fully indexed, reactivatable by cue. Only the user can hard-delete (`forget` tool), and that writes a tombstone. This is strictly better than biology: we get adaptive forgetting's speed benefits with none of its loss.

---

# PART IV — BUILD PLAN

| Phase | Work | Effort | Acceptance test |
|---|---|---|---|
| **0. Instrument** | Make `consolidation_runs`, `reflections`, `entity_edges` actually write. Log every encode/retrieve. | 1–2 days | After 24 h, non-zero rows in all three. Today they are all empty |
| **1. Sensory buffers** | `SensoryRing` for audio/screen/frames | 3–4 days | "What did I say a moment ago?" answered in <1 s |
| **2. Encoding gate** | Salience scorer: prediction error, emphasis, repetition, goal-relevance, affect | 1 week | 100-turn transcript → top 10 salient items judged correct by you, 8/10 |
| **3. Facts v2 + versioning** | Versioned, confidence, provenance, no overwrite | 1 week | "You used to think X" → she returns X, the update, and when it changed |
| **4. Entity graph** | Extraction, typed relations, 2-hop spreading activation | 2 weeks | "Everyone I've discussed Project Falcon with" → correct list |
| **5. Activation + decay** | ACT-R base level, threshold, dormancy, archive | 1 week | Old low-salience fact still retrievable after 90 days of no use |
| **6. Nightly consolidation** | Full NREM/REM job | 2 weeks | `consolidation_runs` shows a run every night for 7 nights; LoCoMo-style score improves |
| **7. Procedural store** | Formalise `skills`, generalise from `cad_v3/vocabulary` | 2 weeks | She performs a learned 5-step workflow without being re-told |
| **8. Metamemory** | Confidence, provenance, refusal | 1 week | On 20 unanswerable questions, 0 fabrications |
| **9. Retrieval quality** | Reranking, multi-signal fusion, tuning | 2 weeks | Beat baseline on LoCoMo / LongMemEval-style local eval |

**Parallel track, cheap and worth doing now:** validate every model ID against each provider's model list. `dots-3-note-preview`, `gemini-3.6-flash`, `nvidia/nemotron-3-ultra-550b`, `voxtral-mini-tts-latest` are all suspect. If they 404, every enrichment call silently fails and the memory pipeline starves without erroring.

---

# PART V — HOW WE PROVE SHE DOESN'T FORGET

Public benchmarks, for calibration: **Mem0** scores 92.5 on LoCoMo and 94.4 on LongMemEval at ~6,900 tokens per query; **Zep** 80.3% at 189 ms; **Letta** 74.0; OpenAI Memory 52.9. At 10M tokens Mem0 drops to 48.6 on BEAM — temporal reasoning is where everyone still loses.

But benchmarks don't test *your* life. Build a **personal eval suite**:

1. **Seed 100 facts** across 8 categories (people, projects, preferences, dates, promises, jargon, places, routines)
2. **Probe at 1 day, 7 days, 30 days, 90 days** with paraphrased, indirect cues — not verbatim
3. **Contradiction test**: assert the opposite of a known fact; she must flag the conflict and keep both versions
4. **Confabulation test**: 20 questions she cannot know; she must refuse
5. **Dormancy test**: a fact untouched for 90 days, then cued — must surface
6. **Nightly-job liveness**: `consolidation_runs` non-empty for 30 consecutive days

**Target: 100% on facts explicitly told to her, at every interval.** Not 92.5%. She has no biological reason to miss one.

---

# PART VI — BETTER THAN HUMAN: THE TWELVE GUARANTEES

1. **Nothing unencoded** — buffers capture before attention decides
2. **Nothing lost to decay** — activation decays, records don't
3. **Nothing destroyed by update** — every correction is a new version
4. **No confabulation** — confidence floor, provenance, refusal
5. **No interference** — pattern separation, dedupe, versioning
6. **No tip-of-the-tongue** — five independent retrieval routes
7. **No sleep debt** — the nightly job never skips
8. **No capacity ceiling** — unbounded, activation-ranked
9. **No emotional distortion** — affect stored as metadata, never mixed into content
10. **Perfect provenance** — who, when, source, confidence, on every fact
11. **Perfect transmission** — exportable, diff-able, portable, encrypted
12. **She knows she doesn't know** — metamemory, which is more than most people manage

Human memory is a **lossy, reconstructive, sleep-dependent, interference-managed system** that evolved for survival, not accuracy. We are not going to copy its limitations — we are going to copy its *architecture*, which is genuinely brilliant, and then drop every defect that came from being made of meat.

**The result should feel like this:** you mention someone once, in passing, in August. In February you ask who they are. Ada answers with what you said, when you said it, what else was happening that week, what she's learned about them since, and how confident she is. No human you know can do that. That is the bar.

---

## Sources

- Engram encoding/consolidation/retrieval, sparsity, PSD-95, IEG tagging, reconsolidation: *Engram Memory Encoding and Retrieval: A Neurocomputational Perspective*, arXiv:2506.01659 (2025); Josselyn & Tonegawa engram work; Nature 2013 false-memory implantation
- Sharp-wave ripples and hippocampo-cortical reactivation: *Neuron* (2025), S0896-6273(25)00756-1; Joo & Frank, *Nat Rev Neurosci* 2018
- Targeted memory reactivation: Liu et al., *Communications Biology* (2025); arXiv:2511.15013
- Pattern separation (DG) / completion (CA3): Bakker et al., *Science* 2008; Hunsaker & Kesner reviews
- Reconsolidation: Nader & Hardt, *Nat Rev Neurosci* 2009 ("A single standard for memory")
- Adaptive forgetting / retrieval-induced forgetting: *Nature* nn.3973 (2014); *Cell Reports* 2023 (retroactive interference, engram expression)
- Working memory: Baddeley & Hitch 1974; Baddeley 2000 (episodic buffer); Cowan 2001 (capacity ≈4)
- Multiple memory systems: Squire, *J Cognitive Neuroscience* 4(3), 1992
- ACT-R base-level activation: Anderson & Schooler 1991; Anderson 2007
- Agent-memory benchmarks: Mem0, *State of AI Agent Memory 2026*; LoCoMo, LongMemEval, BEAM
- Constructive episodic simulation: Schacter & Addis
