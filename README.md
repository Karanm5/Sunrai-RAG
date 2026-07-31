# Multimodal RAG + Knowledge Graph over PubLayNet

A retrieval-augmented generation system for scientific documents that answers
questions using **text, figures, and a knowledge graph** — and measures,
segment by segment, exactly where the multimodal and structured-knowledge
components help and where they do not.

Two systems are built and compared under identical conditions:

| | Baseline | Enhanced |
|---|---|---|
| Text retrieval | dense (MiniLM) | dense (MiniLM) — *identical* |
| Visual retrieval | — | CLIP over figure/table crops |
| Structured knowledge | — | knowledge graph, 1–2 hop expansion |
| Fusion | — | Reciprocal Rank Fusion |
| Generator | shared | shared — *identical prompt* |

Only the evidence differs. Everything else is held constant, so any measured
gap is attributable to the components under test.

---

## Checking the results without running anything

Reproducing the full pipeline needs tesseract, ~2 GB of models, an API key and
about 45 minutes. To confirm the reported numbers instead:

```bash
pip install -r requirements-min.txt   # no models, no API key, no network
sunrai-rag verify --config configs/default.yaml
```

This recomputes every headline metric from `results/retrieval_log.json`, which
records exactly what each system retrieved for each evaluation question. It
takes seconds and runs anywhere.

```bash
pytest tests/ -q                      # ~250 tests, offline, ~1 second
```

## Quick start

```bash
# 1. Environment (conda handles the tesseract binary for you)
conda env create -f environment.yml
conda activate sunrai-rag

# ...or with pip (tesseract must be installed separately, see below)
pip install -r requirements.txt

# 2. Verify the install — offline, no models, no API key, ~1 second
pytest tests/ -q

# 3. Full pipeline — pick a generation backend
export GROQ_API_KEY=...             # free tier
make all CONFIG=configs/groq.yaml

# ...or Anthropic
export ANTHROPIC_API_KEY=...
make all
```

### Choosing a generation backend

One `OpenAICompatibleBackend` covers every hosted provider and any local
server, because they all speak the same chat-completions wire format.
Switching between a hosted model and one running on local hardware is a
config edit, not a code change — which is the point when a platform has to
run either way.

| `llm.backend` | Endpoint | Key | Notes |
|---|---|---|---|
| `groq` | Groq | `GROQ_API_KEY` | free tier; rate-limited, throttled by default |
| `anthropic` | Anthropic | `ANTHROPIC_API_KEY` | paid |
| `together` / `openrouter` | those providers | respective | paid |
| `ollama` | `localhost:11434` | any non-empty | fully local, no network |
| `local` | in-process transformers | none | no server needed, slowest |
| `stub` | none | none | tests and CI only |

`configs/groq.yaml` is tuned for a free tier: rule-based KG extraction (which
removes roughly 400 LLM calls), a 25 rpm client-side throttle, and caching so
a re-run costs nothing. Rate limits are handled by pacing requests rather than
tripping 429s and backing off; `Retry-After` is honoured when the server sends
it, and a 4xx other than 429 fails immediately instead of burning the
remaining quota on a request that will never succeed.

Provider model names change — verify the current list before running. A
retired model name returns HTTP 400, which fails fast with the message.

`make all` runs: `ingest → index → kg → qa → evaluate`, writing
`results/comparison.csv` and `results/results.json`.

### Exact reproduction commands

```bash
make ingest      # stream shards, crop regions, OCR, chunk   -> artifacts/corpus.json
make index       # text + CLIP embeddings                    -> artifacts/{text,image}_index/
make kg          # entity/relation extraction, graph build   -> artifacts/kg.json
make qa          # construct the segmented question set      -> artifacts/qa_set.json
make evaluate    # baseline vs enhanced comparison           -> results/comparison.csv
make demo        # interactive Streamlit demo
```

Single question against either system:

```bash
python -m sunrai_rag.cli ask --system enhanced -q "Which method scored highest?"
python -m sunrai_rag.cli ask --system baseline -q "Which method scored highest?"
```

### Installing tesseract (pip route only)

```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng   # Debian/Ubuntu
brew install tesseract                                  # macOS
```

### Docker

```bash
make docker
docker run --rm -p 8501:8501 -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY sunrai-rag
```

The image runs the test suite at build time, so a successful build is itself
a verification that the environment is correct.

---

## The dataset finding that shapes the design

`lhoestq/small-publaynet-wds` ships each page as a **`.png` render** paired
with a **`.json` layout annotation** (COCO bounding boxes over five classes:
text, title, list, table, figure).

**The annotation contains no text.** PubLayNet is a *layout* dataset, not a
text corpus. Text must therefore be recovered by **OCR over the region
crops** before any retrieval is possible — OCR is a load-bearing pipeline
stage here, not a convenience, and its quality bounds the whole system's
ceiling. The ingestion stage reports OCR usable-rate and mean confidence so
this limitation is quantified rather than hand-waved.

---

## Architecture

```
WebDataset shards (.png + .json)
        │
        ▼
┌─ INGESTION (shared by both systems) ──────────────────────────┐
│  parse layout annotation → crop regions by bbox               │
│  textual regions → OCR → clean → chunk                        │
│  visual regions  → save crops                                 │
│  → validated Corpus (every chunk traceable to its region)     │
└───────────────────────────────────────────────────────────────┘
        │                                    │
        ▼                                    ▼
┌─ BASELINE ─────────────┐    ┌─ ENHANCED ──────────────────────┐
│ dense text retrieval   │    │ dense text retrieval            │
│         │              │    │  + CLIP visual retrieval        │
│         ▼              │    │  + KG entity linking & expansion│
│   generate + cite      │    │         │                       │
└────────────────────────┘    │         ▼ RRF fusion            │
                              │   generate + cite               │
                              └─────────────────────────────────┘
        │                                    │
        └──────────────┬─────────────────────┘
                       ▼
        ┌─ EVALUATION ────────────────────────┐
        │ segmented QA set                     │
        │ Recall@k · Precision@k · nDCG · MRR  │
        │ + BM25 and random floors             │
        │ + cached LLM-as-judge (faithfulness) │
        └──────────────────────────────────────┘
```

### Key design decisions

**Region-aligned chunking.** One chunk per layout region, so a citation maps
to exactly one highlightable box on the page. Exact provenance is worth more
here than marginal retrieval gains from arbitrary windowing.

**RRF rather than weighted score fusion.** Dense cosine, BM25 and CLIP scores
live on incomparable scales; weighting them requires a validation split this
task does not have, risking tuning on the test set. RRF uses only *rank*, so
it is scale-free with one robust hyper-parameter (k=60, Cormack et al. 2009).

**NetworkX rather than a graph database.** At this scale a server-backed
store changes no result and adds operational burden. The graph serialises to
one JSON file, keeping reproduction to a single command. The `expand()`
interface is storage-agnostic, so Neo4j is a backend swap, not a redesign.

**Exact vector search rather than an ANN index.** Approximate search would
introduce recall variance across builds and undermine the determinism the
evaluation rests on. `FaissVectorStore` (also exact, `IndexFlatIP`) is
included as the scaling path.

**Pluggable LLM backend.** A platform that must run *on local systems or in
the cloud* cannot hard-depend on one hosted provider, so the backend is a
first-class config switch: Anthropic, any OpenAI-compatible provider (Groq,
Together, OpenRouter), a local Ollama/vLLM server, an in-process transformers
model, or a deterministic stub for CI. Rate-limited free tiers are supported
with client-side throttling and `Retry-After`-aware backoff.

**Cached LLM calls.** This is what makes an LLM pipeline reproducible: a
re-run replays identical responses rather than resampling, and costs nothing.

---

## Evaluation methodology

PubLayNet ships no QA ground truth, so the question set is constructed. That
creates a **circularity risk**: a question written *from* a chunk inherits
its vocabulary, so retrieval succeeds for the wrong reason and Recall is
inflated for every system. Three safeguards, all reported rather than buried:

1. **Paraphrase constraint + overlap filter.** Questions are generated under
   an instruction to avoid the source's distinctive terms; `lexical_overlap`
   measures compliance and questions above threshold are dropped.
2. **Segmentation by where the answer lives.**
   - `text_answerable` — answer is in body text
   - `visual_requiring` — answer exists **only** in a figure or table
   - `multi_hop` — answer requires two regions
3. **Deltas over absolutes.** Circularity inflates both systems similarly, so
   the *gap* between them survives the bias better than either raw value.

**Sanity floors.** A random retriever and BM25 are evaluated alongside. If
dense retrieval does not clearly beat lexical matching, the "semantic
retrieval" claim is unearned — and without the floors that failure is
invisible in an absolute Recall number.

The central hypothesis, stated so it can fail: *the enhanced system should
win decisively on `visual_requiring` and `multi_hop` questions, and roughly
tie on `text_answerable`.* A tie or loss on the text segment is the expected,
honest result — not a defect.

---

## Testing

```bash
pytest tests/ -q            # ~190 tests, runs offline in about a second
ruff check src/ tests/      # lint
python scripts/smoke_e2e.py # full pipeline on synthetic data, no network
```

The suite runs with **no models, no network and no API key** — heavy
dependencies are injected behind protocols and substituted with deterministic
stubs. Coverage includes: annotation-schema variants, bbox clipping, OCR
statistics and caching, chunk-to-region provenance, vector search
determinism, BM25 correctness, RRF algebraic properties (order-invariance,
rank-monotonicity, scale-freeness), graph expansion and entity-linking edge
cases, hallucinated-relation rejection, metric values against hand-computed
figures, QA circularity filtering, and the id-space mapping between chunks
and regions.

`scripts/smoke_e2e.py` builds a synthetic corpus where some answers exist
*only* in figure regions, then asserts the behaviour the project claims:
baseline recall on visual questions is 0.0 (structurally blind), enhanced is
above 0.0, both beat the random floor, and two runs produce identical output.

> **A bug this caught.** Scoring originally concatenated evidence
> text-then-visual, which pushed every visual hit past position *k* — so
> `Recall@5` could never credit visual retrieval and the enhanced system
> looked blind to figures even when its visual retrieval was perfect.
> `Provenance.ranked_evidence_ids()` interleaves modalities by rank, and
> `test_ranked_evidence_interleaves_modalities` guards the regression.

---

## Reproducibility

- **Seeded** — one `seed` in config drives `random`, `numpy`, and `torch`.
- **Pinned** — all dependencies use `==`; `temperature: 0.0` is enforced by
  config validation, which *raises* if changed, so the determinism claim
  cannot silently lapse.
- **Config-driven** — every number that affects a result lives in
  `configs/default.yaml`. Unknown keys raise rather than being ignored, since
  a silently-ignored typo is a reproducibility bug.
- **Provenance trail** — `results.json` records the seed, backend, model
  names and top-k used for that run.
- **Integrity guard** — `run_comparison` refuses to write headline results
  when the stub backend is active, so a smoke run can never be mistaken for a
  real result.

---

## Known limitations

Stated plainly, because they bound what the results can support:

1. **OCR noise** propagates to every downstream stage and caps achievable
   recall. Usable-rate is reported per run.
2. **Constructed QA** is not human-authored ground truth. The safeguards
   above bound the circularity; they do not remove it.
3. **POC scale** — a subset of shards, not full PubLayNet. Conclusions are
   about the *relative* behaviour of the two designs, not absolute
   state-of-the-art numbers.
4. **Rule-based KG extraction** favours precision over recall, so the graph
   is sparse rather than exhaustive. The LLM extractor raises recall at the
   cost of reproducibility without an API key.
5. **LLM-as-judge** is a single-model, single-run proxy for answer quality;
   scores are reported with 95% intervals rather than as point estimates.
6. **Entity linking** is normalised substring matching, not a learned linker
   — it will miss paraphrased entity mentions.

---

## Repository layout

```
src/sunrai_rag/
├── config.py            # config loading, validation, global seeding
├── schemas.py           # typed data model shared by every stage
├── cli.py               # one entry point per pipeline stage
├── ingest/              # loader, layout parsing, OCR
├── represent/           # text + CLIP embedders (protocol-based)
├── index/               # vector store, BM25, RRF fusion
├── kg/                  # extraction (rule + LLM), NetworkX graph
├── rag/                 # LLM backends, baseline & enhanced pipelines
├── eval/                # metrics, QA construction, comparison runner
└── demo/                # Streamlit app
tests/                   # ~190 offline tests
scripts/smoke_e2e.py     # end-to-end pipeline check, no network
configs/                 # default.yaml (real runs), ci.yaml (offline)
```
