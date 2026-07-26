# Multimodal RAG with Knowledge Graph Augmentation for Scientific Documents

**Karan** · Technical Challenge, AI Platform Engineer (KTP) · August 2026

> **HOW TO USE THIS TEMPLATE**
> Every `<<PLACEHOLDER>>` must be replaced with a number from **your own run**
> of `make evaluate` (`results/comparison.csv`). Do not fill any of them from
> this template, from expectation, or from the smoke test — those numbers are
> synthetic. If a result contradicts the hypothesis, report it as it is: the
> discussion section is written to accommodate that outcome, and an honest
> negative result is stronger than a forced positive one.
> Delete this block and keep the final document to **2 A4 pages**.

## 1. Problem formulation

Scientific documents are multimodal: a claim in the body text is frequently
quantified only in a figure or table, and answering a question can require
linking a method to a result reported elsewhere in the paper. A text-only
retrieval system is structurally incapable of reaching evidence that exists
solely in a figure.

This work tests whether adding **visual retrieval** and a **knowledge graph**
to a conventional RAG pipeline measurably improves retrieval and answer
quality over scientific document pages — and, specifically, *where* it does.
The hypothesis is stated so it can fail: the enhanced system should win
decisively on questions whose answers are visual or require multiple hops,
and roughly tie on questions answerable from body text alone.

## 2. Data and preprocessing

**Dataset.** `lhoestq/small-publaynet-wds` — WebDataset shards of PubLayNet,
each sample a page render (`.png`) plus a COCO-style layout annotation
(`.json`) over five region classes (text, title, list, table, figure).
`<<N_PAGES>>` pages across `<<N_DOCS>>` documents were ingested.

**The design-defining finding.** PubLayNet is a *layout* dataset: the
annotation contains bounding boxes and category ids but **no text**. Text must
therefore be recovered by **OCR over region crops**, making OCR a load-bearing
stage whose quality bounds the system's ceiling. Measured OCR usable rate:
`<<OCR_USABLE_RATE>>`%, mean confidence `<<OCR_CONFIDENCE>>`.

**Preprocessing choices.** Boxes are clipped to page bounds and degenerate
regions dropped. OCR output is normalised conservatively (ligatures,
hyphenated line breaks, whitespace) but *not* spell-corrected, which would
risk silently rewriting scientific terms and units. Chunking is
**region-aligned** — one chunk per layout region — so every citation maps to
exactly one highlightable box on the page. Exact provenance was preferred over
the marginal retrieval gains of arbitrary windowing.

## 3. Architecture: baseline vs enhanced

Both systems share the same corpus, the same dense text retriever, the same
top-*k*, and the same generator and prompt. **Only the evidence differs**, so
any measured gap is attributable to the components under test.

**Baseline (control).** Dense retrieval (`all-MiniLM-L6-v2`) over OCR'd
chunks, exact cosine search, top-*k* passed to the generator with citation
instructions.

**Enhanced.** Three evidence sources, fused by rank:
1. the identical dense text retriever;
2. **CLIP** (`clip-vit-base-patch32`) over figure/table crops, with the query
   text embedded into CLIP's shared space — enabling direct text→image
   retrieval, the capability the baseline structurally lacks;
3. **knowledge-graph expansion** — entities (methods, metrics, datasets,
   findings, figure references) and relations (`evaluated_by`, `has_value`,
   `reported_in`) extracted per region, linked from the query and expanded
   `<<HOPS>>` hop(s). This reaches the region reporting a *result* when the
   query names only a *method* — the case embedding similarity handles worst.

**Fusion: RRF, not weighted score sum.** Dense cosine, BM25 and CLIP scores
occupy incomparable scales; weighting them needs a validation split this task
does not have, risking tuning on the test set. RRF uses only rank, so it is
scale-free with one robust parameter (*k*=60).

Graph: `<<KG_NODES>>` nodes, `<<KG_EDGES>>` edges
(`<<KG_EXTRACTOR>>` extractor).

## 4. Evaluation methodology

PubLayNet has no QA ground truth, so the question set is constructed — which
creates a **circularity risk**: a question written from a chunk inherits its
vocabulary, so retrieval succeeds for the wrong reason. Three safeguards:

1. **Paraphrase constraint + overlap filter.** Questions exceeding a lexical
   overlap of `<<MAX_OVERLAP>>` with their source are discarded
   (`<<N_REJECTED>>` rejected).
2. **Segmentation by where the answer lives** — `text_answerable`,
   `visual_requiring` (answer exists *only* in a figure/table), `multi_hop`.
3. **Deltas over absolutes.** Circularity inflates both systems similarly, so
   the gap between them survives the bias better than either raw value.

**Sanity floors.** Random and BM25 retrievers are evaluated alongside: if
dense retrieval does not clearly beat lexical matching, the semantic-retrieval
claim is unearned, and without floors that failure is invisible.

**Metrics.** Recall@*k*, Precision@*k*, nDCG@*k*, MRR (binary relevance,
macro-averaged); answer quality via cached LLM-as-judge (faithfulness,
relevance) reported with 95% intervals; plus citation validity — a judge-free
check that cited ids actually exist, catching invented citations.

Question set: `<<N_QUESTIONS>>` questions (`<<N_TEXT>>` text, `<<N_VISUAL>>`
visual, `<<N_MULTIHOP>>` multi-hop). Seed `<<SEED>>`, backend
`<<LLM_BACKEND>>`.

## 5. Key results

**Table 1 — Retrieval, by segment (Recall@`<<K>>` / MRR).**

| System | Overall | text_answerable | visual_requiring | multi_hop |
|---|---|---|---|---|
| Random floor | `<<>>` | `<<>>` | `<<>>` | `<<>>` |
| BM25 (lexical) | `<<>>` | `<<>>` | `<<>>` | `<<>>` |
| Baseline (text-only) | `<<>>` | `<<>>` | `<<>>` | `<<>>` |
| **Enhanced (MM+KG)** | `<<>>` | `<<>>` | `<<>>` | `<<>>` |
| **Δ (enhanced − baseline)** | `<<>>` | `<<>>` | `<<>>` | `<<>>` |

**Table 2 — Answer quality and cost.**

| System | Faithfulness (95% CI) | Relevance | Citation validity | Latency (s) |
|---|---|---|---|---|
| Baseline | `<<>>` | `<<>>` | `<<>>` | `<<>>` |
| Enhanced | `<<>>` | `<<>>` | `<<>>` | `<<>>` |

*[Insert the segmented bar chart from `results/`.]*

**Headline** *(rewrite to match your actual numbers)*: on `visual_requiring`
questions the enhanced system improves Recall@`<<K>>` by `<<Δ>>` over the
text-only baseline, which scores `<<BASELINE_VISUAL>>` — the baseline cannot
retrieve figure regions at all. On `text_answerable` questions the two are
`<<within noise / separated by X>>`. The KG contributes most on `multi_hop`
questions (`<<Δ>>`), where the answer requires linking a method to a result
reported in a different region.

## 6. Discussion

**Where the additions earn their cost.** The gain is concentrated, not
uniform, and that concentration is the finding. Visual retrieval helps
precisely where text retrieval is structurally blind — the baseline's
`visual_requiring` score of `<<BASELINE_VISUAL>>` is not a tuning failure but
a capability boundary. The graph helps on multi-hop questions by connecting
regions that share no vocabulary. A single blended metric would have averaged
these gains away into an unremarkable number and hidden the mechanism.

**Where they do not.** On `text_answerable` questions the enhanced system
`<<ties / trails slightly>>`. This is expected: adding graph-derived
candidates perturbs a text ranking that was already correct, and fusion can
displace a well-ranked chunk. The honest reading is that multimodal and KG
augmentation is a *targeted* intervention, not a general-purpose uplift —
which matters for deployment, since both add latency
(`<<BASELINE_LAT>>`s → `<<ENHANCED_LAT>>`s) and infrastructure.

**Limitations.** (i) OCR noise bounds every downstream stage; (ii) constructed
QA is not human ground truth — the safeguards bound circularity but do not
remove it; (iii) POC scale means conclusions concern the *relative* behaviour
of the two designs, not absolute benchmarks; (iv) rule-based extraction
favours precision, so the graph is sparse; (v) LLM-as-judge is a single-model
proxy, hence the reported intervals; (vi) entity linking is substring-based
and will miss paraphrased mentions.

**A measurement bug worth reporting.** Evidence was initially scored by
concatenating text then visual ids, which pushed every visual hit past
position *k* — making Recall@*k* structurally unable to credit visual
retrieval, and showing the enhanced system as blind to figures when its visual
retrieval was in fact perfect. Rank-interleaving across modalities fixed it,
and a regression test guards it. It is included here because the failure was
invisible in the aggregate number and only surfaced through segment-level
inspection — an argument for segmented evaluation as a practice, not just as a
reporting style.

**Insights for a production platform.** Three choices generalise beyond this
task: a pluggable local-or-hosted model backend, so deployment target is a
config switch rather than an architectural commitment; response caching, which
is what makes an LLM pipeline reproducible rather than merely repeatable; and
provenance as a first-class return value, so every answer is auditable —
necessary wherever outputs inform consequential decisions.

**Next steps.** Human-authored evaluation questions to remove circularity;
figure captioning (BLIP) to give visual regions richer text; learned entity
linking; and a document-level graph spanning pages rather than regions.

---
*Reproduction: `make all` with `configs/default.yaml`. Seed `<<SEED>>`.
Full commands, limitations and test coverage in `README.md`.*
