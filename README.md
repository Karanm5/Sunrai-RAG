# Multimodal RAG with a Knowledge Graph, over PubLayNet

A retrieval system for scientific paper pages that answers questions using text,
figures and tables, plus a small knowledge graph. The point of the project is not
just to build it, but to measure honestly where the extra machinery actually helps.

I built two systems that share everything except the evidence they can reach:

| | Baseline | Enhanced |
|---|---|---|
| Text retrieval | dense (MiniLM) | dense (MiniLM), identical |
| Visual retrieval | none | OCR text from figures and tables, plus CLIP |
| Structured knowledge | none | knowledge graph, one hop |
| Fusion | none | Reciprocal Rank Fusion |
| Generator | shared | shared, identical prompt |

Because the corpus, retriever, top-k and prompt are the same on both sides, any gap
in the results comes from the components under test rather than from incidental
differences.

## Headline result

On questions whose answer only appears in a figure or table, the baseline scores
**0.000** recall. That is not a tuning problem. A text-only index cannot return a
figure region at all, so the ceiling is zero by construction. The enhanced system
reaches **0.750** on the same questions.

| Segment | Baseline | Enhanced | Difference |
|---|---|---|---|
| visual_requiring | 0.000 | 0.750 | +0.750 |
| text_answerable | 0.875 | 0.750 | -0.125 |
| multi_hop | 0.688 | 0.563 | -0.125 |
| **overall** | 0.521 | **0.688** | **+0.167** |

The two losses are real and I have left them in. Adding candidates to the fusion
disturbs rankings that were already correct, and that cost is part of the picture.

## Checking the results without running anything

Reproducing the whole pipeline needs tesseract, roughly 2 GB of models, an API key
and about 45 minutes. Nobody handed a repository is going to do that, so I made the
numbers checkable directly:

```bash
pip install -r requirements-min.txt
sunrai-rag verify
```

That recomputes every figure above from `results/retrieval_log.json`, which records
exactly what each system retrieved for each question. It needs no models, no API key
and no network, and takes a couple of seconds.

```bash
pytest tests/ -q
```

268 tests, offline, about four seconds.

## Running it properly

```bash
# 1. Environment. Conda handles the tesseract binary for you.
conda env create -f environment.yml
conda activate sunrai-rag

# or with pip, in which case install tesseract separately (see below)
pip install -e .
pip install -r requirements.txt

# 2. Check the environment before committing to a long run
export GROQ_API_KEY=...
sunrai-rag doctor --config configs/groq.yaml

# 3. Run the stages
sunrai-rag ingest   --config configs/groq.yaml
sunrai-rag index    --config configs/groq.yaml
sunrai-rag kg       --config configs/groq.yaml
sunrai-rag build-qa --config configs/groq.yaml
sunrai-rag evaluate --config configs/groq.yaml
```

Each stage saves its output, so you can iterate on retrieval without repeating the
slow OCR pass.

Ask a single question against either system:

```bash
sunrai-rag ask --system enhanced -q "Which method scored highest?"
sunrai-rag ask --system baseline -q "Which method scored highest?"
```

### Installing tesseract by hand

```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng   # Debian and Ubuntu
brew install tesseract                                  # macOS
```

On Windows, use the installer at https://github.com/UB-Mannheim/tesseract/wiki and
tick the option to add it to PATH. If it still cannot be found, `sunrai-rag doctor`
will say so, and you can set `ingest.tesseract_cmd` in the config to the full path.

## The dataset finding that shaped everything

`lhoestq/small-publaynet-wds` gives you a rendered page image and a COCO style layout
annotation with bounding boxes over five region types.

The annotation contains no text. PubLayNet is a layout dataset, not a text corpus. So
before any retrieval can happen, the text has to be recovered by running OCR over the
region crops. That makes OCR a load bearing stage rather than a convenience, and its
quality sets the ceiling for everything downstream.

There is a second detail that mattered more than I expected. The pages are about 72
DPI, and a typical text block is only 50 pixels tall. At that size Tesseract returned
nothing at all, scoring 0.000 against known ground truth on rendered fixtures.
Upscaling each crop to a minimum height of 200 pixels before OCR brought that back to
0.914, and the gain flattens out at 200 so anything larger just costs time. Without
that step the pipeline produces no text whatsoever.

Measured on the real corpus: of 1,782 textual regions, 1,582 produced usable
text (0.888), none came back completely empty, and 200 fell below the 20
character threshold. Mean Tesseract confidence 0.804.

## Design decisions and why

**Region aligned chunks.** One chunk per layout region, so a citation maps to exactly
one box you could highlight on the page. I preferred that to sliding windows because
exact provenance is worth more here than a marginal retrieval gain.

**RRF rather than weighted score fusion.** Dense cosine, BM25 and CLIP scores sit on
completely different scales. Combining them by weight would need a validation split I
do not have, and tuning those weights against the evaluation set would quietly
invalidate the comparison. RRF only uses rank, so it needs one parameter and no
calibration.

**NetworkX rather than a graph database.** At this scale a server backed store would
not change a single result, and it would add operational weight to a reproduction that
currently runs in one command. The graph serialises to a single JSON file. The
`expand()` interface is storage agnostic, so moving to Neo4j later would be a backend
swap rather than a redesign.

**Exact search rather than an approximate index.** At a few thousand vectors, ANN buys
no measurable latency but introduces recall variance between builds, which would
undermine the reproducibility the evaluation depends on. There is a FAISS backed class
in `index/vector_store.py` for when the corpus is large enough to need it.

**Pluggable model backend.** Anthropic, any OpenAI compatible provider such as Groq or
Together, a local Ollama or vLLM server, an in process transformers model, or a
deterministic stub for CI. Switching between a hosted model and one running on local
hardware is a config edit rather than a code change. Rate limited free tiers are
handled with client side pacing and backoff that respects `Retry-After`.

**Cached model calls.** This is what makes a language model pipeline reproducible
rather than merely repeatable. A rerun replays identical responses instead of
resampling, and costs nothing.

**Batched knowledge graph extraction.** One call per region is the obvious approach and
the wrong one on a rate limited tier, because it spends the request budget resending
the same prompt boilerplate hundreds of times. Sending five excerpts per call took 200
regions from 200 requests down to 40.

## A negative result I kept

CLIP over the raw crops retrieves at chance on this content. It scored 0.000 recall on
visual questions, which is statistically indistinguishable from picking at random among
106 regions.

That is not a bug, it is a distribution mismatch. CLIP is trained on natural
photographs with captions, and a cropped table from a PubMed paper is nothing like
that. Since the information in those regions is almost entirely textual, I now also
embed their OCR text with the text encoder and search that instead. This is still
genuinely cross modal, because the content is recovered from a non text modality and is
deliberately kept out of the baseline's index, but it uses an instrument suited to the
material. CLIP is still there and fused afterwards, since it may catch a purely
pictorial figure with no readable text.

## How the evaluation works

PubLayNet has no question and answer ground truth, so the question set is constructed.
That creates a circularity risk: a question written from a chunk inherits its
vocabulary, and retrieval then succeeds for the wrong reason. Three things guard
against it.

1. Questions are generated under an instruction to paraphrase, and any question sharing
   more than 60 percent of its content words with the source is discarded.
2. Questions are labelled by where the answer lives, so results are read per segment
   rather than as one blended number.
3. Differences are reported alongside absolutes, since circularity inflates both systems
   similarly and the gap survives that bias better than either raw value.

A random retriever and a BM25 retriever are evaluated alongside. Without those floors, a
weak dense retriever would be invisible in an absolute recall number. Dense retrieval
beats BM25 by roughly five times overall, so the embeddings earn their place.

## Testing

```bash
pytest tests/ -q                 # 268 tests, offline, about four seconds
ruff check src/ tests/
python scripts/verify_ocr.py     # real Tesseract against known ground truth
python scripts/smoke_e2e.py      # whole pipeline on synthetic data
```

The suite runs with no models, no network and no API key, because the heavy pieces sit
behind protocols and get substituted with deterministic stubs. That is a deliberate
contract and CI enforces it by installing only the minimal requirements.

Two of the tests exist because of bugs that actually happened. One simulates Pillow
being absent, after an unconditional import inside the OCR path broke CI while passing
locally. Another checks that OCR text from tables never reaches the searchable text
index, because if it did the baseline would gain the exact capability the experiment is
trying to isolate, and the comparison would quietly become meaningless.

## Known limitations

1. OCR noise, about 11 percent of text regions unusable, caps achievable recall.
2. The question set is constructed rather than human written. The safeguards bound the
   circularity but do not remove it.
3. 24 questions, 8 per segment. A difference of 0.125 is one question changing, so the
   two small losses sit inside the noise while the +0.750 clearly does not.
4. The knowledge graph is sparse. It has 477 nodes but only 222 edges, so a lot of
   entities are isolated and cannot support multi-hop expansion. It was also built
   from roughly 200 of 1,782 textual regions, which was a free tier rate limit
   constraint rather than a design choice.
5. Answer quality was not measured. These are retrieval results only.
6. Entity typing from the language model is noisy. Study groups sometimes end up
   labelled as datasets.
7. Entity linking is normalised substring matching, so it misses paraphrased mentions.

## Layout

```
src/sunrai_rag/
  config.py        config loading, validation, seeding
  schemas.py       the typed data model every stage shares
  cli.py           one entry point per stage, plus doctor and verify
  ingest/          shard loader, layout parsing, OCR
  represent/       text and CLIP embedders behind a protocol
  index/           vector store, BM25, RRF fusion
  kg/              extraction and the NetworkX graph
  rag/             model backends, baseline and enhanced pipelines
  eval/            metrics, question construction, comparison runner
  demo/            Streamlit app
tests/             268 offline tests
scripts/           fixture generation, OCR verification, end to end smoke test
configs/           default.yaml, ci.yaml, groq.yaml
results/           comparison.csv, results.json, retrieval_log.json
```
