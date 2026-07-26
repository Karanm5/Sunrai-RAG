# Reproduce everything: make setup && make all
CONFIG ?= configs/default.yaml
PY     := PYTHONPATH=src python -m sunrai_rag.cli

.PHONY: help setup test lint ingest index kg qa evaluate all demo smoke docker clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "  %-12s %s\n",$$1,$$2}'

setup:      ## Install dependencies
	pip install -r requirements.txt

test:       ## Run the full test suite (offline, no models needed)
	PYTHONPATH=src pytest tests/ -v

lint:       ## Lint
	ruff check src/ tests/

ingest:     ## Stage 1: stream shards, crop regions, OCR, chunk
	$(PY) ingest --config $(CONFIG)

index:      ## Stage 2: build text + CLIP indices
	$(PY) index --config $(CONFIG)

kg:         ## Stage 3: build the knowledge graph
	$(PY) kg --config $(CONFIG)

qa:         ## Stage 4: construct the evaluation question set
	$(PY) build-qa --config $(CONFIG)

evaluate:   ## Stage 5: run the baseline-vs-enhanced comparison
	$(PY) evaluate --config $(CONFIG)

all: ingest index kg qa evaluate   ## Full pipeline end to end

smoke:      ## Offline smoke test: no network, no API key
	PYTHONPATH=src pytest tests/ -q && $(PY) ingest --config configs/ci.yaml || true

demo:       ## Launch the Streamlit demo
	PYTHONPATH=src streamlit run src/sunrai_rag/demo/app.py -- --config $(CONFIG)

docker:     ## Build the container
	docker build -t sunrai-rag .

clean:      ## Remove generated artefacts (keeps results/)
	rm -rf artifacts/ data/ .pytest_cache __pycache__
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

fixtures:   ## Regenerate the rendered test pages
	python scripts/make_fixtures.py tests/fixtures/pages

verify-ocr: ## Run real Tesseract against known ground truth
	python scripts/verify_ocr.py

verify: test lint verify-ocr   ## Everything that can be checked offline
	python scripts/smoke_e2e.py
