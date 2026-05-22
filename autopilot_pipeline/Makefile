# ─────────────────────────────────────────────────────────────────────────────
# AutoPilot Video Pipeline — Makefile
# ─────────────────────────────────────────────────────────────────────────────

.DEFAULT_GOAL := help
PYTHON := python
PIP    := pip

.PHONY: help install infra-up infra-down run run-fast run-dev worker-audio worker-gpu worker-render clean test lint

help:
	@echo ""
	@echo "  AutoPilot Video Pipeline"
	@echo ""
	@echo "  Setup:"
	@echo "    make install         Install Python dependencies"
	@echo "    make infra-up        Start Redis, Postgres, Qdrant via Docker"
	@echo "    make infra-down      Stop infrastructure containers"
	@echo ""
	@echo "  Run pipeline:"
	@echo "    make run             Full pipeline (personal_finance niche)"
	@echo "    make run-fast        Pipeline with auto-approve (no human review)"
	@echo "    make run-dev         Dev mode (no DB, auto-approve, console logs)"
	@echo ""
	@echo "  Workers (run in separate terminals):"
	@echo "    make worker-audio    Start Celery audio worker"
	@echo "    make worker-gpu      Start Celery GPU worker"
	@echo "    make worker-render   Start Celery render worker"
	@echo ""
	@echo "  Quality:"
	@echo "    make test            Run test suite"
	@echo "    make lint            Lint with ruff"
	@echo "    make clean           Remove all generated outputs"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────

install:
	$(PIP) install -r requirements.txt
	@echo "✓ Dependencies installed"

infra-up:
	docker compose up -d redis postgres qdrant
	@echo "✓ Infrastructure services started"
	@echo "  Redis   : localhost:6379"
	@echo "  Postgres: localhost:5432"
	@echo "  Qdrant  : localhost:6333"

infra-down:
	docker compose down
	@echo "✓ Infrastructure stopped"

# ── Run ───────────────────────────────────────────────────────────────────────

run:
	$(PYTHON) main.py --niche personal_finance --log-format console

run-fast:
	$(PYTHON) main.py --niche personal_finance --approve --log-format console

run-dev:
	$(PYTHON) main.py --niche personal_finance --no-db --approve --log-format console

run-legal:
	$(PYTHON) main.py --niche legal_tax --no-db --approve --log-format console

run-saas:
	$(PYTHON) main.py --niche saas_tools --no-db --approve --log-format console

# ── Workers ───────────────────────────────────────────────────────────────────

worker-audio:
	celery -A workers.audio_worker worker --queues=audio --concurrency=2 --loglevel=info

worker-gpu:
	celery -A workers.gpu_worker worker --queues=gpu --concurrency=1 --loglevel=info

worker-render:
	celery -A workers.render_worker worker --queues=render --concurrency=1 --loglevel=info

# ── Quality ───────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v --tb=short

lint:
	ruff check . --fix
	ruff format .

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	rm -rf outputs/audio/* outputs/video/* outputs/visual/*
	rm -rf data/clip_cache/index.json
	@echo "✓ Output directories cleared"

clean-all: clean
	rm -rf outputs/ data/clip_cache/ data/vectorstore/ logs/
	@echo "✓ Full cleanup complete"
