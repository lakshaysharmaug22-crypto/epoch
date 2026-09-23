.PHONY: setup setup-gpu test lint demo demo-gpu serve web web-build up

setup:            ## CPU install (works anywhere)
	uv venv -p 3.11 && uv pip install -e ".[dev]"

setup-gpu:        ## + PyTorch/transformers/bitsandbytes/optimum/faiss for the RAG workload's GPU genes
	uv pip install -e ".[dev,gpu]"

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check epoch tests && .venv/bin/mypy

doctor:
	.venv/bin/epoch doctor

demo:             ## full measured pipeline → web/public/replay/bundle.json
	.venv/bin/epoch demo --workloads triage,rag --budget 60 --seeds 3

serve:            ## API for dashboard live mode
	.venv/bin/epoch serve

web:              ## dashboard dev server (live mode if the API is up)
	cd web && npm install && NEXT_PUBLIC_EPOCH_API=http://localhost:8000 npm run dev

web-build:        ## static replay build → web/out (deploy to Vercel)
	cd web && npm install && npm run build

up:
	docker compose up --build
