.PHONY: proto test lint format run-orchestrator run-worker clean

proto:
	bash scripts/generate_proto.sh

test: proto
	python3 -m pytest tests/ -v

lint:
	ruff check .
	black --check .

format:
	black .

run-orchestrator: proto
	python3 -m orchestrator.server

run-worker: proto
	python3 -m worker.daemon

clean:
	rm -rf proto/generated/
	rm -rf build/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
