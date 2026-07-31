.PHONY: proto test test-slow lint format run-orchestrator run-worker clean webui-install webui-dev webui-build

proto:
	bash scripts/generate_proto.sh

# Excludes tests marked @pytest.mark.slow - those load real models and
# are heavy on RAM/time. Use `make test-slow` to run those explicitly.
test: proto
	python3 -m pytest tests/ -v -m "not slow"

test-slow: proto
	python3 -m pytest tests/ -v -m slow

lint:
	ruff check .
	black --check .

format:
	black .

run-orchestrator: proto dashboard/out/index.html
	python3 -m orchestrator.server

# Builds the dashboard only if it hasn't been built yet - run `make
# webui-build` explicitly to rebuild after changing frontend code.
dashboard/out/index.html:
	cd dashboard && npm install && npm run build

run-worker: proto
	python3 -m worker.daemon

webui-install:
	cd dashboard && npm install

webui-dev:
	cd dashboard && npm run dev

webui-build:
	cd dashboard && npm run build

clean:
	rm -rvf proto/generated/
	rm -rvf build/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rvf {} +
