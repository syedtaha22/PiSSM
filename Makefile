.PHONY: proto test test-slow lint format clean setup setup-orch dashboard


# Full local setup: regenerate proto stubs, install the Python package,
# then build the dashboard. Safe to re-run any time - nothing here is
# skipped just because it already ran once before.
setup-orch: setup
	$(MAKE) dashboard

setup: proto
	pip install .

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

# Rebuilds the dashboard unconditionally - use this on its own to pick
# up frontend changes without re-running the rest of setup.
dashboard:
	cd dashboard && npm install && npm run build

clean:
	rm -rvf proto/generated/
	rm -rvf build/ *.egg-info/
	rm -rvf dashboard/out/ dashboard/.next/
	find . -type d -name __pycache__ -exec rm -rvf {} +
