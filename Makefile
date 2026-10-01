.PHONY: install test lint build eval eval-flash clean

install:        ## Install editable with dev extras
	pip install -e ".[dev]"

test:           ## Run the test suite with coverage (integration excluded)
	pytest -q

test-integration:  ## Run integration tests against the real API (needs CLEF_* env)
	pytest -q -m integration --no-cov

lint:           ## Ruff over sources and tests
	ruff check src tests

build:          ## Build wheel + sdist
	python -m build

eval:           ## Benchmark both Clef models on the committed datasets (needs CLEF_* env)
	python evals/run_eval.py

eval-flash:     ## Benchmark clef-flash only
	python evals/run_eval.py --model @cf/cloudflare/clef-flash

clean:          ## Remove build and test artifacts
	rm -rf dist build .pytest_cache .ruff_cache .coverage htmlcov
