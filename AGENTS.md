# Repository Guidelines

## Project Layout
- `mlrun/`: Python SDK, core runtimes, artifacts, and utilities exposed to users.
- `server/py/`: API and controller services (framework utilities plus `services/api`).
- `server/go/`: Go-based helpers such as the log collector.
- `tests/`: unit, integration, system suites; mirrors SDK/server structure.
- `docs/`, `examples/`: user documentation and runnable notebooks.
- Build assets: `Makefile`, `pyproject.toml`, `dockerfiles/`, `automation/` scripts.

## Daily Development
- Bootstrap deps: `make install-dev-requirements` (lint + pytest) or `make install-complete-requirements` for full stack.
- Update locked deps with UV: `make upgrade-mlrun-deps-lock`.
- Lint: `make lint` (Ruff + import-linter) and fix formatting via `make fmt`.
- Unit tests locally: `make test`; tune scope with `UNIT_TESTS_PATH=...` and enable coverage using `RUN_COVERAGE=true` and `COVERAGE_FILE=...`.
- Containerized run: `make test-dockerized` or `make html-docs-dockerized` when host tooling diverges.
- Build distributables: `make package-wheel` (uses `uv build`).

Example: `RUN_COVERAGE=true UNIT_TESTS_PATH=tests/sdk make test` creates a focused coverage report under `tests/coverage_reports/`.

## Style & Naming
- Python target is 3.9; max line length 120 (`pyproject.toml`).
- Ruff handles linting and formatting; add new rules under `[tool.ruff]` if required.
- Respect import layering from `tool.importlinter`: server code may import SDK modules, never the reverse.
- Naming: files `snake_case.py`, tests `test_*.py`, classes PascalCase, functions snake_case, constants UPPER_CASE.
- Markdown code blocks must format with `blacken-docs`; run `make fmt` before committing doc changes.

## Testing Strategy
- Pytest patterns defined in `pyproject.toml`; place server-focused tests under `server/py/*/tests/` to inherit fixtures.
- Mark external or slow tests explicitly and keep them skipped by default.
- Combine coverage artifacts with `make coverage-combine`; CI expects updated reports when coverage flags are used.

## Commit & PR Standards
- Follow existing history: `[Scope] Imperative summary (#issue)` for commits.
- PR checklist: problem statement, solution overview, test evidence (`make lint`, `make test`), docs/examples impacts, linked Jira/GitHub issue.
- Squash local fixups before pushing; rebase on `master` to keep release automation simple.

## Security & Configuration
- Never hard-code credentials; read from env (e.g., `MLRUN_HTTPDB__DSN`, `MLRUN_SECRET_STORES__TEST_MODE_MOCK_SECRETS`).
- Confirm config defaults in `project.yaml` and update `docs/` when behavior changes.
- Use `make pull-cache` before Docker builds if you rely on shared cache images.

