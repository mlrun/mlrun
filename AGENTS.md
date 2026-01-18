<!-- This file provides guidance to AI agents. -->

# Project Overview

**MLRun** is an open-source AI orchestration platform for rapidly building and managing continuous (gen) AI and ML applications across their lifecycle. MLRun automates the delivery of production data pipelines, ML workflows, and online applications, significantly reducing engineering efforts, time to production, and computational resources. It integrates into development and CI/CD environments, breaks silos between data, ML, software, and DevOps/MLOps teams, and supports both community edition (CE) deployments and enterprise features when running in Iguazio systems.

## Repository Structure

This repository contains two major Python codebases plus Go services:

- **`mlrun/`** – SDK and client library (what end-users import); includes projects, runtimes, feature store, model monitoring, data store, serving, launchers, and frameworks integrations.
- **`server/py/`** – Python server components (FastAPI-based API service + alerts service); includes `services/api/` (main MLRun API), `services/alerts/` (events processing), and `framework/` (DB sessions, auth, utilities, rundb implementation).
- **`server/go/`** – Go services; includes `services/logcollector/` (gRPC microservice for streaming logs from Kubernetes pods).
- **`tests/`** – SDK unit tests, integration tests, and system tests (CE and enterprise-marked); uses pytest with shared fixtures in `tests/common_fixtures.py`.
- **`server/py/services/api/tests/`** – Server-side unit tests (note: `pyproject.toml` configures pytest `pythonpath=./server/py` to enable running server tests from repo root).
- **`docs/`** – Sphinx-based documentation (API reference, tutorials, guides, architecture); builds to ReadTheDocs.
- **`pipeline-adapters/`** – Pipeline integration packages (mlrun-pipelines-kfp-common, kfp-v1-8, kfp-v2) with independent `pyproject.toml` files.
- **`automation/`** – CI/CD automation scripts (deployment, system tests, release notes generation, version management).
- **`dockerfiles/`** – Dockerfile definitions for mlrun, mlrun-api, mlrun-gpu, mlrun-kfp, jupyter, test, test-system images.
- **`hack/`** – Local development environment configurations (.env files for various setups, local k8s manifests, benchmarks).
- **`examples/`** – Jupyter notebooks and example scripts demonstrating MLRun features.
- **`.github/`** – GitHub workflows (build, CI, system tests, security scans, release pipelines), issue templates, PR template, CODEOWNERS.


## Build & Development Commands

### Environment Setup

Set up Python environment:

```bash
# Create virtual environment (using venv or uv)
uv venv venv --python 3.11 --seed
source venv/bin/activate

# Install all dependencies
export MLRUN_PYTHON_PACKAGE_INSTALLER=uv
make install-requirements
uv pip install -e '.[complete]'
```

**Configure PYTHONPATH:**

```bash
# Required for server-side development
export PYTHONPATH="$(pwd):$(pwd)/server/py"
```

### Format & Lint

```bash
# Format code (Ruff formatter)
make fmt

# Lint code (Ruff linter)
make lint
```

### Database Migrations

```bash
# Create a new database migration (MySQL)
MLRUN_MIGRATION_MESSAGE="Add new column" make create-migration
```

### Documentation

```bash
# Install docs dependencies
make install-docs-requirements

# Build docs locally
cd docs
make html
# Output in docs/_build/html/index.html
```

### Dependency Management

```bash
# Update Python lock files for mlrun-api image
make upgrade-mlrun-api-deps-lock

# Update specific package only
MLRUN_UV_UPGRADE_FLAG="--upgrade-package <package-name>" make upgrade-mlrun-api-deps-lock
```

## Code Style & Conventions

### Formatter & Linter

- **Ruff** (v0.8.0+) for both formatting and linting (configured in `pyproject.toml`).
- Run `make fmt` before every commit.
- CI enforces linting via `make lint`.

### Naming

- **snake_case**: functions, variables, modules, parameters.
- **CamelCase**: classes.
- Prefer explicit, readable names; avoid unclear abbreviations.

### Imports

- **Internal (repo) imports**: **SHOULD** prefer module imports (`import pkg.mod`) or aliases (`import pkg.mod as mod`) to reduce circular-import risk and make boundaries explicit.
  - Example (preferred): `import mlrun.utils`
  - Example (avoid): `from mlrun.utils import logger`
- **External packages**: `from X import Y` is acceptable when it improves readability.
- **Import boundaries**: `mlrun.common` must NOT import higher-level `mlrun.*` modules (enforced by import-linter in `pyproject.toml`).
- **Forbidden imports**:
  - In `mlrun/`, do NOT import `kfp` directly; use `mlrun_pipelines` adapters instead.
  - In `mlrun/`, do NOT import `server/py` (server-side code).

### Type Hints & Docstrings

- Use type hints for complex data structures and public APIs.
- Public API functions/classes require docstrings (triple-quotes `""" """`).
- Docstring format: brief description, `:param` tags for parameters, `:return` tag for return value.

### Logging

- Use **structured logging** (variables as fields, not f-strings):

```python
from mlrun.utils import logger

# GOOD
logger.debug("Storing function", project=project, name=name, tag=tag)

# BAD: f-string in logs
logger.debug(f"Storing function {project}/{name}:{tag}")
```

- **NEVER log credentials**: passwords, tokens, API keys, secret values, auth headers, session cookies.

### Error Handling

- Stringify exceptions using `mlrun.errors.err_to_str(exc)` instead of `str(exc)`.
- Use `mlrun.errors.raise_for_status(...)` for HTTP response validation.

### Async + Blocking I/O (CRITICAL)

In `async def` contexts (server endpoints, async utilities):

- **NEVER block the event loop with synchronous I/O**.
- Run blocking I/O in a threadpool:

```python
import mlrun.utils


async def handler():
    # GOOD: run blocking work in threadpool
    result = await mlrun.utils.run_in_threadpool(sync_db_call, "arg")
    return result
```

### Commit Messages

- Follow conventional commit format: `[<Scope>] Verb changes made` (e.g., `[API] Add endpoint to list runs`).
- Use imperative verbs (Add, Fix, Update, Refactor).
- Include `fix` or `bug` keywords for bugfix PRs (auto-categorized in release notes).

## Architecture Notes

### High-Level Component Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        User / IDE / CI/CD                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────────┐
        │       MLRun SDK (mlrun/)                  │
        │  - Projects, Runtimes, Feature Store      │
        │  - Data Store, Model Monitoring           │
        │  - Launchers (Local/Remote/Server)        │
        └───────────────┬───────────────────────────┘
                        │ HTTP (via mlrun.db.httpdb.HTTPRunDB)
                        ▼
        ┌───────────────────────────────────────────┐
        │   MLRun API Server (server/py/services/   │
        │              api/)                        │
        │  - FastAPI endpoints (api/endpoints/)     │
        │  - CRUD operations (crud/)                │
        │  - ServerSideLauncher (launcher.py)       │
        └───────────┬───────────────────────────────┘
                    │\
                    │ \ gRPC
                    │  ▼
                    │  ┌──────────────────┐
                    │  │ Log Collector    │
                    │  │ (Go service)     │
                    │  │ (gRPC + K8s API) │
                    │  └──────────────────┘
                    │
        ┌───────────┴────────────┬──────────────────┐
        ▼                        ▼                  ▼
┌──────────────┐      ┌──────────────────┐  ┌──────────────┐
│ Database     │      │ Kubernetes API   │  │ Alerts       │
│ (MySQL/      │      │ (Job/Pod/CRD     │  │ Service      │
│  PostgreSQL) │      │  orchestration)  │  │ (alerts/)    │
└──────────────┘      └──────────────────┘  └──────────────┘
```

### Key Architectural Patterns

**1. SDK ↔ Server Boundary**

- SDK communicates via `mlrun.db.httpdb.HTTPRunDB` (implements `mlrun.db.base.RunDBInterface`).
- Project-scoped interface: `mlrun.projects.project.MlrunProject` (high-level wrapper with project context and enrichment).

**2. Launcher Pattern**

- **BaseLauncher** (`mlrun/launcher/base.py`): abstract interface for running functions.
- **ClientLocalLauncher** (`mlrun/launcher/local.py`): runs locally (user machine or remote with local semantics).
- **ClientRemoteLauncher** (`mlrun/launcher/remote.py`): submits jobs to API server/Kubernetes.
- **ServerSideLauncher** (`server/py/services/api/launcher.py`): server-side run submission with auth context.
- Launcher selection: automatic based on `mlrun.config.is_running_as_api()`, `runtime._is_remote`, and `local=` flag.

**3. Shared Layer (`mlrun.common`)**

- Foundation layer with minimal dependencies.
- Must NOT import higher-level `mlrun.*` modules (enforced by import-linter).

**4. Server Framework (`server/py/framework/`)**

- DB sessions, middlewares, auth utilities, base services.
- Should NOT import specific services (some exceptions exist, tracked in `pyproject.toml`).

### Data Flow Examples

**Function Execution (Remote):**

1. User calls `function.run()` in SDK.
2. `ClientRemoteLauncher` (or `ServerSideLauncher`) submits run via `HTTPRunDB.submit_run(...)`.
3. Server endpoint (`/projects/{project}/functions/{name}`) receives request.
4. Server launcher creates Kubernetes job/pod with runtime spec.
5. Log collector gRPC service streams logs from pod to persistent storage.
6. Run results/artifacts stored in DB and object storage.

## Testing Strategy

### Test Types & Tools

- **Unit Tests**: pytest, mocks (`pytest-mock`), fixtures.
- **Integration Tests**: pytest with Docker containers (MySQL, Postgres via `pytest-mock-resources`), K8s interactions (via test clusters).
- **System Tests**: end-to-end tests against live MLRun CE or Iguazio system (marked with `@pytest.mark.enterprise` for enterprise-only features).
- **Coverage**: tracked via `coverage.py` (configured in `pyproject.toml`).

## Security & Compliance

### Authentication & Authorization (Server APIs)

**Every new FastAPI endpoint MUST:**

1. **Authenticate requests** via `framework.api.deps.authenticate_request`:

```python
from fastapi import Depends
import mlrun.common.schemas
import framework.api.deps


@router.get("/projects/{project}/resource")
async def my_endpoint(
    project: str,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(
        framework.api.deps.authenticate_request
    ),
):
    # endpoint logic
    pass
```

2. **Authorize access** via `framework.utils.auth.verifier.AuthVerifier`:

```python
import framework.utils.auth.verifier
import mlrun.common.schemas


async def verify_permissions(
    project: str,
    resource_name: str,
    auth_info: mlrun.common.schemas.AuthInfo,
):
    # Project-level permission
    await framework.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project_name=project,
        action=mlrun.common.schemas.AuthorizationAction.read,
        auth_info=auth_info,
    )

    # Resource-level permission
    await framework.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        resource_type=mlrun.common.schemas.AuthorizationResourceTypes.function,
        project_name=project,
        resource_name=resource_name,
        action=mlrun.common.schemas.AuthorizationAction.update,
        auth_info=auth_info,
    )
```

### Secrets Handling

- **NEVER log credentials**: passwords, tokens, API keys, secret values, auth headers, session cookies.
- Use `mlrun.secrets` module for secret management (integrates with Kubernetes secrets, Vault).
- Sanitize request/response payloads before logging.

## Agent Guardrails

### Files NEVER to Modify (Without Team Approval)

- **`mlrun/utils/version/version.json`** – Auto-generated by release automation; manual edits will be overwritten.
- **Database migration files** (`server/py/services/api/migrations/versions/*.py`) – Never edit existing migrations; create new ones via `make create-migration`.
- **`**/locked-requirements.txt`** – Auto-generated by uv; use `make upgrade-mlrun-deps-lock` to update.
- **`.github/workflows/*.yaml`** – CI/CD pipelines; changes require maintainer review.

### Required Reviews

- **API schema changes** (FastAPI endpoint modifications, new fields in schemas) – Require backward compatibility review.
- **Database schema migrations** – Require DB team review.
- **Import boundary changes** (`pyproject.toml` import-linter rules) – Require architecture review.
- **Deprecations/removals** – Follow `DEPRECATION.md` process; update Jira ticket.

### Programming Principles (MUST FOLLOW)

- **DRY**: Extract helpers instead of copy/pasting logic across modules.
- **KISS**: Prefer straightforward solutions; minimize abstractions.
- **YAGNI**: Don't add "future-proof" frameworks without concrete need.

### Environment Variables

Key configuration variables (see `mlrun/config.py` and `hack/*.env` for examples):

- `MLRUN_PYTHON_PACKAGE_INSTALLER` - either `pip` or `uv`. prefer `uv`.
- `MLRUN_DBPATH` – MLRun API server URL (e.g., `http://localhost:8080`).
- `MLRUN_VERSION` – Version override for builds.
- `MLRUN_DOCKER_REGISTRY` – Docker registry prefix (default: DockerHub).
- `MLRUN_NO_CACHE` – Disable build/pip caching (set to any value).
- `MLRUN_SKIP_COMPILE_SCHEMAS` – Skip protobuf schema compilation during build.
- `MLRUN_SYSTEM_TESTS_COMPONENT` – Filter system tests by component (or prefix with `no_` to exclude).
- `COVERAGE_FILE` – Coverage data file path (for `RUN_COVERAGE=true`).

### Plugin Points

- **Custom runtimes**: extend `mlrun.runtimes.base.BaseRuntime` and register in `mlrun.runtimes/__init__.py`.
- **Data sources**: implement `mlrun.datastore.base.DataStore` interface for custom storage backends.
- **Serving steps**: extend `mlrun.serving.server.GraphStep` for custom serving graph nodes.
- **Model monitoring apps**: implement custom monitoring apps in `mlrun.model_monitoring.applications/`.

### Feature Flags

- **Import-time feature flags**: check `mlrun.mlconf.<feature>` (e.g., `mlrun.mlconf.igz_version` for Iguazio integration).
- **Runtime feature flags**: controlled via `mlrun.common.schemas.FeatureFlags` (server-side).

## Further Reading

### Documentation

- **Architecture**: `docs/architecture.md` – high-level system design.
- **Contributing**: `CONTRIBUTING.md` – dev environment setup, coding conventions, PR guidelines.
- **Deprecation process**: `DEPRECATION.md` – how to deprecate APIs/parameters/endpoints.
- **Cheat sheet**: `docs/cheat-sheet.md` – quick reference for common SDK operations.

### Code Navigation

- **SDK ↔ server boundary**: `mlrun/db/httpdb.py`, `mlrun/db/base.py`.
- **Project layer**: `mlrun/projects/project.py`, `mlrun/projects/pipelines.py`.
- **Launcher selection**: `mlrun/launcher/factory.py`, `mlrun/launcher/`, `server/py/services/api/launcher.py`.
- **Server endpoints**: `server/py/services/api/api/endpoints/`.
- **Server DB/session infra**: `server/py/framework/db/`, `server/py/framework/rundb/`.
- **Go log collector**: `server/go/services/logcollector/`.

### External Resources

- **MLRun Docs**: [docs.mlrun.org](https://docs.mlrun.org/) (stable version on ReadTheDocs).
- **Tutorials**: [Tutorials](https://docs.mlrun.org/en/stable/tutorials/index.html) (Jupyter notebooks).
- **API Reference**: [API Reference](https://docs.mlrun.org/en/stable/api/index.html) (auto-generated from docstrings).

### Internal Docs

- **Pipeline adapters**: `pipeline-adapters/README.md` (KFP integration details).
- **Automation scripts**: `automation/` (release notes, version management, deployment).
- **Local dev setup**: `hack/local/README.md` (running MLRun locally on Kubernetes).
