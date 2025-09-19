# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLRun is an open-source AI orchestration platform for building and managing continuous AI applications. It provides end-to-end MLOps capabilities including data management, model training, serving, and monitoring for both traditional ML and GenAI workloads.

## Development Commands

### Environment Setup
```bash
make install-requirements        # Install all development requirements
make install-complete-requirements  # Install all requirements for full development
```

### Code Quality
```bash
make lint                       # Run ruff linter and import checks
make fmt                        # Format code with ruff and blacken-docs
make lint-go                    # Lint Go code
make fmt-go                     # Format Go code
```

### Testing
```bash
make test                       # Run unit tests (excludes integration/system tests)
make test-integration          # Run integration tests 
make test-system               # Run system tests
make test-dockerized           # Run unit tests in Docker container
make test-go-unit              # Run Go unit tests
```

### Building
```bash
make build                     # Build all artifacts (Docker images + Python wheel)
make docker-images             # Build all Docker images (api, mlrun, mlrun-gpu, jupyter, etc.)
make package-wheel             # Build Python package wheel
make api                       # Build mlrun-api Docker image
make mlrun                     # Build mlrun Docker image
```

### Documentation
```bash
make html-docs                 # Build HTML documentation
make clean-html-docs          # Clean documentation build artifacts
```

## Architecture Overview

MLRun follows a distributed architecture with clear separation between client SDK and server components:

### Core Components

- **Client SDK** (`mlrun/`): Python SDK for interacting with MLRun services
  - `artifacts/`: Artifact management (models, datasets, documents)
  - `datastore/`: Data source integrations (S3, Azure, GCS, etc.)
  - `feature_store/`: Feature engineering and serving
  - `frameworks/`: ML framework integrations (sklearn, pytorch, xgboost, etc.)
  - `runtimes/`: Execution runtimes (local, Kubernetes, Spark, Dask, etc.)
  - `serving/`: Real-time model serving pipeline
  - `model_monitoring/`: Model drift detection and monitoring

- **Server Components** (`server/`):
  - `py/`: Python-based API server and services
  - `go/`: Go services (log collector, etc.)

- **Runtime Handlers**: Kubernetes job execution, Nuclio functions, Spark jobs
- **Feature Store**: Data transformation, cataloging, and serving
- **Model Monitoring**: Automated drift detection and alerting

### Key Directories

- `mlrun/api/`: API server implementation
- `mlrun/common/`: Shared utilities and schemas
- `mlrun/db/`: Database abstraction layer
- `mlrun/projects/`: Project management and CI/CD integration
- `tests/`: Comprehensive test suite (unit, integration, system)
- `docs/`: Documentation source files
- `examples/`: Usage examples and demos
- `dockerfiles/`: Docker image definitions

## Development Patterns

### Testing Strategy
- Unit tests in `tests/` mirror the source structure
- Integration tests for external service dependencies
- System tests for end-to-end workflows
- Use pytest fixtures extensively for test setup
- Docker-based testing for consistency

### Code Organization
- Modular design with clear separation of concerns
- Common schemas in `mlrun/common/schemas/`
- Framework-specific code in `mlrun/frameworks/`
- Runtime-specific implementations in `mlrun/runtimes/`

### Configuration
- Main config in `mlrun/config.py`
- Environment-based configuration
- Support for multiple deployment targets (local, Kubernetes, cloud)

## Important Notes

- Python 3.9+ required (support for 3.11, with 3.9 being phased out)
- Uses Ruff for linting and formatting (configured in pyproject.toml)
- Uses `uv` as the modern Python package installer (faster than pip)
- Docker-based development workflow with multi-stage builds
- Extensive CI/CD pipeline with automated testing and building
- Go components for performance-critical services
- Supports multiple ML frameworks and cloud providers
- Strict import dependency rules enforced via import-linter (see pyproject.toml)

## Running Single Tests

To run a specific test file:
```bash
python -m pytest tests/path/to/test_file.py -v
```

To run a specific test function:
```bash
python -m pytest tests/path/to/test_file.py::test_function_name -v
```

For server-side tests:
```bash
python -m pytest server/py/services/api/tests/unit/path/to/test_file.py -v
```

## Coverage

To run tests with coverage:
```bash
make test RUN_COVERAGE=true                    # Unit tests with coverage
make test-integration RUN_COVERAGE=true       # Integration tests with coverage
make coverage-combine                         # Combine all coverage reports
```