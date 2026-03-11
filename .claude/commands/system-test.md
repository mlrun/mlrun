---
description: Run MLRun system tests by component, file, or specific test
argument-hint: <test> [-k filter] [--enterprise|--open-source] [--no-clean]
allowed-tools: Bash, Read, Glob, Grep, AskUserQuestion
---

# Run System Test

You are helping a developer run MLRun system tests. Parse the arguments and construct the correct pytest command.

## Arguments

The user provides: `$ARGUMENTS`

Parse the arguments as follows:

- **Test** (first positional arg, required): One of:
  - A **component name** (directory under `tests/system/`): `runtimes`, `projects`, `feature_store`, `datastore`, `alerts`, `api`, `model_monitoring`, `demos`, `examples`, `hub`, `logs`, `backwards_compatibility`
  - A **file path** like `tests/system/runtimes/test_nuclio.py`
  - A **test name** like `test_basic_api_gateway_flow` — search for the file containing it
  - A **class::method** like `TestNuclio::test_basic_api_gateway_flow`

- **Flags** (optional):
  - `-k <expression>` — pytest keyword filter (passed through to pytest `-k`)
  - `--enterprise` — run only enterprise-marked tests
  - `--open-source` or `--oss` — run only non-enterprise tests (adds `--system-test-open-source -m "not enterprise"`)
  - `--no-clean` — set `MLRUN_SYSTEM_TESTS_CLEAN_RESOURCES=false` to skip resource cleanup after tests
  - `--collect-only` — only collect tests, don't run them (useful to verify which tests will run)
  - Any other flags are passed through to pytest directly

## Steps

1. **Validate environment**:
   - Check that `tests/system/env.yml` exists and contains the **required** variable `MLRUN_DBPATH` (uncommented, with a non-empty value). If missing or empty, warn the user.
   - The following variables are **only required for enterprise tests** (tests marked with `@pytest.mark.enterprise`): `V3IO_ACCESS_KEY`, `MLRUN_IGUAZIO_API_URL`, `V3IO_API`. Only warn about these if the user is running enterprise tests (via `--enterprise` flag or auto-inferred enterprise marker).
   - If `MLRUN_DBPATH` contains `vmdev` (a dev environment with self-signed certs), verify that `MLRUN_HTTPDB__HTTP__VERIFY` is set to `false`. If it's missing or not `false`, warn the user that SSL certificate errors will occur and suggest adding `MLRUN_HTTPDB__HTTP__VERIFY: false` to `env.yml`.
   - If `env.yml` doesn't exist at all, also check if `MLRUN_DBPATH` is set as an environment variable. Warn the user if neither is configured and ask if they want to proceed anyway.

2. **Resolve the target**:
   - If target is a component name, use `tests/system/<component>/`
   - If target is a file path, use it directly
   - If target looks like a test function name (starts with `test_`), search for it in `tests/system/` using Grep and resolve the file path. If found in multiple files, ask the user which one.
   - If target contains `::`, it's a `file::class::method` or `class::method` pattern — resolve the file if needed

3. **Auto-infer enterprise marker** (only when the user did NOT explicitly pass `--enterprise` or `--open-source`/`--oss`):
   - When the target resolves to a **specific test function** (not a whole component or file), check whether the test or its containing class has a `@pytest.mark.enterprise` decorator.
   - Read the resolved file and look upward from the test function definition for `@pytest.mark.enterprise` on the function itself or on the class that contains it.
   - If found, automatically add `-m "enterprise"` and inform the user: "Auto-detected `@pytest.mark.enterprise` on this test, adding `-m enterprise` flag."
   - If NOT found, do not add any marker flag (run as a regular test).
   - This step is skipped when targeting a component directory or full file (too broad to infer).

4. **Build the pytest command**:

```bash
python -m pytest -v \
    --capture=no \
    --disable-warnings \
    --durations=100 \
    -rf \
    [MARKER_FLAGS] \
    [KEYWORD_FLAGS] \
    <resolved_target>
```

Where:
- `MARKER_FLAGS`: `-m "enterprise"` if `--enterprise` (explicit or auto-inferred), or `--system-test-open-source -m "not enterprise"` if `--open-source`
- `KEYWORD_FLAGS`: `-k "<expression>"` if `-k` was provided
- Prepend `MLRUN_SYSTEM_TESTS_CLEAN_RESOURCES=false` if `--no-clean`

5. **Run the test** display run command to the user and execute the test. Display:
   - The full pytest command
   - What test target was resolved
   - Any special flags applied
   - Whether the enterprise marker was auto-inferred

## Examples

- `/system-test runtimes` — run all runtime system tests
- `/system-test tests/system/runtimes/test_nuclio.py` — run nuclio test file
- `/system-test test_basic_api_gateway_flow` — find and run a specific test (auto-infers `--enterprise` if decorated)
- `/system-test runtimes -k nuclio` — run runtime tests matching "nuclio"
- `/system-test projects --no-clean` — run project tests without cleanup
- `/system-test feature_store --enterprise` — run only enterprise feature store tests
- `/system-test runtimes --collect-only` — list which runtime tests would run
