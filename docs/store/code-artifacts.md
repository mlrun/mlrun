(code-artifacts)=
# Code artifacts

A code artifact (`kind="code"`) stores a function or workflow source file (or an archive of files) as a versioned MLRun artifact. Once logged, the artifact can be referenced by a `store://` URI as the source for {py:meth}`~mlrun.projects.MlrunProject.set_function` or {py:meth}`~mlrun.projects.MlrunProject.set_workflow`, in addition to a local path, git source, or remote URL. The code is downloaded by the runner pod at runtime — the client never resolves the URI.

**In this section**
- [SDK](#sdk)
- [Log a code artifact](#log-a-code-artifact)
- [Run a function from a code artifact](#run-a-function-from-a-code-artifact)
- [Handler format](#handler-format)
- [Run a workflow from a code artifact](#run-a-workflow-from-a-code-artifact)
- [Requirements and rebuild behavior](#requirements-and-rebuild-behavior)
- [function vs workflow code_type](#function-vs-workflow-code_type)
- [Project export behavior](#project-export-behavior)

**See also**
- {ref}`artifacts`

## SDK
- {py:meth}`~mlrun.projects.MlrunProject.log_code_file`
- {py:meth}`~mlrun.projects.MlrunProject.set_function`
- {py:meth}`~mlrun.projects.MlrunProject.set_workflow`

## Log a code artifact

Use {py:meth}`~mlrun.projects.MlrunProject.log_code_file` to register a Python file or archive as a code artifact:

```python
# Minimal — local file uploaded to the project's artifact path
artifact = project.log_code_file("my_func_code", local_path="./my_func.py")
```

Common variants:

```python
# Upload to a specific remote target
project.log_code_file(
    "my_func_code",
    local_path="./my_func.py",
    target_path="s3://bucket/funcs/my_func.py",
)

# Reference an existing object via a datastore profile (recommended when
# credentials should be centralized server-side)
project.log_code_file(
    "my_func_code",
    target_path="ds://my-profile/funcs/my_func.py",
)

# Inline body (small files only)
project.log_code_file("my_func_code", body=b"def main():\n    return 'ok'\n")

# Archive containing multiple source files; extracted on resolution
project.log_code_file("my_pkg_code", local_path="./my_pkg.zip")

# With an explicit code_type and pip dependencies
project.log_code_file(
    "my_workflow_code",
    local_path="./my_workflow.py",
    code_type="workflow",
    requirements=["pandas>=2.0"],
)
```

`language` is auto-derived from the file suffix when omitted (`.py` / `.ipynb` → `"python"`). `code_type` defaults to `"function"`. See [function vs workflow code_type](#function-vs-workflow-code_type) and [Requirements and rebuild behavior](#requirements-and-rebuild-behavior) for those parameters.

## Run a function from a code artifact

Pass the artifact's `store://` URI as the `func` argument:

```python
fn = project.set_function(
    func="store://artifacts/my-project/my_func_code",
    name="my_func",
    kind="job",
    handler="my_func:main",
)

fn.run(params={"p1": 5})
```

The `store://` URI is stored as-is in the function and in the MLRun DB. The runner pod resolves the artifact and downloads the code at startup ("pull at runtime").

For Nuclio and serving functions, the default is "pull at buildtime" mode: the server downloads the code at deploy and embeds it in the processor image. Set `load_source_on_run=True` on the function spec to switch to "pull at runtime" mode, where an init container fetches the code at pod startup (no image rebuild on code-only changes). See [function storage](../runtimes/function-storage.md) for the general source-loading mechanism.

When running locally (`fn.run(local=True)`), the client resolves the artifact and must have datastore credentials configured.

## Handler format

When the source is a `store://` code artifact, the `handler` argument must be in `"<module>:<function>"` form:

```python
artifact = project.log_code_file("my_func_code", local_path="./my_func.py")

fn = project.set_function(
    func=artifact.uri,
    name="my_func",
    kind="job",
    handler="my_func:main",  # <module>:<function>
)
fn.run()
```

The module segment is the file's basename without the `.py` extension. The colon form is required because the runner downloads the artifact at startup — there is no local `spec.command` pointing at a file, so the handler itself must identify which extracted module to load.

When the artifact is an archive (`.zip` / `.tar.gz`), the module segment refers to a file **inside the extracted archive**, not to the archive itself. For example, if `my_pkg.zip` extracts to `trainer.py` and `utils.py`, the handler is `"trainer:main"` — not `"my_pkg:main"`:

```python
artifact = project.log_code_file("my_pkg_code", local_path="./my_pkg.zip")

fn = project.set_function(
    func=artifact.uri,
    name="trainer",
    kind="job",
    handler="trainer:main",  # file inside the archive, not "my_pkg"
)
```

## Run a workflow from a code artifact

`set_workflow` accepts a `store://` URI as `workflow_path`:

```python
project.set_workflow(
    name="main",
    workflow_path="store://artifacts/my-project/my_workflow_code",
)

project.run(name="main", arguments={"p1": 5})
```

The workflow file is downloaded by the runner pod at execution time. Supported engines: `kfp`, `remote`, `remote:kfp`, `remote:local`, `local`.

## Requirements and rebuild behavior

A code artifact can declare its own pip dependencies:

```python
project.log_code_file(
    "my_func_code",
    local_path="./my_func.py",
    requirements=["pandas>=2.0", "numpy"],
)
```

At deploy or run time the server merges the artifact's `requirements` into `function.spec.build.requirements`. User-set requirements (via `func.with_requirements()`) take priority over artifact requirements, and deduplication is case-insensitive.

A change to the artifact's `requirements` is a build-configuration change, and like any other such change it is not detected automatically (`auto_build` does not pick it up). Rebuild the image as described in {ref}`build-function-image`.

A change to the code only (no requirements change) depends on the source-loading mode described in [Run a function from a code artifact](#run-a-function-from-a-code-artifact). When the code is resolved at build time it is embedded in the image, so a code change is also a build-configuration change and the image must be rebuilt. When the code is resolved at runtime it is fetched by the pod at startup, so a code change is not a build-configuration change and no rebuild is needed. For job and KFP functions the next `run()` picks up the latest artifact version.

## function vs workflow code_type

Code artifacts carry a `code_type` field whose value (`"function"` or `"workflow"`) tells consumers what the artifact is intended for. Using an artifact in the wrong role raises `MLRunInvalidArgumentError` at the `set_function` / `set_workflow` call (and again on the server at deploy time):

```python
project.log_code_file(
    "my_workflow_code",
    local_path="./my_workflow.py",
    code_type="workflow",
)

# This raises MLRunInvalidArgumentError because the artifact is a workflow:
project.set_function(
    func="store://artifacts/my-project/my_workflow_code",
    name="bad",
    kind="job",
    handler="my_workflow:main",
)
```

## Project export behavior

`project.export()` behaves differently depending on whether code artifacts are referenced:

- **Zip export** — the code is retrieved from each referenced artifact and packaged into the zip. On import the function references a local file path inside the unpacked project, so the project is portable across clusters.
- **YAML-only export** (`project.yaml`) — the `store://` URI is preserved as-is. The project is only usable on a cluster that already has the same artifact.
