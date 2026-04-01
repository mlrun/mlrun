(packagers-overview)=
# Packagers overview

Learn about built-in and custom packagers, and how to configure and use them. 

**In this section**
- [What are packagers?](#what-are-packagers)
- [Why use packagers?](#why-use-packagers)
- [How to use packagers](#how-to-use-packagers)
- [Configuration](#configuration)
- [Built-in packagers](#built-in-packagers)
- [Creating a custom packager](#creating-a-custom-packager)


## What are packagers?

Writing a function locally and running it remotely should feel identical. For example, 
running locally, your function accepts a DataFrame and returns a cleaned dataset — objects 
live in memory and everything just works. But when you move that same function to a remote 
job, Python objects can't be sent over the wire, and whatever the function returns simply 
disappears into the void. 

Packagers bridge this gap: they serialize inputs before they reach your code and
capture outputs after it runs, handling all the MLRun-specific I/O behind the scenes so
your function stays pure Python regardless of where it executes.

Every MLRun function has two I/O touch-points:

- **Input parsing** — automatically cast `DataItem` inputs to the type-hinted Python type (e.g. `pd.DataFrame`, 
   `np.ndarray`, `dict`)
- **Output logging** — automatically serialize, log, and upload returned objects as artifacts or results based on log 
   hints

The flow looks like this:

**Input flow:** `inputs={"data": "store://..."}` → `DataItem` → packager `unpack()` → typed Python object → 
your function

**Output flow:** your function `return` → Python object → packager `pack()` → `Artifact` / `Result` → artifact store

## Why use packagers?

Packagers offer several advantages over manual artifact handling and the legacy context-based API.

### Better and faster learning curve

With packagers you don't need to learn about `Artifact`s, `DataItem`s, or the MLRun context object. You write standard 
Python with type hints and returning values — MLRun wraps your existing code without changing it.

**Before** — manual artifact handling:

```python
import mlrun
import pandas as pd


def clean_data(context: mlrun.MLClientCtx, raw_data: mlrun.DataItem):
    # Parse input manually
    df = raw_data.as_df()

    # Drop rows with missing values and duplicates
    cleaned = df.dropna().drop_duplicates()
    row_count = len(cleaned)

    # Log outputs manually
    context.log_result("row_count", row_count)
    context.log_dataset("cleaned_data", df=cleaned, format="parquet")
```

**After** — with packagers:

```python
import pandas as pd


def clean_data(raw_data: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    cleaned = raw_data.dropna().drop_duplicates()
    return len(cleaned), cleaned
```

The function is pure Python — no MLRun imports, no manual serialization. When you run it with:

```python
fn.run(
    handler="clean_data",
    inputs={"raw_data": "store://my-raw-data"},
    returns=["row_count", "cleaned_data : dataset"],
)
```

MLRun automatically converts the `DataItem` to a DataFrame on input and logs the row count as a result and the
cleaned DataFrame as a dataset artifact on output.

### Uniformity of artifacts between users and projects

ML engineers can establish a standardized method for artifact serialization once, ensuring consistent enforcement 
across every development notebook,, CI pipeline, and production project in the organization.
Because packagers standardize the serialization format, artifacts become truly portable — a
DataFrame logged in one project can be consumed by a function in a completely different
project without conversion steps or format mismatches (assuming, of course, that access is allowed across 
these projects).

In a pipeline, functions don't need to agree on file formats or know about MLRun's artifact
API. The producer just `return`s the object and the consumer receives it as a typed
parameter:

```python
# Producer — returns a DataFrame
def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.dropna()


# Consumer — receives a DataFrame directly
def train_model(data: pd.DataFrame): ...
```

No manual `data_item.as_df()` calls, no format negotiation — the same artifact flows
cleanly between functions, projects, and teams.

### Adaptive to user needs

MLRun provides common built-in packagers with rich options and configurations. For
example, you can control the output format with a single log hint string:

```python
returns = ['data : dataset[format="parquet"]']
```

or equivalently using the {py:class}`~mlrun.package.log_hint.LogHint` class for full
control:

```python
from mlrun import LogHint

returns = [
    LogHint(key="data", artifact_type="dataset", packing_kwargs={"format": "parquet"})
]
```

Either form replaces manual pandas I/O and artifact construction.

Beyond built-in packagers, MLRun supports **custom packagers** that you write and
register in your project to handle domain-specific types.
See the {ref}`custom packagers tutorials <custom-packagers-tutorials>`.

## How to use packagers

### Parsing inputs with type hints

Type hints on function parameters tell packagers what Python type each input
should be converted to. When you pass a value via `inputs={}`, it arrives as a
`DataItem`. The packager looks at the type hint and automatically converts it
to the declared type — `pd.DataFrame`, `dict`, `np.ndarray`, etc.

```python
def my_handler(data: pd.DataFrame, config: dict):
    # `data` is already a DataFrame — no .as_df() needed
    # `config` is already a dict — no .get() / json.loads() needed
    ...
```

```python
fn.run(
    handler="my_handler",
    inputs={"data": "store://my-dataset", "config": "store://my-config"},
)
```

Packagers are enabled by default (`mlrun.mlconf.packagers.enabled = True`).
When enabled, the runtime automatically parses all type-hinted arguments
that are passed via `inputs={}`. To disable parsing for a specific run,
set `mlrun.mlconf.packagers.enabled = False`.

### Logging outputs with log hints

A log hint tells MLRun how to log a single returned value — what key to store it
under, what artifact type to use, and any serialization options. Log hints are
passed via the `returns` parameter on `function.run()`:

```python
fn.run(
    handler="train",
    inputs={"dataset": "store://my-dataset"},
    returns=["accuracy", "X_test : dataset", "model : model"],
)
```

Each entry in the `returns` list is a log hint — either a `LogHint` object or a
string shortcut. The sections below cover artifact types, the LogHint class, and
the string shortcut format.

#### Artifact types

The artifact type is a string that determines how an object is serialized and
what metadata is stored. MLRun defines common types in `mlrun.package.ArtifactType`,
but custom packagers can implement any artifact type string they need — these are
just conventions that built-in packagers share:

| Type | Description | Typical objects |
|------|-------------|-----------------|
| `result` | Scalar/simple value stored in run metadata | `int`, `float`, `str`, small `dict`/`list` |
| `dataset` | Tabular data logged as a `DatasetArtifact` | `pd.DataFrame` |
| `file` | Generic file upload | `np.ndarray`, `bytes`, large dicts |
| `model` | ML model artifact | scikit-learn models, torch models |
| `plot` | Visualization | matplotlib figures |
| `object` | Pickle serialization (fallback) | Any Python object |
| `path` | File/directory path | `str`, `pathlib.Path` |

If you don't specify an artifact type, the packager for the object's type chooses
a sensible default. Custom packagers define their own defaults via
`DEFAULT_PACKING_ARTIFACT_TYPE`.

##### Asymmetric (pack-only / unpack-only) artifact types

Packing and unpacking artifact types are discovered independently. A `DefaultPackager`
subclass with `pack_foo` but no `unpack_foo` supports `"foo"` for packing only —
`is_packable` accepts it but `is_unpackable` rejects it. The reverse also applies:
`unpack_bar` without `pack_bar` means `"bar"` is unpack-only.

Common scenarios:

- **Pack-only** — saving plots as images, logging summary metrics as plain results,
  rendering a model to an image (the PNG can't be deserialized back to the original
  object)
- **Unpack-only** — legacy/migration support (e.g. `unpack_v1` reads artifacts from
  an older packager version while new writes always use `pack_v2`); cross-format
  compatibility (e.g. a DataFrame packager can `unpack_csv` to read manually-logged
  CSV artifacts but always `pack_parquet` for new outputs)

#### The LogHint class

A {py:class}`~mlrun.package.log_hint.LogHint` gives you full control over logging —
artifact type, labels, extra data, metrics, and more:

```python
from mlrun import LogHint

returns = [
    LogHint(key="model", artifact_type="model", labels={"version": "1"}),
    LogHint(key="data", artifact_type="dataset", packing_kwargs={"format": "csv"}),
]
```

A `LogHint` has the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `key` | `str` | **Required.** The artifact key to log the object under. |
| `artifact_type` | `str \| None` | The artifact type (e.g. `"dataset"`, `"model"`, `"result"`). If `None`, the packager's default is used. |
| `tag` | `str` | Tag for the artifact. Default: `""`. |
| `itemized` | `bool \| int` | Unbundling control. `False` (default): log as one artifact. `True`: fully unbundle. `int`: unbundle to that depth. |
| `packing_kwargs` | `dict` | Extra keyword arguments passed to the packager's `pack_<type>()` method (e.g. `{"format": "parquet"}`). |
| `labels` | `dict[str, str]` | Labels to add to the logged artifact. |
| `extra_data` | `dict` | Extra data to attach to the artifact. Use `...` (Ellipsis) as a value to link to another package by key. |
| `metrics` | `dict` | Metrics to log alongside a model artifact. Use `...` (Ellipsis) as a value to link to another package by key. |

#### Linking artifacts

When a function returns multiple values, you can **link** them together so that
related outputs are attached to a primary artifact. For example, you might want a
model artifact to carry its evaluation metrics and supporting artifacts (plots, test
data) as part of its metadata. This is done through the `extra_data` and `metrics`
fields of `LogHint`, using Python's `...` (Ellipsis) as a placeholder meaning
"fill this in with the package that has this key."

Consider a training function that returns a model alongside its metrics, a loss plot,
and a test dataset:

```python
def train(dataset: pd.DataFrame):
    # ... training logic ...
    return my_model, some_result, loss_plot, test_dataset


fn.run(
    handler="train",
    inputs={"dataset": "store://my-dataset"},
    returns=[
        LogHint(
            key="my_model",
            artifact_type="model",
            metrics={"some_result": ...},
            extra_data={"loss_plot": ..., "test_dataset": ...},
        ),
        "some_results : result",
        "loss_plot : plot",
        "test_dataset : dataset",
    ],
)
```

After all four values are packed, the packager manager resolves every `...`:

- `"some_result"` is a result (scalar), so it is placed into the model's `metrics`
- `"loss_plot"` and `"test_dataset"` are artifacts, so they are placed into the
  model's `extra_data`

The result is a model artifact with its evaluation metrics and supporting data
attached directly — visible as a single unit in the MLRun UI.

```{note}
**Linking rules:**

* `metrics` is available only on **model** artifacts and can link to **results** only
  (scalar values in run metadata)
* `extra_data` works with any artifact type and can link to both artifacts and results
* If a referenced key is not found among the packed outputs, the entry is removed and
  a warning is logged
* The order of items in `returns` does not matter — linking is resolved after all
  packing is complete
```

#### String shortcut

The most common way to specify a log hint. A string shortcut has up to four parts:

| Part | Syntax | Purpose | Example |
|------|--------|---------|---------|
| **Key** (required) | `"<key>"` | The artifact name | `"accuracy"` |
| **Artifact type** | `"<key> : <type>"` | Override the default type | `"data : dataset"` |
| **Packing kwargs** | `"<key> : <type>[k='v', ...]"` | Pass options to the packager | `'data : dataset[format="parquet"]'` |
| **Itemization prefix** | `"*<key>"` or `"<N>*<key>"` | Unbundle a collection | `"*results"`, `"2*results"` |

Examples and their `LogHint` equivalents:

| String | Equivalent LogHint |
|--------|--------------------|
| `"accuracy"` | `LogHint(key="accuracy")` |
| `"data : dataset"` | `LogHint(key="data", artifact_type="dataset")` |
| `'data : dataset[format="parquet"]'` | `LogHint(key="data", artifact_type="dataset", packing_kwargs={"format": "parquet"})` |
| `"*results"` | `LogHint(key="results", itemized=True)` |
| `"2*results"` | `LogHint(key="results", itemized=2)` |

(unbundling)=
#### Itemization (unbundling)

Unbundling breaks a collection (list or dict) into separate artifacts, each logged individually. This is useful when a 
function returns a dictionary of DataFrames and you want each one as its own dataset artifact.

```python
def evaluate(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Returns per-category evaluation results."""
    results = {}
    for category in data["category"].unique():
        subset = data[data["category"] == category]
        results[category] = compute_metrics(subset)
    return results
```

Without unbundling, the entire dict is logged as a single artifact. With unbundling:

```python
fn.run(handler="evaluate", inputs={"data": "store://eval-data"}, returns=["*results"])
```

Each DataFrame in the dict becomes its own dataset artifact, keyed as `results_<category_name>`.

**Depth control**

- `"*results"` or `itemized=True` — fully recursive unbundling.
- `"2*results"` or `itemized=2` — unbundle up to 2 levels deep. Nested collections beyond that depth are logged as 
  single artifacts.

## Configuration

Packager behavior is controlled by settings under `mlrun.mlconf.packagers`:

| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | `True` | Master switch. When enabled, MLRun automatically wraps every function execution with the packager handler — parsing typed inputs and logging returned outputs. Set to `False` to disable all packager functionality. |
| `auto_unpack_inputs` | `False` | When `True`, inputs that have **no type hint** are still automatically unpacked if they were originally logged via packagers. When `False` (default), un-hinted inputs remain as raw `DataItem` objects. |
| `auto_pack_outputs` | `False` | When `True`, returned objects are packed even if no log hints were provided by the user. The artifact key follows the pattern `<context_name>-<auto_pack_key>-<i>` where `i` is enumerated. When `False` (default), returned objects without log hints are ignored. |
| `auto_pack_key` | `"artifact"` | The base key used in the auto-generated artifact name when `auto_pack_outputs` is enabled. |
| `pack_tuples` | `False` | When `True`, returned tuples are treated as a single tuple object and packed together. When `False` (default), each element of a returned tuple is packed as a separate output — enabling functions to return multiple items via `return a, b, c`. |
| `logging_worker` | `0` | In multi-worker runs, only the worker with this rank packs outputs and logs results/artifacts. Other workers skip logging to avoid overriding each other. Default is `0` (the main worker). |

You can change these settings globally:

```python
import mlrun

mlrun.mlconf.packagers.auto_unpack_inputs = True
```

```{note}
You can also set these options via environment variables. Use the `MLRUN_` prefix
with `__` (double underscore) as the nesting separator:

    MLRUN_PACKAGERS__ENABLED=true
    MLRUN_PACKAGERS__AUTO_PACK_OUTPUTS=true
```

## Built-in packagers

MLRun includes packagers for common Python types. All built-in packagers are available automatically — no registration 
needed.

### Python standard library

Handles `None`, `int`, `float`, `bool`, `str`, `dict`, `list`, `tuple`, `set`,
`frozenset`, `bytes`, `bytearray`, and `pathlib.Path`.

API reference: {py:mod}`~mlrun.package.packagers.python_standard_library_packagers`

### NumPy

Handles `np.ndarray`, `np.number`, and collections of arrays (`list[np.ndarray]`,
`dict[str, np.ndarray]`).

API reference: {py:mod}`~mlrun.package.packagers.numpy_packagers`

### Pandas

Handles `pd.DataFrame` and `pd.Series`.

API reference: {py:mod}`~mlrun.package.packagers.pandas_packagers`

### Default (fallback)

Any unrecognized type is handled by the {py:class}`~mlrun.package.packagers.default_packager.DefaultPackager`, which 
serializes objects using `cloudpickle` (or any pickling module configured). The default artifact type is `object`.

(custom-packagers)=
## Creating a custom packager

When a built-in packager doesn't handle your type (or you want human-readable serialization
instead of pickle), you can write a custom packager. The
{ref}`custom packagers guide <custom-packagers-tutorials>` walks through the full process —
choosing a base class, setting class variables, implementing pack/unpack methods, and
registering the packager in your project.

```{note}
When running remotely, set the project source with `pull_at_runtime=True`
so the packager module can be imported on the remote worker.
See [Setting a project source, either remote or archive](../../projects/automate-project-git-source.ipynb#setting-a-project-source-either-remote-or-archive). The git repo or the archive file needs to include the custom packager files.

The custom packager needs to be available during runtime. See how to do that
in {ref}`Make the packager importable on the remote worker <make-the-packager-importable-on-the-remote-worker>`.
```

**See also**
- {ref}`auto-logging-mlops` — framework-specific auto-logging with `apply_mlrun()`
- {ref}`working-with-data-and-model-artifacts` — manual artifact handling
- [mlrun.package](../../api/mlrun.package/index.rst) — API reference
