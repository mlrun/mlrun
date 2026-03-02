(custom-packagers-tutorial-overview)=
# Custom packager tutorials overview
Learn when custom packagers are required, and how to create them.

**In this section**
- [When to write a custom packager](#when-to-write-a-custom-packager)
- [Choosing a base class: `DefaultPackager` vs `Packager`](#choosing-a-base-class-defaultpackager-vs-packager)
- [The four patterns](#the-four-patterns)
- [Step-by-step guide](#step-by-step-guide)

## When to write a custom packager

Write a custom packager when:

- **Your type isn't handled by a built-in packager** — for example, a PIL Image,
  a LangChain prompt template, or a domain-specific data class
- **You want human-readable serialization** — save as JSON, PNG, CSV, etc.
  instead of an opaque pickle file
- **You need bundling support** — your type is a collection that should be
  decomposed into individual artifacts when unbundled with `"*key"`

## Choosing a base class: `DefaultPackager` vs `Packager`

MLRun provides two base classes for custom packagers. In most cases you should use
`DefaultPackager`.

### `DefaultPackager` (recommended)

{py:class}`~mlrun.package.packagers.default_packager.DefaultPackager` is the recommended
base class for custom packagers. It implements all the abstract methods from `Packager`
with sensible default logic — routing pack/unpack calls to the right method by artifact
type, validating arguments, and falling back to pickle when needed. Instead of overriding
abstract methods, you configure behavior through **class variables** and implement
**named methods** (`pack_<artifact_type>`, `unpack_<artifact_type>`).

The class variables you can set:

| Variable | Default | Description |
|----------|---------|-------------|
| `PACKABLE_OBJECT_TYPE` | `...` (any) | The Python type this packager handles. Used by `is_packable` and `is_unpackable` to match objects and type hints. |
| `PACK_SUBCLASSES` | `False` | When `True`, this packager also handles subclasses of `PACKABLE_OBJECT_TYPE`. |
| `DEFAULT_PACKING_ARTIFACT_TYPE` | `"object"` | The artifact type to use when the user doesn't specify one in the log hint. |
| `DEFAULT_UNPACKING_ARTIFACT_TYPE` | `"object"` | The artifact type to use when unpacking a `DataItem` that wasn't originally packed by this packager (e.g. a manually logged artifact). |
| `BUNDLE_FROM_LIST` | `False` | When `True`, the type can be initialized from a `list` to serve as a bundle container. |
| `BUNDLE_FROM_DICT` | `False` | When `True`, the type can be initialized from a `dict` to serve as a bundle container. |

`DefaultPackager` **auto-discovers** supported artifact types by scanning for methods
independently: `pack_<artifact_type>` methods define packing artifact types and `unpack_<artifact_type>` methods
define unpacking artifact types. If your class has `pack_file` but no `unpack_file`,
then `"file"` is available for packing only — `is_packable` accepts it but
`is_unpackable` rejects it. The `"result"` type is always available for packing
(logging scalar values as run metadata). The `"object"` (pickle) type is always
available for both packing and unpacking.

If needed, you can still override methods like `is_packable`, `is_unpackable`, `get_default_packing_artifact_type`, 
`get_default_unpacking_artifact_type` etc. to customize the default behavior. For example, when the default artifact 
type depends on runtime conditions rather than a fixed value.

### `Packager` (full control)

The base {py:class}`~mlrun.package.packager.Packager` class gives you complete control.
You override `pack()` and `unpack()` directly and manage artifact-type routing, validation,
and fallback behavior yourself. Use this only when `DefaultPackager`'s convention-based
approach doesn't fit your needs.

## The four patterns

Custom packagers follow one of four patterns, depending on what your type needs:

| Pattern | When to use | What to implement |
|---------|-------------|-------------------|
| **Pack-only** | The type is produced as output but never consumed as a typed input. The framework automatically excludes pack-only artifact types from unpacking validation. | `pack_<artifact_type>` methods only. |
| **Unpack-only** | Legacy/migration support — reading artifacts from an older format while new writes use a different artifact type. | `unpack_<artifact_type>` methods only. |
| **Round-trip (pack + unpack)** | The type needs to be saved *and* loaded back in a later function. | Both `pack_<artifact_type>` and `unpack_<artifact_type>` methods. |
| **Bundling & unbundling** | The type is a collection that should decompose into separate artifacts when unbundled. | `pack_<artifact_type>`/`unpack_<artifact_type>` plus `bundle`/`unbundle` methods and `BUNDLE_FROM_LIST`/`BUNDLE_FROM_DICT` flags. |

## Step-by-step guide

### 1. Subclass `DefaultPackager`

```python
from mlrun.package import ArtifactType
from mlrun.package.packagers.default_packager import DefaultPackager


class MyTypePackager(DefaultPackager): ...
```

### 2. Set class variables

At a minimum, set the type your packager handles and the default artifact type:

```python
class MyTypePackager(DefaultPackager):
    PACKABLE_OBJECT_TYPE = MyType
    DEFAULT_PACKING_ARTIFACT_TYPE = ArtifactType.FILE
    DEFAULT_UNPACKING_ARTIFACT_TYPE = ArtifactType.FILE
```

If your type has subclasses that should also be handled by this packager, set
`PACK_SUBCLASSES = True`.

### 3. Implement `pack_<artifact_type>()` methods

Each packing method serializes the object and returns a tuple of `(Artifact, instructions_dict)`.
The `instructions_dict` carries metadata needed to reconstruct the object when unpacking.
Returning an artifact means that you can return any of the common subclasses of `Artifact`, including:
`ModelArtifact`, `DatasetArtifact` and `LLMPromptArtifact`.

```python
from mlrun import Artifact


def pack_file(
    self, obj: MyType, key: str, file_format: str = "json"
) -> tuple[Artifact, dict]:
    # Serialize to a temporary file
    path = f"/tmp/{key}.{file_format}"
    obj.save(path)

    # Create the artifact
    artifact = Artifact(key=key, src_path=path)

    # Clean up the temp file after upload
    self.add_future_clearing_path(path)

    # Return artifact + instructions for unpacking
    return artifact, {"file_format": file_format}
```

```{note}
Inside a `pack_<artifact_type>` method you create and **return** an `Artifact` object — you do not
call `context.log_artifact()` or `context.log_dataset()`. The packager manager handles
the actual logging and uploading; the pack method's job is only to serialize the data and
describe the artifact.
```

The method name determines the artifact type: `pack_file` handles `artifact_type="file"`,
`pack_plot` handles `"plot"`, and so on.

**Important:** Extra parameters like `file_format` above become **packing kwargs** that
users can pass via log hints:

The class variables `DEFAULT_PACKING_ARTIFACT_TYPE` must be equal to one of the artifact types defined by your `pack_<artifact_type>` 
methods, so that when users log without an explicit artifact type, the packager knows which method to call. 

```python
returns = ['my_output : file[file_format="csv"]']
```

All packing kwargs must have default values so users aren't forced to specify them.

#### Result artifact type

A **special case** of a `pack_<artifact_type>()` method is the **result** artifact type — a scalar 
or simple value (`int`, `float`, `str`, `bool`) stored directly in run metadata (visible 
in `run.status.results` and in the MLRun UI without downloading anything). For result 
types, the pack method returns a plain `dict` with the key and value instead of an 
`(Artifact, instructions)` tuple:

```python
def pack_result(self, obj: MyType, key: str) -> dict:
    # Stored as run metadata, not as a file artifact
    return {key: obj.score}
```

`DefaultPackager` already provides a generic `pack_result` implementation, so you only
need to override it if you want custom extraction logic (e.g. pulling a specific field
from your type). The `"result"` type is always available for packing.

### 4. Implement `unpack_<artifact_type>()` methods

Each unpacking method takes a {py:class}`~mlrun.datastore.base.DataItem` and the
instructions that were stored during packing, and returns the reconstructed object:

```python
import mlrun


def unpack_file(self, data_item: mlrun.DataItem, file_format: str = "json") -> MyType:
    # Download the artifact to a local path
    local_path = data_item.local()

    # Reconstruct the object
    return MyType.load(local_path, format=file_format)
```

Each instruction parameter (e.g. `file_format`) must be **optional** (have a default
value) so that the method can also handle objects that were logged manually rather than
through this packager.

The class variables `DEFAULT_UNPACKING_ARTIFACT_TYPE` must be equal to one of the artifact types defined by your 
`unpack_<artifact_type>` 
methods, so that when users log without an explicit artifact type, the packager knows which method to call. 

For pack-only packagers, you can skip implementing `unpack_<artifact_type>` methods entirely —
the artifact type is automatically excluded from unpacking validation, so no extra
configuration is needed. Real-world examples:

- **Pack-only**: PIL Image → PNG (no need to reconstruct the original PIL object
  from the logged PNG)
- **Unpack-only**: reading a legacy serialization format that should no longer be
  written (e.g. `unpack_v1` for backward compatibility while new outputs use
  `pack_v2`)

### 5. Clean up temporary files

If your `pack_<artifact_type>` or `unpack_<artifact_type>` methods write files to disk, call
`self.add_future_clearing_path(path)` so MLRun deletes them after the artifact is
uploaded. This prevents temporary files from accumulating on the worker.

### 6. Set the priority (optional)

The `PRIORITY` class variable (integer 1–10, 1 = highest priority) controls which
packager is selected when multiple packagers can handle the same type. Custom packagers
default to priority **3**, which is higher than the built-in packagers at **5**. You
rarely need to change this unless you have multiple custom packagers competing for the
same type - which is not how the packagers are intended to be used (`XPackager` should 
handle type `x`).

```python
class MyTypePackager(DefaultPackager):
    PRIORITY = 2  # Higher priority than other custom packagers
    ...
```

### 7. Register the packager in your project

Use {py:meth}`~mlrun.projects.project.MlrunProject.add_custom_packager` to register
your packager:

```python
project.add_custom_packager(packager="my_module.MyTypePackager", is_mandatory=True)
```

The `is_mandatory` flag controls what happens when the packager fails to import on a
remote worker:

- `True` — the run fails immediately with an import error
- `False` — the packager is silently skipped and the fallback pickle behavior is used

To remove a registered packager:

```python
project.remove_custom_packager("my_module.MyTypePackager")
```

(make-the-packager-importable-on-the-remote-worker)=
### 8. Make the packager importable on the remote worker

When running remotely, the worker must be able to import your packager module. There
are several ways to achieve this:

* **Pull at runtime** (simplest) — set the project source with `pull_at_runtime=True`
  so the code is fetched before execution:

  ```python
  project.set_source(source="./", pull_at_runtime=True)
  ```

* **Build into the function image** — include the packager source in the function's
  build so it is baked into the container image

* **Shared storage** — place the packager module on a shared volume and configure the
  function's working directory to point there

If the packager module is missing at runtime, the run fails immediately when
`is_mandatory=True`, or falls back to pickle when `is_mandatory=False`.