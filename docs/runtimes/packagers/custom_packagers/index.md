(custom-packagers-tutorials)=
# Creating custom packagers

MLRun's {ref}`built-in packagers <packagers-overview>` cover common Python types — scalars,
collections, NumPy arrays, and Pandas DataFrames. But when your function produces or
consumes a type that isn't built-in, the default behavior is to **pickle** the object
with `cloudpickle` (or any configured pickling module of your choice). Pickle files are opaque, 
not human-readable, and tied to specific Python and module versions — making them fragile and hard to inspect.

A **custom packager** gives you full control over how your type is serialized and
deserialized, producing readable, portable artifacts (PNG images, JSON configs, CSV
tables) instead of pickle blobs.

**Reminder**: Packing applies to function **outputs** (return values → artifacts) and unpacking applies to 
function **inputs** (artifacts → typed Python objects).

```{toctree}
:maxdepth: 1

custom-packagers-tutorial-overview
pack-only-tutorial
round-trip-tutorial
itemization-tutorial
```
