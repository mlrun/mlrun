(packagers)=
# Packagers

Packagers are the recommended way to move data in and out of MLRun functions.
Instead of manually handling `DataItem`s, artifact uploads, and serialization code,
you write standard Python functions with type hints and returning values — and MLRun
automatically handles input parsing, output logging, and artifact creation.

**In this section**

```{toctree}
:maxdepth: 1

packagers-overview
packagers-tutorial
custom_packagers/index
```

