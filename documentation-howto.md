# Documenting mlrun

This document describe how to write the external documentation for `mlrun`, the
one you can view at https://mlrun.readthedocs.io


## Technology

We use [sphinx](https://www.sphinx-doc.org/en/master/) for documentation.
The documentation files are in
[reStructuredText](https://docutils.sourceforge.io/rst.html) format.
The master document is `docs/index.rst` and the configuration is at
`docs/conf.py`.

To build the doc, run `make html-docs`, then open `docs/_build/html/index.html`

## "External" Documentation
In order to avoid duplication, the `setup` function in `docs/conf.py` copies
over some markdown files into `docs/external/`. It also generates HTML from a
notebook in the `examples` directory which is embedded in `docs/examples.rst`.

## readthedocs
There's a git hook in `readthedocs` that builds the documentation.
See https://readthedocs.org/projects/mlrun/ for more details.
Ask @yaronha to add you to the project if you don't have access.

## Documenting

The master file is `docs/index.rst`, every file included in the `.. toctree::`
section will be a separate HTML page.

`docs/api.rst` is the general documentation and `docs/mlrun.rst` contains the
code documentation.

To add a module to be documented. Add it to `docs/mlrun.rst`, for example:
```rst
mlrun.run module
----------------

.. automodule:: mlrun.run
   :members:
   :show-inheritance:
```

When the object is import in `__init__.py` from a sub module, you'll need to
tell sphinx a bit more one how to document it and *must* add the object to
`__all__`. For example:

```rst
mlrun.projects module
---------------------

.. automodule:: mlrun.projects
   :members:
   :show-inheritance:

.. autoclass:: MlrunProject
   :members:
```
