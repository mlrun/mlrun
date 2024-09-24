# Documenting mlrun

This document describes how to write the external documentation for `mlrun`, the
one you can view at https://docs.mlrun.org/en/latest/.

## Technology

We use [sphinx](https://www.sphinx-doc.org/en/master/) for documentation.
The documentation files are in
[reStructuredText](https://docutils.sourceforge.io/rst.html) format.
The master document is `docs/contents.rst`. Every file included in the `.. toctree::`
section publishes as a separate HTML page.

The configuration is at: `docs/conf.py`.

To build the doc, run `make html` from the `docs` folder, then open `docs/_build/html/index.html`

### Documentation linter

In order to check that documentation doesn't contain any typos and correctly formatted run `make lint-docs`.
This command runs `vale` and `blacken-docs`.
Configuration file for `vale` can be found in `.vale.ini` file.

## "External" Documentation
In order to avoid duplication, the `setup` function in `docs/conf.py` copies
over some markdown files into `docs/external/`. It also generates HTML from a
notebook in the `examples` directory which is embedded in `docs/examples.rst`.

## Docs
There's a git hook in `readthedocs` that builds the documentation.
See https://readthedocs.org/projects/mlrun/ for more details.
Ask @yaronha to add you to the project if you don't have access.

## Structure 

The master file is `docs/contents.rst`

## Language (usage) guidelines

**One idea per sentence**<br>
Don’t join multiple ideas with commas. 
This 1 sentence needs to be split into 3. Every comma should be a period:
In many cases the features can have null values (None, NaN, Inf, ..), the Enrichment routers can substitute the null value with fixed or 
statistical value per feature, this is done through the impute_policy parameter which accepts the impute policy per feature (where * is 
used to specify the default), the value can be fixed number for constants or $mean, $max, $min, $std, $count for statistical values.

**Hyphens, dashes**<br>
- To join two nouns, use a - (hyphen). <br>
- Use colons or &mdash; when separating text (not hyphen):
   - Yes: `v3io` storage through API: 
   - Yes: `v3io` storage through API &mdash; (m dashes are always surrounded by spaces)
   - No: `v3io` storage through API - 
- For a range of numbers, use the &ndash;  For example: The range is 2&ndash;4. 

**Commas**<br>
The commas in the docs were (almost) standardized with what is called the Oxford comma. That is the comma before the "and":
- Yes: The feature store supports using Spark for ingesting, transforming, and writing results to data targets.
- No: The feature store supports using Spark for ingesting, transforming and writing results to data targets.
(It’s acceptable to write both with and without that comma. But the docs now use it.)

Use commas before an independent clause:
- Yes: While each of those layers is independent, the integration provides much greater value and simplicity.
- As mentioned above, don't use commas to string together multiple independent phrases to make a (very long) sentence.

**That vs. which**<br>
It depends on your sentence. Use **which** after a comma.
- Yes: There is also an open marketplace that stores many pre-developed functions for...
- Yes: If you update the project object you need to run project.save(), which updates the project.yaml file....
- No: There is also an open marketplace which stores many pre-developed functions for...

**Lists**<br>
- Use numbered lists for steps that are executed in a specific order.
- Use bullets for lists that have no specific order.

**Tense**<br>
Use present, active tense.
- Yes: Use the beat meat for your stew.
- No: The best meat can be used for your stew.
- No: The best meat could be used for your stew.
- No: You’ll use the best meat for your stew.

**Person**<br>
- Use you and yours, not we and ours.
     You can blah blah… Your system blah blah….

**May, might, can**<br>
- Can implies what is possible. For the docs, you can do something because the system supports it.
- Might implies options. 
- May implies permission. Rarely used.<br>
Do not use may in place of can! 
- Yes: You can update the code using...
- Yes: As use-cases evolve, other types of storage access might be needed.
- No: As use-cases evolve, other types of storage access may be needed.

## Cheat sheet

**x-refs**

To x-ref a file by name and not location, use the file id.

Target file starts with:
      (mlops-dev-flow)=
      # MLOps development flow

X-ref looks like one of:
{ref}`mlops-dev-flow`
{ref}`title<mlops-dev-flow>`

## Documenting APIs

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
