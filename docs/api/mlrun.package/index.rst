.. _mlrun.package:

mlrun.package
=============

The ``mlrun.package`` module provides MLRun's packagers system — classes that
automatically serialize and deserialize Python objects moving in and out of MLRun
functions. Your function code stays pure Python; type hints and log hints are all
that's needed.

Packagers perform two tasks:

#. **Parsing inputs** — cast ``inputs`` values to the type-hinted Python type
   (e.g. ``pd.DataFrame``, ``dict``, ``np.ndarray``).
#. **Logging outputs** — serialize returned objects and log them as artifacts or
   results based on the provided log hints (``returns``).

For a full introduction — including usage patterns, log hints, configuration, and
built-in packagers — see :ref:`packagers`.
To create a custom packager, see :ref:`custom-packagers-tutorials`.

.. currentmodule:: mlrun.package

.. autosummary::
   :toctree: ./generated_rsts
   :template: class_summary.rst

   log_hint.LogHint
   packager.Packager
   packagers.default_packager.DefaultPackager
   packagers_manager.PackagersManager

.. autosummary::
   :toctree: ./generated_rsts

   errors

**Built-in packager modules**


MLRun includes the following built-in packager modules. All built-in packagers
subclass :py:class:`~mlrun.package.packagers.default_packager.DefaultPackager` and
are registered automatically at the start of each run.

.. autosummary::
   :toctree: ./generated_rsts
   :template: module_summary.rst

   packagers.python_standard_library_packagers
   packagers.numpy_packagers
   packagers.pandas_packagers
