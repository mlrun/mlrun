.. _mlrun.package:

mlrun.package
=============

MLRun package enables fully-automated experiment and pipeline tracking and reproducibility, and easy
passing of python objects between remote jobs, while not requiring any form of editing to the actual function original code.
Simply set the function code in a project and run it, MLRun takes care of the rest.

MLRun uses packagers: classes that perform 2 tasks:

#. **Parsing inputs** - automatically cast the runtime's inputs (user's input passed to the function via the ``inputs`` parameter of the ``run`` method) to the relevant hinted type.  (Does not require handling of data items.)
#. **Logging outputs** - automatically save, log, and upload the function's returned objects by the provided log hints (user's input passed to the function via the ``returns`` parameter of the ``run`` method). (Does not require handling of files and artifacts.)

.. currentmodule:: mlrun.package

.. autosummary::
   :toctree: ./generated_rsts
   :template: class_summary.rst

   packager.Packager
   packagers.default_packager.DefaultPackager
   packagers_manager.PackagersManager


.. autosummary::
   :toctree: ./generated_rsts
   :recursive:

   errors

.. rubric:: Packagers

MLRun comes with the following list of modules, out of the box. All of the packagers listed here
use the implementation of :ref:`DefaultPackager <mlrun.package.packagers.default\_packager.DefaultPackager>` and are
available by default at the start of each run.

.. autosummary::
   :toctree: ./generated_rsts
   :template: module_summary.rst

   packagers.python_standard_library_packagers
   packagers.numpy_packagers
   packagers.pandas_packagers


