.. _mlrun.package:

mlrun.package
=============

MLRun package enable **fully automated experiment and pipeline tracking and reproducibility**, easy
**passing python objects between remote jobs** while **not requiring any form of editing** to the actual function
original code. Simply set the function code in a project and run, MLRun will take care of the dirty work.

MLRun is using packagers - classes that are performing 2 tasks:

#. **Parsing inputs** - automatically cast runtime's inputs (user's input passed to the function via the ``inputs`` parameter of the ``run`` method) to the relevant hinted type - no need to handle data items.
#. **Logging outputs** - automatically save, log and upload function's returned objects by the provided log hints (user's input passed to the function via the ``returns`` parameter of the ``run`` method) - no need to handle files and artifacts.

To know more about packagers, see an example and how to write your own custom packager, click here (coming soon).

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

Below is a list of all the modules including the packagers MLRun comes with out of the box. All of the packagers here
use the implementation of :ref:`DefaultPackager <mlrun.package.packagers.default\_packager.DefaultPackager>` and are
available by default at the start of each run.

.. autosummary::
   :toctree: ./generated_rsts
   :template: module_summary.rst

   packagers.python_standard_library_packagers
   packagers.numpy_packagers
   packagers.pandas_packagers


