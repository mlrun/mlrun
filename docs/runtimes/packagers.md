(packagers)=
# Packagers
 
 The `mlrun.package` is a sub-package in MLRun for packing returning outputs, logging them to MLRun, 
 unpacking inputs, and parsing data items to their required type. It provides
 
- Full auto experiment tracking and reproducibility.
   - Full auto experiment tracking and reproducibility.
   - Each function and step in a workflow is logged.
- Pipelines can be rerun from a chosen step with full reproducibility since each step is logged automatically.
   - Simpler and faster learning curve for using MLRun
   - No need to know about Artifacts, DataItems and context (MLClientCtx).
- MLRun is integrated into existing code without changing it, making passing objects between runtimes 
a transparent pythonic experience.
   - Uniformity of artifacts between users and projects
   - You can pass artifacts and use them as input in every function without editing them.
   - Artifacts can be imported and exported between projects.
     
See {py:class}`~mlrun.projects.MlrunProject.add_custom_packager`, 
{py:class}`~mlrun.projects.MlrunProject.get_custom_packagers`, [Source code for mlrun.package](../_modules/mlrun/package.html)

## Passing objects between runtimes

To log artifacts and results, you just request to return them from the handler. Since most 
function-based-code return objects, the only change to your code is to specify the log hints in 
the returns argument of the run. Then the outputs are logged according to the hints.

To parse inputs you need to add type hints to them. The artifacts are saved by their original type.

Moreover, if a packaged input is passed, then a type hint is not required since each package is labeled by 
the manager with the original type, artifact type, and packager of it.

## The DefaultPackager
You can also implement the more convenient and flexible DefaultPackager. This class has 
two advantages:

- It implements all of the required abstract methods of the Packager, and it adds a simple logic that 
you can use to implement your own packagers, simplifying your work.
- It is used as the default packager (hence its name) and pickles objects that do not have a 
built-in packager in MLRun, making the mechanism more robust.
  
## Logging an arbitrary number of objects
You can specify, in the log hint key, a prefix of the python unpacking operator:
- ** * ** for a list
- ** ** ** for a dictionary

This indicates that the return object is, in fact, an arbitrary number of objects 
that need to be logged separately as their own artifacts.

## Clearing mechanism
Each packager can specify which outputs to clean at the end of the run to reduce unnecessary outputs 
produced in each run.

## Custom packagers
You can write your own packagers and add them to your 
project for MLRun to use, for logging outputs and parsing inputs of your special python objects. 
Each custom packager is added to a project with an `is_mandatory` flag to specify whether or not 
it must be a collected packager for a run or not. A Mandatory packager raises an error in case it 
fails to be collected.

## Priority
Packagers have a priority from 1 (most important) to 10 (least important) so that you can specify multiple 
custom packagers of the same type.

## Added Packagers
- builtins
   - int
   - float
   - bool
   - str
   - dict
   - list
   - tuple
   - set
   - frozenset
   - bytes
   - bytearray
- pathlib
   - Path (and all inheriting classes)
- numpy
   - numpy.ndarray
   - numpy.number (and all inheriting classes)
   - list[numpy.ndarray]
   - dict[str, numpy.ndarray]
- pandas
   - pandas.Series
   - padnas.DataFrame