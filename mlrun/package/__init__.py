# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# flake8: noqa  - this is until we take care of the F401 violations with respect to __all__ & sphinx

import functools
import inspect
from collections import OrderedDict

# Imports for the `handler` decorator:
from typing import Callable, Dict, List, Type, Union

# `mlrun.package` __init__ file imports:
from .common import ArtifactType, LogHintKey
from .context_handler import ContextHandler
from .packager import Packager
from .packagers import DefaultPackager
from .packagers_manager import PackagersManager


def handler(
    labels: Dict[str, str] = None,
    outputs: List[Union[str, Dict[str, str]]] = None,
    inputs: Union[bool, Dict[str, Union[str, Type]]] = True,
):
    """
    MLRun's handler is a decorator to wrap a function and enable setting labels, parsing inputs (`mlrun.DataItem`) using
    type hints and log returning outputs using log hints.

    :param labels:  Labels to add to the run. Expecting a dictionary with the labels names as keys. Default: None.
    :param outputs: Log hints (logging configurations) for the function's returned values. Expecting a list of the
                    following values:

                    * `str` - A string in the format of '{key}:{artifact_type}'. If a string was given without ':' it
                      will indicate the key and the artifact type will be according to the returned value type. The
                      artifact types can be one of: "dataset", "directory", "file", "object", "plot" and "result".
                    * `Dict[str, str]` - A dictionary of logging configuration. the key 'key' is mandatory for the
                      logged artifact key.
                    * None - Do not log the output.

                    The list length must be equal to the total amount of returned values from the function. Default is
                    None - meaning no outputs will be logged.

    :param inputs: Type hints (parsing configurations) for the arguments passed as inputs via the `run` method of an
                   MLRun function. Can be passed as a boolean value or a dictionary:

                   * True - Parse all found inputs to the assigned type hint in the function's signature. If there is no
                     type hint assigned, the value will remain an `mlrun.DataItem`.
                   * False - Do not parse inputs, leaving the inputs as `mlrun.DataItem`.
                   * Dict[str, Union[Type, str]] - A dictionary with argument name as key and the expected type to parse
                     the `mlrun.DataItem` to. The expected type can be a string as well, idicating the full module path.

                   Default: True.

    Example::

            import mlrun

            @mlrun.handler(outputs=["my_array", None, "my_multiplier"])
            def my_handler(array: np.ndarray, m: int):
                array = array * m
                m += 1
                return array, "I won't be logged", m

            >>> mlrun_function = mlrun.code_to_function("my_code.py", kind="job")
            >>> run_object = mlrun_function.run(
            ...     handler="my_handler",
            ...     inputs={"array": "store://my_array_Artifact"},
            ...     params={"m": 2}
            ... )
            >>> run_object.outputs
            {'my_multiplier': 3, 'my_array': 'store://...'}
    """

    def decorator(func: Callable):
        def wrapper(*args: tuple, **kwargs: dict):
            nonlocal labels
            nonlocal outputs
            nonlocal inputs

            # Set default `inputs` - inspect the full signature and add the user's input on top of it:
            if inputs:
                # Get the available parameters type hints from the function's signature:
                func_signature = inspect.signature(func)
                parameters = OrderedDict(
                    {
                        parameter.name: parameter.annotation
                        for parameter in func_signature.parameters.values()
                    }
                )
                # If user input is given, add it on top of the collected defaults (from signature):
                if isinstance(inputs, dict):
                    parameters.update(inputs)
                inputs = parameters

            # Create a context handler and look for a context:
            cxt_handler = ContextHandler()
            cxt_handler.look_for_context(args=args, kwargs=kwargs)

            # If an MLRun context is found, parse arguments pre-run (kwargs are parsed inplace):
            if cxt_handler.is_context_available() and inputs:
                args = cxt_handler.parse_inputs(
                    args=args, kwargs=kwargs, type_hints=inputs
                )

            # Call the original function and get the returning values:
            func_outputs = func(*args, **kwargs)

            # If an MLRun context is found, set the given labels and log the returning values to MLRun via the context:
            if cxt_handler.is_context_available():
                if labels:
                    cxt_handler.set_labels(labels=labels)
                if outputs:
                    cxt_handler.log_outputs(
                        outputs=func_outputs
                        if isinstance(func_outputs, tuple)
                        else [func_outputs],
                        log_hints=outputs,
                    )
                    return  # Do not return any values as the returning values were logged to MLRun.
            return func_outputs

        # Make sure to pass the wrapped function's signature (argument list, type hints and doc strings) to the wrapper:
        wrapper = functools.wraps(func)(wrapper)

        return wrapper

    return decorator
