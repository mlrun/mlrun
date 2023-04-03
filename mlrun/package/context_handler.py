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
import builtins
import importlib
import inspect
import os
import re
from collections import OrderedDict
from typing import Dict, List, Type, Union

from mlrun.datastore import DataItem
from mlrun.errors import MLRunInvalidArgumentError, MLRunPackagePackagerCollectionError
from mlrun.execution import MLClientCtx
from mlrun.run import get_or_create_ctx
from mlrun.utils import logger

from .constants import ArtifactType, LogHintKey
from .packagers_manager import PackagersManager


class ContextHandler:
    """
    A class for handling a MLRun context of a function that is wrapped in MLRun's `handler` decorator.

    The context handler have 3 duties:
      1. Check if the user used MLRun to run the wrapped function and if so, get the MLRun context.
      2. Parse the user's inputs (MLRun `DataItem`) to the function.
      3. Log the function's outputs to MLRun.

    The context handler uses a packagers manager to unpack (parse) the inputs and pack (log) the outputs.
    """

    def __init__(self):
        """
        Initialize a context handler.
        """
        # Initialize a packagers manager:
        self._packagers_manager = PackagersManager()

        # Set up a variable to hold the context:
        self._context: MLClientCtx = None

    def look_for_context(self, args: tuple, kwargs: dict):
        """
        Look for an MLRun context (`mlrun.MLClientCtx`). The handler will look for a context in the given order:
          1. The given arguments.
          2. The given keyword arguments.
          3. If an MLRun RunTime was used the context will be located via the `mlrun.get_or_create_ctx` method.

        :param args:   The arguments tuple passed to the function.
        :param kwargs: The keyword arguments dictionary passed to the function.
        """
        # Search in the given arguments:
        for argument in args:
            if isinstance(argument, MLClientCtx):
                self._context = argument
                break

        # Search in the given keyword arguments:
        if self._context is None:
            for argument_name, argument_value in kwargs.items():
                if isinstance(argument_value, MLClientCtx):
                    self._context = argument_value
                    break

        # Search if the function was triggered from an MLRun RunTime object by looking at the call stack:
        # Index 0: the current frame.
        # Index 1: the decorator's frame.
        # Index 2-...: If it is from mlrun.runtimes we can be sure it ran via MLRun, otherwise not.
        if self._context is None:
            for callstack_frame in inspect.getouterframes(inspect.currentframe()):
                if (
                    os.path.join("mlrun", "runtimes", "local")
                    in callstack_frame.filename
                ):
                    self._context = get_or_create_ctx("context")
                    break

        # Give the packagers manager custom packagers to collect (if available):
        if self._context is not None:
            # Get the custom packagers property from the project's spec:
            custom_packagers = self._context.get_project_param(
                key="custom_packagers", default=[]
            )
            if custom_packagers:
                # Packager added to collect may not be mandatory, so if it fails we will write a debugging message and
                # continue, but if they are mandatory, we will raise an error:
                for custom_packager, is_mandatory in custom_packagers:
                    try:
                        self._packagers_manager.collect_packagers(
                            packagers=custom_packagers
                        )
                    except (
                        MLRunInvalidArgumentError,
                        MLRunPackagePackagerCollectionError,
                    ) as error:
                        if is_mandatory:
                            raise error
                        else:
                            self._context.logger.debug(
                                f"The packager '{custom_packager}' could not be imported due to the following error:\n"
                                f"'{error}'"
                            )

    def is_context_available(self) -> bool:
        """
        Check if a context was found by the method `look_for_context`.

        :returns: True if a context was found and False otherwise.
        """
        return self._context is not None

    def parse_inputs(
        self,
        args: tuple,
        kwargs: dict,
        type_hints: OrderedDict,
    ) -> tuple:
        """
        Parse the given arguments and keyword arguments data items to the expected types.

        :param args:       The arguments tuple passed to the function.
        :param kwargs:     The keyword arguments dictionary passed to the function.
        :param type_hints: An ordered dictionary of the expected types of arguments.

        :returns: The parsed args (kwargs are parsed inplace).
        """
        # Parse the type hints (in case some were given as strings):
        type_hints = {
            key: self.parse_type_hint(type_hint=value)
            for key, value in type_hints.items()
        }

        # Parse the arguments:
        parsed_args = []
        type_hints_keys = list(type_hints.keys())
        for i, argument in enumerate(args):
            if isinstance(argument, DataItem) and type_hints[
                type_hints_keys[i]
            ] not in [
                inspect._empty,
                DataItem,
            ]:
                parsed_args.append(
                    self._packagers_manager.unpack(
                        data_item=argument,
                        type_hint=type_hints[type_hints_keys[i]],
                    )
                )
            else:
                parsed_args.append(argument)
        parsed_args = tuple(parsed_args)  # `args` is expected to be a tuple.

        # Parse the keyword arguments:
        for key, value in kwargs.items():
            if isinstance(value, DataItem) and type_hints[key] not in [
                inspect._empty,
                DataItem,
            ]:
                kwargs[key] = self._packagers_manager.unpack(
                    data_item=value, type_hint=type_hints[key]
                )

        return parsed_args

    def log_outputs(
        self,
        outputs: list,
        log_hints: List[Union[Dict[str, str], str, None]],
    ):
        """
        Log the given outputs as artifacts with the stored context.

        :param outputs:   List of outputs to log.
        :param log_hints: List of log hints (logging configurations) to use.
        """
        # Verify the outputs and log hints are the same length:
        if len(outputs) != len(log_hints):
            self._context.logger.warn(
                f"The amount of outputs objects returned from the function ({len(outputs)}) does not match the amount "
                f"of provided log hints ({len(log_hints)})."
            )
            if len(outputs) > len(log_hints):
                ignored_outputs = [str(output) for output in outputs[len(log_hints) :]]
                self._context.logger.warn(
                    f"The following outputs will not be logged: {', '.join(ignored_outputs)}"
                )
            if len(outputs) < len(log_hints):
                ignored_log_hints = [
                    str(log_hint) for log_hint in log_hints[len(outputs) :]
                ]
                self._context.logger.warn(
                    f"The following log hints will be ignored: {', '.join(ignored_log_hints)}"
                )

        # Go over the outputs and pack them:
        for obj, log_hint in zip(outputs, log_hints):
            # Check if needed to log (not None):
            if log_hint is None:
                continue
            # Parse the log hint:
            log_hint = self.parse_log_hint(log_hint=log_hint)
            # Check if the object to log is None (None values are only logged if the artifact type is Result):
            if (
                obj is None
                and log_hint.get(LogHintKey.ARTIFACT_TYPE, ArtifactType.RESULT)
                != ArtifactType.RESULT
            ):
                continue
            # Pack the object (we don't catch the returned package as we log it after we pack all the outputs to enable
            # linking extra data of some artifacts):
            self._packagers_manager.pack(obj=obj, log_hint=log_hint)

        # Link packages:
        self._packagers_manager.link_packages(
            additional_artifacts=self._context.artifacts,
            additional_results=self._context.results,
        )

        # Log the packed results and artifacts:
        for result in self._packagers_manager.results:
            self._context.log_results(results=result)
        for artifact in self._packagers_manager.artifacts:
            self._context.log_artifact(item=artifact)

        # Clear packagers outputs:
        self._packagers_manager.clear_packagers_outputs()

    def set_labels(self, labels: Dict[str, str]):
        """
        Set the given labels with the stored context.

        :param labels: The labels to set.
        """
        for key, value in labels.items():
            self._context.set_label(key=key, value=value)

    @staticmethod
    def parse_type_hint(type_hint: Union[Type, str]) -> Type:
        """
        Parse a given type hint from string to its actual hinted type class object. The string must be one of the
        following:

        * Python builtin type - for example: `tuple`, `list`, `set`, `dict` and `bytearray`.
        * Full module import path. An alias (if `import pandas as pd is used`, the type hint cannot be `pd.DataFrame`)
          is not allowed.

        The type class on its own (like `DataFrame`) cannot be used as the scope of this function is not the same as the
        handler itself, hence modules and objects that were imported in the handler's scope are not available. This is
        the same reason import aliases cannot be used as well.

        If the provided type hint is not a string, it will simply be returned as is.

        :param type_hint: The type hint to parse.

        :return: The hinted type.

        :raise MLRunInvalidArgumentError: In case the type hint is not following the 2 options mentioned above.
        """
        if not isinstance(type_hint, str):
            return type_hint

        # Validate the type hint is a valid module path:
        if not bool(
            re.fullmatch(
                r"([a-zA-Z_][a-zA-Z0-9_]*\.)*[a-zA-Z_][a-zA-Z0-9_]*", type_hint
            )
        ):
            raise MLRunInvalidArgumentError(
                f"Invalid type hint. An input type hint must be a valid python class name or its module import path. "
                f"For example: 'list', 'pandas.DataFrame', 'numpy.ndarray', 'sklearn.linear_model.LinearRegression'. "
                f"Type hint given: '{type_hint}'."
            )

        # Look for a builtin type (rest of the builtin types like `int`, `str`, `float` should be treated as results,
        # hence not given as an input to an MLRun function, but as a parameter):
        builtin_types = {
            builtin_name: builtin_type
            for builtin_name, builtin_type in builtins.__dict__.items()
            if isinstance(builtin_type, type)
        }
        if type_hint in builtin_types:
            return builtin_types[type_hint]

        # If it's not a builtin, its should have a full module path, meaning at least one '.' to separate the module and
        # the class. If it doesn't, we will try to get the class from the main module:
        if "." not in type_hint:
            logger.warn(
                f"The type hint string given '{type_hint}' is not a `builtins` python type. MLRun will try to look for "
                f"it in the `__main__` module instead."
            )
            type_hint = f"__main__.{type_hint}"

        # Import the module to receive the hinted type:
        try:
            # Get the module path and the type class (If we'll wish to support inner classes, the `rsplit` won't work):
            module_path, type_hint = type_hint.rsplit(".", 1)
            # Replace alias if needed (alias assumed to be imported already, hence we look in globals):
            # For example:
            # If in handler scope there was `import A.B.C as abc` and user gave a type hint "abc.Something" then:
            # `module_path[0]` will be equal to "abc". Then, because it is an alias, it will appear in the globals, so
            # we'll replace the alias with the full module name in order to import the module.
            module_path = module_path.split(".")
            if module_path[0] in globals():
                module_path[0] = globals()[module_path[0]].__name__
            module_path = ".".join(module_path)
            # Import the module:
            module = importlib.import_module(module_path)
            # Get the class type from the module:
            type_hint = getattr(module, type_hint)
        except ModuleNotFoundError as module_not_found_error:
            # May be raised from `importlib.import_module` in case the module does not exist.
            if "__main__" in type_hint:
                message = (
                    f"MLRun tried to get the type hint '{type_hint.split('.')[1]}' but it can't as it is not a valid "
                    f"builtin Python type (one of `list`, `dict`, `str`, `int`, etc.) nor a locally declared type "
                    f"(from the `__main__` module). Pay attention using only the type as string is not allowed as the "
                    f"handler's scope is different than MLRun's. To properly give a type hint as string, please "
                    f"specify the full module path without aliases. For example: do not use `DataFrame` or "
                    f"`pd.DataFrame`, use `pandas.DataFrame`."
                )
            else:
                message = (
                    f"MLRun tried to get the type hint '{type_hint}' but the module '{module_path}' cannot be "
                    f"imported. Keep in mind that using alias in the module path (meaning: import module as alias) is "
                    f"not allowed. If the module path is correct, please make sure the module package is installed in "
                    f"the python interpreter."
                )
            raise MLRunInvalidArgumentError(message) from module_not_found_error
        except AttributeError as attribute_error:
            # May be raised from `getattr(module, type_hint)` in case the class type cannot be imported directly from
            # the imported module.
            raise MLRunInvalidArgumentError(
                f"MLRun tried to get the type hint '{type_hint}' from the module '{module.__name__}' but it seems it "
                f"doesn't exist. Make sure the class can be imported from the module with the exact module path you "
                f"passed. Notice inner classes (a class inside of a class) are not supported."
            ) from attribute_error

        return type_hint

    @staticmethod
    def parse_log_hint(
        log_hint: Union[Dict[str, str], str, None]
    ) -> Union[Dict[str, str], None]:
        """
        Parse a given log hint from string to a logging configuration dictionary. The string will be read as the
        artifact key ('key' in the dictionary) and if the string have a single colon, the following structure is
        assumed: "<artifact_key> : <artifact_type>".

        If a logging configuration dictionary is received, it will be validated to have a key field.

        None will be returned as None.

        :param log_hint: The log hint to parse.

        :return: The hinted logging configuration.

        :raise MLRunInvalidArgumentError: In case the log hint is not following the string structure or the dictionary
                                          is missing the key field.
        """
        # Check for None value:
        if log_hint is None:
            return None

        # If the log hint was provided as a string, construct a dictionary out of it:
        if isinstance(log_hint, str):
            # Check if only key is given:
            if ":" not in log_hint:
                log_hint = {LogHintKey.KEY: log_hint}
            # Check for valid "<key> : <artifact type>" pattern:
            else:
                if log_hint.count(":") > 1:
                    raise MLRunInvalidArgumentError(
                        f"Incorrect log hint pattern. Log hints can have only a single ':' in them to specify the "
                        f"desired artifact type the returned value will be logged as: "
                        f"'<artifact_key> : <artifact_type>', but given: {log_hint}"
                    )
                # Split into key and type:
                key, artifact_type = log_hint.replace(" ", "").split(":")
                log_hint = {
                    LogHintKey.KEY: key,
                    LogHintKey.ARTIFACT_TYPE: artifact_type,
                }

        # Validate the log hint dictionary has the mandatory key:
        if LogHintKey.KEY not in log_hint:
            raise MLRunInvalidArgumentError(
                f"A log hint dictionary must include the 'key' - the artifact key (it's name). The following log hint "
                f"is missing the key: {log_hint}."
            )

        return log_hint
