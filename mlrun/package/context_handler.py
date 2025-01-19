# Copyright 2023 Iguazio
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
import inspect
import os
from collections import OrderedDict
from typing import Union

from mlrun.datastore import DataItem
from mlrun.errors import MLRunInvalidArgumentError
from mlrun.execution import MLClientCtx
from mlrun.run import get_or_create_ctx

from .errors import MLRunPackageCollectionError, MLRunPackagePackingError
from .packagers_manager import PackagersManager
from .utils import LogHintUtils, TypeHintUtils


class ContextHandler:
    """
    A class for handling a MLRun context of a function that is wrapped in MLRun's `handler` decorator.

    The context handler have 3 duties:
      1. Check if the user used MLRun to run the wrapped function and if so, get the MLRun context.
      2. Parse the user's inputs (MLRun `DataItem`) to the function.
      3. Log the function's outputs to MLRun.

    The context handler uses a packagers manager to unpack (parse) the inputs and pack (log) the outputs. It sets up a
    manager with all the packagers in the `mlrun.package.packagers` directory. Packagers whom are in charge of modules
    that are in the MLRun requirements are mandatory and additional extensions packagers for non-required modules are
    added if the modules are available in the user's interpreter. Once a context is found, project custom packagers will
    be added as well.
    """

    # Mandatory packagers to be collected at initialization time:
    _MLRUN_REQUIREMENTS_PACKAGERS = [
        "python_standard_library",
        "pandas",
        "numpy",
    ]
    # Optional packagers to be collected at initialization time:
    _EXTENDED_PACKAGERS = []  # TODO: Create "matplotlib", "plotly", packagers.
    # Optional packagers from the `mlrun.frameworks` package:
    _MLRUN_FRAMEWORKS_PACKAGERS = []  # TODO: Create frameworks packagers.
    # Default priority values for packagers:
    _BUILTIN_PACKAGERS_DEFAULT_PRIORITY = 5
    _CUSTOM_PACKAGERS_DEFAULT_PRIORITY = 3

    def __init__(self):
        """
        Initialize a context handler.
        """
        # Set up a variable to hold the context:
        self._context: MLClientCtx = None

        # Initialize a packagers manager:
        self._packagers_manager = PackagersManager()

        # Prepare the manager (collect the MLRun builtin standard and optional packagers):
        self._collect_mlrun_packagers()

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

        # Give the packagers manager custom packagers to collect (if a context is found and a project is available):
        if self._context is not None and self._context.project:
            # Get the custom packagers property from the project's spec:
            project = self._context.get_project_object()
            if project and project.spec.custom_packagers:
                # Add the custom packagers taking into account the mandatory flag:
                for custom_packager, is_mandatory in project.spec.custom_packagers:
                    self._collect_packagers(
                        packagers=[custom_packager],
                        is_mandatory=is_mandatory,
                        is_custom_packagers=True,
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
            key: TypeHintUtils.parse_type_hint(type_hint=value)
            for key, value in type_hints.items()
        }

        # Parse the arguments:
        parsed_args = []
        type_hints_keys = list(type_hints.keys())
        for i, argument in enumerate(args):
            if (
                isinstance(argument, DataItem)
                and type_hints[type_hints_keys[i]] is not inspect.Parameter.empty
            ):
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
            if (
                isinstance(value, DataItem)
                and type_hints[key] is not inspect.Parameter.empty
            ):
                kwargs[key] = self._packagers_manager.unpack(
                    data_item=value, type_hint=type_hints[key]
                )

        return parsed_args

    def log_outputs(
        self,
        outputs: list,
        log_hints: list[Union[dict[str, str], str, None]],
    ):
        """
        Log the given outputs as artifacts (or results) with the stored context. Errors raised during the packing will
        be ignored to not fail a run. A warning with the error wil be printed.

        Only the logging worker will pack and log the outputs.

        :param outputs:   List of outputs to log.
        :param log_hints: List of log hints (logging configurations) to use.
        """
        # Pack and log only from the logging worker (in case of multi-workers job like OpenMPI):
        if self._context.is_logging_worker():
            # Verify the outputs and log hints are the same length:
            self._validate_objects_to_log_hints_length(
                outputs=outputs, log_hints=log_hints
            )
            # Go over the outputs and pack them:
            for obj, log_hint in zip(outputs, log_hints):
                try:
                    # Check if needed to log (not None):
                    if log_hint is None:
                        continue
                    # Parse the log hint:
                    log_hint = LogHintUtils.parse_log_hint(log_hint=log_hint)
                    # Pack the object (we don't catch the returned package as we log it after we pack all the outputs to
                    # enable linking extra data of some artifacts):
                    self._packagers_manager.pack(obj=obj, log_hint=log_hint)
                except (MLRunInvalidArgumentError, MLRunPackagePackingError) as error:
                    self._context.logger.warn(
                        f"Skipping logging an object with the log hint '{log_hint}' "
                        f"due to the following error:\n{error}"
                    )
            # Link packages:
            self._packagers_manager.link_packages(
                additional_artifact_uris=self._context.artifact_uris,
                additional_results=self._context.results,
            )
            # Log the packed results and artifacts:
            self._context.log_results(results=self._packagers_manager.results)
            for artifact in self._packagers_manager.artifacts:
                self._context.log_artifact(item=artifact)
        else:
            self._context.logger.debug("Skipping logging - not the logging worker.")

        # Clear packagers outputs:
        self._packagers_manager.clear_packagers_outputs()

    def set_labels(self, labels: dict[str, str]):
        """
        Set the given labels with the stored context.

        :param labels: The labels to set.
        """
        for key, value in labels.items():
            self._context.set_label(key=key, value=value)

    def _collect_packagers(
        self, packagers: list[str], is_mandatory: bool, is_custom_packagers: bool
    ):
        """
        Collect packagers with the stored manager. The collection can ignore errors raised by setting the mandatory flag
        to False.

        :param packagers:           The list of packagers to collect.
        :param is_mandatory:        Whether the packagers are mandatory for the context run.
        :param is_custom_packagers: Whether the packagers to collect are user's custom or MLRun's builtins.
        """
        try:
            self._packagers_manager.collect_packagers(
                packagers=packagers,
                default_priority=self._CUSTOM_PACKAGERS_DEFAULT_PRIORITY
                if is_custom_packagers
                else self._BUILTIN_PACKAGERS_DEFAULT_PRIORITY,
            )
        except MLRunPackageCollectionError as error:
            if is_mandatory:
                raise error
            else:
                # If the packagers to collect were added manually by the user, the logger should write the collection
                # issue as a warning. Otherwise - for mlrun builtin packagers, a debug message will do.
                message = (
                    f"The given optional packagers '{packagers}' could not be imported due to the following error:\n"
                    f"'{error}'"
                )
                if is_custom_packagers:
                    self._context.logger.warn(message)
                else:
                    self._context.logger.debug(message)

    def _collect_mlrun_packagers(self):
        """
        Collect MLRun's builtin packagers. That include all mandatory packagers whom in charge of MLRun's requirements
        libraries, more optional commonly used libraries packagers and more `mlrun.frameworks` packagers. The priority
        will be as follows (from higher to lower priority):

        1. Optional `mlrun.frameworks` packagers
        2. MLRun's optional packagers
        3. MLRun's mandatory packagers (MLRun's requirements)
        """
        # Collect MLRun's requirements packagers (mandatory):
        self._collect_packagers(
            packagers=[
                f"mlrun.package.packagers.{module_name}_packagers.*"
                for module_name in self._MLRUN_REQUIREMENTS_PACKAGERS
            ],
            is_mandatory=True,
            is_custom_packagers=False,
        )

        # Add extra packagers for optional libraries:
        for module_name in self._EXTENDED_PACKAGERS:
            self._collect_packagers(
                packagers=[f"mlrun.package.packagers.{module_name}_packagers.*"],
                is_mandatory=False,
                is_custom_packagers=False,
            )

        # Add extra packagers from `mlrun.frameworks` package:
        for module_name in self._MLRUN_FRAMEWORKS_PACKAGERS:
            self._collect_packagers(
                packagers=[f"mlrun.frameworks.{module_name}.packagers.*"],
                is_mandatory=False,
                is_custom_packagers=False,
            )

    def _validate_objects_to_log_hints_length(
        self,
        outputs: list,
        log_hints: list[Union[dict[str, str], str, None]],
    ):
        """
        Validate the outputs and log hints are the same length. If they are not, warnings will be printed on what will
        be ignored.

        :param outputs:   List of outputs to log.
        :param log_hints: List of log hints (logging configurations) to use.
        """
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
