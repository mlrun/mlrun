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
import inspect
import os
import shutil
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Type, Union

import cloudpickle
import numpy as np
import pandas as pd

from mlrun.datastore import DataItem
from mlrun.errors import MLRunInvalidArgumentError, MLRunRuntimeError
from mlrun.execution import MLClientCtx
from mlrun.utils import logger


# TODO: Move the `ArtifactType` to constants.py
class ArtifactType(Enum):
    """
    Possible artifact types to log using the MLRun `context` decorator.
    """

    # Types:
    DATASET = "dataset"
    DIRECTORY = "directory"
    FILE = "file"
    OBJECT = "object"
    PLOT = "plot"
    RESULT = "result"

    # Constants:
    DEFAULT = RESULT


class InputsParser:
    """
    A static class to hold all the common parsing functions - functions for parsing MLRun DataItem to the user desired
    type.
    """

    @staticmethod
    def parse_pandas_dataframe(data_item: DataItem) -> pd.DataFrame:
        """
        Parse an MLRun `DataItem` to a `pandas.DataFrame`.

        :param data_item: The `DataItem` to parse.

        :returns: The `DataItem` as a `pandas.DataFrame`.
        """
        return data_item.as_df()

    @staticmethod
    def parse_numpy_array(data_item: DataItem) -> np.ndarray:
        """
        Parse an MLRun `DataItem` to a `numpy.ndarray`.

        :param data_item: The `DataItem` to parse.

        :returns: The `DataItem` as a `numpy.ndarray`.
        """
        return data_item.as_df().to_numpy()

    @staticmethod
    def parse_dict(data_item: DataItem) -> dict:
        """
        Parse an MLRun `DataItem` to a `dict`.

        :param data_item: The `DataItem` to parse.

        :returns: The `DataItem` as a `dict`.
        """
        return data_item.as_df().to_dict()

    @staticmethod
    def parse_list(data_item: DataItem) -> list:
        """
        Parse an MLRun `DataItem` to a `list`.

        :param data_item: The `DataItem` to parse.

        :returns: The `DataItem` as a `list`.
        """
        return data_item.as_df().to_numpy().tolist()

    @staticmethod
    def parse_object(data_item: DataItem) -> object:
        """
        Parse an MLRun `DataItem` to its unpickled object. The pickle file will be downloaded to a local temp
        directory and then loaded.

        :param data_item: The `DataItem` to parse.

        :returns: The `DataItem` as the original object that was pickled once it was logged.
        """
        object_file = data_item.local()
        with open(object_file, "rb") as pickle_file:
            obj = cloudpickle.load(pickle_file)
        return obj


class OutputsLogger:
    """
    A static class to hold all the common logging functions - functions for logging different objects by artifact type
    to MLRun.
    """

    @staticmethod
    def log_dataset(
        ctx: MLClientCtx,
        obj: Union[pd.DataFrame, np.ndarray, pd.Series, dict, list],
        key: str,
        logging_kwargs: dict,
    ):
        """
        Log an object as a dataset. The dataset wil lbe cast to a `pandas.DataFrame`. Supporting casting from
        `pandas.Series`, `numpy.ndarray`, `dict` and `list`.

        :param ctx:            The MLRun context to log with.
        :param obj:            The data to log.
        :param key:            The key of the artifact.
        :param logging_kwargs: Additional keyword arguments to pass to the `context.log_dataset`

        :raise MLRunInvalidArgumentError: If the type is not supported for being cast to `pandas.DataFrame`.
        """
        # Check for the object type:
        if not isinstance(obj, pd.DataFrame):
            if isinstance(obj, (np.ndarray, pd.Series, dict, list)):
                obj = pd.DataFrame(obj)
            else:
                raise MLRunInvalidArgumentError(
                    f"The value requested to be logged as a dataset artifact is of type '{type(obj)}' and it "
                    f"cannot be logged as a dataset. Please parse it in your code into one `numpy.ndarray`, "
                    f"`pandas.DataFrame`, `pandas.Series`, `dict`, `list` before returning it so we can log it."
                )

        # Log the DataFrame object as a dataset:
        ctx.log_dataset(**logging_kwargs, key=key, df=obj)

    @staticmethod
    def log_directory(
        ctx: MLClientCtx,
        obj: Union[str, Path],
        key: str,
        logging_kwargs: dict,
    ):
        """
        Log a directory as a zip file. The zip file will be created at the current working directory. Once logged,
        it will be deleted.

        :param ctx:            The MLRun context to log with.
        :param obj:            The directory to zip path.
        :param key:            The key of the artifact.
        :param logging_kwargs: Additional keyword arguments to pass to the `context.log_artifact` method.

        :raises MLRunInvalidArgumentError: In case the given path is not of a directory or do not exist.
        """
        # In case it is a `pathlib` path, parse to str:
        obj = str(obj)

        # Verify the path is of an existing directory:
        if not os.path.isdir(obj):
            raise MLRunInvalidArgumentError(
                f"The given path is not a directory: '{obj}'"
            )
        if not os.path.exists(obj):
            raise MLRunInvalidArgumentError(
                f"The given directory path do not exist: '{obj}'"
            )

        # Zip the directory:
        directory_zip_path = shutil.make_archive(
            base_name=key,
            format="zip",
            root_dir=os.path.abspath(obj),
        )

        # Log the zip file:
        ctx.log_artifact(**logging_kwargs, item=key, local_path=directory_zip_path)

        # Delete the zip file:
        os.remove(directory_zip_path)

    @staticmethod
    def log_file(
        ctx: MLClientCtx,
        obj: Union[str, Path],
        key: str,
        logging_kwargs: dict,
    ):
        """
        Log a file to MLRun.

        :param ctx:            The MLRun context to log with.
        :param obj:            The path of the file to log.
        :param key:            The key of the artifact.
        :param logging_kwargs: Additional keyword arguments to pass to the `context.log_artifact` method.

        :raises MLRunInvalidArgumentError: In case the given path is not of a file or do not exist.
        """
        # In case it is a `pathlib` path, parse to str:
        obj = str(obj)

        # Verify the path is of an existing directory:
        if not os.path.isfile(obj):
            raise MLRunInvalidArgumentError(f"The given path is not a file: '{obj}'")
        if not os.path.exists(obj):
            raise MLRunInvalidArgumentError(
                f"The given directory path do not exist: '{obj}'"
            )

        # Log the zip file:
        ctx.log_artifact(**logging_kwargs, item=key, local_path=os.path.abspath(obj))

    @staticmethod
    def log_object(ctx: MLClientCtx, obj, key: str, logging_kwargs: dict):
        """
        Log an object as a pickle.

        :param ctx:            The MLRun context to log with.
        :param obj:            The object to log.
        :param key:            The key of the artifact.
        :param logging_kwargs: Additional keyword arguments to pass to the `context.log_artifact` method.
        """
        ctx.log_artifact(
            **logging_kwargs,
            item=key,
            body=obj if isinstance(obj, (bytes, bytearray)) else cloudpickle.dumps(obj),
            format="pkl",
        )

    @staticmethod
    def log_plot(ctx: MLClientCtx, obj, key: str, logging_kwargs: dict):
        """
        Log an object as a plot. Currently, supporting plots produced by one the following modules: `matplotlib`,
        `seaborn`, `plotly` and `bokeh`.

        :param ctx:            The MLRun context to log with.
        :param obj:            The plot to log.
        :param key:            The key of the artifact.
        :param logging_kwargs: Additional keyword arguments to pass to the `context.log_artifact`.

        :raise MLRunInvalidArgumentError: If the object type is not supported (meaning the plot was not produced by
                                          one of the supported modules).
        """
        # Create the plot artifact according to the module produced the object:
        artifact = None

        # `matplotlib` and `seaborn`:
        try:
            import matplotlib.pyplot as plt

            from mlrun.artifacts import PlotArtifact

            # Get the figure:
            figure = None
            if isinstance(obj, plt.Figure):
                figure = obj
            elif isinstance(obj, plt.Axes):
                if hasattr(obj, "get_figure"):
                    figure = obj.get_figure()
                elif hasattr(obj, "figure"):
                    figure = obj.figure
                elif hasattr(obj, "fig"):
                    figure = obj.fig

            # Create the artifact:
            if figure is not None:
                artifact = PlotArtifact(key=key, body=figure)
        except ModuleNotFoundError:
            pass

        # `plotly`:
        if artifact is None:
            try:
                import plotly

                from mlrun.artifacts import PlotlyArtifact

                if isinstance(obj, plotly.graph_objs.Figure):
                    artifact = PlotlyArtifact(key=key, figure=obj)
            except ModuleNotFoundError:
                pass

        # `bokeh`:
        if artifact is None:
            try:
                import bokeh.plotting as bokeh_plt

                from mlrun.artifacts import BokehArtifact

                if isinstance(obj, bokeh_plt.Figure):
                    artifact = BokehArtifact(key=key, figure=obj)
            except ModuleNotFoundError:
                pass
            except ImportError:
                logger.warn(
                    "Bokeh installation is ignored. If needed, "
                    "make sure you have the required version with `pip install mlrun[bokeh]`"
                )

        # Log the artifact:
        if artifact is None:
            raise MLRunInvalidArgumentError(
                f"The given plot is of type `{type(obj)}`. We currently support logging plots produced by one of "
                f"the following modules: `matplotlib`, `seaborn`, `plotly` and `bokeh`. You may try to save the "
                f"plot to file and log it as a file instead."
            )
        ctx.log_artifact(**logging_kwargs, item=artifact)

    @staticmethod
    def log_result(
        ctx: MLClientCtx,
        obj: Union[int, float, str, list, tuple, dict, np.ndarray],
        key: str,
        logging_kwargs: dict,
    ):
        """
        Log an object as a result. The objects value will be cast to a serializable version of itself. Supporting:
        int, float, str, list, tuple, dict, numpy.ndarray

        :param ctx:            The MLRun context to log with.
        :param obj:            The value to log.
        :param key:            The key of the artifact.
        :param logging_kwargs: Additional keyword arguments to pass to the `context.log_result` method.
        """
        ctx.log_result(**logging_kwargs, key=key, value=obj)


class ContextHandler:
    """
    Private class for handling an MLRun context of a function that is wrapped in MLRun's `handler` decorator.

    The context handler have 3 duties:
      1. Check if the user used MLRun to run the wrapped function and if so, get the MLRun context.
      2. Parse the user's inputs (MLRun `DataItem`) to the function.
      3. Log the function's outputs to MLRun.

    The context handler use dictionaries to map objects to their logging / parsing function. The maps can be edited
    using the relevant `update_X` class method. If needed to add additional artifacts types, the `ArtifactType` class
    can be inherited and replaced as well using the `update_artifact_type_class` class method.
    """

    # The artifact type enum class to use:
    _ARTIFACT_TYPE_CLASS = ArtifactType
    # The map to use to get default artifact types of objects:
    _DEFAULT_OBJECTS_ARTIFACT_TYPES_MAP = None
    # The map to use for logging an object by its type:
    _OUTPUTS_LOGGING_MAP = None
    # The map to use for parsing an object by its type:
    _INPUTS_PARSING_MAP = None

    @classmethod
    def update_artifact_type_class(cls, artifact_type_class: Type[ArtifactType]):
        """
        Update the artifact type enum class that the handler will use to specify new artifact types to log and parse.

        :param artifact_type_class: An enum inheriting from the `ArtifactType` enum.
        """
        cls._ARTIFACT_TYPE_CLASS = artifact_type_class

    @classmethod
    def update_default_objects_artifact_types_map(
        cls, updates: Dict[type, ArtifactType]
    ):
        """
        Enrich the default objects artifact types map with new objects types to support.

        :param updates: New objects types to artifact types to support.
        """
        if cls._DEFAULT_OBJECTS_ARTIFACT_TYPES_MAP is None:
            cls._init_default_objects_artifact_types_map()
        cls._DEFAULT_OBJECTS_ARTIFACT_TYPES_MAP.update(updates)

    @classmethod
    def update_outputs_logging_map(
        cls,
        updates: Dict[ArtifactType, Callable[[MLClientCtx, Any, str, dict], None]],
    ):
        """
        Enrich the outputs logging map with new artifact types to support. The outputs logging map is a dictionary of
        artifact type enum as key, and a function that will handle the given output. The function must accept 4 keyword
        arguments

        * ctx: `mlrun.MLClientCtx` - The MLRun context to log with.
        * obj: `Any` - The value / object to log.
        * key: `str` - The key of the artifact.
        * logging_kwargs: `dict` - Keyword arguments the user can pass in the instructions tuple.

        :param updates: New artifact types to support - a dictionary of artifact type enum as key, and a function that
                        will handle the given output to update the current map.
        """
        if cls._OUTPUTS_LOGGING_MAP is None:
            cls._init_outputs_logging_map()
        cls._OUTPUTS_LOGGING_MAP.update(updates)

    @classmethod
    def update_inputs_parsing_map(cls, updates: Dict[type, Callable[[DataItem], Any]]):
        """
        Enrich the inputs parsing map with new objects to support. The inputs parsing map is a dictionary of object
        types as key, and a function that will handle the given input. The function must accept 1 keyword argument
        (data_item: `mlrun.DataItem`) and return the relevant parsed object.

        :param updates: New object types to support - a dictionary of artifact type enum as key, and a function that
                        will handle the given input to update the current map.
        """
        if cls._INPUTS_PARSING_MAP is None:
            cls._init_inputs_parsing_map()
        cls._INPUTS_PARSING_MAP.update(updates)

    def __init__(self):
        """
        Initialize a context handler.
        """
        # Initialize the maps:
        if self._DEFAULT_OBJECTS_ARTIFACT_TYPES_MAP is None:
            self._init_default_objects_artifact_types_map()
        if self._OUTPUTS_LOGGING_MAP is None:
            self._init_outputs_logging_map()
        if self._INPUTS_PARSING_MAP is None:
            self._init_inputs_parsing_map()

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
                return

        # Search in the given keyword arguments:
        for argument_name, argument_value in kwargs.items():
            if isinstance(argument_value, MLClientCtx):
                self._context = argument_value
                return

        # Search if the function was triggered from an MLRun RunTime object by looking at the call stack:
        # Index 0: the current frame.
        # Index 1: the decorator's frame.
        # Index 2-...: If it is from mlrun.runtimes we can be sure it ran via MLRun, otherwise not.
        for callstack_frame in inspect.getouterframes(inspect.currentframe()):
            if os.path.join("mlrun", "runtimes", "local") in callstack_frame.filename:
                import mlrun

                self._context = mlrun.get_or_create_ctx("context")
                break

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
        # Parse the arguments:
        parsed_args = []
        type_hints_keys = list(type_hints.keys())
        for i, argument in enumerate(args):
            if (
                isinstance(argument, DataItem)
                and type_hints[type_hints_keys[i]] != inspect._empty
            ):
                parsed_args.append(
                    self._parse_input(
                        data_item=argument,
                        type_hint=type_hints[type_hints_keys[i]],
                    )
                )
                continue
            parsed_args.append(argument)
        parsed_args = tuple(parsed_args)  # `args` is expected to be a tuple.

        # Parse the keyword arguments:
        for key in kwargs.keys():
            if isinstance(kwargs[key], DataItem) and type_hints[key] not in [
                inspect._empty,
                DataItem,
            ]:
                kwargs[key] = self._parse_input(
                    data_item=kwargs[key], type_hint=type_hints[key]
                )

        return parsed_args

    def log_outputs(
        self,
        outputs: list,
        log_hints: List[Union[Dict[str, str], None]],
    ):
        """
        Log the given outputs as artifacts with the stored context.

        :param outputs:   List of outputs to log.
        :param log_hints: List of logging configurations to use.
        """
        for obj, log_hint in zip(outputs, log_hints):
            # Check if needed to log (not None):
            if log_hint is None:
                continue
            # Parse the instructions:
            artifact_type = self._DEFAULT_OBJECTS_ARTIFACT_TYPES_MAP.get(
                type(obj), self._ARTIFACT_TYPE_CLASS.DEFAULT
            ).value
            key = log_hint.pop("key")
            artifact_type = log_hint.pop("artifact_type", artifact_type)
            # Check if the object to log is None (None values are only logged if the artifact type is Result):
            if obj is None and artifact_type != ArtifactType.RESULT.value:
                continue
            # Log:
            self._log_output(
                obj=obj,
                artifact_type=artifact_type,
                key=key,
                logging_kwargs=log_hint,
            )

    def set_labels(self, labels: Dict[str, str]):
        """
        Set the given labels with the stored context.

        :param labels: The labels to set.
        """
        for key, value in labels.items():
            self._context.set_label(key=key, value=value)

    @classmethod
    def _init_default_objects_artifact_types_map(cls):
        """
        Initialize the default objects artifact types map with the basic classes supported by MLRun. In addition, it
        will try to support further common packages that are not required in MLRun.
        """
        # Initialize the map with the default classes:
        cls._DEFAULT_OBJECTS_ARTIFACT_TYPES_MAP = {
            pd.DataFrame: ArtifactType.DATASET,
            pd.Series: ArtifactType.DATASET,
            np.ndarray: ArtifactType.DATASET,
            dict: ArtifactType.RESULT,
            list: ArtifactType.RESULT,
            tuple: ArtifactType.RESULT,
            str: ArtifactType.RESULT,
            int: ArtifactType.RESULT,
            float: ArtifactType.RESULT,
            bytes: ArtifactType.OBJECT,
            bytearray: ArtifactType.OBJECT,
        }

        # Try to enrich it with further classes according ot the user's environment:
        try:
            import matplotlib.pyplot as plt

            cls._DEFAULT_OBJECTS_ARTIFACT_TYPES_MAP[plt.Figure] = ArtifactType.PLOT
            cls._DEFAULT_OBJECTS_ARTIFACT_TYPES_MAP[plt.Axes] = ArtifactType.PLOT
        except ModuleNotFoundError:
            pass
        try:
            import plotly

            cls._DEFAULT_OBJECTS_ARTIFACT_TYPES_MAP[
                plotly.graph_objs.Figure
            ] = ArtifactType.PLOT
        except ModuleNotFoundError:
            pass
        try:
            import bokeh.plotting as bokeh_plt

            cls._DEFAULT_OBJECTS_ARTIFACT_TYPES_MAP[
                bokeh_plt.Figure
            ] = ArtifactType.PLOT
        except ModuleNotFoundError:
            pass
        except ImportError:
            logger.warn(
                "Bokeh installation is ignored. If needed, "
                "make sure you have the required version with `pip install mlrun[bokeh]`"
            )

    @classmethod
    def _init_outputs_logging_map(cls):
        """
        Initialize the outputs logging map for the basic artifact types supported by MLRun.
        """
        cls._OUTPUTS_LOGGING_MAP = {
            ArtifactType.DATASET: OutputsLogger.log_dataset,
            ArtifactType.DIRECTORY: OutputsLogger.log_directory,
            ArtifactType.FILE: OutputsLogger.log_file,
            ArtifactType.OBJECT: OutputsLogger.log_object,
            ArtifactType.PLOT: OutputsLogger.log_plot,
            ArtifactType.RESULT: OutputsLogger.log_result,
        }

    @classmethod
    def _init_inputs_parsing_map(cls):
        """
        Initialize the inputs parsing map with the basic classes supported by MLRun.
        """
        cls._INPUTS_PARSING_MAP = {
            pd.DataFrame: InputsParser.parse_pandas_dataframe,
            np.ndarray: InputsParser.parse_numpy_array,
            dict: InputsParser.parse_dict,
            list: InputsParser.parse_list,
            object: InputsParser.parse_object,
        }

    def _parse_input(self, data_item: DataItem, type_hint: type) -> Any:
        """
        Parse the given data frame to the expected type. By default, it will be parsed to an object (will be treated as
        a pickle).

        :param data_item: The data item to parse.
        :param type_hint: The expected type to parse to.

        :returns: The parsed data item.

        :raises MLRunRuntimeError: If an error was raised during the parsing function.
        """
        if str(type_hint).startswith("typing."):
            return data_item
        try:
            return self._INPUTS_PARSING_MAP.get(
                type_hint, self._INPUTS_PARSING_MAP[object]
            )(data_item=data_item)
        except Exception as exception:
            raise MLRunRuntimeError(
                f"MLRun tried to parse a `DataItem` of type '{type_hint}' but failed. Be sure the item was "
                f"logged correctly - as the type you are trying to parse it back to. In general, python objects should "
                f"be logged as pickles."
            ) from exception

    def _log_output(
        self,
        obj,
        artifact_type: Union[ArtifactType, str],
        key: str,
        logging_kwargs: Dict[str, Any],
    ):
        """
        Log the given object to MLRun as the given artifact type with the provided key. The key can be part of a
        logging keyword arguments to pass to the relevant context logging function.

        :param obj:           The object to log.
        :param artifact_type: The artifact type to log the object as.
        :param key:           The key (name) of the artifact or a logging kwargs to use when logging the artifact.

        :raises MLRunInvalidArgumentError: If a key was provided in the logging kwargs.
        :raises MLRunRuntimeError:         If an error was raised during the logging function.
        """
        # Get the artifact type (will also verify the artifact type is valid):
        artifact_type = self._ARTIFACT_TYPE_CLASS(artifact_type)

        # Check if 'key' or 'item' were given the logging kwargs:
        if "key" in logging_kwargs or "item" in logging_kwargs:
            raise MLRunInvalidArgumentError(
                "When passing logging keyword arguments, both 'key' and 'item' (according to the context method) "
                "cannot be added to the dictionary as the key is given on its own."
            )

        # Use the logging map to log the object:
        try:
            self._OUTPUTS_LOGGING_MAP[artifact_type](
                ctx=self._context,
                obj=obj,
                key=key,
                logging_kwargs=logging_kwargs,
            )
        except Exception as exception:
            raise MLRunRuntimeError(
                f"MLRun tried to log '{key}' as '{artifact_type.value}' but failed. If you didn't provide the artifact "
                f"type and the default one does not fit, try to select the correct type from the enum `ArtifactType`."
            ) from exception
