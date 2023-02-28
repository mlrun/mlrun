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
from pathlib import Path
from typing import Any, Callable, Dict, List, Type, Union

import cloudpickle
import numpy as np
import pandas as pd

from mlrun.datastore import DataItem
from mlrun.errors import MLRunInvalidArgumentError, MLRunRuntimeError
from mlrun.execution import MLClientCtx
from mlrun.utils import logger

from .constants import ArtifactType


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

'''

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
'''