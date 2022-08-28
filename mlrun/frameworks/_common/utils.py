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
import re
from abc import ABC
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple, TypeVar, Union

import numpy as np
import pandas as pd

import mlrun
from mlrun.artifacts import Artifact
from mlrun.data_types import ValueType
from mlrun.datastore import DataItem


class CommonTypes(ABC):
    """
    Common type hints to all frameworks.
    """

    # A generic model type in a handler / interface (examples: tf.keras.Model, torch.Module):
    ModelType = TypeVar("ModelType")

    # A generic input / output samples for reading the inputs / outputs properties:
    IOSampleType = TypeVar("IOSampleType")

    # A generic object type for what can be wrapped with a framework MLRun interface (examples: xgb, xgb.XGBModel):
    MLRunInterfaceableType = TypeVar("MLRunInterfaceableType")

    # Type for a MLRun Interface restoration tuple as returned from 'remove_interface':
    MLRunInterfaceRestorationType = Tuple[
        Dict[str, Any],  # Interface properties.
        Dict[str, Any],  # Replaced properties.
        List[str],  # Replaced methods and functions.
    ]

    # Common dataset type to all frameworks:
    DatasetType = Union[
        list,
        tuple,
        dict,
        np.ndarray,
        pd.DataFrame,
        pd.Series,
        "scipy.sparse.base.spmatrix",  # noqa: F821
    ]

    # A joined type for receiving a path from 'pathlib' or 'os.path':
    PathType = Union[str, Path]

    # A joined type for all trackable values (for logging):
    TrackableType = Union[str, bool, float, int]

    # Types available in the extra data dictionary of an artifact:
    ExtraDataType = Union[str, bytes, Artifact, DataItem]


class LoggingMode(Enum):
    """
    The logging mode options.
    """

    TRAINING = "Training"
    EVALUATION = "Evaluation"


class CommonUtils(ABC):
    """
    Common utilities functions to all frameworks.
    """

    @staticmethod
    def to_array(dataset: CommonTypes.DatasetType) -> np.ndarray:
        """
        Convert the given dataset to np.ndarray.

        :param dataset: The dataset to convert. Must be one of {pd.DataFrame, pd.Series, scipy.sparse.base.spmatrix,
                        list, tuple, dict}.

        :return: The dataset as a ndarray.

        :raise MLRunInvalidArgumentError: If the dataset type is not supported.
        """
        if isinstance(dataset, np.ndarray):
            return dataset
        if isinstance(dataset, (pd.DataFrame, pd.Series)):
            return dataset.to_numpy()
        if isinstance(dataset, (list, tuple)):
            return np.array(dataset)
        if isinstance(dataset, dict):
            return np.array(list(dataset.values()))
        try:
            # SciPy is not in MLRun's requirements but common to all frameworks.
            import scipy.sparse.base as sp

            if isinstance(dataset, sp.spmatrix):
                return dataset.toarray()
        except ModuleNotFoundError:
            # SciPy is not installed.
            pass

        raise mlrun.errors.MLRunInvalidArgumentError(
            f"Could not convert the given dataset into a numpy ndarray. Supporting conversion from: "
            f"{CommonUtils.get_union_typehint_string(CommonTypes.DatasetType)}. "
            f"The given dataset was of type: '{type(dataset)}'"
        )

    @staticmethod
    def to_dataframe(dataset: CommonTypes.DatasetType) -> pd.DataFrame:
        """
        Convert the given dataset to pd.DataFrame.

        :param dataset: The dataset to convert. Must be one of {np.ndarray, pd.Series, scipy.sparse.base.spmatrix, list,
                        tuple, dict}.

        :return: The dataset as a DataFrame.

        :raise MLRunInvalidArgumentError: If the dataset type is not supported.
        """
        if isinstance(dataset, pd.DataFrame):
            return dataset
        if isinstance(dataset, (np.ndarray, pd.Series, list, tuple, dict)):
            return pd.DataFrame(dataset)
        try:
            # SciPy is not in MLRun's requirements but common to all frameworks.
            import scipy.sparse.base as sp

            if isinstance(dataset, sp.spmatrix):
                return pd.DataFrame.sparse.from_spmatrix(dataset)
        except ModuleNotFoundError:
            # SciPy is not installed.
            pass
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"Could not convert the given dataset into a pandas DataFrame. Supporting conversion from: "
            f"{CommonUtils.get_union_typehint_string(CommonTypes.DatasetType)}. "
            f"The given dataset was of type: '{type(dataset)}'"
        )

    @staticmethod
    def convert_value_type_to_np_dtype(
        value_type: str,
    ) -> np.dtype:
        """
        Get the 'numpy.dtype' equivalent to the given MLRun value type.

        :param value_type: The MLRun value type to convert to numpy data type.

        :return: The 'numpy.dtype' equivalent to the given MLRun data type.

        :raise MLRunInvalidArgumentError: If numpy is not supporting the given data type.
        """
        # Initialize the mlrun to numpy data type conversion map:
        conversion_map = {
            ValueType.BOOL: np.bool,
            ValueType.INT8: np.int8,
            ValueType.INT16: np.int16,
            ValueType.INT32: np.int32,
            ValueType.INT64: np.int64,
            ValueType.UINT8: np.uint8,
            ValueType.UINT16: np.uint16,
            ValueType.UINT32: np.uint32,
            ValueType.UINT64: np.uint64,
            ValueType.FLOAT16: np.float16,
            ValueType.FLOAT: np.float32,
            ValueType.DOUBLE: np.float64,
        }

        # Convert and return:
        if value_type in conversion_map:
            return conversion_map[value_type]
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"The ValueType given is not supported in numpy: '{value_type}'"
        )

    @staticmethod
    def convert_np_dtype_to_value_type(np_dtype: Union[np.dtype, type, str]) -> str:
        """
        Convert the given numpy data type to MLRun value type. It is better to use explicit bit namings (for example:
        instead of using 'np.double', use 'np.float64').

        :param np_dtype: The numpy data type to convert to MLRun's value type. Expected to be a 'numpy.dtype', 'type' or
                         'str'.

        :return: The MLRun value type converted from the given data type.

        :raise MLRunInvalidArgumentError: If the numpy data type is not supported by MLRun.
        """
        # Initialize the numpy to mlrun data type conversion map:
        conversion_map = {
            np.bool.__name__: ValueType.BOOL,
            np.byte.__name__: ValueType.INT8,
            np.int8.__name__: ValueType.INT8,
            np.short.__name__: ValueType.INT16,
            np.int16.__name__: ValueType.INT16,
            np.int32.__name__: ValueType.INT32,
            np.int.__name__: ValueType.INT64,
            np.long.__name__: ValueType.INT64,
            np.int64.__name__: ValueType.INT64,
            np.ubyte.__name__: ValueType.UINT8,
            np.uint8.__name__: ValueType.UINT8,
            np.ushort.__name__: ValueType.UINT16,
            np.uint16.__name__: ValueType.UINT16,
            np.uint32.__name__: ValueType.UINT32,
            np.uint.__name__: ValueType.UINT64,
            np.uint64.__name__: ValueType.UINT64,
            np.half.__name__: ValueType.FLOAT16,
            np.float16.__name__: ValueType.FLOAT16,
            np.single.__name__: ValueType.FLOAT,
            np.float32.__name__: ValueType.FLOAT,
            np.double.__name__: ValueType.DOUBLE,
            np.float.__name__: ValueType.DOUBLE,
            np.float64.__name__: ValueType.DOUBLE,
        }

        # Parse the given numpy data type to string:
        if isinstance(np_dtype, np.dtype):
            np_dtype = np_dtype.name
        elif isinstance(np_dtype, type):
            np_dtype = np_dtype.__name__

        # Convert and return:
        if np_dtype in conversion_map:
            return conversion_map[np_dtype]
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"MLRun value type is not supporting the given numpy data type: '{np_dtype}'"
        )

    @staticmethod
    def get_union_typehint_string(union_typehint) -> str:
        """
        Get the string representation of a types.Union typehint object.

        :param union_typehint: The union typehint to get its string representation.

        :return: The union typehint's string.
        """
        return re.sub(r"typing.Union|[\[\]'\"()]|ForwardRef", "", str(union_typehint))
