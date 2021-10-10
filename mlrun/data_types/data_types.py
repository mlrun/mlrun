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

from enum import Enum

# this module is WIP
import pyarrow
from pyarrow.lib import TimestampType


# TODO: Enable when implementing IOLogging.
class DataType(Enum):
    """
    Tensor data type. Used to define inputs (datasets content) and outputs (predictions) of a model.
    """

    BOOL = "bool"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    INT128 = "int128"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
    UINT128 = "uint128"
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"

    @classmethod
    def from_string(cls, data_type_string: str):
        """
        Get the MLRun data type enum by its string value.

        :param data_type_string: The MLRun data type string to convert to MLRun's data type enum.

        :return: The MLRun data type converted from the given string value.

        :raise ValueError: If the string is not a valid value of a MLRun data type.
        """
        # Initialize the string value to mlrun data type conversion map:
        conversion_map = {
            cls[key].value: cls[key]
            for key in cls.__dict__.keys()
            if isinstance(cls.__dict__[key], cls)
        }

        # Convert and return:
        if data_type_string not in conversion_map:
            raise ValueError(
                "There is no '{}' MLRun data type".format(data_type_string)
            )
        return conversion_map[data_type_string]

    def to_numpy_dtype(self):
        """
        Get the 'numpy.dtype' equivalent to this MLRun data type.

        :return: The 'numpy.dtype' equivalent to this MLRun data type.

        :raise ValueError: If numpy is not supporting this data type.
        """
        import numpy as np

        # Initialize the mlrun to numpy data type conversion map:
        conversion_map = {
            DataType.BOOL.value: np.bool,
            DataType.INT8.value: np.int8,
            DataType.INT16.value: np.int16,
            DataType.INT32.value: np.int32,
            DataType.INT64.value: np.int64,
            DataType.UINT8.value: np.uint8,
            DataType.UINT16.value: np.uint16,
            DataType.UINT32.value: np.uint32,
            DataType.UINT64.value: np.uint64,
            DataType.FLOAT16.value: np.float16,
            DataType.FLOAT32.value: np.float32,
            DataType.FLOAT64.value: np.float64,
        }

        # Convert and return:
        return self._to_framework(conversion_map=conversion_map)

    def to_tensorflow_dtype(self):
        """
        Get the 'tensorflow.DType' equivalent to this MLRun data type.

        :return: The 'tensorflow.DType' equivalent to this MLRun data type.

        :raise ValueError: If tensorflow is not supporting this data type.
        """
        import tensorflow as tf

        # Initialize the mlrun to tensorflow data type conversion map:
        conversion_map = {
            DataType.BOOL.value: tf.bool,
            DataType.INT8.value: tf.int8,
            DataType.INT16.value: tf.int16,
            DataType.INT32.value: tf.int32,
            DataType.INT64.value: tf.int64,
            DataType.UINT8.value: tf.uint8,
            DataType.UINT16.value: tf.uint16,
            DataType.UINT32.value: tf.uint32,
            DataType.UINT64.value: tf.uint64,
            DataType.BFLOAT16.value: tf.bfloat16,
            DataType.FLOAT16.value: tf.float16,
            DataType.FLOAT32.value: tf.float32,
            DataType.FLOAT64.value: tf.float64,
        }

        # Convert and return:
        return self._to_framework(conversion_map=conversion_map)

    def to_torch_dtype(self):
        """
        Get the 'torch.dtype' equivalent to this MLRun data type.

        :return: The 'torch.dtype' equivalent to this MLRun data type.

        :raise ValueError: If torch is not supporting this data type.
        """
        import torch

        # Initialize the mlrun to torch data type conversion map:
        conversion_map = {
            DataType.BOOL.value: torch.bool,
            DataType.INT8.value: torch.int8,
            DataType.INT16.value: torch.int16,
            DataType.INT32.value: torch.int32,
            DataType.INT64.value: torch.int64,
            DataType.UINT8.value: torch.uint8,
            DataType.BFLOAT16.value: torch.bfloat16,
            DataType.FLOAT16.value: torch.float16,
            DataType.FLOAT32.value: torch.float32,
            DataType.FLOAT64.value: torch.float64,
        }

        # Convert and return:
        return self._to_framework(conversion_map=conversion_map)

    @staticmethod
    def from_numpy_dtype(numpy_data_type):
        """
        Convert the given numpy data type to MLRun data type. It is better to use explicit bit namings (for example:
        instead of using 'np.double', use 'np.float64').

        :param numpy_data_type: The numpy data type to convert to MLRun's data type. Expected to be a 'numpy.dtype',
                                'type' or 'str'.

        :return: The MLRun data type converted from the given data type.

        :raise ValueError: If the data type is not supported by MLRun.
        """
        import numpy as np

        # Initialize the numpy to mlrun data type conversion map:
        conversion_map = {
            np.bool.__name__: DataType.BOOL,
            np.byte.__name__: DataType.INT8,
            np.int8.__name__: DataType.INT8,
            np.short.__name__: DataType.INT16,
            np.int16.__name__: DataType.INT16,
            np.int32.__name__: DataType.INT32,
            np.int.__name__: DataType.INT64,
            np.long.__name__: DataType.INT64,
            np.int64.__name__: DataType.INT64,
            np.ubyte.__name__: DataType.UINT8,
            np.uint8.__name__: DataType.UINT8,
            np.ushort.__name__: DataType.UINT16,
            np.uint16.__name__: DataType.UINT16,
            np.uint32.__name__: DataType.UINT32,
            np.uint.__name__: DataType.UINT64,
            np.uint64.__name__: DataType.UINT64,
            np.half.__name__: DataType.FLOAT16,
            np.float16.__name__: DataType.FLOAT16,
            np.single.__name__: DataType.FLOAT32,
            np.float32.__name__: DataType.FLOAT32,
            np.double.__name__: DataType.FLOAT64,
            np.float.__name__: DataType.FLOAT64,
            np.float64.__name__: DataType.FLOAT64,
        }

        # Parse the given numpy data type to string:
        if isinstance(numpy_data_type, np.dtype):
            numpy_data_type = numpy_data_type.name
        elif isinstance(numpy_data_type, type):
            numpy_data_type = numpy_data_type.__name__

        # Convert and return:
        return DataType._from_framework(
            conversion_map=conversion_map, data_type=numpy_data_type
        )

    @staticmethod
    def from_tensorflow_dtype(tensorflow_data_type):
        """
        Convert the given tensorflow data type to MLRun data type. All of the CUDA supported data types are supported.
        For more information regarding tensorflow data types visit: https://www.tensorflow.org/api_docs/python/tf/dtypes

        :param tensorflow_data_type: The tensorflow data type to convert to MLRun's data type. Expected to be a
                                     'tensorflow.dtype' or 'str'.

        :return: The MLRun data type converted from the given data type.

        :raise ValueError: If the data type is not supported by MLRun.
        """
        import tensorflow as tf

        # Initialize the tensorflow to mlrun data type conversion map:
        conversion_map = {
            tf.bool.name: DataType.BOOL,
            tf.int8.name: DataType.INT8,
            tf.int16.name: DataType.INT16,
            tf.int32.name: DataType.INT32,
            tf.int64.name: DataType.INT64,
            tf.uint8.name: DataType.UINT8,
            tf.uint16.name: DataType.UINT16,
            tf.uint32.name: DataType.UINT32,
            tf.uint64.name: DataType.UINT64,
            tf.bfloat16.name: DataType.BFLOAT16,
            tf.half.name: DataType.FLOAT16,
            tf.float16.name: DataType.FLOAT16,
            tf.float32.name: DataType.FLOAT32,
            tf.double.name: DataType.FLOAT64,
            tf.float64.name: DataType.FLOAT64,
        }

        # Parse the given tensorflow data type to string:
        if isinstance(tensorflow_data_type, tf.DType):
            tensorflow_data_type = tensorflow_data_type.name

        # Convert and return:
        return DataType._from_framework(
            conversion_map=conversion_map, data_type=tensorflow_data_type
        )

    @staticmethod
    def from_torch_dtype(torch_data_type):
        """
        Convert the given torch data type to MLRun data type. All of the CUDA supported data types are supported. For
        more information regarding torch data types visit: https://pytorch.org/docs/stable/tensors.html#data-types

        :param torch_data_type: The torch data type to convert to MLRun's data type. Expected to be a 'torch.dtype' or
                                'str'.

        :return: The MLRun data type converted from the given data type.

        :raise ValueError: If the data type is not supported by MLRun.
        """
        import torch

        # Initialize the torch to mlrun data type conversion map:
        conversion_map = {
            str(torch.bool): DataType.BOOL,
            str(torch.int8): DataType.INT8,
            str(torch.short): DataType.INT16,
            str(torch.int16): DataType.INT16,
            str(torch.int): DataType.INT32,
            str(torch.int32): DataType.INT32,
            str(torch.long): DataType.INT64,
            str(torch.int64): DataType.INT64,
            str(torch.uint8): DataType.UINT8,
            str(torch.bfloat16): DataType.BFLOAT16,
            str(torch.half): DataType.FLOAT16,
            str(torch.float16): DataType.FLOAT16,
            str(torch.float): DataType.FLOAT32,
            str(torch.float32): DataType.FLOAT32,
            str(torch.double): DataType.FLOAT64,
            str(torch.float64): DataType.FLOAT64,
        }

        # Parse the given torch data type to string:
        if isinstance(torch_data_type, torch.dtype):
            torch_data_type = str(torch_data_type)

        # Convert and return:
        return DataType._from_framework(
            conversion_map=conversion_map, data_type=torch_data_type
        )

    def _to_framework(self, conversion_map: dict):
        """
        Get the equivalent value of this enum according to the given conversion map.

        :param conversion_map: The conversion map to use for getting the equivalent value of this enum. A dictionary of
                               string keys and a supported framework data type as values.

        :return: The equivalent value of this enum according to the given conversion map.

        :raise ValueError: If this data type value is not inside the given map.
        """
        if self.value in conversion_map:
            return conversion_map[self.value]
        raise ValueError(
            "The required framework is not supporting '{}' data type.".format(
                self.value
            )
        )

    @staticmethod
    def _from_framework(conversion_map: dict, data_type: str):
        """
        Look for the given data type in the given conversion map and return the value. If the required data type is not
        in the map, a value error will be raised.

        :param conversion_map: The map to use for the conversion. A dictionary of string as keys and MLRun DataType as
                               values.
        :param data_type:      The data type to convert to MLRun's data type.

        :return: The MLRun data type converted from the given framework data type.

        :raise ValueError: If the data type is not supported by MLRun.
        """
        if data_type in conversion_map:
            return conversion_map[data_type]
        raise ValueError("Unsupported data type: '{}'.".format(data_type))


class ValueType:
    """Feature value type. Used to define data types in Feature Tables."""

    UNKNOWN = ""
    BYTES = "bytes"
    STRING = "str"
    INT32 = "int32"
    INT64 = "int"
    DOUBLE = "float"
    FLOAT = "float32"
    BOOL = "bool"
    DATETIME = "datetime"
    BYTES_LIST = "List[bytes]"
    STRING_LIST = "List[string]"
    INT32_LIST = "List[int32]"
    INT64_LIST = "List[int]"
    DOUBLE_LIST = "List[float]"
    FLOAT_LIST = "List[float32]"
    BOOL_LIST = "List[bool]"


def pd_schema_to_value_type(value):
    type_map = {
        "integer": ValueType.INT64,
        "string": ValueType.STRING,
        "number": ValueType.DOUBLE,
        "datetime": ValueType.DATETIME,
        "boolean": ValueType.BOOL,
        "any": ValueType.STRING,
        "duration": ValueType.INT64,
    }
    return type_map[value]


def pa_type_to_value_type(type_):
    # To catch timestamps with timezones. This also catches timestamps with different units
    if isinstance(type_, TimestampType):
        return ValueType.DATETIME

    type_map = {
        pyarrow.bool_(): ValueType.BOOL,
        pyarrow.int64(): ValueType.INT64,
        pyarrow.int32(): ValueType.INT32,
        pyarrow.float32(): ValueType.FLOAT,
        pyarrow.float64(): ValueType.DOUBLE,
    }
    return type_map.get(type_, ValueType.STRING)


def python_type_to_value_type(value_type):
    type_name = value_type.__name__
    type_map = {
        "int": ValueType.INT64,
        "str": ValueType.STRING,
        "float": ValueType.DOUBLE,
        "bytes": ValueType.BYTES,
        "float64": ValueType.DOUBLE,
        "float32": ValueType.FLOAT,
        "int64": ValueType.INT64,
        "uint64": ValueType.INT64,
        "int32": ValueType.INT32,
        "uint32": ValueType.INT32,
        "uint8": ValueType.INT32,
        "int8": ValueType.INT32,
        "bool": ValueType.BOOL,
        "timedelta": ValueType.INT64,
        "datetime64[ns]": ValueType.INT64,
        "datetime64[ns, tz]": ValueType.INT64,
        "category": ValueType.STRING,
    }

    if type_name in type_map:
        return type_map[type_name]


def spark_to_value_type(data_type):
    type_map = {
        "int": ValueType.INT64,
        "bigint": ValueType.INT64,
        "double": ValueType.DOUBLE,
        "boolean": ValueType.BOOL,
        "timestamp": ValueType.DATETIME,
        "string": ValueType.STRING,
        "array": "list",
        "map": "dict",
    }
    if data_type in type_map:
        return type_map[data_type]
    return data_type


class InferOptions:
    Null = 0
    Entities = 1
    Features = 2
    Index = 4
    Stats = 8
    Histogram = 16
    Preview = 32

    @staticmethod
    def schema():
        return InferOptions.Entities + InferOptions.Features + InferOptions.Index

    @staticmethod
    def all_stats():
        return InferOptions.Stats + InferOptions.Histogram + InferOptions.Preview

    @staticmethod
    def all():
        return (
            InferOptions.schema()
            + InferOptions.Stats
            + InferOptions.Histogram
            + InferOptions.Preview
        )

    @staticmethod
    def default():
        return InferOptions.all()

    @staticmethod
    def get_common_options(one, two):
        return one & two
