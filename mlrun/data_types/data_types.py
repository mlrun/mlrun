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

# this module is WIP
from enum import Enum

import pyarrow
from pyarrow.lib import TimestampType


class ValueType(str, Enum):
    """Feature value type. Used to define data types in Feature Tables."""

    UNKNOWN = ""
    BOOL = "bool"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int"
    INT128 = "int128"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
    UINT128 = "uint128"
    FLOAT16 = "float16"
    FLOAT = "float32"
    DOUBLE = "float"
    BFLOAT16 = "bfloat16"
    BYTES = "bytes"
    STRING = "str"
    DATETIME = "datetime"
    LIST = "List"
    BYTES_LIST = "List[bytes]"
    STRING_LIST = "List[string]"
    INT32_LIST = "List[int32]"
    INT64_LIST = "List[int]"
    DOUBLE_LIST = "List[float]"
    FLOAT_LIST = "List[float32]"
    BOOL_LIST = "List[bool]"
    Tuple = "Tuple"


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

    # pandas category type translates to pyarrow DictionaryType
    # we need to unpack the value type (ML-7868)
    if isinstance(type_, pyarrow.DictionaryType):
        type_ = type_.value_type

    type_map = {
        pyarrow.bool_(): ValueType.BOOL,
        pyarrow.int64(): ValueType.INT64,
        pyarrow.int32(): ValueType.INT32,
        pyarrow.float32(): ValueType.FLOAT,
        pyarrow.float64(): ValueType.DOUBLE,
    }
    return type_map.get(type_, ValueType.STRING)


def python_type_to_value_type(value_type):
    type_name = (
        value_type.__name__ if hasattr(value_type, "__name__") else str(value_type)
    )
    type_map = {
        "int": ValueType.INT64,
        "str": ValueType.STRING,
        "float": ValueType.DOUBLE,
        "bytes": ValueType.BYTES,
        "float64": ValueType.DOUBLE,
        "Float64": ValueType.DOUBLE,
        "float32": ValueType.FLOAT,
        "int64": ValueType.INT64,
        "Int64": ValueType.INT64,
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
        "list": ValueType.LIST,
        "tuple": ValueType.Tuple,
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


def gbq_to_pandas_dtype(gbq_type):
    import pandas as pd

    if isinstance(gbq_type, TimestampType):
        return ValueType.DATETIME

    type_map = {
        "STRING": "object",
        "BOOL": "bool",
        "FLOAT": "float64",
        "INTEGER": pd.Int64Dtype(),
        "TIMESTAMP": "datetime64[ns]",
    }
    return type_map.get(gbq_type, "object")


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
