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

# this module is WIP


class DataType:
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
        "integer": DataType.INT64,
        "string": DataType.STRING,
        "number": DataType.DOUBLE,
        "datetime": DataType.DATETIME,
        "boolean": DataType.BOOL,
    }
    return type_map[value]


def python_type_to_value_type(value_type):
    type_name = value_type.__name__
    type_map = {
        "int": DataType.INT64,
        "str": DataType.STRING,
        "float": DataType.DOUBLE,
        "bytes": DataType.BYTES,
        "float64": DataType.DOUBLE,
        "float32": DataType.FLOAT,
        "int64": DataType.INT64,
        "uint64": DataType.INT64,
        "int32": DataType.INT32,
        "uint32": DataType.INT32,
        "uint8": DataType.INT32,
        "int8": DataType.INT32,
        "bool": DataType.BOOL,
        "timedelta": DataType.INT64,
        "datetime64[ns]": DataType.INT64,
        "datetime64[ns, tz]": DataType.INT64,
        "category": DataType.STRING,
    }

    if type_name in type_map:
        return type_map[type_name]


def spark_to_value_type(data_type):
    type_map = {
        "int": DataType.INT64,
        "bigint": DataType.INT64,
        "double": DataType.DOUBLE,
        "boolean": DataType.BOOL,
        "timestamp": DataType.DATETIME,
        "string": DataType.STRING,
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
