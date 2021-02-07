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
    }
    return type_map[value]


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
