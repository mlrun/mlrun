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

from .data_types import (
    InferOptions,
    ValueType,
    pd_schema_to_value_type,
    python_type_to_value_type,
)
from .infer import DFDataInfer


class BaseDataInfer:
    infer_schema = None
    get_preview = None
    get_stats = None


def is_spark_dataframe(df) -> bool:
    return "rdd" in dir(df)


def get_infer_interface(df) -> BaseDataInfer:
    if is_spark_dataframe(df):
        from .spark import SparkDataInfer

        return SparkDataInfer
    return DFDataInfer
