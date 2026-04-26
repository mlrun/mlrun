# Copyright 2026 Iguazio
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

import pytest

from mlrun.data_types.data_types import ValueType, python_type_to_value_type


@pytest.mark.parametrize(
    "type_name, expected_value_type",
    [
        ("int8", ValueType.INT8),
        ("uint8", ValueType.UINT8),
        ("int32", ValueType.INT32),
        ("uint32", ValueType.INT32),
        ("int64", ValueType.INT64),
        ("uint64", ValueType.INT64),
        ("int", ValueType.INT64),
        ("float", ValueType.DOUBLE),
        ("float32", ValueType.FLOAT),
        ("float64", ValueType.DOUBLE),
        ("str", ValueType.STRING),
        ("bytes", ValueType.BYTES),
        ("bool", ValueType.BOOL),
    ],
)
def test_python_type_to_value_type(type_name, expected_value_type):
    """Test that Python type names map to the correct ValueType enum values."""
    result = python_type_to_value_type(type_name)
    assert result == expected_value_type, (
        f"Expected {type_name!r} to map to {expected_value_type!r}, got {result!r}"
    )
