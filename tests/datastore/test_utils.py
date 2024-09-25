# Copyright 2024 Iguazio
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


import json

import pytest

import mlrun.datastore
import mlrun.datastore.wasbfs
from mlrun.datastore.utils import transform_list_filters_to_tuple


@pytest.mark.parametrize(
    "additional_filters, message",
    [
        ([("x", "=", 3)], ""),
        (
            [[("x", "=", 3), ("x", "=", 4), ("x", "=", 5)]],
            "additional_filters does not support nested list inside filter tuples except in -in- logic.",
        ),
        (
            [[("x", "=", 3), ("x", "=", 4)]],
            "additional_filters does not support nested list inside filter tuples except in -in- logic.",
        ),
        (("x", "=", 3), "mlrun supports additional_filters only as a list of tuples."),
        ([("x", "in", [3, 4]), ("y", "in", [3, 4])], ""),
        ([0], "mlrun supports additional_filters only as a list of tuples."),
        (
            [("age", "=", float("nan"))],
            "using NaN in additional_filters is not supported",
        ),
        (
            [("age", "in", [10, float("nan")])],
            "using NaN in additional_filters is not supported",
        ),
        ([("x", "=", "=", 3), ("y", "in", [3, 4])], "illegal filter tuple length"),
        ([()], ""),
        ([], ""),
    ],
)
def test_transform_list_filters_to_tuple(additional_filters, message):
    back_from_json_serialization = json.loads(json.dumps(additional_filters))

    if message:
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError, match=message):
            transform_list_filters_to_tuple(additional_filters)
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError, match=message):
            transform_list_filters_to_tuple(
                additional_filters=back_from_json_serialization
            )
    else:
        transform_list_filters_to_tuple(additional_filters)
        result = transform_list_filters_to_tuple(back_from_json_serialization)
        assert result == additional_filters
