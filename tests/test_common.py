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


import pytest

import mlrun.common.schemas.common


@pytest.mark.parametrize(
    "labels,expected",
    [
        (None, []),
        ({}, []),
        ([], []),
        (["label1", "label2"], ["label1", "label2"]),
        (["label1"], ["label1"]),
        ({"label1": "value1", "label2": "value2"}, ["label1=value1", "label2=value2"]),
        ({"label1": 1}, ["label1=1"]),
        (["label1=value1"], ["label1=value1"]),
        ({"label1": "value1", "label2": None}, ["label1=value1", "label2"]),
        ("label1=value1,label2", ["label1=value1", "label2"]),
    ],
)
def test_labels_validation(labels, expected):
    labels_result = mlrun.common.schemas.common.LabelsModel(labels=labels).labels
    assert labels_result == expected
