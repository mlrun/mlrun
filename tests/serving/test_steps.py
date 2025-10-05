# Copyright 2025 Iguazio
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

import mlrun
from mlrun.serving.steps import ChoiceByField

# --- Success cases ---


def test_choice_by_field_with_string_value():
    choice = ChoiceByField("fieldA")
    event = {"fieldA": "outlet1"}
    assert choice.select_outlets(event) == ["outlet1"]


def test_choice_by_field_with_list_value():
    choice = ChoiceByField("fieldA")
    event = {"fieldA": ["outlet1", "outlet2"]}
    assert choice.select_outlets(event) == ["outlet1", "outlet2"]


def test_choice_by_field_with_tuple_value():
    choice = ChoiceByField("fieldA")
    event = {"fieldA": ("outlet1", "outlet2")}
    assert choice.select_outlets(event) == ("outlet1", "outlet2")


# --- Error cases ---


def test_choice_by_field_missing_field():
    choice = ChoiceByField("fieldA")
    event = {"fieldB": "outlet1"}  # missing 'fieldA'
    with pytest.raises(
        mlrun.errors.MLRunRuntimeError,
        match=r"Field 'fieldA' is not contained in the event keys \['fieldB'\].",
    ):
        choice.select_outlets(event)


def test_choice_by_field_none_value():
    choice = ChoiceByField("fieldA")
    event = {"fieldA": None}
    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError,
        match=r"Field 'fieldA' exists but its value is None\.",
    ):
        choice.select_outlets(event)


def test_choice_by_field_invalid_type():
    choice = ChoiceByField("fieldA")
    event = {"fieldA": 123}  # not string/list/tuple
    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentTypeError,
        match=r"Field 'fieldA' must be a string or list of strings but is instead of type 'int'\.",
    ):
        choice.select_outlets(event)


def test_choice_by_field_empty_list():
    choice = ChoiceByField("fieldA")
    event = {"fieldA": []}
    with pytest.raises(
        mlrun.errors.MLRunRuntimeError,
        match=r"The value of the key 'fieldA' cannot be an empty list\.",
    ):
        choice.select_outlets(event)
