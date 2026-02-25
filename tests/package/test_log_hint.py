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

from mlrun.errors import MLRunInvalidArgumentError
from mlrun.package.log_hint import LogHint


@pytest.mark.parametrize(
    "log_hint, expected_log_hint",
    [
        # No unbundling
        ("some_key", LogHint(key="some_key")),
        (
            "some_key:artifact",
            LogHint(key="some_key", artifact_type="artifact"),
        ),
        (
            "some_key :artifact",
            LogHint(key="some_key", artifact_type="artifact"),
        ),
        (
            "some_key: artifact",
            LogHint(key="some_key", artifact_type="artifact"),
        ),
        (
            "some_key : artifact",
            LogHint(key="some_key", artifact_type="artifact"),
        ),
        (
            "some_key:",
            "Incorrect log hint pattern. The ':' in a log hint should specify",
        ),
        (
            "some_key : artifact : error",
            "Incorrect log hint pattern. Log hints can have only a single ':' in them",
        ),
        (LogHint(key="some_key"), LogHint(key="some_key")),
        (
            LogHint(key="some_key", artifact_type="artifact"),
            LogHint(key="some_key", artifact_type="artifact"),
        ),
        # Full unbundling (no level specified)
        ("*results", LogHint(key="results", itemized=True)),
        ("* results", LogHint(key="results", itemized=True)),
        (" *results", LogHint(key="results", itemized=True)),
        (" * results", LogHint(key="results", itemized=True)),
        # Level-specific unbundling
        ("1 * results", LogHint(key="results", itemized=1)),
        ("2 *nested", LogHint(key="nested", itemized=2)),
        ("3* deep", LogHint(key="deep", itemized=3)),
        ("10*multi", LogHint(key="multi", itemized=10)),
        # Error case - invalid level
        ("abc*results", "Invalid unbundle level"),
        ("1.5*results", "Invalid unbundle level"),
        # Error case - empty key after asterisk
        ("*", "Key is missing after the unbundle operator '*'"),
        ("  * ", "Key is missing after the unbundle operator '*'"),
        ("1*", "Key is missing after the unbundle operator '*'"),
        # Packing kwargs - success cases
        (
            "key:type[k1=1]",
            LogHint(key="key", artifact_type="type", packing_kwargs={"k1": 1}),
        ),
        (
            "key:type[k1=1, k2='hello']",
            LogHint(
                key="key",
                artifact_type="type",
                packing_kwargs={"k1": 1, "k2": "hello"},
            ),
        ),
        (
            "key : type[n=42, b=True, x=None]",
            LogHint(
                key="key",
                artifact_type="type",
                packing_kwargs={"n": 42, "b": True, "x": None},
            ),
        ),
        (
            "*key:type[k1=1]",
            LogHint(
                key="key",
                artifact_type="type",
                itemized=True,
                packing_kwargs={"k1": 1},
            ),
        ),
        (
            "2*key:type[k1=1]",
            LogHint(
                key="key",
                artifact_type="type",
                itemized=2,
                packing_kwargs={"k1": 1},
            ),
        ),
        # Packing kwargs - error cases
        ("key:type[k1=1", "Incorrect log hint pattern for packing kwargs"),
        ("key:type[k1]", "packing kwarg should be given in the format"),
        ("key:type[k1=undefined_var]", "not a valid Python literal"),
        # Round-trip: LogHint → .dict() → parse_obj() preserves all fields
        (
            LogHint(
                key="deep",
                artifact_type="model",
                itemized=2,
                tag="v1",
                packing_kwargs={"dtype": "float32"},
                labels={"env": "test"},
                extra_data={"schema": "..."},
                metrics={"acc": "..."},
            ).dict(),
            LogHint(
                key="deep",
                artifact_type="model",
                itemized=2,
                tag="v1",
                packing_kwargs={"dtype": "float32"},
                labels={"env": "test"},
                extra_data={"schema": "..."},
                metrics={"acc": "..."},
            ),
        ),
    ],
)
def test_model_validate_from_string(
    log_hint: str | dict, expected_log_hint: str | dict
):
    """
    Test the `LogHint.parse_obj` (will be `model_validate` when API support Pydantic v2) class method for handling
    strings.

    :param log_hint:          The log hint to parse.
    :param expected_log_hint: The expected parsed log hint. A string value indicates the parsing should fail with the
                              provided error message in the variable.
    """
    try:
        parsed_log_hint = LogHint.parse_obj(obj=log_hint)
        assert parsed_log_hint == expected_log_hint
    except MLRunInvalidArgumentError as error:
        if isinstance(expected_log_hint, str):
            assert expected_log_hint in str(error)
        else:
            raise error
