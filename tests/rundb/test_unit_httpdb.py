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
#
# test_httpdb.py actually holds integration tests (that should be migrated to tests/integration/sdk_api/httpdb)
# currently we are running it in the integration tests CI step so adding this file for unit tests for the httpdb
import enum
import unittest.mock

import pytest
import requests

import mlrun.config
import mlrun.db.httpdb


class SomeEnumClass(str, enum.Enum):
    value1 = "value1"
    value2 = "value2"


def test_api_call_enum_conversion():
    db = mlrun.db.httpdb.HTTPRunDB("fake-url")
    db.session = unittest.mock.Mock()

    # ensure not exploding when no headers/params
    db.api_call("GET", "some-path")

    db.api_call(
        "GET",
        "some-path",
        headers={"enum-value": SomeEnumClass.value1, "string-value": "value"},
        params={"enum-value": SomeEnumClass.value2, "string-value": "value"},
    )
    for dict_key in ["headers", "params"]:
        for value in db.session.request.call_args_list[1][1][dict_key].values():
            assert type(value) == str


@pytest.mark.parametrize(
    "feature_config,exception_type,exception_message,call_amount",
    [
        # feature enabled
        ("enabled", Exception, "some-error", 1),
        ("enabled", ConnectionError, "some-error", 1),
        ("enabled", ConnectionResetError, "some-error", 1),
        (
            "enabled",
            ConnectionError,
            "Connection aborted",
            # one try + the max retries
            1 + mlrun.config.config.http_retry_defaults.max_retries,
        ),
        (
            "enabled",
            ConnectionResetError,
            "Connection reset by peer",
            # one try + the max retries
            1 + mlrun.config.config.http_retry_defaults.max_retries,
        ),
        (
            "enabled",
            ConnectionRefusedError,
            "Connection refused",
            # one try + the max retries
            1 + mlrun.config.config.http_retry_defaults.max_retries,
        ),
        (
            "enabled",
            requests.exceptions.ConnectionError,
            "Connection aborted",
            # one try + the max retries
            1 + mlrun.config.config.http_retry_defaults.max_retries,
        ),
        # feature disabled
        ("disabled", Exception, "some-error", 1),
        ("disabled", ConnectionError, "some-error", 1),
        ("disabled", ConnectionResetError, "some-error", 1),
        ("disabled", ConnectionError, "Connection aborted", 1),
        (
            "disabled",
            ConnectionResetError,
            "Connection reset by peer",
            1,
        ),
        (
            "disabled",
            ConnectionRefusedError,
            "Connection refused",
            1,
        ),
        (
            "disabled",
            requests.exceptions.ConnectionError,
            "Connection aborted",
            # one try + the max retries
            1,
        ),
    ],
)
def test_connection_reset_causes_retries(
    feature_config, exception_type, exception_message, call_amount
):
    mlrun.config.config.httpdb.retry_api_call_on_exception = feature_config
    db = mlrun.db.httpdb.HTTPRunDB("fake-url")
    original_request = requests.Session.request
    requests.Session.request = unittest.mock.Mock()
    requests.Session.request.side_effect = exception_type(exception_message)

    # patch sleep to make test faster
    with unittest.mock.patch("time.sleep"):

        # Catching also MLRunRuntimeError as if the exception inherits from requests.RequestException, it will be
        # wrapped with MLRunRuntimeError
        with pytest.raises((exception_type, mlrun.errors.MLRunRuntimeError)):
            db.api_call("GET", "some-path")

    assert requests.Session.request.call_count == call_amount
    requests.Session.request = original_request
