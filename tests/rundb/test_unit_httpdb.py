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
#
# test_httpdb.py actually holds integration tests (that should be migrated to tests/integration/sdk_api/httpdb)
# currently we are running it in the integration tests CI step so adding this file for unit tests for the httpdb
import enum
import unittest.mock

import pytest
import requests
import urllib3
import urllib3.exceptions

import mlrun.artifacts.base
import mlrun.config
import mlrun.db.httpdb


class SomeEnumClass(str, enum.Enum):
    value1 = "value1"
    value2 = "value2"


def test_api_call_enum_conversion():
    db = mlrun.db.httpdb.HTTPRunDB("https://fake-url")
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
    "feature_config,exception_type,exception_args,call_amount",
    [
        # feature enabled
        ("enabled", Exception, ("some-error",), 1),
        ("enabled", ConnectionError, ("some-error",), 1),
        (
            "enabled",
            ConnectionError,
            ("Connection aborted",),
            # one try + the max retries
            1 + mlrun.config.config.http_retry_defaults.max_retries,
        ),
        (
            "enabled",
            ConnectionResetError,
            ("Connection reset by peer",),
            # one try + the max retries
            1 + mlrun.config.config.http_retry_defaults.max_retries,
        ),
        (
            "enabled",
            ConnectionRefusedError,
            ("Connection refused",),
            # one try + the max retries
            1 + mlrun.config.config.http_retry_defaults.max_retries,
        ),
        (
            "enabled",
            ConnectionAbortedError,
            ("Connection aborted",),
            # one try + the max retries
            1 + mlrun.config.config.http_retry_defaults.max_retries,
        ),
        (
            "enabled",
            urllib3.exceptions.ReadTimeoutError,
            (urllib3.HTTPConnectionPool(host="dummy"), "dummy", ""),
            1 + mlrun.config.config.http_retry_defaults.max_retries,
        ),
        (
            "enabled",
            requests.exceptions.ConnectionError,
            ("Connection aborted.",),
            1 + mlrun.config.config.http_retry_defaults.max_retries,
        ),
        # feature disabled
        ("disabled", Exception, ("some-error",), 1),
        ("disabled", ConnectionError, ("some-error",), 1),
        ("disabled", ConnectionResetError, ("some-error",), 1),
        ("disabled", ConnectionError, ("Connection aborted",), 1),
        (
            "disabled",
            ConnectionResetError,
            ("Connection reset by peer",),
            1,
        ),
        (
            "disabled",
            ConnectionRefusedError,
            ("Connection refused",),
            1,
        ),
        (
            "disabled",
            ConnectionAbortedError,
            ("Connection aborted",),
            # one try + the max retries
            1,
        ),
    ],
)
def test_connection_reset_causes_retries(
    feature_config, exception_type, exception_args, call_amount
):
    mlrun.config.config.httpdb.retry_api_call_on_exception = feature_config
    db = mlrun.db.httpdb.HTTPRunDB("https://fake-url")
    original_request = requests.Session.request
    requests.Session.request = unittest.mock.Mock()
    requests.Session.request.side_effect = exception_type(*exception_args)

    # patch sleep to make test faster
    with unittest.mock.patch("time.sleep"):

        # Catching also MLRunRuntimeError as if the exception inherits from requests.RequestException, it will be
        # wrapped with MLRunRuntimeError
        with pytest.raises((exception_type, mlrun.errors.MLRunRuntimeError)):
            db.api_call("GET", "some-path")

    assert requests.Session.request.call_count == call_amount
    requests.Session.request = original_request


@pytest.mark.parametrize(
    "client_value,server_value,expected",
    [
        (None, None, None),
        (True, None, True),
        (False, None, False),
        (None, True, True),
        (None, False, False),
        (True, True, True),
        (True, False, True),
        (False, True, False),
        (False, False, False),
    ],
)
def test_client_spec_generate_target_path_from_artifact_hash_enrichment(
    client_value,
    server_value,
    expected,
):
    mlrun.mlconf.artifacts.generate_target_path_from_artifact_hash = client_value
    db = mlrun.db.httpdb.HTTPRunDB("https://fake-url")

    db.api_call = unittest.mock.Mock()
    db.api_call.return_value = unittest.mock.Mock(
        status_code=201,
        json=lambda: {
            "version": "v1.1.0",
            "generate_artifact_target_path_from_artifact_hash": server_value,
        },
    )

    db.connect()
    assert expected == mlrun.mlconf.artifacts.generate_target_path_from_artifact_hash


def test_resolve_artifacts_to_tag_objects():
    db = mlrun.db.httpdb.HTTPRunDB("https://fake-url")
    artifact = mlrun.artifacts.base.Artifact("some-key", "some-value")
    artifact.metadata.iter = 1
    artifact.metadata.tree = "some-uid"

    tag_objects = db._resolve_artifacts_to_tag_objects([artifact])
    assert len(tag_objects.identifiers) == 1
    assert tag_objects.identifiers[0].key == "some-key"
    assert tag_objects.identifiers[0].iter == 1
    assert tag_objects.identifiers[0].kind == "artifact"
    assert tag_objects.identifiers[0].uid == "some-uid"


@pytest.mark.parametrize(
    "path, call_amount",
    [
        (
            "projects/default/artifacts/uid/tag",
            1 + mlrun.config.config.http_retry_defaults.max_retries,
        ),
        (
            "projects/default/artifacts/8bbaaa9f-919e-4438-8e6c-edbf6d37f3bf/v1",
            1 + mlrun.config.config.http_retry_defaults.max_retries,
        ),
        (
            "/projects/default/artifacts/uid/tag",
            1 + mlrun.config.config.http_retry_defaults.max_retries,
        ),
        ("run/default/uid", 1 + mlrun.config.config.http_retry_defaults.max_retries),
        (
            "run/default/8bbaaa9f-919e-4438-8e6c-edbf6d37f3bf",
            1 + mlrun.config.config.http_retry_defaults.max_retries,
        ),
        ("/run/default/uid", 1 + mlrun.config.config.http_retry_defaults.max_retries),
        ("/not/retriable", 1),
    ],
)
def test_retriable_post_requests(path, call_amount):
    mlrun.config.config.httpdb.retry_api_call_on_exception = "enabled"
    db = mlrun.db.httpdb.HTTPRunDB("fake-url")
    # init the session to make sure it will be reinitialized when needed
    db.session = db._init_session(False)
    original_request = requests.Session.request
    requests.Session.request = unittest.mock.Mock()
    requests.Session.request.side_effect = ConnectionRefusedError(
        "Connection refused",
    )

    # patch sleep to make test faster
    with unittest.mock.patch("time.sleep"):

        # Catching also MLRunRuntimeError as if the exception inherits from requests.RequestException, it will be
        # wrapped with MLRunRuntimeError
        with pytest.raises(ConnectionRefusedError):
            db.api_call("POST", path)

    assert requests.Session.request.call_count == call_amount
    requests.Session.request = original_request
