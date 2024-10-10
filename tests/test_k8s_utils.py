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
from contextlib import nullcontext as does_not_raise
from unittest import mock
from unittest.mock import create_autospec

import kubernetes.client.rest as k8s_client_rest
import kubernetes.dynamic.exceptions as k8s_dynamic_exceptions
import pytest
from kubernetes.client import CoreV1Api, CustomObjectsApi, V1Secret

import mlrun.errors
import mlrun.k8s_utils
from mlrun.common.schemas import SecretEventActions
from server.api.utils.singletons.k8s import K8sHelper


@pytest.mark.parametrize(
    "value, expected",
    [
        ("my-value", "my-value"),
        ("foo%bar", "foo-bar"),
        (
            "very{long}[string](value)#with#$several$|illegal|;characters;'present'",
            "very-long--string--value--with--several--illegal--characters--p",
        ),
        ("0.0.0+unstable", "0.0.0-unstable"),
    ],
)
def test_sanitize_label_value(value: str, expected: str):
    assert mlrun.k8s_utils.sanitize_label_value(value) == expected


@pytest.mark.parametrize(
    "label_key, exception",
    [
        # valid
        ("a/" + "b" * 63, does_not_raise()),
        ("a" * 253 + "/b", does_not_raise()),
        ("a" * 253 + "/" + "b" * 63, does_not_raise()),
        ("my-key", does_not_raise()),
        ("a/b", does_not_raise()),
        ("prefix/valid-key", does_not_raise()),
        ("prefix.with.dots/valid-key", does_not_raise()),
        ("prefix-with-dashes/valid-key", does_not_raise()),
        # preserved
        ("k8s.io/a", pytest.raises(mlrun.errors.MLRunInvalidArgumentError)),
        ("kubernetes.io/a", pytest.raises(mlrun.errors.MLRunInvalidArgumentError)),
        # prefix too long
        (
            "toolong" + "a" * 248 + "/key",
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
        ),
        # name too long
        (
            "a/" + "b" * 64,
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
        ),
        # prefix has invalid character - '_'
        (
            "prefix_with_underscores/valid-key",
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
        ),
        # invalid prefix
        (
            "invalid-prefix-.com/key",
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
        ),
        # trailing slash in key
        ("invalid-prefix/key/", pytest.raises(mlrun.errors.MLRunInvalidArgumentError)),
        # leading slash in key
        ("/invalid-key", pytest.raises(mlrun.errors.MLRunInvalidArgumentError)),
        # empty key
        ("", pytest.raises(mlrun.errors.MLRunInvalidArgumentError)),
        # trailing dash
        ("invalid-key-", pytest.raises(mlrun.errors.MLRunInvalidArgumentError)),
        # trailing underscore
        ("invalid-key_", pytest.raises(mlrun.errors.MLRunInvalidArgumentError)),
        # trailing dot
        ("invalid-key.", pytest.raises(mlrun.errors.MLRunInvalidArgumentError)),
        ("invalid-key/", pytest.raises(mlrun.errors.MLRunInvalidArgumentError)),
    ],
)
def test_verify_label_key(label_key, exception):
    with exception:
        mlrun.k8s_utils.verify_label_key(label_key, allow_k8s_prefix=False)


@pytest.mark.parametrize(
    "node_selectors, expected",
    [
        # Valid cases
        # node selectors that should pass validation
        ({"kubernetes.io/arch": "amd64", "tier": "backend"}, does_not_raise()),
        ({"datacenter/region": "us-west", "role": "worker"}, does_not_raise()),
        ({"team/department": "engineering", "project": "ml-models"}, does_not_raise()),
        ({"test": "", "kubernetes.io/os": "linux"}, does_not_raise()),
        # Invalid cases
        # Invalid key with extra slashes
        (
            {"invalid/key/format/with/slash": "value"},
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
        ),
        # Invalid character '=' in key
        (
            {"key-with-invalid-characters=": "value"},
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
        ),
        # Key with trailing slash
        (
            {"key-with-dash/": "value"},
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
        ),
        # Invalid character '=' in value
        (
            {"key": "value_with_invalid_chars=a"},
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
        ),
        # Value too long
        (
            {
                "key": "value_with_very_long_string_that_exceeds_the_maximum_length_limit_of_63_characters"
            },
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
        ),
        # Invalid character '#' in value
        (
            {"key": "value_with_invalid_character#"},
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
        ),
        # Value starts with a character
        ({"key": ".valid-val"}, pytest.raises(ValueError)),
        # Prefix is too long
        ({"a" * 254 + "/key": "value"}, pytest.raises(ValueError)),
    ],
)
def test_validate_node_selectors(node_selectors, expected):
    with expected:
        mlrun.k8s_utils.validate_node_selectors(node_selectors)


@pytest.fixture
def k8s_helper():
    with mock.patch(
        "server.api.utils.singletons.k8s.K8sHelper._init_k8s_config",
        return_value=None,
    ):
        k8s_helper = K8sHelper()
        k8s_helper.v1api = create_autospec(
            CoreV1Api,
            instance=True,
            spec_set=True,
        )
        k8s_helper.crdapi = create_autospec(
            CustomObjectsApi,
            instance=True,
            spec_set=True,
        )
        k8s_helper._create_secret = mock.MagicMock()
        k8s_helper._update_secret = mock.MagicMock()
        k8s_helper._read_secret = mock.MagicMock()
        return k8s_helper


def test_create_new_secret(k8s_helper):
    k8s_helper._read_secret.side_effect = k8s_dynamic_exceptions.NotFoundError(
        k8s_client_rest.ApiException(status=404)
    )
    result = k8s_helper.store_secrets(
        secret_name="my-secret",
        secrets={"key1": "value1"},
        namespace="default",
    )

    k8s_helper._create_secret.assert_called_once()
    assert result == SecretEventActions.created


def test_conflict_during_create_secret(k8s_helper):
    k8s_helper._read_secret.side_effect = k8s_dynamic_exceptions.NotFoundError(
        k8s_client_rest.ApiException(status=404)
    )
    k8s_helper._create_secret.side_effect = k8s_dynamic_exceptions.api_exception(
        k8s_client_rest.ApiException(status=409)
    )

    with pytest.raises(mlrun.errors.MLRunConflictError):
        k8s_helper.store_secrets(
            secret_name="my-secret",
            secrets={"key1": "value1"},
            namespace="default",
        )

    k8s_helper._create_secret.assert_called_once()


def test_update_existing_secret(k8s_helper):
    k8s_helper._read_secret.return_value = V1Secret()
    k8s_helper._create_secret.side_effect = k8s_dynamic_exceptions.api_exception(
        k8s_client_rest.ApiException(status=409)
    )

    result = k8s_helper.store_secrets(
        secret_name="my-secret",
        secrets={"key1": "value1"},
        namespace="default",
    )

    k8s_helper._update_secret.assert_called_once()
    assert result == SecretEventActions.updated


def test_update_failure(k8s_helper):
    k8s_helper._read_secret.return_value = V1Secret()
    k8s_helper._update_secret.side_effect = k8s_dynamic_exceptions.api_exception(
        k8s_client_rest.ApiException(status=500)
    )

    with pytest.raises(mlrun.errors.MLRunInternalServerError):
        k8s_helper.store_secrets(
            secret_name="my-secret",
            secrets={"key1": "value1"},
            namespace="default",
        )

    k8s_helper._update_secret.assert_called_once()


def test_read_secret_failure(k8s_helper):
    k8s_helper._read_secret.side_effect = k8s_dynamic_exceptions.api_exception(
        k8s_client_rest.ApiException(status=403)
    )

    with pytest.raises(mlrun.errors.MLRunAccessDeniedError):
        k8s_helper.store_secrets(
            secret_name="my-secret",
            secrets={"key1": "value1"},
            namespace="default",
        )

    k8s_helper._read_secret.assert_called_once()
