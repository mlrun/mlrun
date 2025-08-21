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
import base64
import datetime
import unittest.mock
from contextlib import nullcontext as does_not_raise
from unittest import mock
from unittest.mock import create_autospec

import kubernetes.client as k8s_client
import kubernetes.client.rest as k8s_client_rest
import kubernetes.dynamic.exceptions as k8s_dynamic_exceptions
import pytest
import yaml

import mlrun.common.constants as mlrun_constants
import mlrun.common.runtimes
import mlrun.common.schemas
import mlrun.runtimes
from mlrun.common.schemas import SecretEventActions

import framework.utils.runtimes.mpijob
import framework.utils.singletons.k8s


@pytest.fixture
def k8s_helper():
    with mock.patch(
        "framework.utils.singletons.k8s.K8sHelper._init_k8s_config",
        return_value=None,
    ):
        k8s_helper = framework.utils.singletons.k8s.K8sHelper(
            namespace="test-namespace",
            silent=True,
        )
        k8s_helper.v1api = create_autospec(
            k8s_client.CoreV1Api,
            instance=True,
            spec_set=True,
        )
        k8s_helper.crdapi = create_autospec(
            k8s_client.CustomObjectsApi,
            instance=True,
            spec_set=True,
        )
        k8s_helper._create_secret = mock.MagicMock()
        k8s_helper._update_secret = mock.MagicMock()
        return k8s_helper


def test_create_new_secret(k8s_helper):
    k8s_helper.read_secret = mock.MagicMock()
    k8s_helper.read_secret.side_effect = k8s_dynamic_exceptions.NotFoundError(
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
    k8s_helper.read_secret = mock.MagicMock()
    k8s_helper.read_secret.side_effect = k8s_dynamic_exceptions.NotFoundError(
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
    k8s_helper.read_secret = mock.MagicMock()
    k8s_helper.read_secret.return_value = k8s_client.V1Secret()
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
    k8s_helper.read_secret = mock.MagicMock()
    k8s_helper.read_secret.return_value = k8s_client.V1Secret()
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
    k8s_helper.read_secret = mock.MagicMock()
    k8s_helper.read_secret.side_effect = k8s_dynamic_exceptions.api_exception(
        k8s_client_rest.ApiException(status=403)
    )

    with pytest.raises(mlrun.errors.MLRunAccessDeniedError):
        k8s_helper.store_secrets(
            secret_name="my-secret",
            secrets={"key1": "value1"},
            namespace="default",
        )

    k8s_helper.read_secret.assert_called_once()


@pytest.mark.parametrize(
    "labels_in_secret, labels_to_match, expected",
    [
        # Matching labels
        ({"key1": "value1", "key2": "value2"}, {"key1": "value1"}, True),
        # Non-matching labels
        ({"key1": "value1", "key2": "value2"}, {"key1": "wrong"}, False),
        # No labels provided (always accept)
        ({"key1": "value1"}, None, True),
        # Secret has no labels but caller requires them
        ({}, {"key1": "value1"}, False),
    ],
)
def test_read_secret_label_validation(
    k8s_helper, labels_in_secret, labels_to_match, expected
):
    """
    Verify that read_secret correctly validates labels on top of name-based lookup.
    """
    secret_name = "my-secret"
    secret_data = {"secret-key1": "secret-value1"}

    secret_obj = k8s_client.V1Secret(
        metadata=k8s_client.V1ObjectMeta(
            name=secret_name,
            labels=labels_in_secret,
        )
    )
    secret_obj.string_data = secret_data

    # Mock the Kubernetes API return
    k8s_helper.v1api.read_namespaced_secret.return_value = secret_obj

    secret = k8s_helper.read_secret(secret_name=secret_name, labels=labels_to_match)

    assert k8s_helper.v1api.read_namespaced_secret.call_count == 1

    if expected:
        assert secret is secret_obj
    else:
        assert secret is None


@pytest.mark.parametrize(
    "run_type,mpi_version,extra_selector",
    [
        ("job", "", ""),
        ("spark", "", "spark-role=driver"),
        (
            "mpijob",
            "v1",
            f"{mlrun_constants.MLRunInternalLabels.mpi_job_role}=launcher",
        ),
        (
            "mpijob",
            "v1alpha1",
            f"{mlrun_constants.MLRunInternalLabels.mpi_role_type}=launcher",
        ),
    ],
)
def test_get_logger_pods_label_selector(
    k8s_helper, monkeypatch, run_type, mpi_version, extra_selector
):
    monkeypatch.setattr(
        framework.utils.runtimes.mpijob,
        "cached_mpijob_crd_version",
        mpi_version or mlrun.common.runtimes.constants.MPIJobCRDVersions.default(),
    )
    uid = "test-uid"
    project = "test-project"
    selector = (
        f"{mlrun_constants.MLRunInternalLabels.mlrun_class},"
        f"{mlrun_constants.MLRunInternalLabels.project}={project},"
        f"{mlrun_constants.MLRunInternalLabels.uid}={uid}"
    )
    if extra_selector:
        selector += f",{extra_selector}"

    k8s_helper.list_pods = unittest.mock.MagicMock()

    k8s_helper.get_logger_pods(project, uid, run_type)
    k8s_helper.list_pods.assert_called_once_with(
        k8s_helper.namespace, selector=selector
    )


@pytest.mark.parametrize(
    "existing_secret_data,secrets_to_store,expected_data,expected_result",
    [
        # we want to ensure that if the data is None, the function doesn't raise an exception
        (None, {}, {}, None),
        (None, None, {}, None),
        # regular case
        (
            {"a": "b"},
            {"a": "c"},
            {"a": "Yw=="},
            mlrun.common.schemas.SecretEventActions.updated,
        ),
        (
            None,
            {"a": "b"},
            {"a": "Yg=="},
            mlrun.common.schemas.SecretEventActions.created,
        ),
    ],
)
def test_store_secret(
    k8s_helper,
    existing_secret_data: dict,
    secrets_to_store: dict,
    expected_data: dict,
    expected_result: SecretEventActions,
):
    k8s_helper.read_secret = mock.MagicMock()
    if existing_secret_data:
        k8s_helper.read_secret.return_value = k8s_client.V1Secret(
            data=existing_secret_data,
        )
    else:
        k8s_helper.read_secret.side_effect = k8s_dynamic_exceptions.NotFoundError(
            k8s_client_rest.ApiException(status=404)
        )
    result = k8s_helper.store_secrets(
        secret_name="my-secret",
        secrets=secrets_to_store,
    )
    assert result == expected_result
    if secrets_to_store and result == mlrun.common.schemas.SecretEventActions.created:
        data = k8s_helper._create_secret.call_args.kwargs["secrets"]
        assert data == secrets_to_store
    elif secrets_to_store and result == mlrun.common.schemas.SecretEventActions.updated:
        data = k8s_helper._update_secret.call_args.kwargs["secrets"]
        assert data == secrets_to_store


@pytest.mark.parametrize(
    "k8s_secret_data, secrets_data, expected_action, expected_secret_data",
    [
        (
            {"key1": "value1", "key2": "value2"},
            [],
            None,
            {"key1": "value1", "key2": "value2"},
        ),
        (
            {"key1": "value1", "key2": "value2"},
            None,  # delete all secrets
            mlrun.common.schemas.SecretEventActions.deleted,
            {},
        ),
        (
            {"key1": "value1", "key2": "value2"},
            ["key3"],
            None,
            {"key1": "value1", "key2": "value2"},
        ),
        (None, ["key1"], mlrun.common.schemas.SecretEventActions.deleted, {}),
        ({}, ["key1"], mlrun.common.schemas.SecretEventActions.deleted, {}),
        (
            {"key1": "value1"},
            ["key1"],
            mlrun.common.schemas.SecretEventActions.deleted,
            {},
        ),
        (
            {"key1": "value1", "key2": "value2"},
            ["key1"],
            mlrun.common.schemas.SecretEventActions.updated,
            {"key2": "value2"},
        ),
    ],
)
def test_delete_secrets(
    k8s_helper, k8s_secret_data, secrets_data, expected_action, expected_secret_data
):
    k8s_secret_mock = unittest.mock.MagicMock(data=k8s_secret_data)
    k8s_helper.v1api.read_namespaced_secret.return_value = k8s_secret_mock

    result = k8s_helper.delete_secrets("my-secret", secrets_data)
    assert result == expected_action

    k8s_helper.v1api.read_namespaced_secret.assert_called_once_with(
        "my-secret", k8s_helper.namespace
    )

    if expected_action == mlrun.common.schemas.SecretEventActions.updated:
        data = k8s_helper.v1api.replace_namespaced_secret.call_args.args[2].data
        assert data == expected_secret_data


@pytest.mark.parametrize(
    "side_effect, expectation, expected_result",
    [
        (
            [
                k8s_client.ApiException(status=410),
                k8s_client.ApiException(status=410),
                k8s_client.V1PodList(
                    items=[],
                    metadata=k8s_client.V1ListMeta(),
                ),
            ],
            does_not_raise(),
            [],
        ),
        (
            [
                k8s_client.ApiException(status=410),
                k8s_client.ApiException(status=410),
                k8s_client.ApiException(status=410),
                k8s_client.ApiException(status=410),
            ],
            pytest.raises(mlrun.errors.MLRunHTTPError),
            None,
        ),
        (
            [
                k8s_client.ApiException(status=400),
                k8s_client.V1PodList(
                    items=[],
                    metadata=k8s_client.V1ListMeta(),
                ),
            ],
            pytest.raises(mlrun.errors.MLRunBadRequestError),
            None,
        ),
    ],
)
def test_list_paginated_pods_retry(
    k8s_helper, side_effect, expectation, expected_result
):
    k8s_helper.v1api.list_namespaced_pod.side_effect = side_effect
    with expectation:
        result = list(k8s_helper.list_pods_paginated("my-ns"))
        if expected_result is not None:
            assert result == expected_result


@pytest.mark.parametrize(
    "side_effect, expectation, expected_result",
    [
        (
            [
                k8s_client.ApiException(status=410),
                k8s_client.ApiException(status=410),
                {"items": [], "metadata": {"continue": None}},
            ],
            does_not_raise(),
            [],
        ),
        (
            [
                k8s_client.ApiException(status=410),
                k8s_client.ApiException(status=410),
                k8s_client.ApiException(status=410),
                k8s_client.ApiException(status=410),
            ],
            pytest.raises(mlrun.errors.MLRunHTTPError),
            None,
        ),
        (
            [
                k8s_client.ApiException(status=400),
                {},
            ],
            pytest.raises(mlrun.errors.MLRunBadRequestError),
            None,
        ),
        # Ignoring not found - should not raise
        (
            [
                k8s_client.ApiException(status=404),
            ],
            does_not_raise(),
            [],
        ),
    ],
)
def test_list_paginated_crds_retry(
    k8s_helper, side_effect, expectation, expected_result
):
    k8s_helper.crdapi.list_namespaced_custom_object.side_effect = side_effect
    with expectation:
        result = list(k8s_helper.list_crds_paginated("group", "v1", "objects", "my-ns"))
        if expected_result is not None:
            assert result == expected_result


def test_list_pod_events(k8s_helper):
    event = k8s_client.CoreV1Event(
        metadata=k8s_client.V1ObjectMeta(name="pod-event"),
        type="event-type",
        reason="event-reason",
        message="event-message",
        involved_object="my-pod",
        first_timestamp=datetime.datetime.now(),
    )
    with unittest.mock.patch.object(
        k8s_helper.v1api,
        "list_namespaced_event",
        return_value=k8s_client.CoreV1EventList(items=[event]),
    ):
        events = k8s_helper.list_object_events(object_name="my-pod")
        assert events[0].metadata.name == event.metadata.name
        assert events[0].type == event.type
        assert events[0].reason == event.reason
        assert events[0].message == event.message
        assert events[0].first_timestamp == event.first_timestamp


def test_store_user_token_secret_created(k8s_helper):
    k8s_helper.read_secret = mock.MagicMock(return_value=None)

    username = "test-user"
    token_name = "my-token"
    token_value = "abc123"
    expiration = 9999

    result = k8s_helper.store_user_token_secret(
        username=username,
        token_name=token_name,
        token=token_value,
        expiration=expiration,
        namespace="default",
    )

    # Check that the secret creation was triggered
    assert result == mlrun.common.schemas.SecretEventActions.created
    k8s_helper._create_secret.assert_called_once()
    k8s_helper._update_secret.assert_not_called()

    # Verify that the secrets data passed to _create_secret is properly encoded
    secrets_data = k8s_helper._create_secret.call_args.kwargs["secrets"]
    assert "tokensFile" in secrets_data
    assert "tokenExpiration" in secrets_data

    # Decode and verify tokensFile
    decoded_tokens_yaml = base64.b64decode(secrets_data["tokensFile"]).decode()
    tokens_yaml_dict = yaml.safe_load(decoded_tokens_yaml)
    assert tokens_yaml_dict == {
        "secretTokens": [{"name": token_name, "token": token_value}]
    }

    # Decode and verify tokenExpiration
    decoded_expiration = int(base64.b64decode(secrets_data["tokenExpiration"]).decode())
    assert decoded_expiration == expiration


def test_store_user_token_secret_updated(k8s_helper):
    username = "test-user"
    token_name = "my-token"
    token_value = "abc123"
    new_expiration = 2000
    secret_name = k8s_helper._resolve_user_token_secret_name(username, token_name)

    # Existing secret with older expiration
    existing_secret = _make_user_token_secret(
        secret_name, token_name=token_name, token_value=token_value, expiration=1000
    )
    k8s_helper.read_secret = mock.MagicMock(return_value=existing_secret)

    result = k8s_helper.store_user_token_secret(
        username=username,
        token_name=token_name,
        token=token_value,
        expiration=new_expiration,
        namespace="default",
    )

    # Check that the secret update was triggered
    assert result == mlrun.common.schemas.SecretEventActions.updated
    k8s_helper._update_secret.assert_called_once()
    k8s_helper._create_secret.assert_not_called()

    # Verify that the updated secret data is properly encoded
    secrets_data = k8s_helper._update_secret.call_args.kwargs["secrets"]
    assert "tokensFile" in secrets_data
    assert "tokenExpiration" in secrets_data

    # Decode and verify tokensFile
    decoded_tokens_yaml = base64.b64decode(secrets_data["tokensFile"]).decode()
    tokens_yaml_dict = yaml.safe_load(decoded_tokens_yaml)
    assert tokens_yaml_dict == {
        "secretTokens": [{"name": token_name, "token": token_value}]
    }

    # Decode and verify tokenExpiration
    decoded_expiration = int(base64.b64decode(secrets_data["tokenExpiration"]).decode())
    assert decoded_expiration == new_expiration


def test_store_user_token_secret_skipped(k8s_helper):
    username = "test-user"
    token_name = "my-token"
    token_value = "abc123"
    secret_name = k8s_helper._resolve_user_token_secret_name(username, token_name)

    # Existing secret with newer expiration than what we pass -> should skip update
    existing_secret = _make_user_token_secret(
        secret_name,
        token_name=token_name,
        token_value=token_value,
        expiration=5000,  # current expiration is newer
    )

    k8s_helper.read_secret = mock.MagicMock(return_value=existing_secret)

    result = k8s_helper.store_user_token_secret(
        username=username,
        token_name=token_name,
        token=token_value,
        expiration=4000,  # older expiration -> should skip
        namespace="default",
    )

    # Ensure the action is skipped
    assert result == mlrun.common.schemas.SecretEventActions.skipped
    k8s_helper._update_secret.assert_not_called()
    k8s_helper._create_secret.assert_not_called()


def test_get_user_token_secret_value_valid(k8s_helper):
    username = "test-user"
    token_name = "my-token"
    token_value = "abc123"
    secret_name = k8s_helper._resolve_user_token_secret_name(username, token_name)

    # Create a Kubernetes secret with properly encoded tokensFile
    existing_secret = _make_user_token_secret(
        secret_name,
        token_name=token_name,
        token_value=token_value,
        expiration=9999,
    )

    k8s_helper.resolve_namespace = mock.MagicMock(return_value="default")
    k8s_helper.read_secret = mock.MagicMock(return_value=existing_secret)

    token_value_from_k8s = k8s_helper.get_user_token_secret_value(
        username=username,
        token_name=token_name,
        namespace="default",
    )

    assert token_value_from_k8s == token_value
    k8s_helper.read_secret.assert_called_once()


def test_get_user_token_secret_value_not_found(k8s_helper):
    username = "test-user"
    token_name = "my-token"

    k8s_helper.resolve_namespace = mock.MagicMock(return_value="default")
    k8s_helper.read_secret = mock.MagicMock(return_value=None)

    with pytest.raises(mlrun.errors.MLRunNotFoundError):
        k8s_helper.get_user_token_secret_value(
            username, token_name, namespace="default"
        )


def test_get_user_token_secret_value_token_missing(k8s_helper):
    username = "test-user"
    token_name = "my-token"
    secret_name = k8s_helper._resolve_user_token_secret_name(username, token_name)

    # Secret exists but tokensFile does not contain the requested token
    secret = _make_user_token_secret(secret_name, token_name="other-token")
    k8s_helper.resolve_namespace = mock.MagicMock(return_value="default")
    k8s_helper.read_secret = mock.MagicMock(return_value=secret)

    with pytest.raises(mlrun.errors.MLRunNotFoundError):
        k8s_helper.get_user_token_secret_value(
            username, token_name, namespace="default"
        )


def test_get_user_token_secret_value_invalid_base64(k8s_helper):
    username = "test-user"
    token_name = "my-token"
    secret_name = k8s_helper._resolve_user_token_secret_name(username, token_name)

    # Create a secret with an invalid base64 tokensFile
    bad_secret = _make_k8s_secret(secret_name)
    bad_secret.data["tokensFile"] = "!!!invalidbase64!!!"  # invalid base64 content
    bad_secret.data["tokenExpiration"] = base64.b64encode(b"9999").decode()

    k8s_helper.resolve_namespace = mock.MagicMock(return_value="default")
    k8s_helper.read_secret = mock.MagicMock(return_value=bad_secret)

    with pytest.raises(mlrun.errors.MLRunRuntimeError):
        k8s_helper.get_user_token_secret_value(
            username=username,
            token_name=token_name,
            namespace="default",
        )


def test_get_user_token_secret_value_invalid_yaml(k8s_helper):
    username = "test-user"
    token_name = "my-token"
    secret_name = k8s_helper._resolve_user_token_secret_name(username, token_name)

    # Base64 encoded string but invalid YAML
    bad_yaml = base64.b64encode(b"{invalid_yaml: ]").decode()
    bad_secret = _make_k8s_secret(secret_name)
    bad_secret.data["tokensFile"] = bad_yaml
    k8s_helper.resolve_namespace = mock.MagicMock(return_value="default")
    k8s_helper.read_secret = mock.MagicMock(return_value=bad_secret)

    with pytest.raises(mlrun.errors.MLRunRuntimeError):
        k8s_helper.get_user_token_secret_value(
            username, token_name, namespace="default"
        )


def test_delete_user_token_secret_success(k8s_helper):
    username = "test-user"
    token_name = "token1"
    secret_name = k8s_helper._resolve_user_token_secret_name(username, token_name)

    k8s_helper.resolve_namespace = mock.MagicMock(return_value="default")
    k8s_helper.v1api.delete_namespaced_secret = mock.MagicMock()

    k8s_helper.delete_user_token_secret(
        username=username, token_name=token_name, namespace="default"
    )

    k8s_helper.v1api.delete_namespaced_secret.assert_called_once_with(
        name=secret_name,
        namespace="default",
    )


def test_delete_user_token_secret_not_found(k8s_helper):
    username = "test-user"
    token_name = "missing"
    secret_name = k8s_helper._resolve_user_token_secret_name(username, token_name)

    k8s_helper.resolve_namespace = mock.MagicMock(return_value="default")
    k8s_helper.v1api.delete_namespaced_secret = mock.MagicMock(
        side_effect=k8s_client_rest.ApiException(status=404, reason="Not Found")
    )

    with pytest.raises(mlrun.errors.MLRunNotFoundError) as exc:
        k8s_helper.delete_user_token_secret(
            username=username, token_name=token_name, namespace="default"
        )

    assert f"Secret for token '{token_name}' not found" in str(exc.value)

    k8s_helper.v1api.delete_namespaced_secret.assert_called_once_with(
        name=secret_name,
        namespace="default",
    )


def test_delete_user_token_secret_api_error(k8s_helper):
    username = "test-user"
    token_name = "badtoken"
    secret_name = k8s_helper._resolve_user_token_secret_name(username, token_name)

    k8s_helper.resolve_namespace = mock.MagicMock(return_value="default")
    k8s_helper.v1api.delete_namespaced_secret = mock.MagicMock(
        side_effect=k8s_client_rest.ApiException(status=500, reason="Internal Error")
    )

    with pytest.raises(mlrun.errors.MLRunRuntimeError) as exc:
        k8s_helper.delete_user_token_secret(
            username=username, token_name=token_name, namespace="default"
        )

    assert "Failed to delete secret" in str(exc.value)

    k8s_helper.v1api.delete_namespaced_secret.assert_called_once_with(
        name=secret_name,
        namespace="default",
    )


def test_delete_user_token_secret_unexpected_error(k8s_helper):
    username = "test-user"
    token_name = "oops"
    secret_name = k8s_helper._resolve_user_token_secret_name(username, token_name)

    k8s_helper.resolve_namespace = mock.MagicMock(return_value="default")
    k8s_helper.v1api.delete_namespaced_secret = mock.MagicMock(
        side_effect=RuntimeError("dummy-error")
    )

    with pytest.raises(mlrun.errors.MLRunRuntimeError) as exc:
        k8s_helper.delete_user_token_secret(
            username=username, token_name=token_name, namespace="default"
        )

    assert "Unexpected error deleting secret" in str(exc.value)

    k8s_helper.v1api.delete_namespaced_secret.assert_called_once_with(
        name=secret_name,
        namespace="default",
    )


def _make_user_token_secret(
    secret_name,
    token_name="my-token",
    token_value="abc123",
    expiration=None,
    labels=None,
):
    labels = labels or {
        mlrun_constants.MLRunInternalLabels.user_token_secret_label_key: "test-user"
    }
    secret = _make_k8s_secret(secret_name, labels)

    # Add tokensFile
    token_yaml = yaml.safe_dump(
        {"secretTokens": [{"name": token_name, "token": token_value}]}
    )
    secret.data["tokensFile"] = base64.b64encode(token_yaml.encode()).decode()

    # Encode tokenExpiration if provided
    if expiration is not None:
        secret.data["tokenExpiration"] = base64.b64encode(
            str(expiration).encode()
        ).decode()

    return secret


def _make_k8s_secret(name, labels=None):
    metadata = k8s_client.V1ObjectMeta(name=name, labels=labels or {})
    return k8s_client.V1Secret(metadata=metadata, data={})
