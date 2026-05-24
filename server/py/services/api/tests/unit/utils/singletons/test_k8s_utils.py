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

import mlrun
import mlrun.common.constants as mlrun_constants
import mlrun.common.runtimes
import mlrun.common.schemas
import mlrun.errors
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


def test_store_secrets_no_labels(k8s_helper):
    """
    Test ensures that labels param is not passed to read_secret when storing secrets.
    The labels param during read_secret is intended for IG4 secrets only.
    """
    k8s_helper.read_secret = mock.MagicMock(
        side_effect=k8s_dynamic_exceptions.NotFoundError(
            k8s_client_rest.ApiException(status=404)
        )
    )
    k8s_helper.store_secrets(
        secret_name="my-secret",
        secrets={"key1": "value1"},
        namespace="default",
    )

    try:
        k8s_helper.read_secret.assert_called_once_with(
            secret_name="my-secret", namespace="default"
        )
    except AssertionError:
        raise AssertionError(
            "Store secrets should not pass 'labels' to read_secret. Please review params that were "
            "added to the read_secret call."
        )


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
        "my-secret",
        k8s_helper.namespace,
        _request_timeout=framework.utils.singletons.k8s.K8sHelper._resolve_k8s_timeout(
            framework.utils.singletons.k8s.K8S_TIMEOUT_DEFAULT
        ),
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
    k8s_helper.list_secrets = mock.MagicMock(return_value=[])

    user_id = "test-user-id"
    username = "test-username"
    auth_info = mlrun.common.schemas.AuthInfo(user_id=user_id, username=username)
    token_name = "my-token"
    token_value = "abc123"
    issued_at = 1
    expiration = 9999

    result = k8s_helper.store_user_token_secret(
        auth_info=auth_info,
        token_name=token_name,
        token=token_value,
        issued_at=issued_at,
        expiration=expiration,
        namespace="default",
    )

    # Check that the secret creation was triggered
    assert result == mlrun.common.schemas.SecretEventActions.created
    k8s_helper._create_secret.assert_called_once()
    k8s_helper._update_secret.assert_not_called()

    # Verify labels contain raw user_id, hashed username and hashed token_name
    labels = k8s_helper._create_secret.call_args.kwargs["labels"]
    assert labels[mlrun_constants.MLRunInternalLabels.auth_userid] == user_id
    assert labels[
        mlrun_constants.MLRunInternalLabels.auth_username
    ] == k8s_helper._hash_label(username)
    assert labels[
        mlrun_constants.MLRunInternalLabels.auth_token_name
    ] == k8s_helper._hash_label(token_name)

    # Verify annotations contain raw username and token_name
    annotations = k8s_helper._create_secret.call_args.kwargs["annotations"]
    assert annotations[mlrun_constants.InternalAnnotations.auth_username] == username
    assert (
        annotations[mlrun_constants.InternalAnnotations.auth_token_name] == token_name
    )

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

    # Decode and verify tokenIssuedAt
    decoded_issued_at = int(base64.b64decode(secrets_data["tokenIssuedAt"]).decode())
    assert decoded_issued_at == issued_at


@pytest.mark.parametrize(
    "user_id",
    [
        "test-user-id",
        "user123",
        "my-token-user",
    ],
)
def test_store_user_token_secret_stores_user_id_in_label(k8s_helper, user_id):
    """Test that user_id is stored in label when creating token secret."""
    k8s_helper.list_secrets = mock.MagicMock(return_value=[])

    username = "test-username"
    auth_info = mlrun.common.schemas.AuthInfo(user_id=user_id, username=username)
    token_name = "my-token"
    token_value = "abc123"
    expiration = 9999

    result = k8s_helper.store_user_token_secret(
        auth_info=auth_info,
        token_name=token_name,
        token=token_value,
        issued_at=1,
        expiration=expiration,
        namespace="default",
    )

    # Verify creation succeeded
    assert result == mlrun.common.schemas.SecretEventActions.created
    k8s_helper._create_secret.assert_called_once()

    # Verify labels contain raw user_id and hashed token_name
    labels = k8s_helper._create_secret.call_args.kwargs["labels"]
    assert labels[mlrun_constants.MLRunInternalLabels.auth_userid] == user_id
    assert labels[
        mlrun_constants.MLRunInternalLabels.auth_token_name
    ] == k8s_helper._hash_label(token_name)


@pytest.mark.parametrize(
    "username",
    [
        "test-user",
        "user@example.com",
        "user name",
        "user!@#$%^&*()",
        "a" * 100,
    ],
)
def test_store_user_token_secret_username_annotation(k8s_helper, username):
    """Test that the raw username is stored in the annotation and the hashed
    username is stored in the label."""
    import uuid

    k8s_helper.list_secrets = mock.MagicMock(return_value=[])

    user_id = str(uuid.uuid4())
    auth_info = mlrun.common.schemas.AuthInfo(user_id=user_id, username=username)
    token_name = "my-token"
    token_value = "abc123"
    issued_at = 1
    expiration = 9999

    result = k8s_helper.store_user_token_secret(
        auth_info=auth_info,
        token_name=token_name,
        token=token_value,
        issued_at=issued_at,
        expiration=expiration,
        namespace="default",
    )

    # Verify creation succeeded
    assert result == mlrun.common.schemas.SecretEventActions.created
    k8s_helper._create_secret.assert_called_once()

    # Annotation stores raw username (no sanitization)
    annotations = k8s_helper._create_secret.call_args.kwargs["annotations"]
    assert annotations[mlrun_constants.InternalAnnotations.auth_username] == username

    # Label stores hashed username (safe for k8s label constraints)
    labels = k8s_helper._create_secret.call_args.kwargs["labels"]
    assert labels[
        mlrun_constants.MLRunInternalLabels.auth_username
    ] == k8s_helper._hash_label(username)


def test_store_user_token_secret_rejects_missing_username(k8s_helper):
    """Token secret handling requires a username (enterprise-only)."""
    k8s_helper.list_secrets = mock.MagicMock(return_value=[])

    auth_info = mlrun.common.schemas.AuthInfo(user_id="test-user-id", username=None)
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        k8s_helper.store_user_token_secret(
            auth_info=auth_info,
            token_name="my-token",
            token="abc123",
            issued_at=1,
            expiration=9999,
            namespace="default",
        )


def test_store_user_token_secret_secret_naming(k8s_helper):
    """Test that secret name is derived from user_id + token_name hash."""
    k8s_helper.list_secrets = mock.MagicMock(return_value=[])

    user_id = "test-user-id"
    auth_info = mlrun.common.schemas.AuthInfo(user_id=user_id, username="test-username")
    token_name = "my-token"
    token_value = "abc123"
    issued_at = 1
    expiration = 9999

    result = k8s_helper.store_user_token_secret(
        auth_info=auth_info,
        token_name=token_name,
        token=token_value,
        issued_at=issued_at,
        expiration=expiration,
        namespace="default",
    )

    # Verify creation succeeded
    assert result == mlrun.common.schemas.SecretEventActions.created
    k8s_helper._create_secret.assert_called_once()

    # Verify the secret name is derived from user_id + token_name hash
    secret_name = k8s_helper._create_secret.call_args.kwargs["secret_name"]
    expected_secret_name = k8s_helper._resolve_auth_secret_name(user_id, token_name)
    assert secret_name == expected_secret_name


def test_store_user_token_secret_updated(k8s_helper):
    user_id = "test-user-id"
    username = "test-username"
    auth_info = mlrun.common.schemas.AuthInfo(user_id=user_id, username=username)
    token_name = "my-token"
    token_value = "abc123"
    issued_at = 1
    new_expiration = 2000
    secret_name = k8s_helper._resolve_auth_secret_name(user_id, token_name)

    # Existing secret with older expiration
    existing_secret = _make_user_token_secret(
        secret_name,
        token_name=token_name,
        token_value=token_value,
        issued_at=issued_at,
        expiration=1000,
        user_id=user_id,
        username=username,
    )
    k8s_helper.list_secrets = mock.MagicMock(return_value=[existing_secret])

    result = k8s_helper.store_user_token_secret(
        auth_info=auth_info,
        token_name=token_name,
        token=token_value,
        issued_at=issued_at,
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
    assert "tokenIssuedAt" in secrets_data

    # Decode and verify tokensFile
    decoded_tokens_yaml = base64.b64decode(secrets_data["tokensFile"]).decode()
    tokens_yaml_dict = yaml.safe_load(decoded_tokens_yaml)
    assert tokens_yaml_dict == {
        "secretTokens": [{"name": token_name, "token": token_value}]
    }

    # Decode and verify tokenExpiration
    decoded_expiration = int(base64.b64decode(secrets_data["tokenExpiration"]).decode())
    assert decoded_expiration == new_expiration

    # Decode and verify tokenIssuedAt
    decoded_issued_at = int(base64.b64decode(secrets_data["tokenIssuedAt"]).decode())
    assert decoded_issued_at == issued_at


@pytest.mark.parametrize(
    "expiration, force, expected_result, update_called, create_called",
    [
        (4000, False, None, False, False),  # skip update, expiration older
        (
            4000,
            True,
            mlrun.common.schemas.SecretEventActions.updated,
            True,
            False,
        ),  # force update
    ],
)
def test_store_user_token_secret_skipped_and_force_update(
    k8s_helper, expiration, force, expected_result, update_called, create_called
):
    user_id = "test-user-id"
    username = "test-username"
    auth_info = mlrun.common.schemas.AuthInfo(user_id=user_id, username=username)
    token_name = "my-token"
    token_value = "abc123"
    issued_at = 1
    secret_name = k8s_helper._resolve_auth_secret_name(user_id, token_name)

    existing_secret = _make_user_token_secret(
        secret_name,
        token_name=token_name,
        token_value=token_value,
        issued_at=issued_at,
        expiration=5000,
        user_id=user_id,
        username=username,
    )
    k8s_helper.list_secrets = mock.MagicMock(return_value=[existing_secret])

    result = k8s_helper.store_user_token_secret(
        auth_info=auth_info,
        token_name=token_name,
        token=token_value,
        issued_at=issued_at,
        expiration=expiration,
        namespace="default",
        force=force,
    )

    assert result == expected_result

    if update_called:
        k8s_helper._update_secret.assert_called_once()
    else:
        k8s_helper._update_secret.assert_not_called()

    if create_called:
        k8s_helper._create_secret.assert_called_once()
    else:
        k8s_helper._create_secret.assert_not_called()


def test_list_secrets_with_labels(k8s_helper):
    secret1 = _make_k8s_secret(
        "secret1",
        labels={
            mlrun_constants.MLRunInternalLabels.auth_userid: "test-user-id",
        },
    )
    secret2 = _make_k8s_secret(
        "secret2",
        labels={
            mlrun_constants.MLRunInternalLabels.auth_userid: "test-user-id",
        },
    )

    fake_secret_list = mock.MagicMock()
    fake_secret_list.items = [secret1, secret2]

    k8s_helper.v1api.list_namespaced_secret = mock.MagicMock(
        return_value=fake_secret_list
    )

    result = k8s_helper.list_secrets(
        namespace="default",
        labels={mlrun_constants.MLRunInternalLabels.auth_userid: "test-user-id"},
    )

    assert result == [secret1, secret2]
    k8s_helper.v1api.list_namespaced_secret.assert_called_once_with(
        namespace="default",
        label_selector="mlrun/user-id=test-user-id",
        _request_timeout=framework.utils.singletons.k8s.K8sHelper._resolve_k8s_timeout(
            framework.utils.singletons.k8s.K8S_TIMEOUT_LIST
        ),
    )


def test_list_secrets_no_labels(k8s_helper):
    secret = _make_k8s_secret("secret-no-labels")

    fake_secret_list = mock.MagicMock()
    fake_secret_list.items = [secret]

    k8s_helper.resolve_namespace = mock.MagicMock(return_value="default")
    k8s_helper.v1api.list_namespaced_secret = mock.MagicMock(
        return_value=fake_secret_list
    )

    result = k8s_helper.list_secrets(namespace="default")

    assert result == [secret]
    k8s_helper.v1api.list_namespaced_secret.assert_called_once_with(
        namespace="default",
        label_selector=None,
        _request_timeout=framework.utils.singletons.k8s.K8sHelper._resolve_k8s_timeout(
            framework.utils.singletons.k8s.K8S_TIMEOUT_LIST
        ),
    )


def test_list_secrets_empty(k8s_helper):
    fake_secret_list = mock.MagicMock()
    fake_secret_list.items = []

    k8s_helper.resolve_namespace = mock.MagicMock(return_value="default")
    k8s_helper.v1api.list_namespaced_secret = mock.MagicMock(
        return_value=fake_secret_list
    )

    result = k8s_helper.list_secrets(namespace="default")
    assert result == []


def test_list_user_token_secrets_valid(k8s_helper):
    token1_name = "token1"
    token2_name = "token2"
    user_id = "test-user-id"
    username = "test-username"
    iat = 1
    exp1 = 1111
    exp2 = 2222
    secret1_name = k8s_helper._resolve_auth_secret_name(user_id, token1_name)
    secret2_name = k8s_helper._resolve_auth_secret_name(user_id, token2_name)
    secret1 = _make_user_token_secret(
        secret1_name,
        token_name=token1_name,
        issued_at=iat,
        expiration=exp1,
        user_id=user_id,
        username=username,
    )
    secret2 = _make_user_token_secret(
        secret2_name,
        token_name=token2_name,
        issued_at=iat,
        expiration=exp2,
        user_id=user_id,
        username=username,
    )

    k8s_helper.resolve_namespace = mock.MagicMock(return_value="default")
    k8s_helper.list_secrets = mock.MagicMock(return_value=[secret1, secret2])

    result = k8s_helper.list_user_token_secrets(username=username, namespace="default")

    assert len(result) == 2
    assert result[0].name == token1_name
    assert int(result[0].expiration.timestamp()) == exp1
    assert result[1].name == token2_name
    assert int(result[1].expiration.timestamp()) == exp2

    k8s_helper.list_secrets.assert_called_once_with(
        namespace="default",
        labels={
            mlrun_constants.MLRunInternalLabels.auth_token_name: None,
            mlrun_constants.MLRunInternalLabels.auth_username: k8s_helper._hash_label(
                username
            ),
        },
    )


def test_store_user_token_secret_canonicalizes_username_to_lowercase(k8s_helper):
    """Usernames are stored canonically (lowercase) so list lookups match regardless of
    the case the user originally registered with — Keycloak/Iguazio are case-insensitive."""
    k8s_helper.list_secrets = mock.MagicMock(return_value=[])

    mixed_case_username = "Mixed.Case-User"
    auth_info = mlrun.common.schemas.AuthInfo(
        user_id="user-id-1", username=mixed_case_username
    )

    k8s_helper.store_user_token_secret(
        auth_info=auth_info,
        token_name="my-token",
        token="abc123",
        issued_at=1,
        expiration=9999,
        namespace="default",
    )

    labels = k8s_helper._create_secret.call_args.kwargs["labels"]
    annotations = k8s_helper._create_secret.call_args.kwargs["annotations"]
    assert labels[
        mlrun_constants.MLRunInternalLabels.auth_username
    ] == k8s_helper._hash_label(mixed_case_username.lower())
    assert (
        annotations[mlrun_constants.InternalAnnotations.auth_username]
        == mixed_case_username.lower()
    )


def test_list_user_token_secrets_canonicalizes_username_to_lowercase(k8s_helper):
    """Listing with mixed-case username must hit the same hashed label that was written
    using the canonical (lowercase) form."""
    canonical_username = "mixed.case-user"
    secret_name = k8s_helper._resolve_auth_secret_name("user-id-1", "token1")
    secret = _make_user_token_secret(
        secret_name,
        token_name="token1",
        issued_at=1,
        expiration=9999,
        user_id="user-id-1",
        username=canonical_username,
    )
    k8s_helper.resolve_namespace = mock.MagicMock(return_value="default")
    k8s_helper.list_secrets = mock.MagicMock(return_value=[secret])

    result = k8s_helper.list_user_token_secrets(
        username="Mixed.Case-User", namespace="default"
    )

    # The hash-collision post-filter compares against the canonical stored username,
    # so the match succeeds and the token is returned.
    assert len(result) == 1
    assert result[0].name == "token1"

    k8s_helper.list_secrets.assert_called_once_with(
        namespace="default",
        labels={
            mlrun_constants.MLRunInternalLabels.auth_token_name: None,
            mlrun_constants.MLRunInternalLabels.auth_username: k8s_helper._hash_label(
                canonical_username
            ),
        },
    )


def test_list_user_token_secrets_wildcard_not_canonicalized(k8s_helper):
    """The "*" sentinel must not be lowercased or hashed — it skips the username filter."""
    k8s_helper.resolve_namespace = mock.MagicMock(return_value="default")
    k8s_helper.list_secrets = mock.MagicMock(return_value=[])

    k8s_helper.list_user_token_secrets(username="*", namespace="default")

    # No auth_username label key should be present — the wildcard skips the filter
    called_labels = k8s_helper.list_secrets.call_args.kwargs["labels"]
    assert mlrun_constants.MLRunInternalLabels.auth_username not in called_labels


def test_list_user_token_secrets_invalid_expiration(k8s_helper):
    user_id = "test-user-id"
    username = "test-username"
    secret_name = k8s_helper._resolve_auth_secret_name(user_id, "token1")
    bad_secret = _make_user_token_secret(
        secret_name=secret_name,
        issued_at=1,
        expiration=b"not-a-number",
        user_id=user_id,
        username=username,
    )
    k8s_helper.resolve_namespace = mock.MagicMock(return_value="default")
    k8s_helper.list_secrets = mock.MagicMock(return_value=[bad_secret])

    result = k8s_helper.list_user_token_secrets(username=username, namespace="default")
    assert len(result) == 0


def test_list_user_token_secrets_invalid_issued_at(k8s_helper):
    user_id = "test-user-id"
    username = "test-username"
    secret_name = k8s_helper._resolve_auth_secret_name(user_id, "token1")
    bad_secret = _make_user_token_secret(
        secret_name=secret_name,
        issued_at=b"not-a-number",
        expiration=1,
        user_id=user_id,
        username=username,
    )
    k8s_helper.resolve_namespace = mock.MagicMock(return_value="default")
    k8s_helper.list_secrets = mock.MagicMock(return_value=[bad_secret])

    result = k8s_helper.list_user_token_secrets(username=username, namespace="default")
    assert len(result) == 0


def test_get_user_token_secret_value_valid(k8s_helper):
    user_id = "test-user-id"
    token_name = "my-token"
    token_value = "abc123"
    secret_name = k8s_helper._resolve_auth_secret_name(user_id, token_name)

    # Create a Kubernetes secret with properly encoded tokensFile
    existing_secret = _make_user_token_secret(
        secret_name,
        token_name=token_name,
        token_value=token_value,
        issued_at=1,
        expiration=9999,
        user_id=user_id,
    )

    k8s_helper.resolve_namespace = mock.MagicMock(return_value="default")
    k8s_helper.list_secrets = mock.MagicMock(return_value=[existing_secret])

    token_value_from_k8s = k8s_helper.get_user_token_secret_value(
        user_id=user_id,
        token_name=token_name,
        namespace="default",
    )

    assert token_value_from_k8s == token_value
    k8s_helper.list_secrets.assert_called_once()


def test_get_user_token_secret_value_not_found(k8s_helper):
    user_id = "test-user-id"
    token_name = "my-token"

    k8s_helper.resolve_namespace = mock.MagicMock(return_value="default")
    k8s_helper.list_secrets = mock.MagicMock(return_value=[])

    with pytest.raises(mlrun.errors.MLRunNotFoundError):
        k8s_helper.get_user_token_secret_value(user_id, token_name, namespace="default")


@pytest.mark.parametrize("user_id", [None, ""])
def test_get_user_token_secret_value_rejects_empty_user_id(k8s_helper, user_id):
    with pytest.raises(mlrun.errors.MLRunBadRequestError, match="user_id is missing"):
        k8s_helper.get_user_token_secret_value(user_id, "some-token")


def test_get_user_token_secret_value_invalid_base64(k8s_helper):
    user_id = "test-user-id"
    token_name = "my-token"
    secret_name = k8s_helper._resolve_auth_secret_name(user_id, token_name)

    # Create a secret with an invalid base64 tokensFile
    bad_secret = _make_k8s_secret(
        secret_name,
        labels={
            mlrun_constants.MLRunInternalLabels.auth_userid: user_id,
            mlrun_constants.MLRunInternalLabels.auth_token_name: k8s_helper._hash_label(
                token_name
            ),
        },
        annotations={
            mlrun_constants.InternalAnnotations.auth_token_name: token_name,
        },
    )
    bad_secret.data["tokensFile"] = "!!!invalidbase64!!!"  # invalid base64 content
    bad_secret.data["tokenExpiration"] = base64.b64encode(b"9999").decode()

    k8s_helper.resolve_namespace = mock.MagicMock(return_value="default")
    k8s_helper.list_secrets = mock.MagicMock(return_value=[bad_secret])

    with pytest.raises(mlrun.errors.MLRunRuntimeError):
        k8s_helper.get_user_token_secret_value(
            user_id=user_id,
            token_name=token_name,
            namespace="default",
        )


def test_get_user_token_secret_value_invalid_yaml(k8s_helper):
    user_id = "test-user-id"
    token_name = "my-token"
    secret_name = k8s_helper._resolve_auth_secret_name(user_id, token_name)

    # Base64 encoded string but invalid YAML
    bad_yaml = base64.b64encode(b"{invalid_yaml: ]").decode()
    bad_secret = _make_k8s_secret(
        secret_name,
        labels={
            mlrun_constants.MLRunInternalLabels.auth_userid: user_id,
            mlrun_constants.MLRunInternalLabels.auth_token_name: k8s_helper._hash_label(
                token_name
            ),
        },
        annotations={
            mlrun_constants.InternalAnnotations.auth_token_name: token_name,
        },
    )
    bad_secret.data["tokensFile"] = bad_yaml
    k8s_helper.resolve_namespace = mock.MagicMock(return_value="default")
    k8s_helper.list_secrets = mock.MagicMock(return_value=[bad_secret])

    with pytest.raises(mlrun.errors.MLRunRuntimeError):
        k8s_helper.get_user_token_secret_value(user_id, token_name, namespace="default")


def test_delete_user_token_secret_success(k8s_helper):
    user_id = "test-user-id"
    token_name = "token1"
    secret_name = k8s_helper._resolve_auth_secret_name(user_id, token_name)

    k8s_helper.resolve_namespace = mock.MagicMock(return_value="default")
    k8s_helper.v1api.delete_namespaced_secret = mock.MagicMock()

    k8s_helper.delete_user_token_secret(
        user_id=user_id, token_name=token_name, namespace="default"
    )

    k8s_helper.v1api.delete_namespaced_secret.assert_called_once_with(
        name=secret_name,
        namespace="default",
        _request_timeout=framework.utils.singletons.k8s.K8sHelper._resolve_k8s_timeout(
            framework.utils.singletons.k8s.K8S_TIMEOUT_DEFAULT
        ),
    )


def test_delete_user_token_secret_not_found(k8s_helper):
    user_id = "test-user-id"
    token_name = "missing"
    secret_name = k8s_helper._resolve_auth_secret_name(user_id, token_name)

    k8s_helper.resolve_namespace = mock.MagicMock(return_value="default")
    k8s_helper.v1api.delete_namespaced_secret = mock.MagicMock(
        side_effect=k8s_client_rest.ApiException(status=404, reason="Not Found")
    )

    with pytest.raises(mlrun.errors.MLRunNotFoundError) as exc:
        k8s_helper.delete_user_token_secret(
            user_id=user_id, token_name=token_name, namespace="default"
        )

    assert f"Secret for token '{token_name}' not found" in str(exc.value)

    k8s_helper.v1api.delete_namespaced_secret.assert_called_once_with(
        name=secret_name,
        namespace="default",
        _request_timeout=framework.utils.singletons.k8s.K8sHelper._resolve_k8s_timeout(
            framework.utils.singletons.k8s.K8S_TIMEOUT_DEFAULT
        ),
    )


def test_delete_user_token_secret_api_error(k8s_helper):
    user_id = "test-user-id"
    token_name = "badtoken"
    secret_name = k8s_helper._resolve_auth_secret_name(user_id, token_name)

    k8s_helper.resolve_namespace = mock.MagicMock(return_value="default")
    k8s_helper.v1api.delete_namespaced_secret = mock.MagicMock(
        side_effect=k8s_client_rest.ApiException(status=500, reason="Internal Error")
    )

    with pytest.raises(mlrun.errors.MLRunRuntimeError) as exc:
        k8s_helper.delete_user_token_secret(
            user_id=user_id, token_name=token_name, namespace="default"
        )

    assert "Failed to delete secret" in str(exc.value)

    k8s_helper.v1api.delete_namespaced_secret.assert_called_once_with(
        name=secret_name,
        namespace="default",
        _request_timeout=framework.utils.singletons.k8s.K8sHelper._resolve_k8s_timeout(
            framework.utils.singletons.k8s.K8S_TIMEOUT_DEFAULT
        ),
    )


def test_delete_user_token_secret_unexpected_error(k8s_helper):
    user_id = "test-user-id"
    token_name = "oops"
    secret_name = k8s_helper._resolve_auth_secret_name(user_id, token_name)

    k8s_helper.resolve_namespace = mock.MagicMock(return_value="default")
    k8s_helper.v1api.delete_namespaced_secret = mock.MagicMock(
        side_effect=RuntimeError("dummy-error")
    )

    with pytest.raises(mlrun.errors.MLRunRuntimeError) as exc:
        k8s_helper.delete_user_token_secret(
            user_id=user_id, token_name=token_name, namespace="default"
        )

    assert "Unexpected error deleting secret" in str(exc.value)

    k8s_helper.v1api.delete_namespaced_secret.assert_called_once_with(
        name=secret_name,
        namespace="default",
        _request_timeout=framework.utils.singletons.k8s.K8sHelper._resolve_k8s_timeout(
            framework.utils.singletons.k8s.K8S_TIMEOUT_DEFAULT
        ),
    )


def test_get_user_secret_tokens_as_igz_yml_data_single_token(k8s_helper):
    """Test fetching a single token by name (strict mode)."""
    user_id = "test-user-id"
    token_name = "my-token"
    token_value = "abc123"
    secret_name = k8s_helper._resolve_auth_secret_name(user_id, token_name)

    existing_secret = _make_user_token_secret(
        secret_name,
        token_name=token_name,
        token_value=token_value,
        issued_at=1,
        expiration=9999,
        user_id=user_id,
    )
    k8s_helper.list_secrets = mock.MagicMock(return_value=[existing_secret])

    result = k8s_helper.get_user_secret_tokens_as_igz_yml_data(
        user_id=user_id, token_name=token_name
    )

    assert result == [{"name": token_name, "token": token_value}]


def test_get_user_secret_tokens_as_igz_yml_data_single_token_not_found(k8s_helper):
    """Test that MLRunBadRequestError is raised when requested token doesn't exist."""
    user_id = "test-user-id"
    token_name = "missing-token"

    k8s_helper.list_secrets = mock.MagicMock(return_value=[])

    with pytest.raises(mlrun.errors.MLRunBadRequestError):
        k8s_helper.get_user_secret_tokens_as_igz_yml_data(
            user_id=user_id, token_name=token_name
        )


def test_list_user_token_secret_values(k8s_helper):
    """Test listing all token values for a user."""
    user_id = "test-user-id"
    token1_name = "token1"
    token2_name = "token2"
    token1_value = "value1"
    token2_value = "value2"
    secret1_name = k8s_helper._resolve_auth_secret_name(user_id, token1_name)
    secret2_name = k8s_helper._resolve_auth_secret_name(user_id, token2_name)

    secret1 = _make_user_token_secret(
        secret1_name,
        token_name=token1_name,
        token_value=token1_value,
        issued_at=1,
        expiration=1111,
        user_id=user_id,
    )
    secret2 = _make_user_token_secret(
        secret2_name,
        token_name=token2_name,
        token_value=token2_value,
        issued_at=1,
        expiration=2222,
        user_id=user_id,
    )

    k8s_helper.list_secrets = mock.MagicMock(return_value=[secret1, secret2])

    result = k8s_helper.list_user_token_secret_values(user_id=user_id)

    assert len(result) == 2
    token_names = [t.name for t in result]
    token_values = [t.token for t in result]
    assert token1_name in token_names
    assert token2_name in token_names
    assert token1_value in token_values
    assert token2_value in token_values


def test_list_user_token_secret_values_partial_failure(k8s_helper):
    """Test that partial failures are skipped when listing token values."""
    user_id = "test-user-id"
    token1_name = "token1"
    token2_name = "token2"
    token1_value = "value1"
    secret1_name = k8s_helper._resolve_auth_secret_name(user_id, token1_name)
    secret2_name = k8s_helper._resolve_auth_secret_name(user_id, token2_name)

    # Create two secrets - one valid, one with missing tokensFile
    secret1 = _make_user_token_secret(
        secret1_name,
        token_name=token1_name,
        token_value=token1_value,
        issued_at=1,
        expiration=1111,
        user_id=user_id,
    )
    secret2 = _make_user_token_secret(
        secret2_name,
        token_name=token2_name,
        token_value="value2",
        issued_at=1,
        expiration=2222,
        user_id=user_id,
    )
    # Remove tokensFile to simulate extraction failure
    secret2.data.pop("tokensFile", None)

    k8s_helper.list_secrets = mock.MagicMock(return_value=[secret1, secret2])

    result = k8s_helper.list_user_token_secret_values(user_id=user_id)

    # Only token1 should be returned (token2 failed extraction)
    assert len(result) == 1
    assert result[0].name == token1_name
    assert result[0].token == token1_value


def test_list_user_token_secret_values_empty(k8s_helper):
    """Test that an empty list is returned when user has no tokens."""
    user_id = "test-user-id"

    k8s_helper.list_secrets = mock.MagicMock(return_value=[])

    result = k8s_helper.list_user_token_secret_values(user_id=user_id)

    assert result == []


@pytest.mark.parametrize("user_id", [None, ""])
def test_list_user_token_secret_values_rejects_empty_user_id(k8s_helper, user_id):
    with pytest.raises(mlrun.errors.MLRunBadRequestError, match="user_id is missing"):
        k8s_helper.list_user_token_secret_values(user_id=user_id)


def test_get_user_secret_tokens_as_igz_yml_data_no_tokens(k8s_helper):
    """Test that MLRunBadRequestError is raised when user has no tokens."""
    user_id = "test-user-id"

    k8s_helper.list_secrets = mock.MagicMock(return_value=[])

    with pytest.raises(
        mlrun.errors.MLRunBadRequestError,
        match=f"No valid tokens found for user id '{user_id}'",
    ):
        k8s_helper.get_user_secret_tokens_as_igz_yml_data(
            user_id=user_id, token_name=None
        )


def test_get_user_secret_tokens_as_igz_yml_data_all_fail(k8s_helper):
    """Test MLRunBadRequestError when all token extractions fail."""
    user_id = "test-user-id"
    token_name = "bad-token"
    secret_name = k8s_helper._resolve_auth_secret_name(user_id, token_name)

    # Create a secret with missing tokensFile
    bad_secret = _make_user_token_secret(
        secret_name,
        token_name=token_name,
        token_value="value",
        issued_at=1,
        expiration=1111,
        user_id=user_id,
    )
    bad_secret.data.pop("tokensFile", None)

    k8s_helper.list_secrets = mock.MagicMock(return_value=[bad_secret])

    with pytest.raises(
        mlrun.errors.MLRunBadRequestError,
        match=f"No valid tokens found for user id '{user_id}'",
    ):
        k8s_helper.get_user_secret_tokens_as_igz_yml_data(
            user_id=user_id, token_name=None
        )


def _make_user_token_secret(
    secret_name,
    token_name="my-token",
    token_value="abc123",
    expiration=None,
    issued_at=None,
    labels=None,
    annotations=None,
    user_id="test-user-id",
    username="test-username",
):
    if labels is None:
        labels = {
            mlrun_constants.MLRunInternalLabels.auth_userid: user_id,
            mlrun_constants.MLRunInternalLabels.auth_username: framework.utils.singletons.k8s.K8sHelper._hash_label(
                username
            ),
            mlrun_constants.MLRunInternalLabels.auth_token_name: framework.utils.singletons.k8s.K8sHelper._hash_label(
                token_name
            ),
        }
    if annotations is None:
        annotations = {
            mlrun_constants.InternalAnnotations.auth_username: username,
            mlrun_constants.InternalAnnotations.auth_token_name: token_name,
        }
    secret = _make_k8s_secret(secret_name, labels, annotations)

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

    # Encode tokenIssuedAt if provided
    if issued_at is not None:
        secret.data["tokenIssuedAt"] = base64.b64encode(
            str(issued_at).encode()
        ).decode()

    return secret


def _make_k8s_secret(name, labels=None, annotations=None):
    metadata = k8s_client.V1ObjectMeta(
        name=name, labels=labels or {}, annotations=annotations or {}
    )
    return k8s_client.V1Secret(metadata=metadata, data={})


class TestK8sTimeouts:
    """Tests for k8s API call timeout configuration."""

    @pytest.mark.parametrize(
        "timeout_type",
        [
            framework.utils.singletons.k8s.K8S_TIMEOUT_DEFAULT,
            framework.utils.singletons.k8s.K8S_TIMEOUT_LIST,
            framework.utils.singletons.k8s.K8S_TIMEOUT_LOGS,
        ],
    )
    def test_get_k8s_timeout_returns_configured_value(self, timeout_type):
        """Test that each timeout tier returns its configured value."""
        expected = int(getattr(mlrun.mlconf.kubernetes.timeouts, timeout_type))
        resolved = framework.utils.singletons.k8s.K8sHelper._resolve_k8s_timeout(
            timeout_type
        )
        assert resolved == expected

    def test_timeout_zero_disables_timeout_on_api_call(self, k8s_helper, monkeypatch):
        """Test that timeout=0 passes None to _request_timeout, disabling it."""
        monkeypatch.setattr(mlrun.mlconf.kubernetes.timeouts, "default", 0)
        k8s_helper.v1api.read_namespaced_pod.return_value = k8s_client.V1Pod()
        k8s_helper.get_pod(name="test-pod", namespace="test-ns")
        call_kwargs = k8s_helper.v1api.read_namespaced_pod.call_args
        assert call_kwargs.kwargs["_request_timeout"] is None

    def test_raise_for_status_code_catches_read_timeout(self):
        """Test that raise_for_status_code catches MaxRetryError with ReadTimeoutError."""
        import urllib3.exceptions

        @framework.utils.singletons.k8s.raise_for_status_code
        def fake_k8s_call():
            raise urllib3.exceptions.MaxRetryError(
                pool=None,
                url="/api/v1/pods",
                reason=urllib3.exceptions.ReadTimeoutError(
                    pool=None, url="/api/v1/pods", message="Read timed out"
                ),
            )

        with pytest.raises(mlrun.errors.MLRunRuntimeError, match="timed out"):
            fake_k8s_call()

    def test_raise_for_status_code_reraises_non_timeout_max_retry(self):
        """Test that raise_for_status_code re-raises MaxRetryError without ReadTimeoutError."""
        import urllib3.exceptions

        @framework.utils.singletons.k8s.raise_for_status_code
        def fake_k8s_call():
            raise urllib3.exceptions.MaxRetryError(
                pool=None,
                url="/api/v1/pods",
                reason=urllib3.exceptions.ConnectTimeoutError("Connect failed"),
            )

        with pytest.raises(urllib3.exceptions.MaxRetryError):
            fake_k8s_call()
