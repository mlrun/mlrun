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

    # Existing secret with older expiration
    existing_secret = _make_user_token_secret(
        k8s_helper,
        token_name=token_name,
        token_value=token_value,
        expiration=1000,
        username=username,
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
    username = "test-user"
    token_name = "my-token"
    token_value = "abc123"

    existing_secret = _make_user_token_secret(
        k8s_helper,
        token_name=token_name,
        token_value=token_value,
        expiration=5000,
        username=username,
    )
    k8s_helper.read_secret = mock.MagicMock(return_value=existing_secret)

    result = k8s_helper.store_user_token_secret(
        username=username,
        token_name=token_name,
        token=token_value,
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
            mlrun_constants.MLRunInternalLabels.user_token_secret_label_key: "test-user"
        },
    )
    secret2 = _make_k8s_secret(
        "secret2",
        labels={
            mlrun_constants.MLRunInternalLabels.user_token_secret_label_key: "test-user"
        },
    )

    fake_secret_list = mock.MagicMock()
    fake_secret_list.items = [secret1, secret2]

    k8s_helper.v1api.list_namespaced_secret = mock.MagicMock(
        return_value=fake_secret_list
    )

    result = k8s_helper.list_secrets(
        namespace="default",
        labels={
            mlrun_constants.MLRunInternalLabels.user_token_secret_label_key: "test-user"
        },
    )

    assert result == [secret1, secret2]
    k8s_helper.v1api.list_namespaced_secret.assert_called_once_with(
        namespace="default", label_selector="mlrun/user=test-user"
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
        namespace="default", label_selector=None
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
    username = "test-user"
    secret1 = _make_user_token_secret(
        k8s_helper, token_name=token1_name, expiration=1111, username=username
    )
    secret2 = _make_user_token_secret(
        k8s_helper, token_name=token2_name, expiration=2222, username=username
    )

    k8s_helper.resolve_namespace = mock.MagicMock(return_value="default")
    k8s_helper.list_secrets = mock.MagicMock(return_value=[secret1, secret2])

    result = k8s_helper.list_user_token_secrets(username=username, namespace="default")

    assert len(result) == 2
    assert result[0].token_name == token1_name
    assert result[0].expiration == 1111
    assert result[0].username == username
    assert result[1].token_name == token2_name
    assert result[1].expiration == 2222
    assert result[1].username == username

    k8s_helper.list_secrets.assert_called_once_with(
        namespace="default",
        labels={
            mlrun_constants.MLRunInternalLabels.user_token_secret_label_key: "test-user"
        },
    )


def test_list_user_token_secrets_invalid_expiration(k8s_helper):
    username = "test-user"
    bad_secret = _make_user_token_secret(
        k8s_helper, expiration=b"not-a-number", username=username
    )
    k8s_helper.resolve_namespace = mock.MagicMock(return_value="default")
    k8s_helper.list_secrets = mock.MagicMock(return_value=[bad_secret])

    result = k8s_helper.list_user_token_secrets(username=username, namespace="default")
    assert len(result) == 0


def test_list_user_token_secrets_all_users(k8s_helper):
    # Create secrets for different users
    token1 = "token1"
    token2 = "token2"
    user1 = "user1"
    user2 = "user2"
    secret1 = _make_user_token_secret(
        k8s_helper, token_name=token1, expiration=1111, username=user1
    )
    secret2 = _make_user_token_secret(
        k8s_helper, token_name=token2, expiration=2222, username=user2
    )

    # Mock namespace resolution
    k8s_helper.resolve_namespace = mock.MagicMock(return_value="default")

    # Mock list_secrets to return all secrets
    k8s_helper.list_secrets = mock.MagicMock(return_value=[secret1, secret2])

    # Call with username=None to get all users
    result = k8s_helper.list_user_token_secrets(username=None, namespace="default")

    # Check that both secrets are returned
    assert len(result) == 2
    assert result[0].token_name == token1
    assert result[0].username == user1
    assert result[0].expiration == 1111
    assert result[1].token_name == token2
    assert result[1].username == user2
    assert result[1].expiration == 2222

    # Ensure list_secrets was called without filtering by username
    k8s_helper.list_secrets.assert_called_once_with(namespace="default", labels=None)


@pytest.mark.parametrize(
    "secret_username, token_name, expiration, expected_username, expected_token_name, expected_expiration",
    [
        # Normal case
        ("alice", "token1", 1111, "alice", "token1", 1111),
        # Username with dash
        ("user-with-dash", "token1", 2222, "user-with-dash", "token1", 2222),
        # Token name with dash
        ("bob", "my-token", 3333, "bob", "my-token", 3333),
        # Both username and token name with dashes
        (
            "user-with-dash",
            "token-with-dash",
            4444,
            "user-with-dash",
            "token-with-dash",
            4444,
        ),
        # Token name with multiple dashes
        ("alice", "my-long-token-name", 5555, "alice", "my-long-token-name", 5555),
        (
            "user-name",
            "token-name-with-many-dashes",
            6666,
            "user-name",
            "token-name-with-many-dashes",
            6666,
        ),
        # Special characters in token name
        ("alice", "token.with.dots", 7777, "alice", "token.with.dots", 7777),
        ("bob", "token-123", 8888, "bob", "token-123", 8888),
        # Username with dots
        ("user.name", "token1", 9999, "user.name", "token1", 9999),
    ],
)
def test_convert_secret_to_token_info_valid_cases(
    k8s_helper,
    secret_username,
    token_name,
    expiration,
    expected_username,
    expected_token_name,
    expected_expiration,
):
    """Test various valid secret formats"""
    # Create secret using the helper
    secret = _make_user_token_secret(
        k8s_helper,
        token_name=token_name,
        expiration=expiration,
        username=secret_username,
    )

    # Convert secret to token info
    token_info = k8s_helper._convert_secret_to_token_info(secret)

    # Verify results
    assert token_info is not None
    assert token_info.username == expected_username
    assert token_info.token_name == expected_token_name
    assert token_info.expiration == expected_expiration


@pytest.mark.parametrize(
    "secret_name, labels, description",
    [
        # Invalid secret name formats
        (
            "invalid-secret-format",
            {mlrun_constants.MLRunInternalLabels.user_token_secret_label_key: "alice"},
            "completely wrong format",
        ),
        (
            "mlrun-auth-",
            {mlrun_constants.MLRunInternalLabels.user_token_secret_label_key: "alice"},
            "missing username and token",
        ),
        (
            "mlrun-auth-username",
            {
                mlrun_constants.MLRunInternalLabels.user_token_secret_label_key: "username"
            },
            "missing token",
        ),
        (
            "mlrun-xyz-alice-token1",
            {mlrun_constants.MLRunInternalLabels.user_token_secret_label_key: "alice"},
            "wrong prefix",
        ),
        (
            "mlrun-auth-alice-",
            {mlrun_constants.MLRunInternalLabels.user_token_secret_label_key: "alice"},
            "empty token name",
        ),
        # Missing or invalid username label
        ("mlrun-auth-alice-token1", {}, "no labels at all"),
        ("mlrun-auth-user-with-dash-token1", {}, "no labels with dashed username"),
        ("mlrun-auth-bob-token2", {"some-other-label": "value"}, "wrong label key"),
        (
            "mlrun-auth-charlie-token3",
            {"mlrun/other": "value"},
            "similar but wrong label",
        ),
        (
            "mlrun-auth-dave-token4",
            {mlrun_constants.MLRunInternalLabels.user_token_secret_label_key: ""},
            "empty username in label",
        ),
    ],
)
def test_convert_secret_to_token_info_invalid_cases(
    k8s_helper, secret_name, labels, description
):
    """Test that secrets with invalid format or missing username label return None"""
    secret = _make_k8s_secret(secret_name, labels=labels)

    # Add valid expiration so failure is not due to expiration
    secret.data["tokenExpiration"] = base64.b64encode(str(1234).encode()).decode()

    token_info = k8s_helper._convert_secret_to_token_info(secret)

    assert token_info is None, f"Expected None for case: {description}"


@pytest.mark.parametrize(
    "username, token_name, expiration_value",
    [
        ("alice", "token1", None),  # None expiration
        ("bob", "token2", "invalid"),  # Invalid expiration format
        ("charlie", "token3", ""),  # Empty expiration
    ],
)
def test_convert_secret_to_token_info_invalid_expiration(
    k8s_helper, username, token_name, expiration_value
):
    """Test that secrets with invalid expiration return None"""
    # Create secret with invalid expiration
    secret = _make_user_token_secret(
        k8s_helper,
        token_name=token_name,
        expiration=expiration_value,
        username=username,
    )

    token_info = k8s_helper._convert_secret_to_token_info(secret)

    assert token_info is None


def test_convert_secret_to_token_info_no_metadata_labels(k8s_helper):
    """Test secret with None metadata.labels attribute"""
    secret_name = "mlrun-auth-alice-token1"
    secret = k8s_client.V1Secret(
        metadata=k8s_client.V1ObjectMeta(
            name=secret_name,
            labels=None,  # Explicitly None
        ),
        data={"tokenExpiration": base64.b64encode(str(1234).encode()).decode()},
    )

    token_info = k8s_helper._convert_secret_to_token_info(secret)

    assert token_info is None


def test_get_user_token_secret_value_valid(k8s_helper):
    username = "test-user"
    token_name = "my-token"
    token_value = "abc123"

    # Create a Kubernetes secret with properly encoded tokensFile
    existing_secret = _make_user_token_secret(
        k8s_helper,
        token_name=token_name,
        token_value=token_value,
        expiration=9999,
        username=username,
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

    # Secret exists but tokensFile does not contain the requested token
    secret = _make_user_token_secret(
        k8s_helper, token_name="other-token", username=username
    )
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
    k8s_helper,
    token_name="my-token",
    token_value="abc123",
    expiration=None,
    username: str = "test-user",
):
    secret_name = k8s_helper._resolve_user_token_secret_name(username, token_name)
    labels = {mlrun_constants.MLRunInternalLabels.user_token_secret_label_key: username}
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
