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

import collections
import json
import unittest.mock

import deepdiff
import fastapi.testclient
import jwt
import pytest
import sqlalchemy.orm

import mlrun.common.schemas
import mlrun.errors
import mlrun.runtimes.base

import services.api.crud
import services.api.tests.unit.conftest


def test_store_project_secrets_verifications(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
):
    project = "project-name"
    provider = mlrun.common.schemas.SecretProviderName.kubernetes
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        services.api.crud.Secrets().store_project_secrets(
            project,
            mlrun.common.schemas.SecretsData(
                provider=provider, secrets={"invalid/key": "value"}
            ),
        )

    with pytest.raises(mlrun.errors.MLRunAccessDeniedError):
        services.api.crud.Secrets().store_project_secrets(
            project,
            mlrun.common.schemas.SecretsData(
                provider=provider, secrets={"mlrun.internal.key": "value"}
            ),
        )


def test_store_project_secrets_with_key_map_verifications(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    k8s_secrets_mock: services.api.tests.unit.conftest.APIK8sSecretsMock,
):
    project = "project-name"
    provider = mlrun.common.schemas.SecretProviderName.kubernetes
    key_map_secret_key = (
        services.api.crud.Secrets().generate_client_key_map_project_secret_key(
            services.api.crud.SecretsClientType.schedules
        )
    )
    # not allowed to edit key map
    with pytest.raises(mlrun.errors.MLRunAccessDeniedError):
        services.api.crud.Secrets().store_project_secrets(
            project,
            mlrun.common.schemas.SecretsData(
                provider=provider, secrets={key_map_secret_key: "value"}
            ),
        )

    # not allowed with provider other than k8s
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        services.api.crud.Secrets().store_project_secrets(
            project,
            mlrun.common.schemas.SecretsData(
                provider=mlrun.common.schemas.SecretProviderName.vault,
                secrets={"invalid/key": "value"},
            ),
        )

    # invalid key map name (wrong prefix)
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        services.api.crud.Secrets().store_project_secrets(
            project,
            mlrun.common.schemas.SecretsData(
                provider=provider, secrets={"invalid/key": "value"}
            ),
            key_map_secret_key="invalid-key-map-secret-key",
        )

    # invalid key map name but with correct prefix
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        services.api.crud.Secrets().store_project_secrets(
            project,
            mlrun.common.schemas.SecretsData(
                provider=provider, secrets={"invalid/key": "value"}
            ),
            allow_internal_secrets=True,
            key_map_secret_key=f"{services.api.crud.Secrets().key_map_secrets_key_prefix}invalid/key",
        )

    # Internal must be allowed when using key maps, verify that without internal allowed we fail
    with pytest.raises(mlrun.errors.MLRunAccessDeniedError):
        services.api.crud.Secrets().store_project_secrets(
            project,
            mlrun.common.schemas.SecretsData(
                provider=provider, secrets={"valid-key": "value"}
            ),
            key_map_secret_key=key_map_secret_key,
        )


def test_get_project_secret_verifications(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    k8s_secrets_mock: services.api.tests.unit.conftest.APIK8sSecretsMock,
):
    project = "project-name"
    provider = mlrun.common.schemas.SecretProviderName.kubernetes
    key_map_secret_key = (
        services.api.crud.Secrets().generate_client_key_map_project_secret_key(
            services.api.crud.SecretsClientType.schedules
        )
    )

    # verifications check
    # not allowed from k8s
    with pytest.raises(mlrun.errors.MLRunAccessDeniedError):
        services.api.crud.Secrets().get_project_secret(
            project, provider, "does-not-exist-key"
        )

    # key map with provider other than k8s
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        services.api.crud.Secrets().get_project_secret(
            project,
            mlrun.common.schemas.SecretProviderName.vault,
            "does-not-exist-key",
            key_map_secret_key=key_map_secret_key,
        )


def test_get_project_secret(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    k8s_secrets_mock: services.api.tests.unit.conftest.APIK8sSecretsMock,
):
    _mock_secrets_crud_uuid_generation()
    project = "project-name"
    provider = mlrun.common.schemas.SecretProviderName.kubernetes
    key_map_secret_key = (
        services.api.crud.Secrets().generate_client_key_map_project_secret_key(
            services.api.crud.SecretsClientType.schedules
        )
    )
    invalid_secret_key = "invalid/key"
    invalid_secret_value = "some-value"
    invalid_secret_2_key = "invalid/key/2"
    invalid_secret_2_value = "some-value-3"
    valid_secret_key = "valid-key"
    valid_secret_value = "some-value-5"

    # sanity - none returned on keys that does not exist
    assert (
        services.api.crud.Secrets().get_project_secret(
            project, provider, "does-not-exist-key", allow_secrets_from_k8s=True
        )
        is None
    )
    assert (
        services.api.crud.Secrets().get_project_secret(
            project,
            provider,
            "does-not-exist-key",
            allow_secrets_from_k8s=True,
            allow_internal_secrets=True,
            key_map_secret_key=key_map_secret_key,
        )
        is None
    )

    services.api.crud.Secrets().store_project_secrets(
        project,
        mlrun.common.schemas.SecretsData(
            provider=provider,
            secrets={
                valid_secret_key: valid_secret_value,
                invalid_secret_key: invalid_secret_value,
                invalid_secret_2_key: invalid_secret_2_value,
            },
        ),
        allow_internal_secrets=True,
        key_map_secret_key=key_map_secret_key,
    )

    assert (
        services.api.crud.Secrets().get_project_secret(
            project, provider, valid_secret_key, allow_secrets_from_k8s=True
        )
        == valid_secret_value
    )
    assert (
        services.api.crud.Secrets().get_project_secret(
            project,
            provider,
            invalid_secret_key,
            allow_secrets_from_k8s=True,
            allow_internal_secrets=True,
            key_map_secret_key=key_map_secret_key,
        )
        == invalid_secret_value
    )
    assert (
        services.api.crud.Secrets().get_project_secret(
            project,
            provider,
            invalid_secret_2_key,
            allow_secrets_from_k8s=True,
            allow_internal_secrets=True,
            key_map_secret_key=key_map_secret_key,
        )
        == invalid_secret_2_value
    )


def test_delete_project_secret_verifications(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    k8s_secrets_mock: services.api.tests.unit.conftest.APIK8sSecretsMock,
):
    project = "project-name"
    provider = mlrun.common.schemas.SecretProviderName.kubernetes
    key_map_secret_key = (
        services.api.crud.Secrets().generate_client_key_map_project_secret_key(
            services.api.crud.SecretsClientType.schedules
        )
    )
    internal_key = services.api.crud.Secrets().generate_client_project_secret_key(
        services.api.crud.SecretsClientType.schedules, "some-name", "access_key"
    )

    # verifications check
    # internal key without allow
    with pytest.raises(mlrun.errors.MLRunAccessDeniedError):
        services.api.crud.Secrets().delete_project_secret(
            project, provider, internal_key
        )

    # vault provider
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        services.api.crud.Secrets().delete_project_secret(
            project, mlrun.common.schemas.SecretProviderName.vault, "valid-key"
        )

    # key map with provider other than k8s
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        services.api.crud.Secrets().delete_project_secret(
            project,
            mlrun.common.schemas.SecretProviderName.vault,
            "invalid/key",
            key_map_secret_key=key_map_secret_key,
        )

    # key map without allow from k8s provider
    with pytest.raises(mlrun.errors.MLRunAccessDeniedError):
        services.api.crud.Secrets().delete_project_secret(
            project, provider, "invalid/key", key_map_secret_key=key_map_secret_key
        )


def test_delete_project_secret(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    k8s_secrets_mock: services.api.tests.unit.conftest.APIK8sSecretsMock,
):
    _mock_secrets_crud_uuid_generation()
    project = "project-name"
    provider = mlrun.common.schemas.SecretProviderName.kubernetes
    key_map_secret_key = (
        services.api.crud.Secrets().generate_client_key_map_project_secret_key(
            services.api.crud.SecretsClientType.schedules
        )
    )
    invalid_secret_key = "invalid/key"
    invalid_secret_value = "some-value"
    invalid_secret_2_key = "invalid/key/2"
    invalid_secret_2_value = "some-value-3"
    valid_secret_key = "valid-key"
    valid_secret_value = "some-value-5"

    # sanity - do not explode on deleting key that doesn't exist
    services.api.crud.Secrets().delete_project_secret(
        project, provider, "does-not-exist-key", allow_secrets_from_k8s=True
    )

    services.api.crud.Secrets().store_project_secrets(
        project,
        mlrun.common.schemas.SecretsData(
            provider=provider,
            secrets=collections.OrderedDict(
                {
                    valid_secret_key: valid_secret_value,
                    invalid_secret_key: invalid_secret_value,
                    invalid_secret_2_key: invalid_secret_2_value,
                }
            ),
        ),
        allow_internal_secrets=True,
        key_map_secret_key=key_map_secret_key,
    )

    k8s_secrets_mock.assert_project_secrets(
        project,
        {
            valid_secret_key: valid_secret_value,
            0: invalid_secret_value,
            1: invalid_secret_2_value,
            key_map_secret_key: json.dumps(
                {invalid_secret_key: 0, invalid_secret_2_key: 1}
            ),
        },
    )

    services.api.crud.Secrets().delete_project_secret(
        project, provider, valid_secret_key, allow_secrets_from_k8s=True
    )

    k8s_secrets_mock.assert_project_secrets(
        project,
        {
            0: invalid_secret_value,
            1: invalid_secret_2_value,
            key_map_secret_key: json.dumps(
                {invalid_secret_key: 0, invalid_secret_2_key: 1}
            ),
        },
    )

    services.api.crud.Secrets().delete_project_secret(
        project,
        provider,
        invalid_secret_key,
        allow_secrets_from_k8s=True,
        allow_internal_secrets=True,
        key_map_secret_key=key_map_secret_key,
    )
    k8s_secrets_mock.assert_project_secrets(
        project,
        {
            1: invalid_secret_2_value,
            key_map_secret_key: json.dumps({invalid_secret_2_key: 1}),
        },
    )

    services.api.crud.Secrets().delete_project_secret(
        project,
        provider,
        invalid_secret_2_key,
        allow_secrets_from_k8s=True,
        allow_internal_secrets=True,
        key_map_secret_key=key_map_secret_key,
    )
    k8s_secrets_mock.assert_project_secrets(project, {})


def test_store_project_secrets_with_key_map_success(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    k8s_secrets_mock: services.api.tests.unit.conftest.APIK8sSecretsMock,
):
    _mock_secrets_crud_uuid_generation()
    project = "project-name"
    provider = mlrun.common.schemas.SecretProviderName.kubernetes
    key_map_secret_key = (
        services.api.crud.Secrets().generate_client_key_map_project_secret_key(
            services.api.crud.SecretsClientType.schedules
        )
    )
    invalid_secret_key = "invalid/key"
    invalid_secret_value = "some-value"
    invalid_secret_value_2 = "some-value-2"
    invalid_secret_2_key = "invalid/key/2"
    invalid_secret_2_value = "some-value-3"
    invalid_secret_2_value_2 = "some-value-4"
    valid_secret_key = "valid-key"
    valid_secret_value = "some-value-5"
    valid_secret_value_2 = "some-value-6"

    # store secret with valid key - map shouldn't be used
    services.api.crud.Secrets().store_project_secrets(
        project,
        mlrun.common.schemas.SecretsData(
            provider=provider, secrets={valid_secret_key: valid_secret_value}
        ),
        allow_internal_secrets=True,
        key_map_secret_key=key_map_secret_key,
    )
    k8s_secrets_mock.assert_project_secrets(
        project, {valid_secret_key: valid_secret_value}
    )

    # store secret with invalid key - map should be used
    services.api.crud.Secrets().store_project_secrets(
        project,
        mlrun.common.schemas.SecretsData(
            provider=provider, secrets={invalid_secret_key: invalid_secret_value}
        ),
        allow_internal_secrets=True,
        key_map_secret_key=key_map_secret_key,
    )
    k8s_secrets_mock.assert_project_secrets(
        project,
        {
            valid_secret_key: valid_secret_value,
            0: invalid_secret_value,
            key_map_secret_key: json.dumps({invalid_secret_key: 0}),
        },
    )

    # store secret with the same invalid key and different value
    services.api.crud.Secrets().store_project_secrets(
        project,
        mlrun.common.schemas.SecretsData(
            provider=provider, secrets={invalid_secret_key: invalid_secret_value_2}
        ),
        allow_internal_secrets=True,
        key_map_secret_key=key_map_secret_key,
    )
    k8s_secrets_mock.assert_project_secrets(
        project,
        {
            valid_secret_key: valid_secret_value,
            0: invalid_secret_value_2,
            key_map_secret_key: json.dumps({invalid_secret_key: 0}),
        },
    )

    # store secret with the different invalid key and value - do it twice - nothing should change
    for _ in range(2):
        services.api.crud.Secrets().store_project_secrets(
            project,
            mlrun.common.schemas.SecretsData(
                provider=provider,
                secrets={invalid_secret_2_key: invalid_secret_2_value},
            ),
            allow_internal_secrets=True,
            key_map_secret_key=key_map_secret_key,
        )
        k8s_secrets_mock.assert_project_secrets(
            project,
            {
                valid_secret_key: valid_secret_value,
                0: invalid_secret_value_2,
                1: invalid_secret_2_value,
                key_map_secret_key: json.dumps(
                    {invalid_secret_key: 0, invalid_secret_2_key: 1}
                ),
            },
        )

    # change values to all secrets
    services.api.crud.Secrets().store_project_secrets(
        project,
        mlrun.common.schemas.SecretsData(
            provider=provider,
            secrets={
                valid_secret_key: valid_secret_value_2,
                invalid_secret_key: invalid_secret_value,
                invalid_secret_2_key: invalid_secret_2_value_2,
            },
        ),
        allow_internal_secrets=True,
        key_map_secret_key=key_map_secret_key,
    )
    k8s_secrets_mock.assert_project_secrets(
        project,
        {
            valid_secret_key: valid_secret_value_2,
            0: invalid_secret_value,
            1: invalid_secret_2_value_2,
            key_map_secret_key: json.dumps(
                {invalid_secret_key: 0, invalid_secret_2_key: 1}
            ),
        },
    )


def _mock_secrets_crud_uuid_generation():
    uuids_iter = iter(range(10000))

    def _mock_generate_uuid():
        return next(uuids_iter)

    services.api.crud.Secrets()._generate_uuid = unittest.mock.Mock(
        side_effect=_mock_generate_uuid
    )


def test_secrets_crud_internal_project_secrets(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    k8s_secrets_mock: services.api.tests.unit.conftest.APIK8sSecretsMock,
):
    project = "project-name"
    provider = mlrun.common.schemas.SecretProviderName.kubernetes
    regular_secret_key = "key"
    regular_secret_value = "value"
    internal_secret_key = (
        f"{services.api.crud.Secrets().internal_secrets_key_prefix}internal-key"
    )
    internal_secret_value = "internal-value"

    # store regular secret - pass
    services.api.crud.Secrets().store_project_secrets(
        project,
        mlrun.common.schemas.SecretsData(
            provider=provider, secrets={regular_secret_key: regular_secret_value}
        ),
    )

    # store internal secret - fail
    with pytest.raises(mlrun.errors.MLRunAccessDeniedError):
        services.api.crud.Secrets().store_project_secrets(
            project,
            mlrun.common.schemas.SecretsData(
                provider=provider, secrets={internal_secret_key: internal_secret_value}
            ),
        )

    # store internal secret with allow - pass
    services.api.crud.Secrets().store_project_secrets(
        project,
        mlrun.common.schemas.SecretsData(
            provider=provider, secrets={internal_secret_key: internal_secret_value}
        ),
        allow_internal_secrets=True,
    )

    # list keys without allow - regular only
    secret_keys_data = services.api.crud.Secrets().list_project_secret_keys(
        project, provider
    )
    assert secret_keys_data.secret_keys == [regular_secret_key]

    # list keys with allow - regular and internal
    secret_keys_data = services.api.crud.Secrets().list_project_secret_keys(
        project, provider, allow_internal_secrets=True
    )
    assert secret_keys_data.secret_keys == [regular_secret_key, internal_secret_key]

    # list data without allow - regular only
    secrets_data = services.api.crud.Secrets().list_project_secrets(
        project, provider, allow_secrets_from_k8s=True
    )
    assert (
        deepdiff.DeepDiff(
            secrets_data.secrets,
            {regular_secret_key: regular_secret_value},
            ignore_order=True,
        )
        == {}
    )

    # list data with allow - regular and internal
    secrets_data = services.api.crud.Secrets().list_project_secrets(
        project, provider, allow_secrets_from_k8s=True, allow_internal_secrets=True
    )
    assert (
        deepdiff.DeepDiff(
            secrets_data.secrets,
            {
                regular_secret_key: regular_secret_value,
                internal_secret_key: internal_secret_value,
            },
            ignore_order=True,
        )
        == {}
    )

    # delete regular secret - pass
    services.api.crud.Secrets().delete_project_secrets(
        project,
        provider,
        [regular_secret_key],
    )

    # delete with empty list (delete all) - shouldn't delete internal
    services.api.crud.Secrets().delete_project_secrets(
        project,
        provider,
        [],
    )
    # list to verify - only internal should remain
    secrets_data = services.api.crud.Secrets().list_project_secrets(
        project,
        provider,
        allow_secrets_from_k8s=True,
        allow_internal_secrets=True,
    )
    assert (
        deepdiff.DeepDiff(
            secrets_data.secrets,
            {internal_secret_key: internal_secret_value},
            ignore_order=True,
        )
        == {}
    )

    # delete internal secret without allow - fail
    with pytest.raises(mlrun.errors.MLRunAccessDeniedError):
        services.api.crud.Secrets().delete_project_secrets(
            project,
            provider,
            [internal_secret_key],
        )

    # delete internal secret with allow - pass
    services.api.crud.Secrets().delete_project_secrets(
        project, provider, [internal_secret_key], allow_internal_secrets=True
    )
    # list to verify - there should be no secrets
    secrets_data = services.api.crud.Secrets().list_project_secrets(
        project, provider, allow_secrets_from_k8s=True
    )
    assert (
        deepdiff.DeepDiff(
            secrets_data.secrets,
            {},
            ignore_order=True,
        )
        == {}
    )

    # store internal secret again to verify deletion with empty list with allow - pass
    services.api.crud.Secrets().store_project_secrets(
        project,
        mlrun.common.schemas.SecretsData(
            provider=provider, secrets={internal_secret_key: internal_secret_value}
        ),
        allow_internal_secrets=True,
    )
    # delete with empty list (delete all) with allow - nothing should remain
    services.api.crud.Secrets().delete_project_secrets(
        project,
        provider,
        [],
        allow_internal_secrets=True,
    )
    # list to verify
    secrets_data = services.api.crud.Secrets().list_project_secrets(
        project, provider, allow_secrets_from_k8s=True
    )
    assert (
        deepdiff.DeepDiff(
            secrets_data.secrets,
            {},
            ignore_order=True,
        )
        == {}
    )


def test_store_auth_secret_verifications(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
):
    # not allowed with provider other than k8s
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        services.api.crud.Secrets().store_auth_secret(
            mlrun.common.schemas.AuthSecretData(
                provider=mlrun.common.schemas.SecretProviderName.vault,
                username="some-username",
                access_key="some-access-key",
            ),
        )


def test_store_auth_secret(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    k8s_secrets_mock: services.api.tests.unit.conftest.APIK8sSecretsMock,
):
    username = "some-username"
    access_key = "some-access-key"
    secret_name = services.api.crud.Secrets().store_auth_secret(
        mlrun.common.schemas.AuthSecretData(
            provider=mlrun.common.schemas.SecretProviderName.kubernetes,
            username=username,
            access_key=access_key,
        ),
    )
    k8s_secrets_mock.assert_auth_secret(secret_name, username, access_key)


@pytest.fixture
def mock_iguazio_client():
    with unittest.mock.patch(
        "framework.utils.clients.iguazio.v4.Client"
    ) as mock_client_cls:
        mock_client_instance = unittest.mock.MagicMock()
        mock_client_instance.refresh_access_tokens.return_value = None
        mock_client_instance.revoke_offline_token.return_value = None
        mock_client_cls.return_value = mock_client_instance
        yield mock_client_instance


@pytest.mark.parametrize("tokens", [[], None])
def test_store_secret_tokens_missing_tokens(
    tokens,
):
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        services.api.crud.Secrets().store_secret_tokens(
            tokens,
            mlrun.common.schemas.AuthInfo(
                username="dummy-username", user_id="user-id-123"
            ),
        )


def test_store_secret_tokens_incorrect_user_id():
    token_payload = {"exp": 9999999999, "sub": "user-id-123"}
    secret_tokens = [
        mlrun.common.schemas.SecretToken(
            name="token1", token=_generate_token(token_payload)
        ),
    ]

    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError,
        match="Offline token 'token1' does not match the authenticated user ID. Stored tokens can only belong to the"
        " authenticated user.",
    ):
        services.api.crud.Secrets().store_secret_tokens(
            secret_tokens,
            mlrun.common.schemas.AuthInfo(
                username="dummy-username", user_id="different-user-id"
            ),
        )


def test_store_secret_tokens_duplicate_names():
    token_payload = {"exp": 9999999999, "sub": "user-id-123"}

    secret_tokens = [
        mlrun.common.schemas.SecretToken(
            name="dup-token", token=_generate_token(token_payload)
        ),
        mlrun.common.schemas.SecretToken(
            name="dup-token", token=_generate_token(token_payload)
        ),
    ]

    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError, match="Invalid or duplicate token name"
    ):
        services.api.crud.Secrets().store_secret_tokens(
            secret_tokens,
            mlrun.common.schemas.AuthInfo(
                username="dummy-username", user_id="user-id-123"
            ),
        )


def test_store_secret_tokens_invalid_offline_token_jwt_decode(mock_iguazio_client):
    secret_tokens = [
        mlrun.common.schemas.SecretToken(name="bad", token="this-is-not-a-jwt"),
    ]

    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError,
        match="Failed to decode offline token",
    ):
        services.api.crud.Secrets().store_secret_tokens(
            secret_tokens,
            mlrun.common.schemas.AuthInfo(
                username="dummy-username", user_id="user-id-123"
            ),
        )


@pytest.mark.parametrize(
    "payload, expected_err_msg",
    [
        ({"sub": "user-id-123"}, r"missing the 'exp' \(expiration\) claim"),  # no exp
        (
            {"sub": "user-id-123", "exp": None},
            r"missing the 'exp' \(expiration\) claim",
        ),  # exp is None
        (
            {"sub": "user-id-123", "exp": ""},
            r"missing the 'exp' \(expiration\) claim",
        ),  # exp is empty
        ({"exp": 9999999999}, r"missing the 'sub' \(subject\) claim"),  # no sub
        (
            {"sub": None, "exp": 9999999999},
            r"missing the 'sub' \(subject\) claim",
        ),  # sub is None
        (
            {"sub": "", "exp": 9999999999},
            r"missing the 'sub' \(subject\) claim",
        ),  # sub is empty
    ],
)
def test_store_secret_tokens_missing_required_claims_in_offline_token(
    mock_iguazio_client, payload, expected_err_msg
):
    token = _generate_token(payload)
    secret_tokens = [
        mlrun.common.schemas.SecretToken(name="bad-token", token=token),
    ]

    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError,
        match=expected_err_msg,
    ):
        services.api.crud.Secrets().store_secret_tokens(
            secret_tokens,
            mlrun.common.schemas.AuthInfo(
                username="dummy-username", user_id="user-id-123"
            ),
        )


def test_store_secret_tokens_return_values(mock_iguazio_client):
    token_payload = {"sub": "user-id-123", "exp": 9999999999}
    secret_tokens = [
        mlrun.common.schemas.SecretToken(
            name="token1", token=_generate_token(token_payload)
        ),
        mlrun.common.schemas.SecretToken(
            name="token2", token=_generate_token(token_payload)
        ),
        mlrun.common.schemas.SecretToken(
            name="token3", token=_generate_token(token_payload)
        ),
    ]

    mock_secrets_provider = unittest.mock.Mock()
    services.api.crud.Secrets().secrets_provider = mock_secrets_provider
    mock_secrets_provider.store_user_token_secret.side_effect = [
        mlrun.common.schemas.SecretEventActions.created,
        mlrun.common.schemas.SecretEventActions.updated,
        None,
    ]

    result = services.api.crud.Secrets().store_secret_tokens(
        secret_tokens,
        mlrun.common.schemas.AuthInfo(username="dummy-username", user_id="user-id-123"),
    )

    assert result == {
        "created_tokens": ["token1"],
        "updated_tokens": ["token2"],
    }

    assert mock_secrets_provider.store_user_token_secret.call_count == 3
    assert mock_iguazio_client.refresh_access_tokens.call_count == 1


def test_store_secret_tokens_refresh_access_tokens_failure(mock_iguazio_client):
    mock_iguazio_client.refresh_access_tokens.side_effect = (
        mlrun.errors.MLRunUnauthorizedError("Refresh failed")
    )

    secret_tokens = [
        mlrun.common.schemas.SecretToken(
            name="token1",
            token=_generate_token({"sub": "user-id-123", "exp": 9999999999}),
        ),
    ]

    with pytest.raises(mlrun.errors.MLRunUnauthorizedError, match="Refresh failed"):
        services.api.crud.Secrets().store_secret_tokens(
            secret_tokens,
            mlrun.common.schemas.AuthInfo(
                username="dummy-username", user_id="user-id-123"
            ),
        )

    mock_iguazio_client.refresh_access_tokens.assert_called_once_with(secret_tokens)


def test_list_secret_tokens_returns_tokens():
    username = "dummy-user"
    expected_tokens = [
        mlrun.common.schemas.SecretTokenInfo(name="jupyter", expiration=1750979191),
        mlrun.common.schemas.SecretTokenInfo(name="my-token", expiration=1754966400),
    ]

    mock_secrets_provider = unittest.mock.Mock()
    services.api.crud.Secrets().secrets_provider = mock_secrets_provider
    services.api.crud.Secrets().secrets_provider.list_user_token_secrets = (
        unittest.mock.Mock(return_value=expected_tokens)
    )

    response = services.api.crud.Secrets().list_secret_tokens(
        authenticated_username=username
    )

    assert isinstance(response, mlrun.common.schemas.ListSecretTokensResponse)
    assert len(response.secret_tokens) == 2
    assert response.secret_tokens[0].name == "jupyter"
    assert response.secret_tokens[0].expiration == 1750979191
    assert response.secret_tokens[1].name == "my-token"
    assert response.secret_tokens[1].expiration == 1754966400

    mock_secrets_provider.list_user_token_secrets.assert_called_once_with(
        username=username
    )


def test_revoke_secret_token_success(mock_iguazio_client):
    username = "dummy-user"
    token_name = "my-token"
    fake_token = "jwt-token-123"
    request_headers = {
        mlrun.common.schemas.HeaderNames.authorization: f"{mlrun.common.schemas.AuthorizationHeaderPrefixes.bearer}123",
    }

    mock_secrets_provider = unittest.mock.Mock()
    services.api.crud.Secrets().secrets_provider = mock_secrets_provider

    mock_secrets_provider.get_user_token_secret_value.return_value = fake_token
    mock_secrets_provider.delete_user_token_secret = unittest.mock.Mock()

    services.api.crud.Secrets().revoke_secret_token(
        token_name=token_name,
        authenticated_username=username,
        request_headers=request_headers,
    )

    mock_secrets_provider.get_user_token_secret_value.assert_called_once_with(
        username=username, token_name=token_name
    )
    mock_iguazio_client.revoke_offline_token.assert_called_once_with(
        fake_token, request_headers
    )
    mock_secrets_provider.delete_user_token_secret.assert_called_once_with(
        username=username, token_name=token_name
    )


def test_revoke_secret_token_secret_not_found(mock_iguazio_client):
    username = "dummy-user"
    token_name = "missing"

    mock_secrets_provider = unittest.mock.Mock()
    services.api.crud.Secrets().secrets_provider = mock_secrets_provider

    mock_secrets_provider.get_user_token_secret_value.side_effect = (
        mlrun.errors.MLRunNotFoundError("Token not found")
    )

    services.api.crud.Secrets().revoke_secret_token(
        token_name=token_name, authenticated_username=username
    )


def test_revoke_secret_token_iguazio_failure(mock_iguazio_client):
    username = "dummy-user"
    token_name = "badtoken"
    fake_token = "jwt-token-456"

    mock_secrets_provider = unittest.mock.Mock()
    services.api.crud.Secrets().secrets_provider = mock_secrets_provider
    mock_secrets_provider.get_user_token_secret_value.return_value = fake_token

    mock_iguazio_client.revoke_offline_token.side_effect = RuntimeError("Iguazio error")

    with pytest.raises(RuntimeError, match="Iguazio error"):
        services.api.crud.Secrets().revoke_secret_token(
            token_name=token_name, authenticated_username=username
        )


def test_revoke_secret_token_delete_failure(mock_iguazio_client):
    username = "dummy-user"
    token_name = "fail-delete"
    fake_token = "jwt-token-789"

    mock_secrets_provider = unittest.mock.Mock()
    services.api.crud.Secrets().secrets_provider = mock_secrets_provider
    mock_secrets_provider.get_user_token_secret_value.return_value = fake_token
    mock_secrets_provider.delete_user_token_secret.side_effect = RuntimeError(
        "Delete failed"
    )

    with pytest.raises(
        mlrun.errors.MLRunRuntimeError,
        match="revoked, but failed to delete associated secret",
    ):
        services.api.crud.Secrets().revoke_secret_token(
            token_name=token_name, authenticated_username=username
        )


def test_get_secret_token_success():
    username = "dummy-user"
    token_name = "my-token"
    fake_token_value = "jwt-fake-token"

    mock_secrets_provider = unittest.mock.Mock()
    services.api.crud.Secrets().secrets_provider = mock_secrets_provider
    mock_secrets_provider.get_user_token_secret_value.return_value = fake_token_value

    result = services.api.crud.Secrets().get_secret_token(
        token_name=token_name,
        authenticated_username=username,
    )

    mock_secrets_provider.get_user_token_secret_value.assert_called_once_with(
        username=username,
        token_name=token_name,
    )

    assert isinstance(result, mlrun.common.schemas.SecretToken)
    assert result.name == token_name
    assert result.token == fake_token_value


def test_get_secret_token_not_found():
    username = "dummy-user"
    token_name = "missing-token"

    mock_secrets_provider = unittest.mock.Mock()
    services.api.crud.Secrets().secrets_provider = mock_secrets_provider
    mock_secrets_provider.get_user_token_secret_value.side_effect = (
        mlrun.errors.MLRunNotFoundError("Token not found")
    )

    with pytest.raises(mlrun.errors.MLRunNotFoundError, match="Token not found"):
        services.api.crud.Secrets().get_secret_token(
            token_name=token_name,
            authenticated_username=username,
        )


def _generate_token(payload: dict) -> str:
    return jwt.encode(payload, key="dummy", algorithm="HS256")
