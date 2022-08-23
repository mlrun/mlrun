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
import collections
import json
import unittest.mock

import deepdiff
import fastapi.testclient
import pytest
import sqlalchemy.orm

import mlrun.api.crud
import mlrun.api.schemas
import mlrun.errors
import tests.api.conftest


def test_store_project_secrets_verifications(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
):
    project = "project-name"
    provider = mlrun.api.schemas.SecretProviderName.kubernetes
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        mlrun.api.crud.Secrets().store_project_secrets(
            project,
            mlrun.api.schemas.SecretsData(
                provider=provider, secrets={"invalid/key": "value"}
            ),
        )

    with pytest.raises(mlrun.errors.MLRunAccessDeniedError):
        mlrun.api.crud.Secrets().store_project_secrets(
            project,
            mlrun.api.schemas.SecretsData(
                provider=provider, secrets={"mlrun.internal.key": "value"}
            ),
        )


def test_store_project_secrets_with_key_map_verifications(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
):
    project = "project-name"
    provider = mlrun.api.schemas.SecretProviderName.kubernetes
    key_map_secret_key = (
        mlrun.api.crud.Secrets().generate_client_key_map_project_secret_key(
            mlrun.api.crud.SecretsClientType.schedules
        )
    )
    # not allowed to edit key map
    with pytest.raises(mlrun.errors.MLRunAccessDeniedError):
        mlrun.api.crud.Secrets().store_project_secrets(
            project,
            mlrun.api.schemas.SecretsData(
                provider=provider, secrets={key_map_secret_key: "value"}
            ),
        )

    # not allowed with provider other than k8s
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        mlrun.api.crud.Secrets().store_project_secrets(
            project,
            mlrun.api.schemas.SecretsData(
                provider=mlrun.api.schemas.SecretProviderName.vault,
                secrets={"invalid/key": "value"},
            ),
        )

    # invalid key map name (wrong prefix)
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        mlrun.api.crud.Secrets().store_project_secrets(
            project,
            mlrun.api.schemas.SecretsData(
                provider=provider, secrets={"invalid/key": "value"}
            ),
            key_map_secret_key="invalid-key-map-secret-key",
        )

    # invalid key map name but with correct prefix
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        mlrun.api.crud.Secrets().store_project_secrets(
            project,
            mlrun.api.schemas.SecretsData(
                provider=provider, secrets={"invalid/key": "value"}
            ),
            allow_internal_secrets=True,
            key_map_secret_key=f"{mlrun.api.crud.Secrets().key_map_secrets_key_prefix}invalid/key",
        )

    # Internal must be allowed when using key maps, verify that without internal allowed we fail
    with pytest.raises(mlrun.errors.MLRunAccessDeniedError):
        mlrun.api.crud.Secrets().store_project_secrets(
            project,
            mlrun.api.schemas.SecretsData(
                provider=provider, secrets={"valid-key": "value"}
            ),
            key_map_secret_key=key_map_secret_key,
        )


def test_get_project_secret_verifications(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
):
    project = "project-name"
    provider = mlrun.api.schemas.SecretProviderName.kubernetes
    key_map_secret_key = (
        mlrun.api.crud.Secrets().generate_client_key_map_project_secret_key(
            mlrun.api.crud.SecretsClientType.schedules
        )
    )

    # verifications check
    # not allowed from k8s
    with pytest.raises(mlrun.errors.MLRunAccessDeniedError):
        mlrun.api.crud.Secrets().get_project_secret(
            project, provider, "does-not-exist-key"
        )

    # key map with provider other than k8s
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        mlrun.api.crud.Secrets().get_project_secret(
            project,
            mlrun.api.schemas.SecretProviderName.vault,
            "does-not-exist-key",
            key_map_secret_key=key_map_secret_key,
        )


def test_get_project_secret(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
):
    _mock_secrets_crud_uuid_generation()
    project = "project-name"
    provider = mlrun.api.schemas.SecretProviderName.kubernetes
    key_map_secret_key = (
        mlrun.api.crud.Secrets().generate_client_key_map_project_secret_key(
            mlrun.api.crud.SecretsClientType.schedules
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
        mlrun.api.crud.Secrets().get_project_secret(
            project, provider, "does-not-exist-key", allow_secrets_from_k8s=True
        )
        is None
    )
    assert (
        mlrun.api.crud.Secrets().get_project_secret(
            project,
            provider,
            "does-not-exist-key",
            allow_secrets_from_k8s=True,
            allow_internal_secrets=True,
            key_map_secret_key=key_map_secret_key,
        )
        is None
    )

    mlrun.api.crud.Secrets().store_project_secrets(
        project,
        mlrun.api.schemas.SecretsData(
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
        mlrun.api.crud.Secrets().get_project_secret(
            project, provider, valid_secret_key, allow_secrets_from_k8s=True
        )
        == valid_secret_value
    )
    assert (
        mlrun.api.crud.Secrets().get_project_secret(
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
        mlrun.api.crud.Secrets().get_project_secret(
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
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
):
    project = "project-name"
    provider = mlrun.api.schemas.SecretProviderName.kubernetes
    key_map_secret_key = (
        mlrun.api.crud.Secrets().generate_client_key_map_project_secret_key(
            mlrun.api.crud.SecretsClientType.schedules
        )
    )
    internal_key = mlrun.api.crud.Secrets().generate_client_project_secret_key(
        mlrun.api.crud.SecretsClientType.schedules, "some-name", "access_key"
    )

    # verifications check
    # internal key without allow
    with pytest.raises(mlrun.errors.MLRunAccessDeniedError):
        mlrun.api.crud.Secrets().delete_project_secret(project, provider, internal_key)

    # vault provider
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        mlrun.api.crud.Secrets().delete_project_secret(
            project, mlrun.api.schemas.SecretProviderName.vault, "valid-key"
        )

    # key map with provider other than k8s
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        mlrun.api.crud.Secrets().delete_project_secret(
            project,
            mlrun.api.schemas.SecretProviderName.vault,
            "invalid/key",
            key_map_secret_key=key_map_secret_key,
        )

    # key map without allow from k8s provider
    with pytest.raises(mlrun.errors.MLRunAccessDeniedError):
        mlrun.api.crud.Secrets().delete_project_secret(
            project, provider, "invalid/key", key_map_secret_key=key_map_secret_key
        )


def test_delete_project_secret(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
):
    _mock_secrets_crud_uuid_generation()
    project = "project-name"
    provider = mlrun.api.schemas.SecretProviderName.kubernetes
    key_map_secret_key = (
        mlrun.api.crud.Secrets().generate_client_key_map_project_secret_key(
            mlrun.api.crud.SecretsClientType.schedules
        )
    )
    invalid_secret_key = "invalid/key"
    invalid_secret_value = "some-value"
    invalid_secret_2_key = "invalid/key/2"
    invalid_secret_2_value = "some-value-3"
    valid_secret_key = "valid-key"
    valid_secret_value = "some-value-5"

    # sanity - do not explode on deleting key that doesn't exist
    mlrun.api.crud.Secrets().delete_project_secret(
        project, provider, "does-not-exist-key", allow_secrets_from_k8s=True
    )

    mlrun.api.crud.Secrets().store_project_secrets(
        project,
        mlrun.api.schemas.SecretsData(
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

    mlrun.api.crud.Secrets().delete_project_secret(
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

    mlrun.api.crud.Secrets().delete_project_secret(
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

    mlrun.api.crud.Secrets().delete_project_secret(
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
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
):
    _mock_secrets_crud_uuid_generation()
    project = "project-name"
    provider = mlrun.api.schemas.SecretProviderName.kubernetes
    key_map_secret_key = (
        mlrun.api.crud.Secrets().generate_client_key_map_project_secret_key(
            mlrun.api.crud.SecretsClientType.schedules
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
    mlrun.api.crud.Secrets().store_project_secrets(
        project,
        mlrun.api.schemas.SecretsData(
            provider=provider, secrets={valid_secret_key: valid_secret_value}
        ),
        allow_internal_secrets=True,
        key_map_secret_key=key_map_secret_key,
    )
    k8s_secrets_mock.assert_project_secrets(
        project, {valid_secret_key: valid_secret_value}
    )

    # store secret with invalid key - map should be used
    mlrun.api.crud.Secrets().store_project_secrets(
        project,
        mlrun.api.schemas.SecretsData(
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
    mlrun.api.crud.Secrets().store_project_secrets(
        project,
        mlrun.api.schemas.SecretsData(
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
        mlrun.api.crud.Secrets().store_project_secrets(
            project,
            mlrun.api.schemas.SecretsData(
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
    mlrun.api.crud.Secrets().store_project_secrets(
        project,
        mlrun.api.schemas.SecretsData(
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

    mlrun.api.crud.Secrets()._generate_uuid = unittest.mock.Mock(
        side_effect=_mock_generate_uuid
    )


def test_secrets_crud_internal_project_secrets(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
):
    project = "project-name"
    provider = mlrun.api.schemas.SecretProviderName.kubernetes
    regular_secret_key = "key"
    regular_secret_value = "value"
    internal_secret_key = (
        f"{mlrun.api.crud.Secrets().internal_secrets_key_prefix}internal-key"
    )
    internal_secret_value = "internal-value"

    # store regular secret - pass
    mlrun.api.crud.Secrets().store_project_secrets(
        project,
        mlrun.api.schemas.SecretsData(
            provider=provider, secrets={regular_secret_key: regular_secret_value}
        ),
    )

    # store internal secret - fail
    with pytest.raises(mlrun.errors.MLRunAccessDeniedError):
        mlrun.api.crud.Secrets().store_project_secrets(
            project,
            mlrun.api.schemas.SecretsData(
                provider=provider, secrets={internal_secret_key: internal_secret_value}
            ),
        )

    # store internal secret with allow - pass
    mlrun.api.crud.Secrets().store_project_secrets(
        project,
        mlrun.api.schemas.SecretsData(
            provider=provider, secrets={internal_secret_key: internal_secret_value}
        ),
        allow_internal_secrets=True,
    )

    # list keys without allow - regular only
    secret_keys_data = mlrun.api.crud.Secrets().list_project_secret_keys(
        project, provider
    )
    assert secret_keys_data.secret_keys == [regular_secret_key]

    # list keys with allow - regular and internal
    secret_keys_data = mlrun.api.crud.Secrets().list_project_secret_keys(
        project, provider, allow_internal_secrets=True
    )
    assert secret_keys_data.secret_keys == [regular_secret_key, internal_secret_key]

    # list data without allow - regular only
    secrets_data = mlrun.api.crud.Secrets().list_project_secrets(
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
    secrets_data = mlrun.api.crud.Secrets().list_project_secrets(
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
    mlrun.api.crud.Secrets().delete_project_secrets(
        project,
        provider,
        [regular_secret_key],
    )

    # delete with empty list (delete all) - shouldn't delete internal
    mlrun.api.crud.Secrets().delete_project_secrets(
        project,
        provider,
        [],
    )
    # list to verify - only internal should remain
    secrets_data = mlrun.api.crud.Secrets().list_project_secrets(
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
        mlrun.api.crud.Secrets().delete_project_secrets(
            project,
            provider,
            [internal_secret_key],
        )

    # delete internal secret with allow - pass
    mlrun.api.crud.Secrets().delete_project_secrets(
        project, provider, [internal_secret_key], allow_internal_secrets=True
    )
    # list to verify - there should be no secrets
    secrets_data = mlrun.api.crud.Secrets().list_project_secrets(
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
    mlrun.api.crud.Secrets().store_project_secrets(
        project,
        mlrun.api.schemas.SecretsData(
            provider=provider, secrets={internal_secret_key: internal_secret_value}
        ),
        allow_internal_secrets=True,
    )
    # delete with empty list (delete all) with allow - nothing should remain
    mlrun.api.crud.Secrets().delete_project_secrets(
        project,
        provider,
        [],
        allow_internal_secrets=True,
    )
    # list to verify
    secrets_data = mlrun.api.crud.Secrets().list_project_secrets(
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
        mlrun.api.crud.Secrets().store_auth_secret(
            mlrun.api.schemas.AuthSecretData(
                provider=mlrun.api.schemas.SecretProviderName.vault,
                username="some-username",
                access_key="some-access-key",
            ),
        )


def test_store_auth_secret(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
):
    username = "some-username"
    access_key = "some-access-key"
    secret_name = mlrun.api.crud.Secrets().store_auth_secret(
        mlrun.api.schemas.AuthSecretData(
            provider=mlrun.api.schemas.SecretProviderName.kubernetes,
            username=username,
            access_key=access_key,
        ),
    )
    k8s_secrets_mock.assert_auth_secret(secret_name, username, access_key)
