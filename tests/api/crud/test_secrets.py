import deepdiff
import fastapi.testclient
import pytest
import sqlalchemy.orm

import mlrun.api.crud
import mlrun.api.schemas
import mlrun.errors
import tests.api.conftest


def test_store_secrets_verifications(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
):
    project = "project-name"
    provider = mlrun.api.schemas.SecretProviderName.kubernetes
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        mlrun.api.crud.Secrets().store_secrets(
            project,
            mlrun.api.schemas.SecretsData(
                provider=provider, secrets={"invalid/key": "value"}
            ),
        )

    with pytest.raises(mlrun.errors.MLRunAccessDeniedError):
        mlrun.api.crud.Secrets().store_secrets(
            project,
            mlrun.api.schemas.SecretsData(
                provider=provider, secrets={"mlrun.internal.key": "value"}
            ),
        )


def test_secrets_crud_internal_secrets(
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
    mlrun.api.crud.Secrets().store_secrets(
        project,
        mlrun.api.schemas.SecretsData(
            provider=provider, secrets={regular_secret_key: regular_secret_value}
        ),
    )

    # store internal secret - fail
    with pytest.raises(mlrun.errors.MLRunAccessDeniedError):
        mlrun.api.crud.Secrets().store_secrets(
            project,
            mlrun.api.schemas.SecretsData(
                provider=provider, secrets={internal_secret_key: internal_secret_value}
            ),
        )

    # store internal secret with allow - pass
    mlrun.api.crud.Secrets().store_secrets(
        project,
        mlrun.api.schemas.SecretsData(
            provider=provider, secrets={internal_secret_key: internal_secret_value}
        ),
        allow_internal_secrets=True,
    )

    # list keys without allow - regular only
    secret_keys_data = mlrun.api.crud.Secrets().list_secret_keys(project, provider)
    assert secret_keys_data.secret_keys == [regular_secret_key]

    # list keys with allow - regular and internal
    secret_keys_data = mlrun.api.crud.Secrets().list_secret_keys(
        project, provider, allow_internal_secrets=True
    )
    assert secret_keys_data.secret_keys == [regular_secret_key, internal_secret_key]

    # list data without allow - regular only
    secrets_data = mlrun.api.crud.Secrets().list_secrets(
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
    secrets_data = mlrun.api.crud.Secrets().list_secrets(
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
    mlrun.api.crud.Secrets().delete_secrets(
        project, provider, [regular_secret_key],
    )

    # delete with empty list (delete all) - shouldn't delete internal
    mlrun.api.crud.Secrets().delete_secrets(
        project, provider, [],
    )
    # list to verify - only internal should remain
    secrets_data = mlrun.api.crud.Secrets().list_secrets(
        project, provider, allow_secrets_from_k8s=True, allow_internal_secrets=True,
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
        mlrun.api.crud.Secrets().delete_secrets(
            project, provider, [internal_secret_key],
        )

    # delete internal secret with allow - pass
    mlrun.api.crud.Secrets().delete_secrets(
        project, provider, [internal_secret_key], allow_internal_secrets=True
    )
    # list to verify - there should be no secrets
    secrets_data = mlrun.api.crud.Secrets().list_secrets(
        project, provider, allow_secrets_from_k8s=True
    )
    assert deepdiff.DeepDiff(secrets_data.secrets, {}, ignore_order=True,) == {}

    # store internal secret again to verify deletion with empty list with allow - pass
    mlrun.api.crud.Secrets().store_secrets(
        project,
        mlrun.api.schemas.SecretsData(
            provider=provider, secrets={internal_secret_key: internal_secret_value}
        ),
        allow_internal_secrets=True,
    )
    # delete with empty list (delete all) with allow - nothing should remain
    mlrun.api.crud.Secrets().delete_secrets(
        project, provider, [], allow_internal_secrets=True,
    )
    # list to verify
    secrets_data = mlrun.api.crud.Secrets().list_secrets(
        project, provider, allow_secrets_from_k8s=True
    )
    assert deepdiff.DeepDiff(secrets_data.secrets, {}, ignore_order=True,) == {}
