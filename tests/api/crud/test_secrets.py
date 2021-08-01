import fastapi.testclient
import pytest
import sqlalchemy.orm

import mlrun.api.crud
import mlrun.api.schemas
import mlrun.errors


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
