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

import unittest.mock

import fastapi.testclient
import pytest
import sqlalchemy.orm

import mlrun.common.schemas
import server.api.crud


def test_delete_artifact_data(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    k8s_secrets_mock,
) -> None:
    path = "s3://somebucket/some/path/file"
    project = "proj1"

    env_secrets = {"V3IO_ACCESS_KEY": None}
    project_secrets = {"secret1": "value1", "secret2": "value2"}
    full_secrets = project_secrets.copy()
    full_secrets.update(env_secrets)
    k8s_secrets_mock.store_project_secrets(project, project_secrets)
    with unittest.mock.patch(
        "mlrun.datastore.store_manager.object"
    ) as store_manager_object_mock:
        server.api.crud.Files().delete_artifact_data(
            mlrun.common.schemas.AuthInfo(), project, path
        )
        store_manager_object_mock.assert_called_once_with(
            url=path, secrets=full_secrets, project=project
        )
        store_manager_object_mock.reset_mock()

        # user supplied secrets - use the same key to override project secrets
        user_secrets = {"secret1": "user-secret"}
        override_secrets = full_secrets.copy()
        override_secrets.update(user_secrets)
        server.api.crud.Files().delete_artifact_data(
            mlrun.common.schemas.AuthInfo(), project, path, secrets=user_secrets
        )
        store_manager_object_mock.assert_called_once_with(
            url=path, secrets=override_secrets, project=project
        )


def test_delete_artifact_data_internal_secret(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    k8s_secrets_mock,
) -> None:
    path = "s3://somebucket/some/path/file"
    project = "proj1"
    user_secrets = {"mlrun.secret1": "user-secret"}

    with pytest.raises(mlrun.errors.MLRunAccessDeniedError) as exc:
        server.api.crud.Files().delete_artifact_data(
            mlrun.common.schemas.AuthInfo(), project, path, secrets=user_secrets
        )
    assert (
        str(exc.value)
        == "Not allowed to create/update internal secrets (key starts with mlrun.)"
    )


def test_delete_artifact_data_local_path(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    k8s_secrets_mock,
) -> None:
    path = "/some-local-path"
    project = "proj1"

    with pytest.raises(mlrun.errors.MLRunAccessDeniedError) as exc:
        server.api.crud.Files().delete_artifact_data(
            mlrun.common.schemas.AuthInfo(), project, path
        )
    assert str(exc.value) == "Unauthorized path"
