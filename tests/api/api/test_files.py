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
import unittest.mock
from http import HTTPStatus

import fastapi.exceptions
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

# fixtures for test, aren't used directly so we need to ignore the lint here
import mlrun
import server.api.api.endpoints.files
from tests.common_fixtures import (  # noqa: F401
    patch_file_forbidden,
    patch_file_not_found,
)


@pytest.mark.usefixtures("patch_file_forbidden")
def test_files_forbidden(db: Session, client: TestClient, k8s_secrets_mock) -> None:
    validate_files_status_code(client, HTTPStatus.FORBIDDEN.value)


@pytest.mark.usefixtures("patch_file_not_found")
def test_files_not_found(db: Session, client: TestClient, k8s_secrets_mock) -> None:
    validate_files_status_code(client, HTTPStatus.NOT_FOUND.value)


def validate_files_status_code(client: TestClient, status_code: int):
    resp = client.get("projects/{project}/files?schema=v3io&path=mybucket/files.txt")
    assert resp.status_code == status_code

    resp = client.get("projects/{project}/files?schema=v3io&path=mybucket/")
    assert resp.status_code == status_code

    resp = client.get("projects/{project}/filestat?schema=v3io&path=mybucket/files.txt")
    assert resp.status_code == status_code


class DatastoreObjectMock:
    def get(self, size, offset):
        return "dummy body"

    def listdir(self):
        return ["file1", "file2", "dir1/file3"]


@pytest.fixture
def files_mock():
    old_object = mlrun.store_manager.object
    mlrun.store_manager.object = unittest.mock.Mock(return_value=DatastoreObjectMock())

    yield mlrun.store_manager.object

    mlrun.store_manager.object = old_object


def test_files(db: Session, client: TestClient, files_mock, k8s_secrets_mock) -> None:
    path = "s3://somebucket/some/path/file"
    project = "proj1"

    env_secrets = {"V3IO_ACCESS_KEY": None}
    project_secrets = {"secret1": "value1", "secret2": "value2"}
    full_secrets = project_secrets.copy()
    full_secrets.update(env_secrets)
    k8s_secrets_mock.store_project_secrets(project, project_secrets)

    resp = client.get(f"projects/{project}/files?path={path}")
    assert resp
    files_mock.assert_called_once_with(url=path, secrets=full_secrets, project="proj1")
    files_mock.reset_mock()

    resp = client.get(f"projects/wrong-project/files?path={path}")
    assert resp
    files_mock.assert_called_once_with(
        url=path, secrets=env_secrets, project="wrong-project"
    )
    files_mock.reset_mock()

    resp = client.get(f"projects/{project}/files?path={path}&use-secrets=false")
    assert resp
    files_mock.assert_called_once_with(url=path, secrets=env_secrets, project="proj1")
    files_mock.reset_mock()


def test_files_max_chunk_size_exceeded():
    with pytest.raises(fastapi.exceptions.HTTPException) as exc:
        server.api.api.endpoints.files._get_files(
            unittest.mock.Mock(),
            "s3://somebucket/some/path/file",
            "user1",
            mlrun.mlconf.artifacts.limits.max_chunk_size + 1,
            0,
            unittest.mock.Mock(),
        )

    assert exc.value.status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE.value
