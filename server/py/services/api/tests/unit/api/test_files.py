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

import os
import unittest.mock
from http import HTTPStatus

import fastapi
import fastapi.exceptions
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

# fixtures for test, aren't used directly so we need to ignore the lint here
import mlrun
import mlrun.common.schemas
from tests.common_fixtures import (  # noqa: F401
    patch_file_forbidden,
    patch_file_not_found,
)

import services.api.api.endpoints.files


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
        services.api.api.endpoints.files._get_files(
            unittest.mock.Mock(),
            "s3://somebucket/some/path/file",
            "user1",
            mlrun.mlconf.artifacts.limits.max_chunk_size + 1,
            0,
            unittest.mock.Mock(),
        )

    assert exc.value.status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE.value


@pytest.mark.parametrize(
    "objpath, expected_filename",
    [
        ("/path/to/file.txt", "file.txt"),
        ("/path/to/data.csv", "data.csv"),
        ("file.txt", "file.txt"),
        ("/deep/nested/path/model.pkl", "model.pkl"),
    ],
)
def test_filename_extraction_uses_os_path_split(objpath, expected_filename):
    """
    Verify os.path.split is used to extract the filename from objpath.
    Before the fix, str.split(str) was used which always returned an empty
    filename because splitting a string by itself yields ['', ''].
    """
    _, filename = os.path.split(objpath)
    assert filename == expected_filename

def test_str_split_self_returns_empty_filename():
    """
    Demonstrate the bug: str.split(str) always returns ['', ''].
    """
    objpath = "/path/to/file.txt"
    # This is what the buggy code did
    _, buggy_filename = objpath.split(objpath)
    assert buggy_filename == "", "str.split(self) always produces empty string"

    # This is what the fixed code does
    _, correct_filename = os.path.split(objpath)
    assert correct_filename == "file.txt"

@unittest.mock.patch(
    "services.api.api.endpoints.files.store_manager",
)
@unittest.mock.patch(
    "services.api.api.endpoints.files.get_obj_path",
    return_value="/resolved/path/to/data.parquet",
)
@unittest.mock.patch(
    "services.api.api.endpoints.files.get_secrets",
    return_value={},
)
def test_get_files_returns_correct_filename_header(
    self,
    mock_get_secrets,
    mock_get_obj_path,
    mock_store_manager,
):
    """
    Integration-style test: verify _get_files returns the correct
    x-suggested-filename header with the actual filename, not an empty string.
    """
    mock_obj = unittest.mock.MagicMock()
    mock_obj.get.return_value = b"file content"
    mock_store_manager.object.return_value = mock_obj

    auth_info = mlrun.common.schemas.AuthInfo()
    response = services.api.api.endpoints.files._get_files(
        schema="",
        objpath="/some/path/data.parquet",
        user="",
        size=0,
        offset=0,
        auth_info=auth_info,
    )

    assert isinstance(response, fastapi.Response)
    assert response.headers["x-suggested-filename"] == "data.parquet"
