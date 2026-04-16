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

import json
from http import HTTPStatus
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.artifacts
import mlrun.common.schemas

import services.api.crud
from services.api.tests.unit.conftest import K8sSecretsMock

project = "prj"
datastore = {
    "project": "prj",
    "name": "ds",
    "type": "nosql",
    "object": "http://some_url_example/pp",
    "private": None,
}
legacy_api_projects_path = "projects"
api_datastore_path = f"projects/{project}/datastore-profiles"


def _create_project(client: TestClient, project_name: str = project):
    project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=project_name),
        spec=mlrun.common.schemas.ProjectSpec(
            description="banana", source="source", goals="some goals"
        ),
    )
    resp = client.post(legacy_api_projects_path, json=project.dict())
    assert resp.status_code == HTTPStatus.CREATED.value
    return resp


def test_datastore_profile_create_ok(
    db: Session, client: TestClient, k8s_secrets_mock: K8sSecretsMock
):
    _create_project(client)
    resp = client.put(
        api_datastore_path,
        json=datastore,
    )
    assert resp.status_code == HTTPStatus.OK.value

    expected_return = {"project": project, **datastore}

    resp = client.get(
        api_datastore_path + "/" + datastore["name"],
    )
    assert resp.status_code == HTTPStatus.OK.value
    assert json.loads(resp._content) == expected_return


def test_datastore_profile_update_ok(
    db: Session, client: TestClient, k8s_secrets_mock: K8sSecretsMock
):
    _create_project(client)
    resp = client.put(
        api_datastore_path,
        json=datastore,
    )
    assert resp.status_code == HTTPStatus.OK.value
    datastore_updated = datastore
    datastore_updated["object"] = "another version of body"
    resp = client.put(
        api_datastore_path,
        json=datastore_updated,
    )
    assert resp.status_code == HTTPStatus.OK.value

    expected_return = {"project": project, **datastore_updated}

    resp = client.get(
        api_datastore_path + "/" + datastore["name"],
    )
    assert resp.status_code == HTTPStatus.OK.value
    assert json.loads(resp._content) == expected_return


def test_datastore_profile_create_fail(
    db: Session, client: TestClient, k8s_secrets_mock: K8sSecretsMock
):
    # No project created
    resp = client.put(
        api_datastore_path,
        json=datastore,
    )
    assert resp.status_code == HTTPStatus.NOT_FOUND.value

    # Empty data
    _create_project(client)
    resp = client.put(
        api_datastore_path,
        json={},
    )
    assert resp.status_code == HTTPStatus.UNPROCESSABLE_ENTITY.value


def test_datastore_profile_get_fail(
    db: Session, client: TestClient, k8s_secrets_mock: K8sSecretsMock
):
    # No project created
    resp = client.get(
        api_datastore_path + "/" + datastore["name"],
    )
    assert resp.status_code == HTTPStatus.NOT_FOUND.value

    # Not existing profile
    _create_project(client)
    client.put(
        api_datastore_path,
        json={},
    )
    resp = client.get(
        api_datastore_path + "/invalid",
    )
    assert resp.status_code == HTTPStatus.NOT_FOUND.value


def test_datastore_profile_delete_wrong_project(
    db: Session, client: TestClient, k8s_secrets_mock: K8sSecretsMock
):
    # No project created
    resp = client.delete(
        api_datastore_path + "/" + datastore["name"],
    )
    assert resp.status_code == HTTPStatus.NOT_FOUND.value


def test_datastore_profile_delete_not_exist(
    db: Session, client: TestClient, k8s_secrets_mock: K8sSecretsMock
):
    # Not existing profile
    _create_project(client)
    resp = client.delete(
        api_datastore_path + "/" + datastore["name"],
    )
    assert resp.status_code == HTTPStatus.NOT_FOUND.value


def test_datastore_profile_delete(
    db: Session, client: TestClient, k8s_secrets_mock: K8sSecretsMock
):
    # Not existing profile
    _create_project(client)

    # Create the profile
    resp = client.put(
        api_datastore_path,
        json=datastore,
    )
    assert resp.status_code == HTTPStatus.OK.value

    # Get the profile OK
    resp = client.get(
        api_datastore_path + "/" + datastore["name"],
    )
    assert resp.status_code == HTTPStatus.OK.value

    # Delete the profile
    resp = client.delete(
        api_datastore_path + "/" + datastore["name"],
    )
    assert resp.status_code == HTTPStatus.OK.value

    # Get the nonexistent profile
    resp = client.delete(
        api_datastore_path + "/" + datastore["name"],
    )
    assert resp.status_code == HTTPStatus.NOT_FOUND.value


def test_datastore_profile_list(
    db: Session, client: TestClient, k8s_secrets_mock: K8sSecretsMock
):
    # No project created
    resp = client.get(
        api_datastore_path,
    )
    assert resp.status_code == HTTPStatus.NOT_FOUND.value

    # Project with no datasource profiles
    _create_project(client)
    resp = client.get(
        api_datastore_path,
    )
    assert resp.status_code == HTTPStatus.OK.value
    assert json.loads(resp._content) == []

    # Create the profile
    client.put(
        api_datastore_path,
        json=datastore,
    )

    expected_return = [{"project": project, **datastore}]

    resp = client.get(
        api_datastore_path,
    )
    assert resp.status_code == HTTPStatus.OK.value
    assert json.loads(resp._content) == expected_return


def test_delete_secret_passes_correct_arguments():
    """
    Verify that _delete_secret calls delete_project_secret with the
    provider as a SecretProviderName enum and the secret key as a string,
    rather than bundling them into a SecretsData object.
    """
    profiles = services.api.crud.DatastoreProfiles()
    project = "my-project"
    profile_name = "my-profile"

    with (
        patch.object(profiles, "_in_k8s", return_value=True),
        patch("services.api.crud.Secrets") as mock_secrets_cls,
    ):
        mock_secrets_instance = MagicMock()
        mock_secrets_cls.return_value = mock_secrets_instance

        profiles._delete_secret(project, profile_name)

        mock_secrets_instance.delete_project_secret.assert_called_once()
        call_args = mock_secrets_instance.delete_project_secret.call_args

        # The second positional arg must be the provider enum, not a SecretsData object
        provider_arg = call_args[0][1]
        assert provider_arg == mlrun.common.schemas.SecretProviderName.kubernetes, (
            f"Expected provider to be SecretProviderName.kubernetes, "
            f"got {type(provider_arg).__name__}: {provider_arg}"
        )

        # The third positional arg must be the secret key string
        secret_key_arg = call_args[0][2]
        assert isinstance(secret_key_arg, str), (
            f"Expected secret_key to be a string, got {type(secret_key_arg).__name__}"
        )

        # Verify allow_internal_secrets is passed
        assert call_args[1].get("allow_internal_secrets") is True
