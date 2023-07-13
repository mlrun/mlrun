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
import json
from http import HTTPStatus

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.artifacts
import mlrun.common.schemas

PROJECT = "prj"
DATASTORE = {
    "project": "prj",
    "name": "ds",
    "type": "nosql",
    "body": "body or url ds://host:1234",
}
LEGACY_API_PROJECTS_PATH = "projects"
API_DATASTORE_PATH = "/api/v1/projects/{project}/datastore_profiles"


def _create_project(client: TestClient, project_name: str = PROJECT):
    project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=project_name),
        spec=mlrun.common.schemas.ProjectSpec(
            description="banana", source="source", goals="some goals"
        ),
    )
    resp = client.post(LEGACY_API_PROJECTS_PATH, json=project.dict())
    assert resp.status_code == HTTPStatus.CREATED.value
    return resp


def test_datastore_profile_create_ok(db: Session, client: TestClient):
    _create_project(client)
    resp = client.post(
        API_DATASTORE_PATH.format(project=PROJECT),
        data=json.dumps(DATASTORE),
    )
    assert resp.status_code == HTTPStatus.OK.value

    expected_return = {"project": PROJECT, **DATASTORE}

    resp = client.get(
        API_DATASTORE_PATH.format(project=PROJECT) + "/" + DATASTORE["name"],
    )
    assert resp.status_code == HTTPStatus.OK.value
    assert json.loads(resp._content) == expected_return


def test_datastore_profile_create_fail(db: Session, client: TestClient):
    # No project created
    resp = client.post(
        API_DATASTORE_PATH.format(project=PROJECT),
        data=json.dumps(DATASTORE),
    )
    assert resp.status_code == HTTPStatus.NOT_FOUND.value

    # Empty data
    _create_project(client)
    resp = client.post(
        API_DATASTORE_PATH.format(project=PROJECT),
        data={},
    )
    assert resp.status_code == HTTPStatus.UNPROCESSABLE_ENTITY.value


def test_datastore_profile_get_fail(db: Session, client: TestClient):
    # No project created
    resp = client.get(
        API_DATASTORE_PATH.format(project=PROJECT) + "/" + DATASTORE["name"],
    )
    assert resp.status_code == HTTPStatus.NOT_FOUND.value

    # Not existing profile
    _create_project(client)
    resp = client.post(
        API_DATASTORE_PATH.format(project=PROJECT),
        data={},
    )
    resp = client.get(
        API_DATASTORE_PATH.format(project=PROJECT) + "/invalid",
    )
    assert resp.status_code == HTTPStatus.NOT_FOUND.value


def test_datastore_profile_delete(db: Session, client: TestClient):
    # No project created
    resp = client.delete(
        API_DATASTORE_PATH.format(project=PROJECT) + "/" + DATASTORE["name"],
    )
    assert resp.status_code == HTTPStatus.NOT_FOUND.value

    # Not existing profile
    _create_project(client)
    resp = client.delete(
        API_DATASTORE_PATH.format(project=PROJECT) + "/" + DATASTORE["name"],
    )
    assert resp.status_code == HTTPStatus.NOT_FOUND.value

    # Create the profile
    resp = client.post(
        API_DATASTORE_PATH.format(project=PROJECT),
        data=json.dumps(DATASTORE),
    )
    assert resp.status_code == HTTPStatus.OK.value

    # Get the profile OK
    resp = client.get(
        API_DATASTORE_PATH.format(project=PROJECT) + "/" + DATASTORE["name"],
    )
    assert resp.status_code == HTTPStatus.OK.value

    # Delete the profile
    resp = client.delete(
        API_DATASTORE_PATH.format(project=PROJECT) + "/" + DATASTORE["name"],
    )
    assert resp.status_code == HTTPStatus.OK.value

    # Get the non existing project
    resp = client.delete(
        API_DATASTORE_PATH.format(project=PROJECT) + "/" + DATASTORE["name"],
    )
    assert resp.status_code == HTTPStatus.NOT_FOUND.value


def test_datastore_profile_list(db: Session, client: TestClient):
    # No project created
    resp = client.get(
        API_DATASTORE_PATH.format(project=PROJECT),
    )
    assert resp.status_code == HTTPStatus.NOT_FOUND.value

    # Project with no datasource profiles
    _create_project(client)
    resp = client.get(
        API_DATASTORE_PATH.format(project=PROJECT),
    )
    assert resp.status_code == HTTPStatus.OK.value
    assert json.loads(resp._content) == []

    # Create the profile
    resp = client.post(
        API_DATASTORE_PATH.format(project=PROJECT),
        data=json.dumps(DATASTORE),
    )

    expected_return = [{"project": PROJECT, **DATASTORE}]

    resp = client.get(
        API_DATASTORE_PATH.format(project=PROJECT),
    )
    assert resp.status_code == HTTPStatus.OK.value
    assert json.loads(resp._content) == expected_return
