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
from http import HTTPStatus

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.api.schemas

PROJECT = "prj"
KEY = "some-key/with-slash"
UID = "some-uid"
TAG = "some-tag"
LEGACY_API_PROJECTS_PATH = "projects"
LEGACY_API_ARTIFACT_PATH = "artifact"
LEGACY_API_ARTIFACTS_PATH = "artifacts"

API_ARTIFACTS_PATH = "projects/{project}/artifacts"
STORE_API_ARTIFACTS_PATH = API_ARTIFACTS_PATH + "/" + "{uid}/{key}?tag={tag}"


def test_list_artifact_tags(db: Session, client: TestClient) -> None:
    resp = client.get(f"{LEGACY_API_PROJECTS_PATH}/{PROJECT}/artifact-tags")
    assert resp.status_code == HTTPStatus.OK.value, "status"
    assert resp.json()["project"] == PROJECT, "project"


def _create_project(client: TestClient, project_name: str = PROJECT):
    project = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name=project_name),
        spec=mlrun.api.schemas.ProjectSpec(
            description="banana", source="source", goals="some goals"
        ),
    )
    resp = client.post(LEGACY_API_PROJECTS_PATH, json=project.dict())
    assert resp.status_code == HTTPStatus.CREATED.value
    return resp


def test_store_artifact_with_empty_dict(db: Session, client: TestClient):
    _create_project(client)

    resp = client.post(
        f"{LEGACY_API_ARTIFACT_PATH}/{PROJECT}/{UID}/{KEY}?tag={TAG}", data="{}"
    )
    assert resp.status_code == HTTPStatus.OK.value

    resp = client.get(f"{LEGACY_API_PROJECTS_PATH}/{PROJECT}/artifact/{KEY}?tag={TAG}")
    assert resp.status_code == HTTPStatus.OK.value


def test_delete_artifacts_after_storing_empty_dict(db: Session, client: TestClient):
    _create_project(client)
    empty_artifact = "{}"

    resp = client.post(
        f"{LEGACY_API_ARTIFACT_PATH}/{PROJECT}/{UID}/{KEY}?tag={TAG}",
        data=empty_artifact,
    )
    assert resp.status_code == HTTPStatus.OK.value

    resp = client.post(
        f"{LEGACY_API_ARTIFACT_PATH}/{PROJECT}/{UID}2/{KEY}2?tag={TAG}",
        data=empty_artifact,
    )
    assert resp.status_code == HTTPStatus.OK.value

    project_artifacts_path = f"{LEGACY_API_ARTIFACTS_PATH}?project={PROJECT}"

    resp = client.get(project_artifacts_path)
    assert len(resp.json()["artifacts"]) == 2

    resp = client.delete(project_artifacts_path)
    assert resp.status_code == HTTPStatus.OK.value

    resp = client.get(project_artifacts_path)
    assert len(resp.json()["artifacts"]) == 0


def test_list_artifacts(db: Session, client: TestClient) -> None:
    _create_project(client)

    for artifact_path in [
        f"{LEGACY_API_ARTIFACTS_PATH}?project={PROJECT}",
        API_ARTIFACTS_PATH.format(project=PROJECT),
    ]:
        resp = client.get(artifact_path)
        assert resp.status_code == HTTPStatus.OK.value
        assert len(resp.json()["artifacts"]) == 0

    resp = client.post(
        f"{LEGACY_API_ARTIFACT_PATH}/{PROJECT}/{UID}/{KEY}?tag={TAG}", data="{}"
    )
    assert resp.status_code == HTTPStatus.OK.value

    resp = client.post(
        STORE_API_ARTIFACTS_PATH.format(
            project=PROJECT, uid=f"{UID}2", key=f"{KEY}2", tag=TAG
        ),
        data="{}",
    )
    assert resp.status_code == HTTPStatus.OK.value

    for artifact_path in [
        f"{LEGACY_API_ARTIFACTS_PATH}?project={PROJECT}",
        API_ARTIFACTS_PATH.format(project=PROJECT),
    ]:
        resp = client.get(artifact_path)
        assert resp.status_code == HTTPStatus.OK.value
        assert len(resp.json()["artifacts"]) == 2
