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
import http
import json
import uuid

import fastapi.testclient
import sqlalchemy.orm

import mlrun.api.schemas

API_PROJECTS_PATH = "projects"
API_ARTIFACTS_PATH = "projects/{project}/artifacts"
API_ARTIFACTS_PATH_WITH_TAG = API_ARTIFACTS_PATH + "?tag={tag}"
API_TAGS_PATH = "projects/{project}/tags/{tag}"
STORE_API_ARTIFACTS_PATH = API_ARTIFACTS_PATH + "/" + "{uid}/{key}?tag={tag}"


class TestArtifactTags:
    project = "test-project"

    def test_overwrite_artifact_tags_by_name_identifier(
        self, db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
    ):
        self._create_project(client)

        tag = "tag1"
        overwrite_tag = "tag2"
        artifact1_labels = {"artifact_name": "artifact1"}
        _, _, artifact1_name, artifact1_uid, artifact1_key, _, _ = self._store_artifact(
            client, tag=tag, labels=artifact1_labels
        )
        _, _, artifact2_name, artifact2_uid, artifact2_key, _, _ = self._store_artifact(
            client, tag=tag
        )

        response = client.post(
            API_TAGS_PATH.format(project=self.project, tag=overwrite_tag),
            json={
                "kind": "artifact",
                "identifiers": [
                    {
                        "name": artifact1_key,
                    },
                ],
            },
        )
        assert response.status_code == http.HTTPStatus.OK.value

        response_body = self._list_artifacts_and_assert(
            client, tag=tag, expected_number_of_artifacts=1
        )
        assert response_body["artifacts"][0]["name"] == artifact2_name

        response_body = self._list_artifacts_and_assert(
            client, tag=overwrite_tag, expected_number_of_artifacts=1
        )
        assert response_body["artifacts"][0]["name"] == artifact1_name

    def test_overwrite_artifact_tags_by_multiple_name_identifiers(
        self, db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
    ):
        self._create_project(client)

        tag = "tag1"
        overwrite_tag = "tag2"
        artifact1_labels = {"artifact_name": "artifact1"}
        _, _, artifact1_name, artifact1_uid, artifact1_key, _, _ = self._store_artifact(
            client, tag=tag, labels=artifact1_labels
        )
        _, _, artifact2_name, artifact2_uid, artifact2_key, _, _ = self._store_artifact(
            client, tag=tag
        )

        response = client.post(
            API_TAGS_PATH.format(project=self.project, tag=overwrite_tag),
            json={
                "kind": "artifact",
                "identifiers": [
                    {
                        "name": artifact1_key,
                    },
                    {
                        "name": artifact2_key,
                    },
                ],
            },
        )
        assert response.status_code == http.HTTPStatus.OK.value

        self._list_artifacts_and_assert(client, tag=tag, expected_number_of_artifacts=0)

        response_body = self._list_artifacts_and_assert(
            client, tag=overwrite_tag, expected_number_of_artifacts=2
        )
        for artifact in response_body["artifacts"]:
            assert artifact["name"] in [artifact1_name, artifact2_name]

    def test_append_artifact_tags_by_name_identifier(
        self, db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
    ):
        self._create_project(client)

        tag = "tag1"
        new_tag = "tag2"
        artifact1_labels = {"artifact_name": "artifact1"}
        _, _, artifact1_name, artifact1_uid, artifact1_key, _, _ = self._store_artifact(
            client, tag=tag, labels=artifact1_labels
        )
        _, _, artifact2_name, artifact2_uid, artifact2_key, _, _ = self._store_artifact(
            client, tag=tag
        )

        response = client.put(
            API_TAGS_PATH.format(project=self.project, tag=new_tag),
            json={
                "kind": "artifact",
                "identifiers": [
                    {
                        "name": artifact1_key,
                    },
                ],
            },
        )
        assert response.status_code == http.HTTPStatus.OK.value

        response_body = self._list_artifacts_and_assert(
            client, tag=tag, expected_number_of_artifacts=2
        )
        for artifact in response_body["artifacts"]:
            assert artifact["name"] in [artifact1_name, artifact2_name]

        response_body = self._list_artifacts_and_assert(
            client, tag=new_tag, expected_number_of_artifacts=1
        )
        assert response_body["artifacts"][0]["name"] == artifact1_name

    def test_append_artifact_tags_by_multiple_name_identifiers(
        self, db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
    ):
        self._create_project(client)

        tag = "tag1"
        new_tag = "tag2"
        artifact1_labels = {"artifact_name": "artifact1"}
        _, _, artifact1_name, artifact1_uid, artifact1_key, _, _ = self._store_artifact(
            client, tag=tag, labels=artifact1_labels
        )
        _, _, artifact2_name, artifact2_uid, artifact2_key, _, _ = self._store_artifact(
            client, tag=tag
        )

        response = client.put(
            API_TAGS_PATH.format(project=self.project, tag=new_tag),
            json={
                "kind": "artifact",
                "identifiers": [
                    {
                        "name": artifact1_key,
                    },
                    {
                        "name": artifact2_key,
                    },
                ],
            },
        )
        assert response.status_code == http.HTTPStatus.OK.value

        response_body = self._list_artifacts_and_assert(
            client, tag=tag, expected_number_of_artifacts=2
        )
        for artifact in response_body["artifacts"]:
            assert artifact["name"] in [artifact1_name, artifact2_name]

        response_body = self._list_artifacts_and_assert(
            client, tag=new_tag, expected_number_of_artifacts=2
        )
        for artifact in response_body["artifacts"]:
            assert artifact["name"] in [artifact1_name, artifact2_name]

    def test_append_artifact_existing_tag(
        self, db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
    ):
        self._create_project(client)

        tag = "tag1"
        artifact1_labels = {"artifact_name": "artifact1"}
        _, _, artifact1_name, artifact1_uid, artifact1_key, _, _ = self._store_artifact(
            client, tag=tag, labels=artifact1_labels
        )
        _, _, artifact2_name, artifact2_uid, artifact2_key, _, _ = self._store_artifact(
            client, tag=tag
        )

        response = client.put(
            API_TAGS_PATH.format(project=self.project, tag=tag),
            json={
                "kind": "artifact",
                "identifiers": [
                    {
                        "name": artifact1_key,
                    },
                ],
            },
        )
        assert response.status_code == http.HTTPStatus.OK.value

        response_body = self._list_artifacts_and_assert(
            client, tag=tag, expected_number_of_artifacts=2
        )
        for artifact in response_body["artifacts"]:
            assert artifact["name"] in [artifact1_name, artifact2_name]

    def test_delete_artifact_tag_by_name_identifier(
        self, db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
    ):
        self._create_project(client)

        tag = "tag1"
        artifact1_labels = {"artifact_name": "artifact1"}
        _, _, artifact1_name, artifact1_uid, artifact1_key, _, _ = self._store_artifact(
            client, tag=tag, labels=artifact1_labels
        )
        _, _, artifact2_name, artifact2_uid, artifact2_key, _, _ = self._store_artifact(
            client, tag=tag
        )

        response = client.delete(
            API_TAGS_PATH.format(project=self.project, tag=tag),
            json={
                "kind": "artifact",
                "identifiers": [
                    {
                        "name": artifact1_key,
                    },
                ],
            },
        )
        assert response.status_code == http.HTTPStatus.NO_CONTENT.value

        response_body = self._list_artifacts_and_assert(
            client, tag=tag, expected_number_of_artifacts=1
        )
        assert response_body["artifacts"][0]["name"] == artifact2_name

    def test_delete_artifact_tag_by_multiple_name_identifiers(
        self, db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
    ):
        self._create_project(client)

        tag = "tag1"
        artifact1_labels = {"artifact_name": "artifact1"}
        _, _, artifact1_name, artifact1_uid, artifact1_key, _, _ = self._store_artifact(
            client, tag=tag, labels=artifact1_labels
        )
        _, _, artifact2_name, artifact2_uid, artifact2_key, _, _ = self._store_artifact(
            client, tag=tag
        )

        response = client.delete(
            API_TAGS_PATH.format(project=self.project, tag=tag),
            json={
                "kind": "artifact",
                "identifiers": [
                    {
                        "name": artifact1_key,
                    },
                    {
                        "name": artifact2_key,
                    },
                ],
            },
        )
        assert response.status_code == http.HTTPStatus.NO_CONTENT.value

        self._list_artifacts_and_assert(client, tag=tag, expected_number_of_artifacts=0)

    def test_delete_artifact_tag_but_artifact_has_no_tag(
        self, db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
    ):
        self._create_project(client)

        tag = "tag1"
        artifact1_labels = {"artifact_name": "artifact1"}
        _, _, artifact1_name, artifact1_uid, artifact1_key, _, _ = self._store_artifact(
            client, labels=artifact1_labels
        )
        _, _, artifact2_name, artifact2_uid, artifact2_key, _, _ = self._store_artifact(
            client
        )

        self._list_artifacts_and_assert(client, tag=tag, expected_number_of_artifacts=0)

        response = client.delete(
            API_TAGS_PATH.format(project=self.project, tag=tag),
            json={
                "kind": "artifact",
                "identifiers": [
                    {
                        "name": artifact1_key,
                    },
                    {
                        "name": artifact2_key,
                    },
                ],
            },
        )
        assert response.status_code == http.HTTPStatus.NO_CONTENT.value

        self._list_artifacts_and_assert(client, tag=tag, expected_number_of_artifacts=0)

    def _list_artifacts_and_assert(
        self,
        client,
        tag: str,
        expected_number_of_artifacts: int,
        expected_status_code: http.HTTPStatus = http.HTTPStatus.OK,
        assert_tag: bool = False,
    ):
        response = self._list_artifacts(client, tag=tag)
        response_body = response.json()
        assert response.status_code == expected_status_code
        assert len(response_body["artifacts"]) == expected_number_of_artifacts
        if assert_tag and expected_number_of_artifacts > 0:
            self._assert_tag(artifacts=response_body["artifacts"], expected_tag=tag)
        return response_body

    @staticmethod
    def _assert_tag(artifacts, expected_tag):
        for artifact in artifacts:
            artifact_tag = mlrun.utils.get_in_artifact(artifact, "tag")
            assert artifact_tag == expected_tag

    def _create_project(
        self, client: fastapi.testclient.TestClient, project_name: str = None
    ):
        project = mlrun.api.schemas.Project(
            metadata=mlrun.api.schemas.ProjectMetadata(
                name=project_name or self.project
            ),
            spec=mlrun.api.schemas.ProjectSpec(
                description="banana", source="source", goals="some goals"
            ),
        )
        response = client.post(API_PROJECTS_PATH, json=project.dict())
        assert response.status_code == http.HTTPStatus.CREATED.value
        return response

    def _list_artifacts(self, client, project: str = None, tag: str = None):
        project = project or self.project
        if tag:
            return client.get(
                API_ARTIFACTS_PATH_WITH_TAG.format(project=project, tag=tag)
            )
        return client.get(API_ARTIFACTS_PATH.format(project=project))

    def _store_artifact(
        self,
        client: fastapi.testclient.TestClient,
        name: str = None,
        project: str = None,
        uid: str = None,
        key: str = None,
        tag: str = None,
        data: dict = None,
        labels: dict = None,
        kind: str = "artifact",
    ):
        uid = uid or str(uuid.uuid4())
        key = key or str(uuid.uuid4())
        name = name or str(uuid.uuid4())

        if not data:
            data = {
                "metadata": {"name": name},
                "spec": {"src_path": "/some/path"},
                "kind": kind,
                "status": {"bla": "blabla"},
            }
        if labels:
            data["metadata"]["labels"] = labels

        response = client.post(
            STORE_API_ARTIFACTS_PATH.format(
                project=project or self.project, uid=uid, key=key, tag=tag
            ),
            data=json.dumps(data),
        )
        assert response.status_code == http.HTTPStatus.OK.value
        return response, project, name, uid, key, tag, data
