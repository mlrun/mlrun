import http
import uuid

import fastapi.testclient
import sqlalchemy.orm

import mlrun.api.schemas

API_PROJECTS_PATH = "projects"
API_ARTIFACTS_PATH = "projects/{project}/artifacts"
API_ARTIFACTS_PATH_WITH_TAG = API_ARTIFACTS_PATH + "?tag={tag}"
API_TAGS_PATH = "projects/{project}/tags/{tag}"
STORE_API_ARTIFACTS_PATH = API_ARTIFACTS_PATH + "/" + "{uid}/{key}?tag={tag}"


class TestTags:
    project = "test-project"

    def test_overwrite_tags(
        self, db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
    ):
        self._create_project(client)
        tag = "tag1"
        self._store_artifact(client, tag=tag)
        self._store_artifact(client, tag=tag)

        resp = client.get(
            API_ARTIFACTS_PATH_WITH_TAG.format(project=self.project, tag=tag)
        )
        assert resp.status_code == http.HTTPStatus.OK.value
        assert len(resp.json()["artifacts"]) == 2
        self._assert_tag(resp.json()["artifacts"], tag)

        overwrite_tag = "tag2"
        resp = client.post(
            API_TAGS_PATH.format(project=self.project, tag=overwrite_tag),
            json={
                "objects": [
                    {
                        "kind": "artifact",
                        "identifiers": [
                            {
                                "tag": tag,
                            }
                        ],
                    }
                ]
            },
        )
        assert resp.status_code == http.HTTPStatus.OK.value

        resp = client.get(
            API_ARTIFACTS_PATH_WITH_TAG.format(project=self.project, tag=tag)
        )
        assert resp.status_code == http.HTTPStatus.OK.value
        assert len(resp.json()["artifacts"]) == 0

        resp = client.get(
            API_ARTIFACTS_PATH_WITH_TAG.format(project=self.project, tag=overwrite_tag)
        )
        assert resp.status_code == http.HTTPStatus.OK.value
        assert len(resp.json()["artifacts"]) == 2
        self._assert_tag(resp.json()["artifacts"], overwrite_tag)

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
        resp = client.post(API_PROJECTS_PATH, json=project.dict())
        assert resp.status_code == http.HTTPStatus.CREATED.value
        return resp

    def _store_artifact(
        self,
        client: fastapi.testclient.TestClient,
        project: str = None,
        uid: str = None,
        key: str = None,
        tag: str = None,
        data: str = "{}",
    ):
        uid = uid or str(uuid.uuid4())
        key = key or str(uuid.uuid4())

        resp = client.post(
            STORE_API_ARTIFACTS_PATH.format(
                project=project or self.project, uid=uid, key=key, tag=tag
            ),
            data=data,
        )
        assert resp.status_code == http.HTTPStatus.OK.value
        return resp
