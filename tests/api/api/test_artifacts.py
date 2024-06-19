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
import tempfile
import unittest.mock
import uuid
from http import HTTPStatus

import deepdiff
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.artifacts
import mlrun.common.schemas
from mlrun.utils.helpers import is_legacy_artifact

PROJECT = "prj"
KEY = "some-key"
UID = "some-uid"
TAG = "some-tag"

API_ARTIFACTS_PATH = "projects/{project}/artifacts"
STORE_API_ARTIFACTS_PATH = API_ARTIFACTS_PATH + "/{uid}/{key}?tag={tag}"
GET_API_ARTIFACT_PATH = API_ARTIFACTS_PATH + "/{key}?tag={tag}"
LIST_API_ARTIFACTS_PATH_WITH_TAG = API_ARTIFACTS_PATH + "?tag={tag}"
DELETE_API_ARTIFACTS_PATH = API_ARTIFACTS_PATH + "/{key}?tag={tag}"

# V2 endpoints
V2_PREFIX = "v2/"
STORE_API_ARTIFACTS_V2_PATH = V2_PREFIX + API_ARTIFACTS_PATH
LIST_API_ARTIFACTS_V2_PATH = V2_PREFIX + API_ARTIFACTS_PATH


def test_list_artifact_tags(db: Session, client: TestClient) -> None:
    resp = client.get(f"projects/{PROJECT}/artifact-tags")
    assert resp.status_code == HTTPStatus.OK.value, "status"
    assert resp.json()["project"] == PROJECT, "project"


def test_store_artifact_with_invalid_key(db: Session, client: TestClient):
    _create_project(client)
    key_path = "some-key/with-slash"

    # sanity, valid key path works
    resp = client.post(
        STORE_API_ARTIFACTS_PATH.format(project=PROJECT, uid=UID, key=KEY, tag=TAG),
        data="{}",
    )
    assert resp.status_code == HTTPStatus.OK.value

    # use invalid key path, expect its validation to fail
    resp = client.post(
        STORE_API_ARTIFACTS_PATH.format(
            project=PROJECT, uid=UID, key=key_path, tag=TAG
        ),
        data="{}",
    )
    assert resp.status_code == HTTPStatus.BAD_REQUEST.value


def test_store_artifact_backwards_compatibility(db: Session, client: TestClient):
    _create_project(client)
    # Invalid key name
    key_path = "some-key/with-slash"

    # Creating two artifacts with different bodies and the same invalid key
    artifact = mlrun.artifacts.Artifact(key=key_path, body="123")
    artifact2 = mlrun.artifacts.Artifact(key=key_path, body="1234")

    # Store an artifact with invalid key (by mocking the regex)
    with unittest.mock.patch(
        "mlrun.utils.helpers.verify_field_regex", return_value=True
    ):
        resp = client.post(
            STORE_API_ARTIFACTS_PATH.format(
                project=PROJECT, uid=UID, key=key_path, tag="latest"
            ),
            data=artifact.to_json(),
        )
        assert resp.status_code == HTTPStatus.OK.value

    # Ascertain that the artifact exists in the database and that it can be retrieved
    resp = client.get(API_ARTIFACTS_PATH.format(project=PROJECT))
    assert (
        resp.status_code == HTTPStatus.OK.value and len(resp.json()["artifacts"]) == 1
    ), "Expected a successful request and an existing record"

    # Make a store request to an existing artifact with an invalid key name, and ensure that it is successful
    resp = client.post(
        STORE_API_ARTIFACTS_PATH.format(
            project=PROJECT, uid=UID, key=key_path, tag="latest"
        ),
        data=artifact2.to_json(),
    )
    assert resp.status_code == HTTPStatus.OK.value

    resp = client.get(API_ARTIFACTS_PATH.format(project=PROJECT))
    assert (
        resp.status_code == HTTPStatus.OK.value and len(resp.json()["artifacts"]) == 1
    )


def test_store_artifact_with_invalid_tag(db: Session, client: TestClient):
    _create_project(client)
    tag = "test_tag_with_characters@#$#%^"

    resp = client.post(
        STORE_API_ARTIFACTS_PATH.format(project=PROJECT, uid=UID, key=KEY, tag=TAG),
        data="{}",
    )
    assert resp.status_code == HTTPStatus.OK.value

    # test overwriting tags object with an invalid tag
    resp = client.post(
        "projects/{project}/tags/{tag}".format(project=PROJECT, tag=tag),
        json={
            "kind": "artifact",
            "identifiers": [(mlrun.common.schemas.ArtifactIdentifier(key=KEY).dict())],
        },
    )

    assert resp.status_code == HTTPStatus.UNPROCESSABLE_ENTITY.value

    # test append invalid tag to artifact's tags
    resp = client.put(
        "projects/{project}/tags/{tag}".format(project=PROJECT, tag=tag),
        json={
            "kind": "artifact",
            "identifiers": [(mlrun.common.schemas.ArtifactIdentifier(key=KEY).dict())],
        },
    )
    assert resp.status_code == HTTPStatus.UNPROCESSABLE_ENTITY.value


def test_store_artifact_with_empty_dict(db: Session, client: TestClient):
    _create_project(client)

    resp = client.post(
        STORE_API_ARTIFACTS_PATH.format(project=PROJECT, uid=UID, key=KEY, tag=TAG),
        data="{}",
    )
    assert resp.status_code == HTTPStatus.OK.value

    resp = client.get(GET_API_ARTIFACT_PATH.format(project=PROJECT, key=KEY, tag=TAG))
    assert resp.status_code == HTTPStatus.OK.value


def test_create_artifact(db: Session, unversioned_client: TestClient):
    _create_project(unversioned_client, prefix="v1")
    data = {
        "kind": "artifact",
        "metadata": {
            "description": "",
            "labels": {},
            "key": "some-key",
            "project": PROJECT,
            "tree": "some-tree",
        },
        "spec": {
            "db_key": "some-key",
            "producer": {"kind": "api", "uri": "my-uri:3000"},
            "target_path": "memory://aaa/aaa",
        },
        "status": {},
    }
    url = "v2/projects/{project}/artifacts".format(project=PROJECT)
    resp = unversioned_client.post(
        url,
        json=data,
    )

    response_data = resp.json()

    assert resp.status_code == HTTPStatus.CREATED.value
    assert response_data["metadata"]["key"] == data["metadata"]["key"]
    assert response_data["metadata"]["tree"] == data["metadata"]["tree"]
    assert response_data["spec"]["target_path"] == data["spec"]["target_path"]


def test_delete_artifacts_after_storing_empty_dict(db: Session, client: TestClient):
    _create_project(client)
    empty_artifact = "{}"

    resp = client.post(
        STORE_API_ARTIFACTS_PATH.format(project=PROJECT, uid=UID, key=KEY, tag=TAG),
        data=empty_artifact,
    )
    assert resp.status_code == HTTPStatus.OK.value

    resp = client.post(
        STORE_API_ARTIFACTS_PATH.format(
            project=PROJECT, uid=f"{UID}2", key=f"{KEY}2", tag=TAG
        ),
        data=empty_artifact,
    )
    assert resp.status_code == HTTPStatus.OK.value

    resp = client.get(API_ARTIFACTS_PATH.format(project=PROJECT, key=KEY))
    assert (
        deepdiff.DeepDiff(
            [
                artifact.get("metadata", {}).get("tag", None)
                for artifact in resp.json()["artifacts"]
            ],
            ["latest", "latest", TAG, TAG],
            ignore_order=True,
        )
        == {}
    )

    resp = client.delete(API_ARTIFACTS_PATH.format(project=PROJECT, key=KEY))
    assert resp.status_code == HTTPStatus.OK.value

    resp = client.get(API_ARTIFACTS_PATH.format(project=PROJECT, key=KEY))
    assert len(resp.json()["artifacts"]) == 0


def test_list_artifacts(db: Session, client: TestClient) -> None:
    _create_project(client)

    artifact_path = API_ARTIFACTS_PATH.format(project=PROJECT)
    resp = client.get(artifact_path)
    assert resp.status_code == HTTPStatus.OK.value
    assert len(resp.json()["artifacts"]) == 0

    resp = client.post(
        STORE_API_ARTIFACTS_PATH.format(project=PROJECT, uid=UID, key=KEY, tag=TAG),
        data="{}",
    )
    assert resp.status_code == HTTPStatus.OK.value

    resp = client.post(
        STORE_API_ARTIFACTS_PATH.format(
            project=PROJECT, uid=f"{UID}2", key=f"{KEY}2", tag=TAG
        ),
        data="{}",
    )
    assert resp.status_code == HTTPStatus.OK.value

    artifact_path = API_ARTIFACTS_PATH.format(project=PROJECT)
    resp = client.get(artifact_path)
    assert resp.status_code == HTTPStatus.OK.value
    assert (
        deepdiff.DeepDiff(
            [
                artifact.get("metadata", {}).get("tag", None)
                for artifact in resp.json()["artifacts"]
            ],
            ["latest", "latest", TAG, TAG],
            ignore_order=True,
        )
        == {}
    )


def test_list_artifacts_with_limits(
    db: Session, unversioned_client: TestClient
) -> None:
    _create_project(unversioned_client, prefix="v1")

    def ensure_endpoint_limit(limit: int):
        for route in unversioned_client.app.routes:
            if route.path.endswith(LIST_API_ARTIFACTS_V2_PATH):
                for qp in route.dependant.query_params:
                    if qp.name == "limit":
                        qp.default = limit
                        break

    default_endpoint_limit = 2
    ensure_endpoint_limit(default_endpoint_limit)

    for i in range(default_endpoint_limit + 1):
        data = {
            "kind": "artifact",
            "metadata": {
                "description": "",
                "labels": {},
                "key": KEY,
                "project": PROJECT,
                "tree": str(uuid.uuid4()),
            },
            "spec": {
                "db_key": "some-key",
                "producer": {"kind": "api"},
                "target_path": "memory://aaa/aaa",
            },
            "status": {},
        }
        resp = unversioned_client.post(
            STORE_API_ARTIFACTS_V2_PATH.format(project=PROJECT),
            json=data,
        )
        assert resp.status_code == HTTPStatus.CREATED.value

    artifact_path = LIST_API_ARTIFACTS_V2_PATH.format(project=PROJECT)
    resp = unversioned_client.get(f"{artifact_path}?limit={default_endpoint_limit-1}")
    assert resp.status_code == HTTPStatus.OK.value
    artifacts = resp.json()["artifacts"]
    assert len(artifacts) == 1

    # Get all artifacts
    resp = unversioned_client.get(artifact_path)
    assert resp.status_code == HTTPStatus.OK.value
    artifacts = resp.json()["artifacts"]
    assert len(artifacts) == default_endpoint_limit

    ensure_endpoint_limit(1000)


def test_list_artifacts_with_producer_uri(
    db: Session, unversioned_client: TestClient
) -> None:
    _create_project(unversioned_client, prefix="v1")
    producer_uri_1 = f"{PROJECT}/abc"
    producer_uri_2 = f"{PROJECT}/def"
    producer_uris = [producer_uri_1, producer_uri_1, producer_uri_2, ""]
    for producer_uri in producer_uris:
        data = {
            "kind": "artifact",
            "metadata": {
                "description": "",
                "labels": {},
                "key": KEY,
                "project": PROJECT,
                "tree": str(uuid.uuid4()),
            },
            "spec": {
                "db_key": "some-key",
                "producer": {"kind": "api", "uri": producer_uri},
                "target_path": "memory://aaa/aaa",
            },
            "status": {},
        }
        resp = unversioned_client.post(
            STORE_API_ARTIFACTS_V2_PATH.format(project=PROJECT),
            json=data,
        )
        assert resp.status_code == HTTPStatus.CREATED.value

    artifact_path = LIST_API_ARTIFACTS_V2_PATH.format(project=PROJECT)
    resp = unversioned_client.get(f"{artifact_path}?producer_uri={producer_uri_1}")
    assert resp.status_code == HTTPStatus.OK.value
    artifacts = resp.json()["artifacts"]
    assert len(artifacts) == 2
    for artifact in artifacts:
        assert artifact["spec"]["producer"]["uri"] == producer_uri_1

    resp = unversioned_client.get(f"{artifact_path}?producer_uri={producer_uri_2}")
    assert resp.status_code == HTTPStatus.OK.value
    artifacts = resp.json()["artifacts"]
    assert len(artifacts) == 1
    assert artifacts[0]["spec"]["producer"]["uri"] == producer_uri_2

    # Get all artifacts
    resp = unversioned_client.get(artifact_path)
    assert resp.status_code == HTTPStatus.OK.value
    artifacts = resp.json()["artifacts"]
    assert len(artifacts) == 4


def test_list_artifacts_with_format_query(db: Session, client: TestClient) -> None:
    _create_project(client)
    artifact = mlrun.artifacts.Artifact(key=KEY, body="123")

    resp = client.post(
        STORE_API_ARTIFACTS_PATH.format(project=PROJECT, uid=UID, key=KEY, tag=TAG),
        data=artifact.to_json(),
    )
    assert resp.status_code == HTTPStatus.OK.value

    # default format is "full"
    artifact_path = API_ARTIFACTS_PATH.format(project=PROJECT)
    resp = client.get(artifact_path)
    assert resp.status_code == HTTPStatus.OK.value

    artifacts = resp.json()["artifacts"]
    assert (
        deepdiff.DeepDiff(
            [artifact["metadata"]["tag"] for artifact in resp.json()["artifacts"]],
            ["latest", TAG],
            ignore_order=True,
        )
        == {}
    )
    assert not is_legacy_artifact(artifacts[0])

    # request legacy format - expect failure (legacy format is not supported anymore)
    artifact_path = f"{API_ARTIFACTS_PATH.format(project=PROJECT)}?format=legacy"
    resp = client.get(artifact_path)
    assert resp.status_code == HTTPStatus.UNPROCESSABLE_ENTITY.value

    # explicitly request full format
    artifact_path = f"{API_ARTIFACTS_PATH.format(project=PROJECT)}?format=full"
    resp = client.get(artifact_path)
    assert resp.status_code == HTTPStatus.OK.value

    artifacts = resp.json()["artifacts"]
    assert (
        deepdiff.DeepDiff(
            [artifact["metadata"]["tag"] for artifact in resp.json()["artifacts"]],
            ["latest", TAG],
            ignore_order=True,
        )
        == {}
    )
    assert not is_legacy_artifact(artifacts[0])


def test_get_artifact_with_format_query(db: Session, client: TestClient) -> None:
    _create_project(client)
    artifact = mlrun.artifacts.Artifact(key=KEY, body="123")

    resp = client.post(
        STORE_API_ARTIFACTS_PATH.format(project=PROJECT, uid=UID, key=KEY, tag=TAG),
        data=artifact.to_json(),
    )
    assert resp.status_code == HTTPStatus.OK.value

    # default format is "full"
    artifact_path = f"{GET_API_ARTIFACT_PATH.format(project=PROJECT, key=KEY, tag=TAG)}"
    resp = client.get(artifact_path)
    assert resp.status_code == HTTPStatus.OK.value

    artifact = resp.json()
    assert not is_legacy_artifact(artifact["data"])

    # request legacy format - expect failure (legacy format is not supported anymore)
    artifact_path = f"{GET_API_ARTIFACT_PATH.format(project=PROJECT, key=KEY, tag=TAG)}&format=legacy"
    resp = client.get(artifact_path)
    assert resp.status_code == HTTPStatus.UNPROCESSABLE_ENTITY.value

    # explicitly request full format
    artifact_path = (
        f"{GET_API_ARTIFACT_PATH.format(project=PROJECT, key=KEY, tag=TAG)}&format=full"
    )
    resp = client.get(artifact_path)
    assert resp.status_code == HTTPStatus.OK.value

    artifact = resp.json()
    assert not is_legacy_artifact(artifact["data"])


def test_list_artifact_with_multiple_tags(db: Session, client: TestClient):
    _create_project(client)

    tag = "tag1"
    new_tag = "tag2"

    artifact = mlrun.artifacts.Artifact(key=KEY, body="123")
    resp = client.post(
        STORE_API_ARTIFACTS_PATH.format(project=PROJECT, uid=UID, key=KEY, tag=tag),
        data=artifact.to_json(),
    )
    assert resp.status_code == HTTPStatus.OK.value

    # tag the artifact with a new tag
    client.put(
        "projects/{project}/tags/{tag}".format(project=PROJECT, tag=new_tag),
        json={
            "kind": "artifact",
            "identifiers": [(mlrun.common.schemas.ArtifactIdentifier(key=KEY).dict())],
        },
    )

    # ensure default flow (no tag) and flow where tag is '*' produce all tags
    for artifacts_path in [
        API_ARTIFACTS_PATH.format(project=PROJECT),
        LIST_API_ARTIFACTS_PATH_WITH_TAG.format(project=PROJECT, tag="*"),
    ]:
        # list all artifacts
        resp = client.get(artifacts_path)
        assert resp.status_code == HTTPStatus.OK.value

        # expected to return three artifacts with the same key but different tags (latest, tag1, tag2)
        artifacts = resp.json()["artifacts"]
        assert len(artifacts) == 3

        # verify that the artifacts returned contains different tags
        assert (
            deepdiff.DeepDiff(
                [artifact["metadata"]["tag"] for artifact in artifacts],
                [tag, new_tag, "latest"],
                ignore_order=True,
            )
        ) == {}


def test_store_artifact_calculate_size(db: Session, client: TestClient):
    _create_project(client)

    # create a temp file with some content
    tmp_file = tempfile.NamedTemporaryFile(mode="w")
    file_path = tmp_file.name
    with open(file_path, "w") as fp:
        fp.write("1234567")
    file_length = 7

    # mock the get_allowed_path_prefixes_list function since we use a local path here for the testing
    with unittest.mock.patch(
        "server.api.api.utils.get_allowed_path_prefixes_list", return_value="/"
    ):
        # create the artifact
        artifact = mlrun.artifacts.Artifact(key=KEY, target_path=file_path)
        resp = client.post(
            STORE_API_ARTIFACTS_PATH.format(project=PROJECT, uid=UID, key=KEY, tag=TAG),
            data=artifact.to_json(),
        )
        assert resp.status_code == HTTPStatus.OK.value

    # validate that the size of the artifact is calculated
    resp = client.get(GET_API_ARTIFACT_PATH.format(project=PROJECT, key=KEY, tag=TAG))
    assert resp.status_code == HTTPStatus.OK.value
    artifact = resp.json()

    assert artifact["data"]["spec"]["size"] == file_length


def test_legacy_get_artifact_with_tree_as_tag_fallback(
    db: Session, client: TestClient
) -> None:
    _create_project(client)
    artifact = mlrun.artifacts.Artifact(key=KEY, body="123")

    # store artifact with tree as tag, which was referred to as uid in the legacy API
    tree = "my-tree"
    resp = client.post(
        STORE_API_ARTIFACTS_PATH.format(project=PROJECT, uid=tree, key=KEY, tag=TAG),
        data=artifact.to_json(),
    )
    assert resp.status_code == HTTPStatus.OK.value

    # get the artifact with the tree as tag, and expect it to be returned properly,
    # due to the fallback in the legacy API
    artifact_path = GET_API_ARTIFACT_PATH.format(project=PROJECT, key=KEY, tag=tree)
    resp = client.get(artifact_path)
    assert resp.status_code == HTTPStatus.OK.value

    artifact = resp.json()
    assert not is_legacy_artifact(artifact["data"])
    assert artifact["data"]["metadata"]["key"] == KEY
    assert artifact["data"]["metadata"]["tree"] == tree


def test_legacy_list_artifact_with_tree_as_tag_fallback(
    db: Session, client: TestClient
) -> None:
    _create_project(client)
    artifact = mlrun.artifacts.Artifact(key=KEY, body="123")

    # store artifact with tree as tag, which was referred to as uid in the legacy API
    tree = "my-tree"
    resp = client.post(
        STORE_API_ARTIFACTS_PATH.format(project=PROJECT, uid=tree, key=KEY, tag=TAG),
        data=artifact.to_json(),
    )
    assert resp.status_code == HTTPStatus.OK.value

    # list the artifacts with the tree as tag, and expect it to be returned properly,
    # due to the fallback in the legacy API
    artifact_path = LIST_API_ARTIFACTS_PATH_WITH_TAG.format(project=PROJECT, tag=tree)
    resp = client.get(artifact_path)
    assert resp.status_code == HTTPStatus.OK.value

    artifact = resp.json()["artifacts"][0]
    assert not is_legacy_artifact(artifact)
    assert artifact["metadata"]["key"] == KEY
    assert artifact["metadata"]["tree"] == tree


def _create_project(
    client: TestClient, project_name: str = PROJECT, prefix: str = None
):
    project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=project_name),
        spec=mlrun.common.schemas.ProjectSpec(
            description="banana", source="source", goals="some goals"
        ),
    )
    url = "projects" if prefix is None else f"{prefix}/projects"
    resp = client.post(url, json=project.dict())
    assert resp.status_code == HTTPStatus.CREATED.value
    return resp
