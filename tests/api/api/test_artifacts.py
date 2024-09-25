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
from datetime import datetime, timedelta
from http import HTTPStatus

import deepdiff
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.artifacts
import mlrun.common.schemas
import mlrun.utils
import server.api.db.sqldb.models
from mlrun.common.constants import MYSQL_MEDIUMBLOB_SIZE_BYTES

PROJECT = "prj"
KEY = "some-key"
UID = "some-uid"
TAG = "some-tag"

API_ARTIFACTS_PATH = "projects/{project}/artifacts"
STORE_API_ARTIFACTS_PATH = API_ARTIFACTS_PATH + "/{uid}/{key}?tag={tag}"
GET_API_ARTIFACT_PATH = API_ARTIFACTS_PATH + "/{key}?tag={tag}"
LIST_API_ARTIFACTS_PATH_WITH_TAG = API_ARTIFACTS_PATH + "?tag={tag}"
DELETE_API_ARTIFACTS_PATH = API_ARTIFACTS_PATH + "/{key}"

# V2 endpoints
V2_PREFIX = "v2/"
DELETE_API_ARTIFACTS_V2_PATH = V2_PREFIX + DELETE_API_ARTIFACTS_PATH
STORE_API_ARTIFACTS_V2_PATH = V2_PREFIX + API_ARTIFACTS_PATH
LIST_API_ARTIFACTS_V2_PATH = V2_PREFIX + API_ARTIFACTS_PATH
GET_API_ARTIFACT_V2_PATH = V2_PREFIX + API_ARTIFACTS_PATH + "/{key}"


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
        f"projects/{PROJECT}/tags/{tag}",
        json={
            "kind": "artifact",
            "identifiers": [(mlrun.common.schemas.ArtifactIdentifier(key=KEY).dict())],
        },
    )

    assert resp.status_code == HTTPStatus.UNPROCESSABLE_ENTITY.value

    # test append invalid tag to artifact's tags
    resp = client.put(
        f"projects/{PROJECT}/tags/{tag}",
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


def test_store_artifact_with_iteration(db: Session, unversioned_client: TestClient):
    _create_project(unversioned_client)
    iteration = 3
    json = _generate_artifact_body(iteration=iteration)
    resp = unversioned_client.put(
        STORE_API_ARTIFACTS_V2_PATH.format(project=PROJECT) + f"/{KEY}?tag={TAG}",
        json=json,
    )
    assert resp.status_code == HTTPStatus.OK.value

    # Change a spec that is not included in UID hash
    json["metadata"]["labels"]["a"] = "b"
    resp = unversioned_client.put(
        STORE_API_ARTIFACTS_V2_PATH.format(project=PROJECT) + f"/{KEY}?tag={TAG}",
        json=json,
    )
    assert resp.status_code == HTTPStatus.OK.value
    artifact_dict = resp.json()
    assert artifact_dict["metadata"]["labels"]["a"] == json["metadata"]["labels"]["a"]
    assert artifact_dict["metadata"]["iter"] == iteration

    artifacts_path = (
        LIST_API_ARTIFACTS_V2_PATH.format(project=PROJECT) + f"?iter={iteration}"
    )
    resp = unversioned_client.get(artifacts_path)
    assert resp.status_code == HTTPStatus.OK.value
    artifacts = resp.json()["artifacts"]
    assert len(artifacts) == 2  # latest and TAG
    assert artifacts[0]["metadata"]["iter"] == iteration


def test_create_artifact(db: Session, unversioned_client: TestClient):
    _create_project(unversioned_client, prefix="v1")
    data = _generate_artifact_body(tree="some-tree")
    url = V2_PREFIX + API_ARTIFACTS_PATH.format(project=PROJECT)
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


@pytest.mark.parametrize(
    "deletion_strategy, expected_status_code",
    [
        (
            mlrun.common.schemas.artifact.ArtifactsDeletionStrategies.data_optional,
            HTTPStatus.NO_CONTENT.value,
        ),
        (
            mlrun.common.schemas.artifact.ArtifactsDeletionStrategies.data_force,
            HTTPStatus.INTERNAL_SERVER_ERROR.value,
        ),
    ],
)
def test_fails_deleting_artifact_data(
    deletion_strategy, expected_status_code, db: Session, unversioned_client: TestClient
):
    # This test attempts to delete the artifact data, but fails - the request should
    # be failed or succeeded by the deletion strategy.
    _create_project(unversioned_client)
    artifact = mlrun.artifacts.Artifact(key=KEY, body="123", target_path="dummy-path")

    resp = unversioned_client.post(
        STORE_API_ARTIFACTS_PATH.format(project=PROJECT, uid=UID, key=KEY, tag=TAG),
        data=artifact.to_json(),
    )
    assert resp.status_code == HTTPStatus.OK.value

    url = DELETE_API_ARTIFACTS_V2_PATH.format(project=PROJECT, key=KEY)
    url_with_deletion_strategy = url + "?deletion_strategy={deletion_strategy}"

    with unittest.mock.patch(
        "server.api.crud.files.Files.delete_artifact_data",
        side_effect=mlrun.errors.MLRunInternalServerError("some error"),
    ):
        resp = unversioned_client.delete(
            url_with_deletion_strategy.format(deletion_strategy=deletion_strategy)
        )
    assert resp.status_code == expected_status_code


def test_delete_artifact_data_default_deletion_strategy(
    db: Session, unversioned_client: TestClient
):
    with unittest.mock.patch(
        "server.api.crud.Files.delete_artifact_data"
    ) as delete_artifact_data:
        # checking metadata-only as default deletion_strategy
        url = DELETE_API_ARTIFACTS_V2_PATH.format(project=PROJECT, key=KEY)
        resp = unversioned_client.delete(url)
        delete_artifact_data.assert_not_called()
        delete_artifact_data.reset_mock()
        assert resp.status_code == HTTPStatus.NO_CONTENT.value


def test_delete_artifact_with_uid(db: Session, unversioned_client: TestClient):
    _create_project(unversioned_client)

    # create an artifact
    data = _generate_artifact_body()
    resp = unversioned_client.post(
        STORE_API_ARTIFACTS_V2_PATH.format(project=PROJECT),
        json=data,
    )
    assert resp.status_code == HTTPStatus.CREATED.value

    # get the artifact to extract the created uid
    artifacts_path = LIST_API_ARTIFACTS_V2_PATH.format(project=PROJECT)
    resp = unversioned_client.get(artifacts_path)
    assert resp.status_code == HTTPStatus.OK.value
    artifacts = resp.json()["artifacts"]
    assert len(artifacts) == 1

    # delete the artifact by uid
    artifact_uid = artifacts[0]["metadata"]["uid"]
    url = DELETE_API_ARTIFACTS_V2_PATH.format(project=PROJECT, key=KEY)
    url_with_uid = url + f"?object-uid={artifact_uid}"
    resp = unversioned_client.delete(url_with_uid)
    assert resp.status_code == HTTPStatus.NO_CONTENT.value

    # verify the artifact was deleted
    resp = unversioned_client.get(artifacts_path)
    assert resp.status_code == HTTPStatus.OK.value
    artifacts = resp.json()["artifacts"]
    assert len(artifacts) == 0


@pytest.mark.parametrize(
    "artifact_kind",
    [
        mlrun.artifacts.DatasetArtifact,
        mlrun.artifacts.ModelArtifact,
        mlrun.artifacts.DirArtifact,
    ],
)
def test_fails_deleting_artifact_data_by_artifact_kind(
    artifact_kind, db: Session, unversioned_client: TestClient
):
    _create_project(unversioned_client)
    artifact = artifact_kind(key=KEY, body="123", target_path="dummy-path")

    resp = unversioned_client.post(
        STORE_API_ARTIFACTS_PATH.format(project=PROJECT, uid=UID, key=KEY, tag=TAG),
        data=artifact.to_json(),
    )
    assert resp.status_code == HTTPStatus.OK.value

    url = DELETE_API_ARTIFACTS_V2_PATH.format(project=PROJECT, key=KEY)
    url_with_deletion_strategy = url + "?deletion_strategy={deletion_strategy}"

    resp = unversioned_client.delete(
        url_with_deletion_strategy.format(
            deletion_strategy=mlrun.common.schemas.artifact.ArtifactsDeletionStrategies.data_force
        )
    )
    assert resp.status_code == HTTPStatus.NOT_IMPLEMENTED.value


@pytest.mark.parametrize(
    "target_path",
    [
        "dummy-path.parquet",
        "dummy-path.pq",
    ],
)
def test_deleting_dataset_artifact_data_includes_one_file(
    target_path, db: Session, unversioned_client: TestClient
):
    _create_project(unversioned_client)
    artifact = mlrun.artifacts.DatasetArtifact(
        key=KEY, body="123", target_path=target_path
    )

    resp = unversioned_client.post(
        STORE_API_ARTIFACTS_PATH.format(project=PROJECT, uid=UID, key=KEY, tag=TAG),
        data=artifact.to_json(),
    )
    assert resp.status_code == HTTPStatus.OK.value

    url = DELETE_API_ARTIFACTS_V2_PATH.format(project=PROJECT, key=KEY)
    url_with_deletion_strategy = url + "?deletion_strategy={deletion_strategy}"

    with unittest.mock.patch(
        "server.api.crud.files.Files.delete_artifact_data",
    ):
        resp = unversioned_client.delete(
            url_with_deletion_strategy.format(
                deletion_strategy=mlrun.common.schemas.artifact.ArtifactsDeletionStrategies.data_force
            )
        )
    assert resp.status_code == HTTPStatus.NO_CONTENT.value


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


@pytest.fixture
def list_limit_unversioned_client(
    unversioned_client: TestClient, request
) -> TestClient:
    def ensure_endpoint_limit(limit_: int = None):
        for route in unversioned_client.app.routes:
            if route.path.endswith(LIST_API_ARTIFACTS_V2_PATH):
                for qp in route.dependant.query_params:
                    if qp.name == "limit":
                        qp.default = limit_
                        break

    try:
        ensure_endpoint_limit(request.param)
        yield request.param, unversioned_client
    finally:
        ensure_endpoint_limit(None)


@pytest.mark.parametrize("list_limit_unversioned_client", [2], indirect=True)
def test_list_artifacts_with_limits(
    db: Session, list_limit_unversioned_client: TestClient
) -> None:
    list_limit, unversioned_client = list_limit_unversioned_client
    _create_project(unversioned_client, prefix="v1")

    for i in range(list_limit + 1):
        data = _generate_artifact_body()
        resp = unversioned_client.post(
            STORE_API_ARTIFACTS_V2_PATH.format(project=PROJECT),
            json=data,
        )
        assert resp.status_code == HTTPStatus.CREATED.value

    artifact_path = LIST_API_ARTIFACTS_V2_PATH.format(project=PROJECT)
    resp = unversioned_client.get(f"{artifact_path}?limit={list_limit-1}")
    assert resp.status_code == HTTPStatus.OK.value
    artifacts = resp.json()["artifacts"]
    assert len(artifacts) == list_limit - 1

    # Get all artifacts
    resp = unversioned_client.get(artifact_path)
    assert resp.status_code == HTTPStatus.OK.value
    artifacts = resp.json()["artifacts"]
    assert len(artifacts) == list_limit


def test_list_artifacts_with_producer_uri(
    db: Session, unversioned_client: TestClient
) -> None:
    _create_project(unversioned_client, prefix="v1")
    producer_uri_1 = f"{PROJECT}/abc"
    producer_uri_2 = f"{PROJECT}/def"
    producer_uris = [producer_uri_1, producer_uri_1, producer_uri_2, ""]
    for producer_uri in producer_uris:
        data = _generate_artifact_body(producer={"kind": "api", "uri": producer_uri})
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
    artifact = mlrun.artifacts.Artifact(key=KEY, body="123", src_path="some-path")

    resp = client.post(
        STORE_API_ARTIFACTS_PATH.format(project=PROJECT, uid=UID, key=KEY, tag=TAG),
        data=artifact.to_json(),
    )
    assert resp.status_code == HTTPStatus.OK.value

    # default format is "full"
    artifact_path = API_ARTIFACTS_PATH.format(project=PROJECT)
    resp = client.get(artifact_path)
    assert resp.status_code == HTTPStatus.OK.value

    assert (
        deepdiff.DeepDiff(
            [artifact["metadata"]["tag"] for artifact in resp.json()["artifacts"]],
            ["latest", TAG],
            ignore_order=True,
        )
        == {}
    )

    # request legacy format - expect failure (legacy format is not supported anymore)
    artifact_path = f"{API_ARTIFACTS_PATH.format(project=PROJECT)}?format=legacy"
    resp = client.get(artifact_path)
    assert resp.status_code == HTTPStatus.BAD_REQUEST.value

    # test request minimal format
    artifact_path = f"{API_ARTIFACTS_PATH.format(project=PROJECT)}?format=minimal"
    resp = client.get(artifact_path)
    assert resp.status_code == HTTPStatus.OK.value

    assert all(
        ["src_path" not in artifact["spec"] for artifact in resp.json()["artifacts"]]
    )

    # explicitly request full format
    artifact_path = f"{API_ARTIFACTS_PATH.format(project=PROJECT)}?format=full"
    resp = client.get(artifact_path)
    assert resp.status_code == HTTPStatus.OK.value

    assert (
        deepdiff.DeepDiff(
            [artifact["metadata"]["tag"] for artifact in resp.json()["artifacts"]],
            ["latest", TAG],
            ignore_order=True,
        )
        == {}
    )


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

    # request legacy format - expect failure (legacy format is not supported anymore)
    artifact_path = f"{GET_API_ARTIFACT_PATH.format(project=PROJECT, key=KEY, tag=TAG)}&format=legacy"
    resp = client.get(artifact_path)
    assert resp.status_code == HTTPStatus.BAD_REQUEST.value

    # test minimal format
    artifact_path = f"{GET_API_ARTIFACT_PATH.format(project=PROJECT, key=KEY, tag=TAG)}&format=minimal"
    resp = client.get(artifact_path)
    assert resp.status_code == HTTPStatus.OK.value

    assert "src_path" not in resp.json()["data"]["spec"]

    # explicitly request full format
    artifact_path = (
        f"{GET_API_ARTIFACT_PATH.format(project=PROJECT, key=KEY, tag=TAG)}&format=full"
    )
    resp = client.get(artifact_path)
    assert resp.status_code == HTTPStatus.OK.value


def test_get_artifact_validate_tag_exists_in_the_response(
    db: Session, unversioned_client: TestClient
) -> None:
    _create_project(unversioned_client)

    # Create artifact with tag "v1"
    artifact_data = _generate_artifact_body(tag="v1")
    resp = unversioned_client.post(
        STORE_API_ARTIFACTS_V2_PATH.format(project=PROJECT),
        json=artifact_data,
    )
    assert resp.status_code == HTTPStatus.CREATED.value
    artifact_v1 = resp.json()

    # Get artifact using UID and tag "v1"
    url_with_uid_and_tag_v1 = _get_artifact_url(
        artifact_v1["metadata"]["uid"], tag="v1"
    )
    resp = unversioned_client.get(url_with_uid_and_tag_v1)
    assert resp.status_code == HTTPStatus.OK.value
    artifact = resp.json()
    assert artifact["metadata"]["tag"] == "v1"

    # Get the same artifact using UID without specifying a tag
    url_with_uid = _get_artifact_url(artifact_v1["metadata"]["uid"])
    resp = unversioned_client.get(url_with_uid)
    assert resp.status_code == HTTPStatus.OK.value
    artifact = resp.json()
    assert artifact["metadata"].get("tag") is None

    # Get the same artifact using UID and tag "latest"
    url_with_uid_and_latest = _get_artifact_url(
        artifact_v1["metadata"]["uid"], tag="latest"
    )
    resp = unversioned_client.get(url_with_uid_and_latest)
    assert resp.status_code == HTTPStatus.OK.value
    artifact = resp.json()
    assert artifact["metadata"]["tag"] == "latest"

    # Get the same artifact using tag "latest" without UID
    url_tag_latest = _get_artifact_url(tag="latest")
    resp = unversioned_client.get(url_tag_latest)
    assert resp.status_code == HTTPStatus.OK.value
    artifact = resp.json()
    assert artifact["metadata"]["tag"] == "latest"
    assert artifact["metadata"]["uid"] == artifact_v1["metadata"]["uid"]

    # Create another artifact with tag "v2" -> now this artifact is the latest
    artifact_data = _generate_artifact_body(tag="v2")
    resp = unversioned_client.post(
        STORE_API_ARTIFACTS_V2_PATH.format(project=PROJECT),
        json=artifact_data,
    )
    assert resp.status_code == HTTPStatus.CREATED.value
    artifact_v2 = resp.json()

    # Get the second artifact using tag "latest" without UID
    url_tag_latest = _get_artifact_url(tag="latest")
    resp = unversioned_client.get(url_tag_latest)
    assert resp.status_code == HTTPStatus.OK.value
    artifact = resp.json()
    assert artifact["metadata"]["tag"] == "latest"
    assert artifact["metadata"]["uid"] == artifact_v2["metadata"]["uid"]

    # Get the first artifact (v1) using UID and tag "latest"
    url_with_uid_tag_latest = _get_artifact_url(
        uid=artifact_v1["metadata"]["uid"], tag="latest"
    )
    resp = unversioned_client.get(url_with_uid_tag_latest)
    assert resp.status_code == HTTPStatus.OK.value
    artifact = resp.json()
    assert artifact["metadata"].get("tag") is None


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
        f"projects/{PROJECT}/tags/{new_tag}",
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
    assert artifact["metadata"]["key"] == KEY
    assert artifact["metadata"]["tree"] == tree


@pytest.mark.parametrize(
    "body_size,is_inline,body_char,expected_status_code",
    [
        # Body size exceeds limit, expect 400
        (
            MYSQL_MEDIUMBLOB_SIZE_BYTES + 1,
            True,
            "a",
            HTTPStatus.BAD_REQUEST.value,
        ),
        # Body size within limit, expect 200
        (
            MYSQL_MEDIUMBLOB_SIZE_BYTES - 1,
            True,
            "a",
            HTTPStatus.OK.value,
        ),
        # Not inline artifact, expect 200
        (
            MYSQL_MEDIUMBLOB_SIZE_BYTES + 1,
            False,
            "a",
            HTTPStatus.OK.value,
        ),
        # Bytes artifact, expect 400
        (
            MYSQL_MEDIUMBLOB_SIZE_BYTES + 1,
            True,
            b"\x86",
            HTTPStatus.BAD_REQUEST.value,
        ),
    ],
)
def test_store_oversized_artifact(
    db: Session,
    client: TestClient,
    body_size,
    is_inline,
    body_char,
    expected_status_code,
) -> None:
    _create_project(client)
    artifact = mlrun.artifacts.Artifact(
        key=KEY, body=body_char * body_size, is_inline=is_inline
    )
    resp = client.post(
        STORE_API_ARTIFACTS_PATH.format(project=PROJECT, uid=UID, key=KEY, tag=TAG),
        data=artifact.to_json(),
    )

    assert resp.status_code == expected_status_code


def test_list_artifacts_with_time_filters(db: Session, unversioned_client: TestClient):
    _create_project(unversioned_client, prefix="v1")
    t1 = datetime(2020, 2, 16)
    t2 = t1 + timedelta(days=7)
    t3 = t2 + timedelta(days=7)
    start = datetime.now()

    key1 = "key1"
    key2 = "key2"
    key3 = "key3"
    key4 = "key4"
    old_artifact_record = server.api.db.sqldb.models.ArtifactV2(
        key=key1,
        project=PROJECT,
        created=t1,
        updated=t1,
        full_object={
            "metadata": {
                "key": key1,
            }
        },
    )
    recent_artifact_record = server.api.db.sqldb.models.ArtifactV2(
        key=key2,
        project=PROJECT,
        created=t2,
        updated=t2,
        full_object={
            "metadata": {
                "key": key2,
            }
        },
    )
    new_artifact_record = server.api.db.sqldb.models.ArtifactV2(
        key=key3,
        project=PROJECT,
        created=start,
        updated=start,
        full_object={
            "metadata": {
                "key": key3,
            }
        },
    )
    recently_updated_artifact_record = server.api.db.sqldb.models.ArtifactV2(
        key=key4,
        project=PROJECT,
        created=t2,
        updated=t3,
        full_object={
            "metadata": {
                "key": key4,
            }
        },
    )
    for artifact in [
        old_artifact_record,
        recent_artifact_record,
        new_artifact_record,
        recently_updated_artifact_record,
    ]:
        db.add(artifact)
        db.commit()

    artifact_path = LIST_API_ARTIFACTS_V2_PATH.format(project=PROJECT)
    resp = unversioned_client.get(artifact_path)
    assert resp.status_code == HTTPStatus.OK.value
    artifacts = resp.json()["artifacts"]
    assert len(artifacts) == 4

    artifact_path = LIST_API_ARTIFACTS_V2_PATH.format(project=PROJECT)
    resp = unversioned_client.get(
        artifact_path, params={"since": mlrun.utils.datetime_to_iso(t2)}
    )
    assert resp.status_code == HTTPStatus.OK.value
    artifacts = resp.json()["artifacts"]
    assert len(artifacts) == 3, "since t2 filter did not return 3 artifacts"
    artifact_keys = [artifact["metadata"]["key"] for artifact in artifacts]
    assert (
        artifact_keys.sort() == [key2, key3, key4].sort()
    ), "since t2 filter returned the wrong artifacts"

    artifact_path = LIST_API_ARTIFACTS_V2_PATH.format(project=PROJECT)
    resp = unversioned_client.get(
        artifact_path,
        params={
            "since": mlrun.utils.datetime_to_iso(t2),
            "until": mlrun.utils.datetime_to_iso(t3),
        },
    )
    assert resp.status_code == HTTPStatus.OK.value
    artifacts = resp.json()["artifacts"]
    assert len(artifacts) == 2, "since t2 until t3 filter did not return 2 artifacts"
    artifact_keys = [artifact["metadata"]["key"] for artifact in artifacts]
    assert (
        artifact_keys.sort() == [key2, key4].sort()
    ), "since t2 until t3 filter returned the wrong artifacts"

    artifact_path = LIST_API_ARTIFACTS_V2_PATH.format(project=PROJECT)
    resp = unversioned_client.get(
        artifact_path,
        params={
            "since": mlrun.utils.datetime_to_iso(t3),
            "until": mlrun.utils.datetime_to_iso(start),
        },
    )
    assert resp.status_code == HTTPStatus.OK.value
    artifacts = resp.json()["artifacts"]
    assert len(artifacts) == 2, "since t3 until start filter did not return 2 artifacts"
    artifact_keys = [artifact["metadata"]["key"] for artifact in artifacts]
    assert (
        artifact_keys.sort() == [key3, key4].sort()
    ), "since t3 until start filter returned the wrong artifacts"

    artifact_path = LIST_API_ARTIFACTS_V2_PATH.format(project=PROJECT)
    resp = unversioned_client.get(
        artifact_path, params={"since": mlrun.utils.datetime_to_iso(start)}
    )
    assert resp.status_code == HTTPStatus.OK.value
    artifacts = resp.json()["artifacts"]
    assert len(artifacts) == 1, "since start filter did not return 1 artifacts"
    artifact_keys = [artifact["metadata"]["key"] for artifact in artifacts]
    assert (
        artifact_keys.sort() == [key4].sort()
    ), "since start filter returned the wrong artifacts"

    artifact_path = LIST_API_ARTIFACTS_V2_PATH.format(project=PROJECT)
    resp = unversioned_client.get(
        artifact_path, params={"until": mlrun.utils.datetime_to_iso(start)}
    )
    assert resp.status_code == HTTPStatus.OK.value
    artifacts = resp.json()["artifacts"]
    assert len(artifacts) == 4, "until start filter did not return 4 artifacts"
    artifact_keys = [artifact["metadata"]["key"] for artifact in artifacts]
    assert (
        artifact_keys.sort() == [key1, key2, key3, key4].sort()
    ), "until start filter returned the wrong artifacts"

    artifact_path = LIST_API_ARTIFACTS_V2_PATH.format(project=PROJECT)
    resp = unversioned_client.get(
        artifact_path, params={"since": mlrun.utils.datetime_to_iso(datetime.now())}
    )
    assert resp.status_code == HTTPStatus.OK.value
    artifacts = resp.json()["artifacts"]
    assert len(artifacts) == 0, "since now filter returned artifacts unexpectedly"


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


def _generate_artifact_body(
    key=KEY,
    project=PROJECT,
    tree=None,
    tag=None,
    body=None,
    producer=None,
    iteration=None,
):
    tree = tree or str(uuid.uuid4())
    producer = producer or {"kind": "api", "uri": "my-uri:3000"}
    data = {
        "kind": "artifact",
        "metadata": {
            "description": "",
            "labels": {},
            "key": key,
            "project": project,
            "tree": tree,
        },
        "spec": {
            "db_key": key,
            "producer": producer,
            "target_path": "memory://aaa/aaa",
        },
        "status": {},
    }
    if tag:
        data["metadata"]["tag"] = tag
    if iteration is not None:
        data["metadata"]["iter"] = iteration
    if body:
        data["spec"] = {"body": body}

    return data


def _get_artifact_url(uid: str = None, tag: str = None) -> str:
    url = GET_API_ARTIFACT_V2_PATH.format(project=PROJECT, key=KEY)
    params = []

    if uid:
        params.append(f"object-uid={uid}")
    if tag:
        params.append(f"tag={tag}")

    return f"{url}?{'&'.join(params)}" if params else url
