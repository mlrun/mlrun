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
import typing
import unittest.mock
from http import HTTPStatus
from uuid import uuid4

import deepdiff
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.common.schemas
import server.api.api.endpoints.feature_store
import server.api.utils.auth.verifier
import tests.api.api.utils

from .base import (
    _assert_diff_as_expected_except_for_specific_metadata,
    _list_and_assert_objects,
    _list_tags_and_assert,
    _patch_object,
    _test_partition_by_for_feature_store_objects,
)


def _generate_feature_vector(name):
    return {
        "kind": "FeatureVector",
        "metadata": {
            "name": name,
            "labels": {"owner": "nobody", "group": "dev"},
            "tag": "latest",
            "extra_metadata": 100,
        },
        "spec": {
            "features": [
                "feature_set.*",
                "feature_set.something",
                "feature_set.just_a_feature",
            ],
            "description": "just a bunch of features",
            "extra_spec": True,
        },
        "status": {
            "state": "created",
            "target": "parquet",
            "extra_status": {"field1": "value1", "field2": "value2"},
        },
    }


def _create_and_assert_feature_vector(
    client: TestClient, project, feature_vector, versioned=True
):
    response = client.post(
        f"projects/{project}/feature-vectors?versioned={versioned}",
        json=feature_vector,
    )
    assert response.status_code == HTTPStatus.OK.value
    return response.json()


def test_feature_vector_create(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    tests.api.api.utils.create_project(client, project_name)

    name = "feature_set1"
    feature_vector = _generate_feature_vector(name)
    feature_vector["metadata"]["project"] = project_name
    feature_vector["metadata"].pop("tag")

    feature_vector_response = _create_and_assert_feature_vector(
        client, project_name, feature_vector, True
    )
    allowed_added_fields = ["uid", "created", "updated", "tag"]
    assert feature_vector_response["metadata"]["tag"] == "latest"

    _assert_diff_as_expected_except_for_specific_metadata(
        feature_vector, feature_vector_response, allowed_added_fields
    )
    uid = feature_vector_response["metadata"]["uid"]

    feature_vector_response = client.get(
        f"projects/{project_name}/feature-vectors/{name}/references/latest"
    )
    assert feature_vector_response.status_code == HTTPStatus.OK.value
    _assert_diff_as_expected_except_for_specific_metadata(
        feature_vector, feature_vector_response.json(), allowed_added_fields
    )

    feature_vector_response = client.get(
        f"projects/{project_name}/feature-vectors/{name}/references/{uid}"
    )
    assert feature_vector_response.status_code == HTTPStatus.OK.value
    # When querying by uid, tag will not be returned
    _assert_diff_as_expected_except_for_specific_metadata(
        feature_vector, feature_vector_response.json(), allowed_added_fields
    )


def test_list_feature_vectors(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    tests.api.api.utils.create_project(client, project_name)

    count = 10
    dead_count = 0
    blue_labels_count = 0
    ooga_name_count = 0
    not_latest_count = 0
    for i in range(count):
        name = f"feature_vector_{i}"
        feature_set = _generate_feature_vector(name)
        # generate some variations
        if i % 2 == 0:
            feature_set["status"]["state"] = "dead"
            dead_count = dead_count + 1
        if i % 3 == 0:
            feature_set["metadata"]["labels"] = {"owner": "somebody", "color": "blue"}
            blue_labels_count = blue_labels_count + 1
        if i % 4 == 0:
            feature_set["metadata"]["name"] = f"ooga_booga_{i}"
            ooga_name_count = ooga_name_count + 1
        if i > 4:
            feature_set["metadata"]["tag"] = "just_a_tag"
            not_latest_count = not_latest_count + 1
        _create_and_assert_feature_vector(client, project_name, feature_set)

    _list_and_assert_objects(client, "feature_vectors", project_name, None, count)
    _list_and_assert_objects(
        client, "feature_vectors", project_name, "name=~ooga", ooga_name_count
    )
    _list_and_assert_objects(
        client,
        "feature_vectors",
        project_name,
        "label=color=blue&label=owner",
        blue_labels_count,
    )
    _list_and_assert_objects(
        client, "feature_vectors", project_name, "label=owner", count
    )
    _list_and_assert_objects(
        client, "feature_vectors", project_name, "state=dead", dead_count
    )
    _list_and_assert_objects(
        client, "feature_vectors", project_name, "tag=just_a_tag", not_latest_count
    )
    _list_and_assert_objects(
        client,
        "feature_vectors",
        project_name,
        "state=dead&name=~booga",
        ooga_name_count,
    )
    _list_and_assert_objects(client, "feature_vectors", "wrong_project", None, 0)


def _store_feature_vector(
    client: TestClient, project, name, reference, feature_vector, versioned=True
):
    response = client.put(
        f"projects/{project}/feature-vectors/{name}/references/{reference}?versioned={versioned}",
        json=feature_vector,
    )
    assert response
    return response.json()


def test_feature_vector_store(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    tests.api.api.utils.create_project(client, project_name)

    name = "feature_vector1"
    feature_vector = _generate_feature_vector(name)

    # Put a new object - verify it's created
    response = _store_feature_vector(client, project_name, name, "tag1", feature_vector)
    assert response["metadata"]["tag"] == "tag1"
    uid = response["metadata"]["uid"]

    # Put same object using uid - should not return tag
    response = _store_feature_vector(client, project_name, name, uid, feature_vector)
    assert response["metadata"]["tag"] is None

    # Change fields that will not affect the uid, verify object is overwritten
    feature_vector["status"]["state"] = "modified"

    response = _store_feature_vector(client, project_name, name, "tag1", feature_vector)
    assert response["metadata"]["uid"] == uid
    assert response["status"]["state"] == "modified"

    # Now modify in a way that will affect uid, add a field to the metadata.
    # Since referencing the object as "latest", a new version (with new uid) should be created.
    feature_vector["metadata"]["new_metadata"] = True
    response = _store_feature_vector(
        client, project_name, name, "latest", feature_vector
    )
    modified_uid = response["metadata"]["uid"]
    assert modified_uid != uid
    assert response["metadata"]["new_metadata"] is True

    _list_and_assert_objects(client, "feature_vectors", project_name, f"name={name}", 2)

    # Do the same, but reference the object by its uid - this should fail the request
    feature_vector["metadata"]["new_metadata"] = "something else"
    response = client.put(
        f"projects/{project_name}/feature-vectors/{name}/references/{modified_uid}",
        json=feature_vector,
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST.value


def test_feature_vector_re_tag_using_store(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    tests.api.api.utils.create_project(client, project_name)

    name = "feature_vector1"
    feature_vector = _generate_feature_vector(name)

    _store_feature_vector(client, project_name, name, "tag1", feature_vector)

    _store_feature_vector(client, project_name, name, "tag2", feature_vector)

    response = _list_and_assert_objects(
        client, "feature_vectors", project_name, f"name={name}", 2
    )["feature_vectors"]
    expected_tags = {"tag1", "tag2"}
    returned_tags = set()
    for feature_vector_response in response:
        returned_tags.add(feature_vector_response["metadata"]["tag"])
    assert expected_tags == returned_tags

    # Storing object with same tag - should just update
    feature_vector["metadata"]["extra_metadata"] = 200
    _store_feature_vector(client, project_name, name, "tag2", feature_vector)

    _list_and_assert_objects(client, "feature_vectors", project_name, f"name={name}", 2)
    response = _list_and_assert_objects(
        client, "feature_vectors", project_name, f"name={name}&tag=tag2", 1
    )["feature_vectors"]
    assert response[0]["metadata"]["extra_metadata"] == 200


def test_feature_vector_patch(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    tests.api.api.utils.create_project(client, project_name)

    name = "feature_vector_1"
    feature_vector = _generate_feature_vector(name)
    _create_and_assert_feature_vector(client, project_name, feature_vector)

    # Update a feature-set
    feature_vector_patch = {
        "spec": {"extra_spec": "extra"},
        "metadata": {"labels": {"new-label": "new-value", "owner": "someone-else"}},
    }

    patched_feature_vector = _patch_object(
        client, project_name, name, feature_vector_patch, "feature-vectors"
    )
    patched_feature_vector_metadata = patched_feature_vector["metadata"]
    assert (
        # New label should be added
        len(patched_feature_vector_metadata["labels"]) == 3
        and "new-label" in patched_feature_vector_metadata["labels"]
        and patched_feature_vector_metadata["labels"]["owner"] == "someone-else"
    ), "update corrupted data - got wrong results for labels from DB after update"
    assert (
        patched_feature_vector["spec"]["extra_spec"]
        == feature_vector_patch["spec"]["extra_spec"]
    )

    # use additive strategy, the new label should be added
    feature_vector_patch = {
        "metadata": {"labels": {"another": "one"}},
    }
    patched_feature_vector = _patch_object(
        client,
        project_name,
        name,
        feature_vector_patch,
        "feature-vectors",
        additive=True,
    )
    assert len(patched_feature_vector["metadata"]["labels"]) == 4


def test_feature_vector_delete(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    tests.api.api.utils.create_project(client, project_name)

    count = 5
    for i in range(count):
        name = f"feature_vector_{i}"
        feature_vector = _generate_feature_vector(name)
        _create_and_assert_feature_vector(client, project_name, feature_vector)

    _list_and_assert_objects(client, "feature_vectors", project_name, None, count)

    # Delete the last feature vector
    response = client.delete(
        f"projects/{project_name}/feature-vectors/feature_vector_{count-1}"
    )
    assert response.status_code == HTTPStatus.NO_CONTENT.value
    _list_and_assert_objects(client, "feature_vectors", project_name, None, count - 1)

    # Delete the first feature set
    response = client.delete(
        f"projects/{project_name}/feature-vectors/feature_vector_0"
    )
    assert response.status_code == HTTPStatus.NO_CONTENT.value
    _list_and_assert_objects(client, "feature_vectors", project_name, None, count - 2)


def test_feature_vector_delete_version(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    tests.api.api.utils.create_project(client, project_name)

    name = "feature_vector"
    feature_vector = _generate_feature_vector(name)

    count = 5
    uids = {}
    for i in range(count):
        # Store different copies of the feature set with different uids and tags
        feature_vector["metadata"]["extra_metadata"] = i * 100
        tag = f"tag{i}"
        result = _store_feature_vector(
            client, project_name, name, f"tag{i}", feature_vector
        )
        uids[result["metadata"]["uid"]] = tag

    _list_and_assert_objects(
        client, "feature_vectors", project_name, f"name={name}", count
    )

    delete_by_tag = True
    objects_left = count
    for uid, tag in uids.items():
        reference = tag if delete_by_tag else uid
        delete_by_tag = not delete_by_tag

        response = client.delete(
            f"projects/{project_name}/feature-vectors/{name}/references/{reference}"
        )
        assert response.status_code == HTTPStatus.NO_CONTENT.value
        objects_left = objects_left - 1
        _list_and_assert_objects(
            client, "feature_vectors", project_name, f"name={name}", objects_left
        )

    for i in range(count):
        feature_vector["metadata"]["extra_metadata"] = i * 100
        _store_feature_vector(client, project_name, name, f"tag{i}", feature_vector)

    # Now delete by name
    response = client.delete(f"projects/{project_name}/feature-vectors/{name}")
    assert response.status_code == HTTPStatus.NO_CONTENT.value
    _list_and_assert_objects(client, "feature_vectors", project_name, f"name={name}", 0)


def test_unversioned_feature_vector_actions(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    tests.api.api.utils.create_project(client, project_name)

    name = "feature_vector1"
    feature_vector = _generate_feature_vector(name)
    feature_vector_response = _create_and_assert_feature_vector(
        client, project_name, feature_vector, versioned=False
    )

    allowed_added_fields = ["uid", "created", "updated", "tag", "project"]
    _assert_diff_as_expected_except_for_specific_metadata(
        feature_vector, feature_vector_response, allowed_added_fields
    )
    assert feature_vector_response["metadata"]["uid"] is None

    feature_vector_response = _store_feature_vector(
        client, project_name, name, "latest", feature_vector, versioned=False
    )

    _assert_diff_as_expected_except_for_specific_metadata(
        feature_vector, feature_vector_response, allowed_added_fields
    )
    assert feature_vector_response["metadata"]["uid"] is None

    feature_vector_patch = {"status": {"patched": "yes"}}
    patched_feature_vector = _patch_object(
        client,
        project_name,
        name,
        feature_vector_patch,
        "feature-vectors",
        reference="latest",
    )

    expected_diff = {"dictionary_item_added": ["root['status']['patched']"]}
    _assert_diff_as_expected_except_for_specific_metadata(
        feature_vector_response,
        patched_feature_vector,
        allowed_added_fields,
        expected_diff,
    )


def test_list_feature_vectors_tags(db: Session, client: TestClient) -> None:
    project_name = "some-project"
    tests.api.api.utils.create_project(client, project_name)

    name = "feature_vector-1"
    name_2 = "feature_vector-2"
    feature_vector_1 = _generate_feature_vector(name)
    feature_vector_2 = _generate_feature_vector(name_2)

    tags = ["tag-1", "tag-2", "tag-3", "tag-4"]
    for feature_vector in [feature_vector_1, feature_vector_2]:
        for tag in tags:
            _store_feature_vector(
                client,
                project_name,
                feature_vector["metadata"]["name"],
                tag,
                feature_vector,
            )
    _list_tags_and_assert(
        client,
        "feature_vectors",
        project_name,
        tags,
    )


def test_feature_vector_list_partition_by(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    tests.api.api.utils.create_project(client, project_name)

    count = 5
    for i in range(count):
        name = f"feature_vector_{i}"
        feature_vector = _generate_feature_vector(name)
        _store_feature_vector(client, project_name, name, "older", feature_vector)
        # Must change the uid, otherwise it will just re-tag the same object
        feature_vector["metadata"]["extra_metadata"] = 200
        _store_feature_vector(client, project_name, name, "newer", feature_vector)
        feature_vector["metadata"]["extra_metadata"] = 300
        _store_feature_vector(client, project_name, name, "newest", feature_vector)

    _test_partition_by_for_feature_store_objects(
        client, "feature_vectors", project_name, count
    )


@pytest.mark.asyncio
async def test_verify_feature_vector_features_permissions(
    db: Session, client: TestClient
) -> None:
    project = "some-project"
    features = [
        "without-project.*",
        "without-project.with-feature-name",
        "without-project.with-feature-alias as some-alias",
        "without-project:with-tag.*",
        "with-project/name.*",
        "with-project/name:and-tag.*",
        "with-project/name@and-uid.*",
        "store://feature-sets/with-project/name:and-tag.*",
        "store://feature-sets/with-project/name@and-uid.*",
        "store://feature-sets/without-project.with-feature-alias as some-alias",
    ]
    label_feature = "some-feature-set.some-feature"

    def _verify_queried_resources(
        resource_type: mlrun.common.schemas.AuthorizationResourceTypes,
        resources: list,
        project_and_resource_name_extractor: typing.Callable,
        action: mlrun.common.schemas.AuthorizationAction,
        auth_info: mlrun.common.schemas.AuthInfo,
        raise_on_forbidden: bool = True,
    ):
        expected_resources = [
            (project, "without-project"),
            ("with-project", "name"),
            (project, "some-feature-set"),
        ]
        assert (
            deepdiff.DeepDiff(
                expected_resources,
                resources,
                ignore_order=True,
            )
            == {}
        )

    server.api.utils.auth.verifier.AuthVerifier().query_project_resources_permissions = unittest.mock.AsyncMock(
        side_effect=_verify_queried_resources
    )
    await server.api.api.endpoints.feature_store._verify_feature_vector_features_permissions(
        mlrun.common.schemas.AuthInfo(),
        project,
        {"spec": {"features": features, "label_feature": label_feature}},
    )
