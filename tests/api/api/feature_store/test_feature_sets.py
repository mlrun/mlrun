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
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import tests.api.api.utils

from .base import (
    _assert_diff_as_expected_except_for_specific_metadata,
    _list_and_assert_objects,
    _list_tags_and_assert,
    _patch_object,
    _test_partition_by_for_feature_store_objects,
)


def _generate_feature_set(name, extra_feature_name="extra"):
    return {
        "kind": "FeatureSet",
        "metadata": {
            "name": name,
            "labels": {"owner": "saarc", "group": "dev"},
            "tag": "latest",
            "extra_metadata": 100,
        },
        "spec": {
            "entities": [
                {
                    "name": "ticker",
                    "value_type": "str",
                    "labels": {"label1": "value1"},
                    "extra_entity_field": "here",
                }
            ],
            "features": [
                {
                    "name": "time",
                    "value_type": "datetime",
                    "labels": {"label2": "value2"},
                    "extra_feature_field": "there",
                },
                {"name": "bid", "value_type": "float", "labels": {"label3": "value3"}},
                {"name": "ask", "value_type": "time", "labels": {"label4": "value4"}},
                {
                    "name": extra_feature_name,
                    "value_type": "str",
                    "labels": {"extra_label": "extra"},
                },
            ],
            "extra_spec": True,
        },
        "status": {
            "state": "created",
            "stats": {
                "time": {
                    "count": "8",
                    "unique": "7",
                    "top": "2016-05-25 13:30:00.222222",
                }
            },
            "extra_status": {"field1": "value1", "field2": "value2"},
        },
    }


def _feature_set_create_and_assert(
    client: TestClient, project, feature_set, versioned=True
):
    response = client.post(
        f"projects/{project}/feature-sets?versioned={versioned}", json=feature_set
    )
    assert response.status_code == HTTPStatus.OK.value
    return response.json()


def _store_and_assert_feature_set(
    client: TestClient, project, name, reference, feature_set, versioned=True
):
    response = client.put(
        f"projects/{project}/feature-sets/{name}/references/{reference}?versioned={versioned}",
        json=feature_set,
    )
    assert response
    return response.json()


def _assert_extra_fields_exist(json_response):
    # Make sure we get all the out-of-schema fields properly
    assert json_response["metadata"]["extra_metadata"] == 100
    assert json_response["spec"]["entities"][0]["extra_entity_field"] == "here"
    assert json_response["spec"]["features"][0]["extra_feature_field"] == "there"
    assert json_response["spec"]["extra_spec"] is True
    assert json_response["status"]["extra_status"]["field1"] == "value1"
    assert json_response["status"]["extra_status"]["field2"] == "value2"


def test_feature_set_put_with_tag(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    tests.api.api.utils.create_project(client, project_name)

    name = "feature_set1"
    tag = "my_tag1"
    feature_set = _generate_feature_set(name)
    feature_set["metadata"]["tag"] = tag

    result = _store_and_assert_feature_set(client, project_name, name, tag, feature_set)
    assert result["metadata"]["tag"] == tag
    uid = result["metadata"]["uid"]

    result = _store_and_assert_feature_set(client, project_name, name, uid, feature_set)
    assert result["metadata"]["tag"] is None


def test_feature_set_create_without_tag(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    tests.api.api.utils.create_project(client, project_name)

    name = "feature_set1"
    feature_set = _generate_feature_set(name)
    feature_set["metadata"].pop("tag")

    response = _feature_set_create_and_assert(client, project_name, feature_set)
    assert response["metadata"]["tag"] == "latest"


def test_feature_set_create_with_extra_fields(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    tests.api.api.utils.create_project(client, project_name)

    name = "feature_set1"
    feature_set = _generate_feature_set(name)
    _feature_set_create_and_assert(client, project_name, feature_set)

    response = client.get(
        f"projects/{project_name}/feature-sets/{name}/references/latest"
    )
    assert response.status_code == HTTPStatus.OK.value
    json_response = response.json()
    _assert_extra_fields_exist(json_response)

    # Make sure extra fields outside of the metadata/spec/status/kind fields are not stored
    feature_set = _generate_feature_set("feature_set2")
    feature_set["something_else"] = {"extra_field": "extra_value"}

    response = _feature_set_create_and_assert(client, project_name, feature_set)
    assert (
        len(response) == 4 and "kind" in response and "something_else" not in response
    )


def test_feature_set_create_and_list(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    tests.api.api.utils.create_project(client, project_name)

    name = "feature_set1"
    feature_set = _generate_feature_set(name)
    _feature_set_create_and_assert(client, project_name, feature_set)

    response = client.get(
        f"projects/{project_name}/feature-sets/{name}/references/latest"
    )
    assert response.status_code == HTTPStatus.OK.value

    name = "feature_set2"
    feature_set = _generate_feature_set(name)
    feature_set["metadata"]["labels"]["color"] = "red"
    _feature_set_create_and_assert(client, project_name, feature_set)

    name = "feat_3"
    feature_set = _generate_feature_set(name)
    feature_set["spec"]["entities"] = [
        {"name": "buyer", "value_type": "str", "extra_entity_field": "here"}
    ]
    feature_set["metadata"]["labels"]["owner"] = "bob"
    feature_set["metadata"]["labels"]["color"] = "blue"
    _feature_set_create_and_assert(client, project_name, feature_set)

    response = _list_and_assert_objects(client, "feature_sets", project_name, None, 3)
    # Verify list query returns full objects, including extra fields
    for feature_set_json in response["feature_sets"]:
        _assert_extra_fields_exist(feature_set_json)

    _list_and_assert_objects(client, "feature_sets", project_name, "name=~feature", 2)
    _list_and_assert_objects(client, "feature_sets", project_name, "entity=buyer", 1)
    _list_and_assert_objects(
        client, "feature_sets", project_name, "entity=ticker&entity=bid", 2
    )
    _list_and_assert_objects(
        client, "feature_sets", project_name, "name=~feature&entity=buyer", 0
    )
    # Test various label filters
    _list_and_assert_objects(
        client, "feature_sets", project_name, "label=owner=saarc", 2
    )
    _list_and_assert_objects(client, "feature_sets", project_name, "label=color", 2)
    # handling multiple label queries has issues right now - needs to fix and re-run this test.
    # _assert_list_objects(client, "feature_sets", project_name, "label=owner=bob&label=color=red", 2)


def test_feature_set_list_partition_by(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    tests.api.api.utils.create_project(client, project_name)

    count = 5
    for i in range(count):
        name = f"feature_set_{i}"
        feature_set = _generate_feature_set(name)
        _store_and_assert_feature_set(client, project_name, name, "older", feature_set)
        # Must change the uid, otherwise it will just re-tag the same object
        feature_set["metadata"]["extra_metadata"] = 200
        _store_and_assert_feature_set(client, project_name, name, "newer", feature_set)
        feature_set["metadata"]["extra_metadata"] = 300
        _store_and_assert_feature_set(client, project_name, name, "newest", feature_set)

    _test_partition_by_for_feature_store_objects(
        client, "feature_sets", project_name, count
    )


def test_feature_set_patch(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    tests.api.api.utils.create_project(client, project_name)

    name = "feature_set1"
    feature_set = _generate_feature_set(name)
    _feature_set_create_and_assert(client, project_name, feature_set)

    # Update a feature-set
    feature_set_patch = {
        "spec": {
            "entities": [
                {
                    "name": "market_cap",
                    "value_type": "integer",
                    "labels": {},
                    "extra_field": "val1",
                }
            ]
        },
        "metadata": {"labels": {"new-label": "new-value", "owner": "someone-else"}},
    }

    patched_feature_set = _patch_object(
        client, project_name, name, feature_set_patch, "feature-sets"
    )
    patched_feature_set_metadata = patched_feature_set["metadata"]
    assert (
        # New label should be added
        len(patched_feature_set_metadata["labels"]) == 3
        and "new-label" in patched_feature_set_metadata["labels"]
        and patched_feature_set_metadata["labels"]["owner"] == "someone-else"
    ), "update corrupted data - got wrong results for labels from DB after update"
    patched_feature_set_spec = patched_feature_set["spec"]
    # Since entities is a list, entity will be replaced, so there should be only one.
    assert patched_feature_set_spec["entities"] == feature_set_patch["spec"]["entities"]

    # update with no labels, ensure labels are not deleted
    feature_set_patch = {
        "spec": {"features": [{"name": "dividend", "value_type": "float"}]}
    }
    patched_feature_set = _patch_object(
        client, project_name, name, feature_set_patch, "feature-sets"
    )
    patched_feature_set_metadata = patched_feature_set["metadata"]
    assert (
        len(patched_feature_set_metadata["labels"]) == 3
        and "new-label" in patched_feature_set_metadata["labels"]
        and patched_feature_set_metadata["labels"]["owner"] == "someone-else"
    ), "patch corrupted data - got wrong results for labels from DB after update"

    # use additive strategy, the new feature should be added
    feature_set_patch = {
        "spec": {
            "features": [{"name": "looks", "value_type": "str", "description": "good"}],
        }
    }
    patched_feature_set = _patch_object(
        client, project_name, name, feature_set_patch, "feature-sets", additive=True
    )
    assert len(patched_feature_set["spec"]["features"]) == 2


def test_feature_set_get_by_reference(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    tests.api.api.utils.create_project(client, project_name)

    name = "feature_set1"
    feature_set = _generate_feature_set(name)

    added_feature_set = _feature_set_create_and_assert(
        client, project_name, feature_set
    )
    uid = added_feature_set["metadata"]["uid"]

    response = client.get(
        f"projects/{project_name}/feature-sets/{name}/references/latest"
    )
    assert response.status_code == HTTPStatus.OK.value
    assert response.json()["metadata"]["uid"] == uid

    response = client.get(
        f"projects/{project_name}/feature-sets/{name}/references/{uid}"
    )
    assert response.status_code == HTTPStatus.OK.value
    assert response.json()["metadata"]["name"] == name


def test_feature_set_delete(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    tests.api.api.utils.create_project(client, project_name)

    count = 5
    for i in range(count):
        name = f"feature_set_{i}"
        feature_set = _generate_feature_set(name)
        _feature_set_create_and_assert(client, project_name, feature_set)

    _list_and_assert_objects(client, "feature_sets", project_name, None, count)

    # Delete the last feature set
    response = client.delete(
        f"projects/{project_name}/feature-sets/feature_set_{count-1}"
    )
    assert response.status_code == HTTPStatus.NO_CONTENT.value
    _list_and_assert_objects(client, "feature_sets", project_name, None, count - 1)

    # Delete the first feature set
    response = client.delete(f"projects/{project_name}/feature-sets/feature_set_0")
    assert response.status_code == HTTPStatus.NO_CONTENT.value
    _list_and_assert_objects(client, "feature_sets", project_name, None, count - 2)


def test_feature_set_delete_version(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    tests.api.api.utils.create_project(client, project_name)

    name = "feature_set"
    feature_set = _generate_feature_set(name)

    count = 5
    uids = {}
    for i in range(count):
        # Store different copies of the feature set with different uids and tags
        feature_set["metadata"]["extra_metadata"] = i * 100
        tag = f"tag{i}"
        result = _store_and_assert_feature_set(
            client, project_name, name, f"tag{i}", feature_set
        )
        uids[result["metadata"]["uid"]] = tag

    _list_and_assert_objects(
        client, "feature_sets", project_name, f"name={name}", count
    )

    delete_by_tag = True
    objects_left = count
    for uid, tag in uids.items():
        reference = tag if delete_by_tag else uid
        delete_by_tag = not delete_by_tag

        response = client.delete(
            f"projects/{project_name}/feature-sets/{name}/references/{reference}"
        )
        assert response.status_code == HTTPStatus.NO_CONTENT.value
        objects_left = objects_left - 1
        _list_and_assert_objects(
            client, "feature_sets", project_name, f"name={name}", objects_left
        )

    for i in range(count):
        feature_set["metadata"]["extra_metadata"] = i * 100
        _store_and_assert_feature_set(
            client, project_name, name, f"tag{i}", feature_set
        )

    # Now delete by name
    response = client.delete(f"projects/{project_name}/feature-sets/{name}")
    assert response.status_code == HTTPStatus.NO_CONTENT.value
    _list_and_assert_objects(client, "feature_sets", project_name, f"name={name}", 0)


def test_feature_set_create_failure_already_exists(
    db: Session, client: TestClient
) -> None:
    project_name = f"prj-{uuid4().hex}"
    tests.api.api.utils.create_project(client, project_name)

    name = "feature_set1"
    feature_set = _generate_feature_set(name)

    _feature_set_create_and_assert(client, project_name, feature_set, versioned=True)

    response = client.post(
        f"projects/{project_name}/feature-sets?versioned=True", json=feature_set
    )
    assert response.status_code == HTTPStatus.CONFLICT.value

    # Now test not-versioned add
    name = "feature_set2"
    feature_set = _generate_feature_set(name)
    added_feature_set = _feature_set_create_and_assert(
        client, project_name, feature_set, versioned=False
    )
    assert added_feature_set["metadata"]["uid"] is None

    response = client.post(
        f"projects/{project_name}/feature-sets?versioned=False", json=feature_set
    )
    assert response.status_code == HTTPStatus.CONFLICT.value


def test_feature_set_multiple_creates_and_patches(
    db: Session, client: TestClient
) -> None:
    project_name = f"prj-{uuid4().hex}"
    tests.api.api.utils.create_project(client, project_name)

    count = 5
    for i in range(count):
        name = f"fs_{i}"
        feature_set = _generate_feature_set(name)
        _feature_set_create_and_assert(client, project_name, feature_set)

    feature_set_patch = {
        "metadata": {"labels": {"new-label": "new-value", "owner": "someone-else"}}
    }

    response = client.patch(
        f"projects/{project_name}/feature-sets/{name}/references/latest",
        json=feature_set_patch,
    )
    assert response.status_code == HTTPStatus.OK.value

    response = _list_and_assert_objects(
        client, "feature_sets", project_name, None, count
    )
    for feature_set in response["feature_sets"]:
        if feature_set["metadata"]["name"] == name:
            labels = feature_set["metadata"]["labels"]
            assert len(labels) == 3
            assert labels["new-label"] == "new-value"
            assert labels["owner"] == "someone-else"


def test_feature_set_store(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    tests.api.api.utils.create_project(client, project_name)

    name = "feature_set1"
    feature_set = _generate_feature_set(name)

    # Put a new object - verify it's created
    response = _store_and_assert_feature_set(
        client, project_name, name, "latest", feature_set
    )
    uid = response["metadata"]["uid"]
    # Change fields that will not affect the uid, verify object is overwritten
    feature_set["status"]["state"] = "modified"

    response = _store_and_assert_feature_set(
        client, project_name, name, "latest", feature_set
    )
    assert response["metadata"]["uid"] == uid
    assert response["status"]["state"] == "modified"

    _list_and_assert_objects(
        client, "feature_sets", project_name, "name=feature_set1", 1
    )

    # Now modify in a way that will affect uid, add a field to the metadata.
    # Since referencing the object as "latest", a new version (with new uid) should be created.
    feature_set["metadata"]["new_metadata"] = True
    response = _store_and_assert_feature_set(
        client, project_name, name, "latest", feature_set
    )
    modified_uid = response["metadata"]["uid"]
    assert modified_uid != uid
    assert response["metadata"]["new_metadata"] is True

    _list_and_assert_objects(
        client, "feature_sets", project_name, "name=feature_set1", 2
    )

    # Do the same, but reference the object by its uid - this should fail the request
    feature_set["metadata"]["new_metadata"] = "something else"
    response = client.put(
        f"projects/{project_name}/feature-sets/{name}/references/{modified_uid}",
        json=feature_set,
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST.value


def test_feature_set_tagging_with_re_store(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    tests.api.api.utils.create_project(client, project_name)

    name = "feature_set1"
    feature_set = _generate_feature_set(name)

    # Put a new object - verify it's created
    response = _store_and_assert_feature_set(
        client, project_name, name, "tag1", feature_set
    )
    uid = response["metadata"]["uid"]

    # Put the same object with a different tag - this should result in just adding a tag
    response = _store_and_assert_feature_set(
        client, project_name, name, "tag2", feature_set
    )
    assert response["metadata"]["uid"] == uid

    response = _list_and_assert_objects(
        client, "feature_sets", project_name, f"name={name}", 2
    )["feature_sets"]

    expected_tags = {"tag1", "tag2"}
    returned_tags = set()
    for feature_set_response in response:
        returned_tags.add(feature_set_response["metadata"]["tag"])
    assert expected_tags == returned_tags

    # Storing object with same tag - should just update
    feature_set["metadata"]["extra_metadata"] = 200
    _store_and_assert_feature_set(client, project_name, name, "tag2", feature_set)

    _list_and_assert_objects(client, "feature_sets", project_name, f"name={name}", 2)

    response = _list_and_assert_objects(
        client, "feature_sets", project_name, f"name={name}&tag=tag2", 1
    )["feature_sets"]
    assert response[0]["metadata"]["extra_metadata"] == 200


def test_list_feature_sets_tags(db: Session, client: TestClient) -> None:
    project_name = "some-project"
    tests.api.api.utils.create_project(client, project_name)

    name = "feature_set-1"
    name_2 = "feature_set-2"
    feature_set_1 = _generate_feature_set(name)
    feature_set_2 = _generate_feature_set(name_2)

    tags = ["tag-1", "tag-2", "tag-3", "tag-4"]
    for feature_set in [feature_set_1, feature_set_2]:
        for tag in tags:
            _store_and_assert_feature_set(
                client, project_name, feature_set["metadata"]["name"], tag, feature_set
            )
    _list_tags_and_assert(
        client,
        "feature_sets",
        project_name,
        tags,
    )


def test_feature_set_create_without_labels(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    tests.api.api.utils.create_project(client, project_name)

    name = "feature_set1"
    feature_set = _generate_feature_set(name)

    feature_set["metadata"].pop("labels")
    _feature_set_create_and_assert(client, project_name, feature_set)

    feature_set_update = {
        "metadata": {"labels": {"label1": "value1", "label2": "value2"}}
    }
    feature_set_response = _patch_object(
        client, project_name, name, feature_set_update, "feature-sets"
    )
    assert (
        len(feature_set_response["metadata"]["labels"]) == 2
    ), "Labels didn't get updated"


def test_feature_set_project_name_mismatch_failure(
    db: Session, client: TestClient
) -> None:
    project_name = f"prj-{uuid4().hex}"
    tests.api.api.utils.create_project(client, project_name)

    name = "feature_set1"
    feature_set = _generate_feature_set(name)
    feature_set["metadata"]["project"] = "booboo"
    # Calling POST with a different project name in object metadata should fail
    response = client.post(f"projects/{project_name}/feature-sets", json=feature_set)
    assert response.status_code == HTTPStatus.BAD_REQUEST.value

    # When POSTing without project name, project name should be implanted in the response
    feature_set["metadata"].pop("project")
    feature_set_response = _feature_set_create_and_assert(
        client, project_name, feature_set
    )
    assert feature_set_response["metadata"]["project"] == project_name

    feature_set["metadata"]["project"] = "woohoo"
    # Calling PUT with a different project name in object metadata should fail
    response = client.put(
        f"projects/{project_name}/feature-sets/{name}/references/latest",
        json=feature_set,
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST.value


def test_feature_set_wrong_kind_failure(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    tests.api.api.utils.create_project(client, project_name)

    name = "feature_set1"
    feature_set = _generate_feature_set(name)
    feature_set["kind"] = "wrong"
    response = client.post(f"projects/{project_name}/feature-sets", json=feature_set)
    assert response.status_code != HTTPStatus.OK.value


def test_entities_list(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    tests.api.api.utils.create_project(client, project_name)

    name = "feature_set"
    count = 5
    colors = ["red", "blue"]
    for idx in range(count):
        feature_set = _generate_feature_set(f"{name}_{idx}")
        feature_set["spec"]["entities"] = [
            {
                "name": f"entity_{idx}",
                "value_type": "str",
                "labels": {"color": colors[idx % 2], "id": f"id_{idx}"},
            },
        ]

        _feature_set_create_and_assert(client, project_name, feature_set)
    _list_and_assert_objects(client, "entities", project_name, "name=entity_0", 1)
    _list_and_assert_objects(client, "entities", project_name, "name=~entity", count)
    _list_and_assert_objects(client, "entities", project_name, "label=color", count)
    _list_and_assert_objects(
        client, "entities", project_name, f"label=color={colors[1]}", count // 2
    )
    _list_and_assert_objects(
        client, "entities", project_name, "name=~entity&label=id=id_0", 1
    )

    # set a new tag
    tag = "my-new-tag"
    feature_set["metadata"]["tag"] = tag
    _store_and_assert_feature_set(
        client, project_name, f"{name}_{idx}", tag, feature_set
    )
    # Now expecting to get 2 objects, one with "latest" tag and one with "my-new-tag"
    entities_response = _list_and_assert_objects(
        client, "entities", project_name, f"name=entity_{idx}", 2
    )
    assert (
        entities_response["entities"][0]["feature_set_digest"]["metadata"]["tag"]
        == "latest"
    )
    assert (
        entities_response["entities"][1]["feature_set_digest"]["metadata"]["tag"] == tag
    )


def test_features_list(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    tests.api.api.utils.create_project(client, project_name)

    name = "feature_set1"
    feature_set = _generate_feature_set(name)
    feature_set["spec"]["features"] = [
        {"name": "feature1", "value_type": "str"},
        {"name": "feature2", "value_type": "float"},
    ]
    _feature_set_create_and_assert(client, project_name, feature_set)
    name = "feature_set2"
    feature_set = _generate_feature_set(name)
    feature_set["spec"]["features"] = [
        {"name": "feature3", "value_type": "bool", "labels": {"owner": "me"}},
        {"name": "feature4", "value_type": "datetime", "labels": {"color": "red"}},
    ]
    _feature_set_create_and_assert(client, project_name, feature_set)

    _list_and_assert_objects(client, "features", project_name, "name=feature1", 1)
    # name is a like query, so expecting all 4 features to return
    _list_and_assert_objects(client, "features", project_name, "name=~feature", 4)
    _list_and_assert_objects(client, "features", project_name, "label=owner=me", 1)

    # set a new tag
    tag = "my-new-tag"
    feature_set["metadata"]["tag"] = tag
    _store_and_assert_feature_set(client, project_name, name, tag, feature_set)
    # Now expecting to get 2 objects, one with "latest" tag and one with "my-new-tag"
    features_response = _list_and_assert_objects(
        client, "features", project_name, "name=feature3", 2
    )
    assert (
        features_response["features"][0]["feature_set_digest"]["metadata"]["tag"]
        == "latest"
    )
    assert (
        features_response["features"][1]["feature_set_digest"]["metadata"]["tag"] == tag
    )


def test_no_feature_leftovers_when_storing_feature_sets(
    db: Session, client: TestClient
) -> None:
    project_name = f"prj-{uuid4().hex}"
    tests.api.api.utils.create_project(client, project_name)

    count = 5
    name = "feature_set"
    # Make sure no leftover features remain in the DB after doing multi-store on the same object
    for i in range(count):
        feature_set = _generate_feature_set(name)

        _store_and_assert_feature_set(
            client, project_name, name, "latest", feature_set, versioned=False
        )
        _list_and_assert_objects(
            client, "features", project_name, None, len(feature_set["spec"]["features"])
        )

    # Now create different features each time we store, make sure no leftovers remain
    for i in range(count):
        feature_set = _generate_feature_set(name, f"feature_{i}")
        _store_and_assert_feature_set(
            client, project_name, name, "latest", feature_set, versioned=False
        )
        _list_and_assert_objects(
            client, "features", project_name, None, len(feature_set["spec"]["features"])
        )

    response = client.delete(f"projects/{project_name}/feature-sets/{name}")
    assert response.status_code == HTTPStatus.NO_CONTENT.value

    # When working on a versioned object, features will be multiplied, since they belong to different versions
    # (different features change the uid)
    expected_number_of_features = 0
    for i in range(count):
        feature_set = _generate_feature_set(name, f"feature_{i}")
        _store_and_assert_feature_set(
            client, project_name, name, "latest", feature_set, versioned=True
        )
        expected_number_of_features = expected_number_of_features + len(
            feature_set["spec"]["features"]
        )
        _list_and_assert_objects(
            client, "features", project_name, None, expected_number_of_features
        )


def test_unversioned_feature_set_actions(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    tests.api.api.utils.create_project(client, project_name)

    name = "feature_set_1"
    feature_set = _generate_feature_set(name)
    feature_set_response = _feature_set_create_and_assert(
        client, project_name, feature_set, versioned=False
    )

    allowed_added_fields = ["created", "updated", "tag", "uid", "project"]
    _assert_diff_as_expected_except_for_specific_metadata(
        feature_set, feature_set_response, allowed_added_fields
    )
    assert feature_set_response["metadata"]["uid"] is None

    feature_set_patch = {"status": {"patched": "yes"}}
    # Since the function calls both PATCH and GET on the same reference, it tests both cases
    patched_feature_set = _patch_object(
        client,
        project_name,
        name,
        feature_set_patch,
        "feature-sets",
        reference=feature_set_response["metadata"]["tag"],
    )

    expected_diff = {"dictionary_item_added": ["root['status']['patched']"]}
    _assert_diff_as_expected_except_for_specific_metadata(
        feature_set_response, patched_feature_set, allowed_added_fields, expected_diff
    )
    assert patched_feature_set["metadata"]["uid"] is None

    # Now attempt to PUT the object again
    feature_set_response = _store_and_assert_feature_set(
        client, project_name, name, "latest", feature_set, versioned=False
    )

    _assert_diff_as_expected_except_for_specific_metadata(
        feature_set, feature_set_response, allowed_added_fields
    )
    assert feature_set_response["metadata"]["uid"] is None

    # Verify we still have just 1 object in the DB
    _list_and_assert_objects(client, "feature_sets", project_name, f"name={name}", 1)


def test_feature_set_name_exact_and_fuzzy_list(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    tests.api.api.utils.create_project(client, project_name)

    name = "FeatureSET123"
    feature_set = _generate_feature_set(name)
    _feature_set_create_and_assert(client, project_name, feature_set)
    _list_and_assert_objects(client, "feature_sets", project_name, f"name={name}", 1)
    _list_and_assert_objects(
        client, "feature_sets", project_name, f"name={name.lower()}", 0
    )
    _list_and_assert_objects(
        client, "feature_sets", project_name, f"name=~{name.lower()}", 1
    )
    _list_and_assert_objects(client, "feature_sets", project_name, "name=~set", 1)
    _list_and_assert_objects(client, "feature_sets", project_name, "name=~SET", 1)
    _list_and_assert_objects(client, "feature_sets", project_name, "name=set", 0)
    _list_and_assert_objects(client, "feature_sets", project_name, "name=SET", 0)


def test_multi_label_query(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    tests.api.api.utils.create_project(client, project_name)

    total_objects = 5
    for i in range(total_objects):
        name = f"feature_set_{i}"
        feature_set = _generate_feature_set(name)
        feature_set["metadata"]["labels"]["serial_number"] = str(i)
        feature_set["metadata"]["labels"]["another"] = "label"
        _feature_set_create_and_assert(
            client, project_name, feature_set, versioned=False
        )

    _list_and_assert_objects(
        client,
        "feature_sets",
        project_name,
        "label=owner=saarc&label=group=dev",
        total_objects,
    )

    for i in range(total_objects):
        _list_and_assert_objects(
            client, "feature_sets", project_name, f"label=serial_number={i}", 1
        )
        _list_and_assert_objects(
            client,
            "feature_sets",
            project_name,
            f"label=owner=saarc&label=another&label=serial_number={i}",
            1,
        )

    _list_and_assert_objects(
        client,
        "feature_sets",
        project_name,
        "label=owner=saarc&label=another",
        total_objects,
    )

    _list_and_assert_objects(
        client,
        "feature_sets",
        project_name,
        "label=owner&label=owner=saarc",
        total_objects,
    )
    _list_and_assert_objects(
        client,
        "feature_sets",
        project_name,
        "label=serial_number=0&label=serial_number=1",
        0,
    )
