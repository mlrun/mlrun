from http import HTTPStatus
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from deepdiff import DeepDiff


def _generate_feature_set(name):
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
                {"name": "bid", "value_type": "float"},
                {"name": "ask", "value_type": "time"},
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
            "features": ["feature_set:*", "feature_set:something", "just_a_feature"],
            "description": "just a bunch of features",
            "extra_spec": True,
        },
        "status": {
            "state": "created",
            "target": "parquet",
            "extra_status": {"field1": "value1", "field2": "value2"},
        },
    }


def _assert_list_objects(
    client: TestClient, entity_name, project, query, expected_number_of_entities
):
    entity_url_name = entity_name.replace("_", "-")
    url = f"/api/projects/{project}/{entity_url_name}"
    if query:
        url = url + f"?{query}"
    response = client.get(url)
    assert response.status_code == HTTPStatus.OK.value
    response_body = response.json()
    assert entity_name in response_body
    assert (
        len(response_body[entity_name]) == expected_number_of_entities
    ), f"wrong number of {entity_name} entities in response"
    return response_body


def _feature_set_create_and_assert(
    client: TestClient, project, feature_set, versioned=True
):
    response = client.post(
        f"/api/projects/{project}/feature-sets?versioned={versioned}", json=feature_set
    )
    assert response.status_code == HTTPStatus.OK.value
    return response.json()


def _feature_vector_create_and_assert(
    client: TestClient, project, feature_vector, versioned=True
):
    response = client.post(
        f"/api/projects/{project}/feature-vectors?versioned={versioned}",
        json=feature_vector,
    )
    assert response.status_code == HTTPStatus.OK.value
    return response.json()


def _assert_store_feature_set(
    client: TestClient, project, name, reference, feature_set, versioned=True
):
    response = client.put(
        f"/api/projects/{project}/feature-sets/{name}/references/{reference}?versioned={versioned}",
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


def test_feature_set_create_with_extra_fields(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"

    name = "feature_set1"
    feature_set = _generate_feature_set(name)
    _feature_set_create_and_assert(client, project_name, feature_set)

    response = client.get(
        f"/api/projects/{project_name}/feature-sets/{name}/references/latest"
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

    name = "feature_set1"
    feature_set = _generate_feature_set(name)
    _feature_set_create_and_assert(client, project_name, feature_set)

    response = client.get(
        f"/api/projects/{project_name}/feature-sets/{name}/references/latest"
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

    response = _assert_list_objects(client, "feature_sets", project_name, None, 3)
    # Verify list query returns full objects, including extra fields
    for feature_set_json in response["feature_sets"]:
        _assert_extra_fields_exist(feature_set_json)

    _assert_list_objects(client, "feature_sets", project_name, "name=feature", 2)
    _assert_list_objects(client, "feature_sets", project_name, "entity=buyer", 1)
    _assert_list_objects(
        client, "feature_sets", project_name, "entity=ticker&entity=bid", 2
    )
    _assert_list_objects(
        client, "feature_sets", project_name, "name=feature&entity=buyer", 0
    )
    # Test various label filters
    _assert_list_objects(client, "feature_sets", project_name, "label=owner=saarc", 2)
    _assert_list_objects(client, "feature_sets", project_name, "label=color", 2)
    # handling multiple label queries has issues right now - needs to fix and re-run this test.
    # _assert_list_objects(client, "feature_sets", project_name, "label=owner=bob&label=color=red", 2)


def _patch_object(
    client: TestClient,
    project_name,
    name,
    object_update,
    object_url_path,
    additive=False,
):
    patch_mode = "replace"
    if additive:
        patch_mode = "additive"
    response = client.patch(
        f"/api/projects/{project_name}/{object_url_path}/{name}/references/latest?patch-mode={patch_mode}",
        json=object_update,
    )
    assert response.status_code == HTTPStatus.OK.value
    response = client.get(
        f"/api/projects/{project_name}/{object_url_path}/{name}/references/latest"
    )
    return response.json()


def test_feature_set_patch(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"

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

    name = "feature_set1"
    feature_set = _generate_feature_set(name)

    added_feature_set = _feature_set_create_and_assert(
        client, project_name, feature_set
    )
    uid = added_feature_set["metadata"]["uid"]

    response = client.get(
        f"/api/projects/{project_name}/feature-sets/{name}/references/latest"
    )
    assert response.status_code == HTTPStatus.OK.value
    assert response.json()["metadata"]["uid"] == uid

    response = client.get(
        f"/api/projects/{project_name}/feature-sets/{name}/references/{uid}"
    )
    assert response.status_code == HTTPStatus.OK.value
    assert response.json()["metadata"]["name"] == name


def test_feature_set_delete(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    count = 5
    for i in range(count):
        name = f"feature_set_{i}"
        feature_set = _generate_feature_set(name)
        _feature_set_create_and_assert(client, project_name, feature_set)

    _assert_list_objects(client, "feature_sets", project_name, None, count)

    # Delete the last fs
    response = client.delete(
        f"/api/projects/{project_name}/feature-sets/feature_set_{count-1}"
    )
    assert response.status_code == HTTPStatus.NO_CONTENT.value
    _assert_list_objects(client, "feature_sets", project_name, None, count - 1)

    # Delete the first fs
    response = client.delete(f"/api/projects/{project_name}/feature-sets/feature_set_0")
    assert response.status_code == HTTPStatus.NO_CONTENT.value
    _assert_list_objects(client, "feature_sets", project_name, None, count - 2)


def test_feature_set_create_failure_already_exists(
    db: Session, client: TestClient
) -> None:
    project_name = f"prj-{uuid4().hex}"
    name = "feature_set1"
    feature_set = _generate_feature_set(name)

    _feature_set_create_and_assert(client, project_name, feature_set, versioned=True)

    response = client.post(
        f"/api/projects/{project_name}/feature-sets?versioned=True", json=feature_set
    )
    assert response.status_code == HTTPStatus.CONFLICT.value

    # Now test not-versioned add
    name = "feature_set2"
    feature_set = _generate_feature_set(name)
    added_feature_set = _feature_set_create_and_assert(
        client, project_name, feature_set, versioned=False
    )
    uid = added_feature_set["metadata"]["uid"]
    assert uid.startswith("unversioned")

    response = client.post(
        f"/api/projects/{project_name}/feature-sets?versioned=False", json=feature_set
    )
    assert response.status_code == HTTPStatus.CONFLICT.value


def test_feature_set_multiple_creates_and_patches(
    db: Session, client: TestClient
) -> None:
    project_name = f"prj-{uuid4().hex}"
    count = 5
    for i in range(count):
        name = f"fs_{i}"
        feature_set = _generate_feature_set(name)
        _feature_set_create_and_assert(client, project_name, feature_set)

    feature_set_patch = {
        "metadata": {"labels": {"new-label": "new-value", "owner": "someone-else"}}
    }

    response = client.patch(
        f"/api/projects/{project_name}/feature-sets/{name}/references/latest",
        json=feature_set_patch,
    )
    assert response.status_code == HTTPStatus.OK.value

    response = _assert_list_objects(client, "feature_sets", project_name, None, count)
    for feature_set in response["feature_sets"]:
        if feature_set["metadata"]["name"] == name:
            labels = feature_set["metadata"]["labels"]
            assert len(labels) == 3
            assert labels["new-label"] == "new-value"
            assert labels["owner"] == "someone-else"


def test_feature_set_store(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    name = "feature_set1"
    feature_set = _generate_feature_set(name)

    # Put a new object - verify it's created
    response = _assert_store_feature_set(
        client, project_name, name, "latest", feature_set
    )
    uid = response["metadata"]["uid"]
    # Change fields that will not affect the uid, verify object is overwritten
    feature_set["status"]["state"] = "modified"

    response = _assert_store_feature_set(
        client, project_name, name, "latest", feature_set
    )
    assert response["metadata"]["uid"] == uid
    assert response["status"]["state"] == "modified"

    _assert_list_objects(client, "feature_sets", project_name, "name=feature_set1", 1)

    # Now modify in a way that will affect uid, add a field to the metadata.
    # Since referencing the object as "latest", a new version (with new uid) should be created.
    feature_set["metadata"]["new_metadata"] = True
    response = _assert_store_feature_set(
        client, project_name, name, "latest", feature_set
    )
    modified_uid = response["metadata"]["uid"]
    assert modified_uid != uid
    assert response["metadata"]["new_metadata"] is True

    _assert_list_objects(client, "feature_sets", project_name, "name=feature_set1", 2)

    # Do the same, but reference the object by its uid - this should fail the request
    feature_set["metadata"]["new_metadata"] = "something else"
    response = client.put(
        f"/api/projects/{project_name}/feature-sets/{name}/references/{modified_uid}",
        json=feature_set,
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST.value


def test_feature_set_create_without_labels(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
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
    name = "feature_set1"
    feature_set = _generate_feature_set(name)
    feature_set["metadata"]["project"] = "booboo"
    # Calling POST with a different project name in object metadata should fail
    response = client.post(
        f"/api/projects/{project_name}/feature-sets", json=feature_set
    )
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
        f"/api/projects/{project_name}/feature-sets/{name}/references/latest",
        json=feature_set,
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST.value


def test_features_list(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"

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

    _assert_list_objects(client, "features", project_name, "name=feature1", 1)
    # name is a like query, so expecting all 4 features to return
    _assert_list_objects(client, "features", project_name, "name=feature", 4)
    _assert_list_objects(client, "features", project_name, "label=owner=me", 1)

    # set a new tag
    tag = "my-new-tag"
    query = {"feature_sets": {"name": name}}
    resp = client.post(f"/api/{project_name}/tag/{tag}", json=query)
    assert resp.status_code == HTTPStatus.OK.value
    # Now expecting to get 2 objects, one with "latest" tag and one with "my-new-tag"
    features_response = _assert_list_objects(
        client, "features", project_name, "name=feature3", 2
    )
    assert (
        features_response["features"][0]["feature_set_digest"]["metadata"]["tag"]
        == "latest"
    )
    assert (
        features_response["features"][1]["feature_set_digest"]["metadata"]["tag"] == tag
    )


# There will be fields added (uid for example), but we don't allow any other changes
def _assert_diff_contains_only_additions(original_object, new_object):
    diff = DeepDiff(original_object, new_object, ignore_order=True,)
    diff.pop("dictionary_item_added")
    assert diff == {}


def test_feature_vector_create(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    name = "feature_set1"
    feature_vector = _generate_feature_vector(name)
    feature_vector["metadata"]["project"] = project_name

    feature_vector_response = _feature_vector_create_and_assert(
        client, project_name, feature_vector, True
    )
    _assert_diff_contains_only_additions(feature_vector, feature_vector_response)
    uid = feature_vector_response["metadata"]["uid"]

    feature_vector_response = client.get(
        f"/api/projects/{project_name}/feature-vectors/{name}/references/latest"
    )
    assert feature_vector_response.status_code == HTTPStatus.OK.value
    _assert_diff_contains_only_additions(feature_vector, feature_vector_response.json())

    feature_vector_response = client.get(
        f"/api/projects/{project_name}/feature-vectors/{name}/references/{uid}"
    )
    assert feature_vector_response.status_code == HTTPStatus.OK.value
    # When querying by uid, tag will not be returned
    feature_vector["metadata"].pop("tag")
    _assert_diff_contains_only_additions(feature_vector, feature_vector_response.json())


def test_list_feature_vectors(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    count = 10
    dead_count = 0
    blue_lables_count = 0
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
            blue_lables_count = blue_lables_count + 1
        if i % 4 == 0:
            feature_set["metadata"]["name"] = f"ooga_booga_{i}"
            ooga_name_count = ooga_name_count + 1
        if i > 4:
            feature_set["metadata"]["tag"] = "just_a_tag"
            not_latest_count = not_latest_count + 1
        _feature_vector_create_and_assert(client, project_name, feature_set)

    _assert_list_objects(client, "feature_vectors", project_name, None, count)
    _assert_list_objects(
        client, "feature_vectors", project_name, "name=ooga", ooga_name_count
    )
    _assert_list_objects(
        client, "feature_vectors", project_name, "label=color=blue", blue_lables_count
    )
    _assert_list_objects(client, "feature_vectors", project_name, "label=owner", count)
    _assert_list_objects(
        client, "feature_vectors", project_name, "state=dead", dead_count
    )
    _assert_list_objects(
        client, "feature_vectors", project_name, "tag=just_a_tag", not_latest_count
    )
    _assert_list_objects(
        client,
        "feature_vectors",
        project_name,
        "state=dead&name=booga",
        ooga_name_count,
    )
    _assert_list_objects(client, "feature_vectors", "wrong_project", None, 0)


def _assert_store_feature_vector(
    client: TestClient, project, name, reference, feature_vector, versioned=True
):
    response = client.put(
        f"/api/projects/{project}/feature-vectors/{name}/references/{reference}?versioned={versioned}",
        json=feature_vector,
    )
    assert response
    return response.json()


def test_feature_vector_store(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    name = "feature_vector1"
    feature_vector = _generate_feature_vector(name)

    # Put a new object - verify it's created
    response = _assert_store_feature_vector(
        client, project_name, name, "latest", feature_vector
    )
    uid = response["metadata"]["uid"]
    # Change fields that will not affect the uid, verify object is overwritten
    feature_vector["status"]["state"] = "modified"

    response = _assert_store_feature_vector(
        client, project_name, name, "latest", feature_vector
    )
    assert response["metadata"]["uid"] == uid
    assert response["status"]["state"] == "modified"

    _assert_list_objects(client, "feature_vectors", project_name, f"name={name}", 1)

    # Now modify in a way that will affect uid, add a field to the metadata.
    # Since referencing the object as "latest", a new version (with new uid) should be created.
    feature_vector["metadata"]["new_metadata"] = True
    response = _assert_store_feature_vector(
        client, project_name, name, "latest", feature_vector
    )
    modified_uid = response["metadata"]["uid"]
    assert modified_uid != uid
    assert response["metadata"]["new_metadata"] is True

    _assert_list_objects(client, "feature_vectors", project_name, f"name={name}", 2)

    # Do the same, but reference the object by its uid - this should fail the request
    feature_vector["metadata"]["new_metadata"] = "something else"
    response = client.put(
        f"/api/projects/{project_name}/feature-vectors/{name}/references/{modified_uid}",
        json=feature_vector,
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST.value


def test_feature_vector_patch(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"

    name = "feature_vector_1"
    feature_vector = _generate_feature_vector(name)
    _feature_vector_create_and_assert(client, project_name, feature_vector)

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
    count = 5
    for i in range(count):
        name = f"feature_vector_{i}"
        feature_vector = _generate_feature_vector(name)
        _feature_vector_create_and_assert(client, project_name, feature_vector)

    _assert_list_objects(client, "feature_vectors", project_name, None, count)

    # Delete the last feature vector
    response = client.delete(
        f"/api/projects/{project_name}/feature-vectors/feature_vector_{count-1}"
    )
    assert response.status_code == HTTPStatus.NO_CONTENT.value
    _assert_list_objects(client, "feature_vectors", project_name, None, count - 1)

    # Delete the first fs
    response = client.delete(
        f"/api/projects/{project_name}/feature-vectors/feature_vector_0"
    )
    assert response.status_code == HTTPStatus.NO_CONTENT.value
    _assert_list_objects(client, "feature_vectors", project_name, None, count - 2)
