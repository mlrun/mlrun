from http import HTTPStatus
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session


def _generate_feature_set(name):
    return {
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
        },
        "something_else": {"field1": "value1", "field2": "value2"},
    }


def _assert_list_objects(
    client: TestClient, entity_name, project, query, expected_number_of_entities
):
    url = f"/api/projects/{project}/{entity_name}"
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


def _assert_add_feature_set(client: TestClient, project, feature_set, versioned=True):
    response = client.post(
        f"/api/projects/{project}/feature_sets?versioned={versioned}", json=feature_set
    )
    assert response.status_code == HTTPStatus.OK.value
    return response.json()


def _assert_put_feature_set(
    client: TestClient, project, name, reference, feature_set, versioned=True
):
    response = client.put(
        f"/api/projects/{project}/feature_sets/{name}/references/{reference}?versioned={versioned}",
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
    assert json_response["something_else"]["field1"] == "value1"
    assert json_response["something_else"]["field2"] == "value2"


def test_feature_set_create_extra_fields(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"

    name = "feature_set1"
    feature_set = _generate_feature_set(name)
    _assert_add_feature_set(client, project_name, feature_set)

    response = client.get(
        f"/api/projects/{project_name}/feature_sets/{name}/references/latest"
    )
    assert response.status_code == HTTPStatus.OK.value
    json_response = response.json()
    _assert_extra_fields_exist(json_response)


def test_feature_set_create_and_list(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"

    name = "feature_set1"
    feature_set = _generate_feature_set(name)
    _assert_add_feature_set(client, project_name, feature_set)

    response = client.get(
        f"/api/projects/{project_name}/feature_sets/{name}/references/latest"
    )
    assert response.status_code == HTTPStatus.OK.value

    name = "feature_set2"
    feature_set = _generate_feature_set(name)
    feature_set["metadata"]["labels"]["color"] = "red"
    _assert_add_feature_set(client, project_name, feature_set)

    name = "feat_3"
    feature_set = _generate_feature_set(name)
    feature_set["spec"]["entities"] = [
        {"name": "buyer", "value_type": "str", "extra_entity_field": "here"}
    ]
    feature_set["metadata"]["labels"]["owner"] = "bob"
    feature_set["metadata"]["labels"]["color"] = "blue"
    _assert_add_feature_set(client, project_name, feature_set)

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


def _perform_patch(
    client: TestClient, project_name, name, feature_set_update, additive=False
):
    response = client.patch(
        f"/api/projects/{project_name}/feature_sets/{name}/references/latest?additive={additive}",
        json=feature_set_update,
    )
    assert response.status_code == HTTPStatus.OK.value
    response = client.get(
        f"/api/projects/{project_name}/feature_sets/{name}/references/latest"
    )
    return response.json()


def test_feature_set_updates(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"

    name = "feature_set1"
    feature_set = _generate_feature_set(name)
    _assert_add_feature_set(client, project_name, feature_set)

    # Update a feature-set
    feature_set_update = {
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
        "something_else": {"field3": "new_value3"},
    }

    updated_resp = _perform_patch(client, project_name, name, feature_set_update)
    feature_set_metadata = updated_resp["metadata"]
    assert (
        # New label should be added
        len(feature_set_metadata["labels"]) == 3
        and "new-label" in feature_set_metadata["labels"]
        and feature_set_metadata["labels"]["owner"] == "someone-else"
    ), "update corrupted data - got wrong results for labels from DB after update"
    feature_set_spec = updated_resp["spec"]
    # Since entities is a list, entity will be replaced, so there should be only one.
    assert feature_set_spec["entities"] == feature_set_update["spec"]["entities"]

    # update with no labels, ensure labels are not deleted
    feature_set_update = {
        "spec": {"features": [{"name": "dividend", "value_type": "float"}]}
    }
    updated_resp = _perform_patch(client, project_name, name, feature_set_update)
    feature_set_resp = updated_resp["metadata"]
    assert (
        len(feature_set_resp["labels"]) == 3
        and "new-label" in feature_set_resp["labels"]
        and feature_set_resp["labels"]["owner"] == "someone-else"
    ), "update corrupted data - got wrong results for labels from DB after update"

    # use additive strategy, the new feature should be added
    feature_set_update = {
        "spec": {
            "features": [{"name": "looks", "value_type": "str", "description": "good"}],
        }
    }
    updated_resp = _perform_patch(
        client, project_name, name, feature_set_update, additive=True
    )
    assert len(updated_resp["spec"]["features"]) == 2


def test_get_feature_set_by_reference(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"

    name = "feature_set1"
    feature_set = _generate_feature_set(name)

    added_feature_set = _assert_add_feature_set(client, project_name, feature_set)
    uid = added_feature_set["metadata"]["uid"]

    response = client.get(
        f"/api/projects/{project_name}/feature_sets/{name}/references/latest"
    )
    assert response.status_code == HTTPStatus.OK.value
    assert response.json()["metadata"]["uid"] == uid

    response = client.get(
        f"/api/projects/{project_name}/feature_sets/{name}/references/{uid}"
    )
    assert response.status_code == HTTPStatus.OK.value
    assert response.json()["metadata"]["name"] == name


def test_delete_feature_set(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    count = 5
    for i in range(count):
        name = f"feature_set_{i}"
        feature_set = _generate_feature_set(name)
        _assert_add_feature_set(client, project_name, feature_set)

    _assert_list_objects(client, "feature_sets", project_name, None, count)

    # Delete the last fs
    response = client.delete(
        f"/api/projects/{project_name}/feature_sets/feature_set_{count-1}"
    )
    assert response.status_code == HTTPStatus.NO_CONTENT.value
    _assert_list_objects(client, "feature_sets", project_name, None, count - 1)

    # Delete the first fs
    response = client.delete(f"/api/projects/{project_name}/feature_sets/feature_set_0")
    assert response.status_code == HTTPStatus.NO_CONTENT.value
    _assert_list_objects(client, "feature_sets", project_name, None, count - 2)


def test_add_multiple_times_same_uid(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    name = "feature_set1"
    feature_set = _generate_feature_set(name)

    _assert_add_feature_set(client, project_name, feature_set, versioned=True)

    response = client.post(
        f"/api/projects/{project_name}/feature_sets?versioned=True", json=feature_set
    )
    assert response.status_code != HTTPStatus.OK.value

    # Now test not-versioned add
    name = "feature_set2"
    feature_set = _generate_feature_set(name)
    added_feature_set = _assert_add_feature_set(
        client, project_name, feature_set, versioned=False
    )
    uid = added_feature_set["metadata"]["uid"]
    assert uid.startswith("unversioned")

    response = client.post(
        f"/api/projects/{project_name}/feature_sets?versioned=False", json=feature_set
    )
    assert response.status_code != HTTPStatus.OK.value


def test_multi_inserts_and_updates(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    count = 5
    for i in range(count):
        name = f"fs_{i}"
        feature_set = _generate_feature_set(name)
        _assert_add_feature_set(client, project_name, feature_set)

    feature_set_update = {
        "metadata": {"labels": {"new-label": "new-value", "owner": "someone-else"}}
    }

    response = client.patch(
        f"/api/projects/{project_name}/feature_sets/{name}/references/latest",
        json=feature_set_update,
    )
    assert response.status_code == HTTPStatus.OK.value

    response = _assert_list_objects(client, "feature_sets", project_name, None, count)
    for feature_set in response["feature_sets"]:
        if feature_set["metadata"]["name"] == name:
            labels = feature_set["metadata"]["labels"]
            assert len(labels) == 3
            assert labels["new-label"] == "new-value"
            assert labels["owner"] == "someone-else"


def test_put_feature_sets(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"
    name = "feature_set1"
    feature_set = _generate_feature_set(name)

    # Put a new object - verify it's created
    response = _assert_put_feature_set(
        client, project_name, name, "latest", feature_set
    )
    uid = response["metadata"]["uid"]
    # Change fields that will not affect the uid, verify object is overwritten
    feature_set["status"]["state"] = "modified"

    response = _assert_put_feature_set(
        client, project_name, name, "latest", feature_set
    )
    assert response["metadata"]["uid"] == uid
    assert response["status"]["state"] == "modified"

    _assert_list_objects(client, "feature_sets", project_name, "name=feature_set1", 1)

    # Now modify in a way that will affect uid, add a field to the metadata
    feature_set["metadata"]["new_metadata"] = True
    response = _assert_put_feature_set(
        client, project_name, name, "latest", feature_set
    )
    assert response["metadata"]["uid"] != uid
    assert response["metadata"]["new_metadata"] is True

    _assert_list_objects(client, "feature_sets", project_name, "name=feature_set1", 2)


def test_list_features(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"

    name = "feature_set1"
    feature_set = _generate_feature_set(name)
    feature_set["spec"]["features"] = [
        {"name": "feature1", "value_type": "str"},
        {"name": "feature2", "value_type": "float"},
    ]
    _assert_add_feature_set(client, project_name, feature_set)
    name = "feature_set2"
    feature_set = _generate_feature_set(name)
    feature_set["spec"]["features"] = [
        {"name": "feature3", "value_type": "bool", "labels": {"owner": "me"}},
        {"name": "feature4", "value_type": "datetime", "labels": {"color": "red"}},
    ]
    _assert_add_feature_set(client, project_name, feature_set)

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
