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
        },
        "spec": {
            "entities": [{"name": "ticker", "value_type": "str"}],
            "features": [
                {"name": "time", "value_type": "datetime"},
                {"name": "bid", "value_type": "float"},
                {"name": "ask", "value_type": "time"},
            ],
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
    assert entity_name in response_body, "no feature sets"
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
    feature_set["spec"]["entities"] = [{"name": "buyer", "value_type": "str"}]
    feature_set["metadata"]["labels"]["owner"] = "bob"
    feature_set["metadata"]["labels"]["color"] = "blue"
    _assert_add_feature_set(client, project_name, feature_set)

    _assert_list_objects(client, "feature_sets", project_name, None, 3)
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

    # Update a feature-set
    feature_set_update = {
        "entities": [{"name": "market_cap", "value_type": "integer"}],
        "labels": {"new-label": "new-value", "owner": "someone-else"},
    }
    response = client.put(
        f"/api/projects/{project_name}/feature_sets/{name}/references/latest",
        json=feature_set_update,
    )
    assert response.status_code == HTTPStatus.OK.value
    response = client.get(
        f"/api/projects/{project_name}/feature_sets/{name}/references/latest"
    )
    assert response.status_code == HTTPStatus.OK.value
    updated_resp = response.json()
    feature_set_resp = updated_resp["metadata"]
    assert (
        len(feature_set_resp["labels"]) == 2
        and "new-label" in feature_set_resp["labels"]
    ), "update corrupted data - got wrong number of labels from get after update"


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
        client, "features", project_name, "name=feature3", 1
    )
    feature_set_digests = features_response["features"][0]["feature_set_digests"]
    assert len(feature_set_digests) == 2


def test_extra_fields(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"

    name = "feature_set1"
    feature_set = _generate_feature_set(name)
    feature_set["metadata"]["test1"] = 200
    feature_set["metadata"]["test2"] = "a test"

    _assert_add_feature_set(client, project_name, feature_set)
