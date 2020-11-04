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


def _assert_list_feature_sets(
    client: TestClient, project, query, expected_number_of_entities
):
    url = f"/api/projects/{project}/feature_sets"
    if query:
        url = url + f"?{query}"
    resp = client.get(url)
    assert resp.status_code == HTTPStatus.OK.value
    json_resp = resp.json()
    assert "feature_sets" in json_resp, "no feature sets"
    assert (
        len(json_resp["feature_sets"]) == expected_number_of_entities
    ), "wrong number of feature sets in response"


def test_feature_set(db: Session, client: TestClient) -> None:
    project_name = f"prj-{uuid4().hex}"

    name = "feature_set1"
    feature_set = _generate_feature_set(name)
    resp = client.post(
        f"/api/projects/{project_name}/feature_sets?versioned=1", json=feature_set
    )
    assert resp.status_code == HTTPStatus.OK.value

    resp = client.get(
        f"/api/projects/{project_name}/feature_sets/{name}/references/latest"
    )
    assert resp.status_code == HTTPStatus.OK.value

    name = "feature_set2"
    feature_set = _generate_feature_set(name)
    resp = client.post(f"/api/projects/{project_name}/feature_sets", json=feature_set)
    assert resp.status_code == HTTPStatus.OK.value

    name = "feat_3"
    feature_set = _generate_feature_set(name)
    feature_set["spec"]["entities"] = [{"name": "buyer", "value_type": "str"}]

    resp = client.post(f"/api/projects/{project_name}/feature_sets", json=feature_set)
    assert resp.status_code == HTTPStatus.OK.value

    _assert_list_feature_sets(client, project_name, None, 3)
    _assert_list_feature_sets(client, project_name, "name=feature", 2)
    _assert_list_feature_sets(client, project_name, "entity=buyer", 1)
    _assert_list_feature_sets(client, project_name, "entity=ticker&entity=bid", 2)
    _assert_list_feature_sets(client, project_name, "name=feature&entity=buyer", 0)

    # Update a feature-set
    feature_set_update = {
        "entities": [{"name": "market_cap", "value_type": "integer"}],
        "labels": {"new-label": "new-value", "owner": "someone-else"},
    }
    resp = client.put(
        f"/api/projects/{project_name}/feature_sets/{name}/references/latest",
        json=feature_set_update,
    )
    assert resp.status_code == HTTPStatus.OK.value
    resp = client.get(
        f"/api/projects/{project_name}/feature_sets/{name}/references/latest"
    )
    assert resp.status_code == HTTPStatus.OK.value
    updated_resp = resp.json()
    feature_set_resp = updated_resp["metadata"]
    assert (
        len(feature_set_resp["labels"]) == 3
        and "new-label" in feature_set_resp["labels"]
    ), "update corrupted data - got wrong number of labels from get after update"

    # Delete the last fs
    resp = client.delete(f"/api/projects/{project_name}/feature_sets/{name}")
    assert resp.status_code == HTTPStatus.NO_CONTENT.value

    # Now try to list - expect only 2 fs
    _assert_list_feature_sets(client, project_name, None, 2)
