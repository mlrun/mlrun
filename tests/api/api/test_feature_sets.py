from http import HTTPStatus
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

fs = {
    "metadata": {
        "name": "dummy",
        "labels": {
            "owner": "saarc",
            "group": "dev",
        },
    },
    "spec": {
        "entities": [
            {
                "name": "ticker",
                "value_type": "str",
            },
        ],
        "features": [
            {
                "name": "time",
                "value_type": "datetime",
            },
            {
                "name": "bid",
                "value_type": "float",
            },
            {
                "name": "ask",
                "value_type": "time",
            },
        ],
    },
    "status": {
        "state": "created",
        "stats": {
            "time": {
                "count": "8",
                "unique": "7",
                "top": "2016-05-25 13:30:00.222222"
            }
        },
    },
}


def test_list(client: TestClient, proj, query, num_entities):
    url = f"/api/projects/{proj}/feature_sets"
    if query:
        url = url + f"?{query}"
    resp = client.get(url)
    assert resp.status_code == HTTPStatus.OK.value, "list"
    json_resp = resp.json()
    print("Response is: {}".format(json_resp))
    assert "feature_sets" in json_resp, "no feature sets"
    assert len(json_resp["feature_sets"]) == num_entities, "wrong number of feature sets in response"


def test_feature_set(db: Session, client: TestClient) -> None:
    proj_name = f"prj-{uuid4().hex}"

    name = "feature_set1"
    fs["metadata"]["name"] = name
    resp = client.post(f"/api/projects/{proj_name}/feature_sets", json=fs)
    assert resp.status_code == HTTPStatus.OK.value, "add"

    resp = client.get(f"/api/projects/{proj_name}/feature_sets/{name}")
    assert resp.status_code == HTTPStatus.OK.value, "get"
    print("Response is: {}".format(resp.json()))

    name = "feature_set2"
    fs["metadata"]["name"] = name
    resp = client.post(f"/api/projects/{proj_name}/feature_sets", json=fs)
    assert resp.status_code == HTTPStatus.OK.value, "add"

    name = "feat_3"
    fs["metadata"]["name"] = name
    fs["spec"]["entities"] = [{
        "name": "buyer",
        "value_type": "str",
        }]

    resp = client.post(f"/api/projects/{proj_name}/feature_sets", json=fs)
    assert resp.status_code == HTTPStatus.OK.value, "add"

    test_list(client, proj_name, None, 3)
    test_list(client, proj_name, "name=feature", 2)
    test_list(client, proj_name, "entity=buyer", 1)
    test_list(client, proj_name, "entity=ticker&entity=bid", 2)
    test_list(client, proj_name, "name=feature&entity=buyer", 0)

    # Update a feature-set
    fs_update = {
        "entities": [
            {
                "name": "market_cap",
                "value_type": "integer",
            },
        ],
        "labels": {
            "new-label": "new-value",
            "owner": "someone-else",
        }
    }
    resp = client.put(f"/api/projects/{proj_name}/feature_sets/{name}", json=fs_update)
    assert resp.status_code == HTTPStatus.OK.value, "update"
    resp = client.get(f"/api/projects/{proj_name}/feature_sets/{name}")
    assert resp.status_code == HTTPStatus.OK.value, "get"
    updated_resp = resp.json()
    fs_resp = updated_resp["feature_set"]["metadata"]
    assert len(fs_resp["labels"]) == 3 and "new-label" in fs_resp["labels"], "update corrupt data"

    # Delete the last fs
    resp = client.delete(f"/api/projects/{proj_name}/feature_sets/{name}")
    assert resp.status_code == HTTPStatus.NO_CONTENT.value, "delete"

    # Now try to list - expect only 2 fs
    test_list(client, proj_name, None, 2)

