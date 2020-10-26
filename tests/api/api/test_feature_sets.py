from http import HTTPStatus
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

fs = {"name": "dummy",
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
      "labels": {
          "owner": "saarc",
          "group": "dev",
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


def test_feature_set(db: Session, client: TestClient) -> None:
    proj_name = f"prj-{uuid4().hex}"

    fs["name"] = "feature_set1"
    resp = client.post(f"/api/projects/{proj_name}/feature_sets", json=fs)
    assert resp.status_code == HTTPStatus.OK.value, "add"

    resp = client.get(f"/api/projects/{proj_name}/feature_sets/{fs['name']}")
    assert resp.status_code == HTTPStatus.OK.value, "get"
    print("Response is: {}".format(resp.json()))

    fs["name"] = "feature_set2"
    resp = client.post(f"/api/projects/{proj_name}/feature_sets", json=fs)
    assert resp.status_code == HTTPStatus.OK.value, "add"

    resp = client.get(f"/api/projects/{proj_name}/feature_sets")
    assert resp.status_code == HTTPStatus.OK.value, "list"
    json_resp = resp.json()
    print("Response is: {}".format(json_resp))
    assert "feature_sets" in json_resp, "no feature sets"
    assert len(json_resp["feature_sets"]) == 2, "not enough feature sets in response"

    # Update a feature-set
    fs_update = {
        "entities": [
            {
                "name": "market_cap",
                "description": "money!",
                "value_type": "integer",
            },
        ],
        "labels": {
            "new-label": "new-value",
            "owner": "someone-else",
        }
    }
    resp = client.put(f"/api/projects/{proj_name}/feature_sets/{fs['name']}", json=fs_update)
    assert resp.status_code == HTTPStatus.OK.value, "update"
    resp = client.get(f"/api/projects/{proj_name}/feature_sets/{fs['name']}")
    assert resp.status_code == HTTPStatus.OK.value, "get"
    updated_resp = resp.json()
    fs_resp = updated_resp["feature_set"]
    assert len(fs_resp["labels"]) == 3 and "new-label" in fs_resp["labels"], "update corrupt data"

    # Delete the 2nd fs
    resp = client.delete(f"/api/projects/{proj_name}/feature_sets/{fs['name']}")
    assert resp.status_code == HTTPStatus.NO_CONTENT.value, "delete"

    # Now try to list - expect only 1 fs
    resp = client.get(f"/api/projects/{proj_name}/feature_sets/")
    assert resp.status_code == HTTPStatus.OK.value, "list"
    json_resp = resp.json()
    print("Response is: {}".format(json_resp))
    assert "feature_sets" in json_resp, "no feature sets"
    assert len(json_resp["feature_sets"]) == 1, "too many feature sets in response after deletion"
