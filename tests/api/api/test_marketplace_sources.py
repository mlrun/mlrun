from http import HTTPStatus

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session


def _generate_source_dict(order, name):
    return {
        "order": order,
        "source": {
            "kind": "MarketplaceSource",
            "metadata": {"name": name, "description": "A test", "labels": None},
            "spec": {"path": "https://www.functionhub.com/myhub", "credentials": None},
            "status": {"state": "created"},
        },
    }


def test_marketplace(db: Session, client: TestClient) -> None:
    response = client.get("/api/marketplace/sources")
    assert response.status_code == HTTPStatus.OK.value
    json_response = response.json()
    print(json_response)

    new_source = _generate_source_dict(1, "source_1")
    response = client.post("/api/marketplace/sources", json=new_source)
    assert response.status_code == HTTPStatus.CREATED.value

    new_source["source"]["metadata"]["something_new"] = 42
    response = client.put("/api/marketplace/sources/source_1", json=new_source)
    assert response.status_code == HTTPStatus.OK.value

    new_source = _generate_source_dict(1, "source_2")
    response = client.put("/api/marketplace/sources/source_2", json=new_source)
    assert response.status_code == HTTPStatus.OK.value

    new_source = _generate_source_dict(3, "source_3")
    response = client.post("/api/marketplace/sources", json=new_source)
    assert response.status_code == HTTPStatus.CREATED.value

    response = client.get("/api/marketplace/sources")
    assert response.status_code == HTTPStatus.OK.value

    response = client.delete("/api/marketplace/sources/source_2")
    assert response.status_code == HTTPStatus.NO_CONTENT.value
    response = client.get("/api/marketplace/sources")
    assert response.status_code == HTTPStatus.OK.value
    json_response = response.json()

    response = client.get("/api/marketplace/sources/source_3")
    assert response.status_code == HTTPStatus.OK.value
    json_response = response.json()
    pass
