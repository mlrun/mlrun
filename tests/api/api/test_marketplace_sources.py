from http import HTTPStatus

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from mlrun.api.schemas.marketplace import MarketplaceSource, OrderedMarketplaceSource


def _generate_source_dict(order, name):
    return {
        "order": order,
        "source": {
            "kind": "MarketplaceSource",
            "metadata": {"name": name, "description": "A test", "labels": None,},
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

    new_source = _generate_source_dict(1, "source_2")
    response = client.post("/api/marketplace/sources", json=new_source)
    assert response.status_code == HTTPStatus.CREATED.value

    new_source = _generate_source_dict(3, "source_3")
    response = client.post("/api/marketplace/sources", json=new_source)
    assert response.status_code == HTTPStatus.CREATED.value

    response = client.get("/api/marketplace/sources")
    assert response.status_code == HTTPStatus.OK.value
    json_response = response.json()
    print(json_response)
