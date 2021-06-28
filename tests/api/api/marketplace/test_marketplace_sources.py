import pathlib
import unittest.mock
from http import HTTPStatus

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.api.schemas
from mlrun.api.crud.function_marketplace import MarketplaceItemsManager
from mlrun.api.utils.singletons.k8s import get_k8s
from mlrun.config import config


def _generate_source_dict(order, name):
    path = str(pathlib.Path(__file__).absolute().parent)

    return {
        "order": order,
        "source": {
            "kind": "MarketplaceSource",
            "metadata": {"name": name, "description": "A test", "labels": None},
            "spec": {"path": path, "credentials": None},
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


class K8sMock:
    def __init__(self):
        self._mock_secrets = {}

    def store_project_secrets(self, project, secrets, namespace=""):
        self._mock_secrets.update(secrets)

    def delete_project_secrets(self, project, secrets, namespace=""):
        for key in secrets:
            self._mock_secrets.pop(key, None)

    def get_project_secrets(self, project, namespace=""):
        return list(self._mock_secrets.keys())

    def get_project_secret_values(self, project, secret_keys=None, namespace=""):
        return self._mock_secrets


def _mock_k8s_secrets(mock_object):
    get_k8s().get_project_secrets = unittest.mock.Mock(
        side_effect=mock_object.get_project_secrets
    )
    get_k8s().get_project_secret_values = unittest.mock.Mock(
        side_effect=mock_object.get_project_secret_values
    )
    get_k8s().store_project_secrets = unittest.mock.Mock(
        side_effect=mock_object.store_project_secrets
    )
    get_k8s().delete_project_secrets = unittest.mock.Mock(
        side_effect=mock_object.delete_project_secrets
    )


def test_marketplace_source_manager() -> None:
    config.namespace = "default-tenant"
    k8s_mock = K8sMock()
    _mock_k8s_secrets(k8s_mock)

    manager = MarketplaceItemsManager()
    source_dict = _generate_source_dict(1, "source_1")
    source_object = mlrun.api.schemas.MarketplaceSource(**source_dict["source"])
    secrets = {"secret1": "value1", "secret2": "value2"}
    source_object.spec.credentials = secrets
    manager.add_source(source_object)
    catalog = manager.get_source_catalog(source_object)
    pass
