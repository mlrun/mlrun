import json
import os
import string
from random import randint, choice
from typing import Dict, Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from mlrun.api.api.endpoints.model_endpoints import (
    _get_endpoint_by_id,
    _encode_labels,
    ENDPOINTS_KV_TABLE,
)
from mlrun.config import config
from mlrun.utils.helpers import get_model_endpoint_id
from mlrun.utils.v3io_clients import get_v3io_client


def is_env_params_exist() -> bool:
    return not all((os.environ.get(r, False) for r in ["V3IO_ACCESS_KEY", "V3IO_API"]))


@pytest.mark.skipif(
    is_env_params_exist(),
    reason="Either V3IO_ACCESS_KEY or V3IO_API environment params not found",
)
def test_create_endpoint(db: Session, client: TestClient):
    endpoint_details = get_random_endpoint_details()

    response = client.post(
        f"/api/projects/{endpoint_details['project']}/model-endpoints",
        json=endpoint_details,
    )

    assert response.status_code == 200

    endpoint_id = get_model_endpoint_id(
        endpoint_details["project"],
        endpoint_details["model"],
        endpoint_details["function"],
        endpoint_details["tag"],
    )
    endpoint = _get_endpoint_by_id(endpoint_id)
    labels = endpoint_details.pop("labels")
    labels = _encode_labels(labels)
    endpoint_details.update(labels)

    assert endpoint == endpoint_details


@pytest.mark.skipif(
    is_env_params_exist(),
    reason="Either V3IO_ACCESS_KEY or V3IO_API environment params not found",
)
def test_delete_endpoint(db: Session, client: TestClient):
    endpoint_details = get_random_endpoint_details()

    project = endpoint_details["project"]

    response = client.post(
        f"/api/projects/{project}/model-endpoints",
        json=endpoint_details,
    )

    assert response.status_code == 200

    endpoint_id = get_model_endpoint_id(
        endpoint_details["project"],
        endpoint_details["model"],
        endpoint_details["function"],
        endpoint_details["tag"],
    )

    endpoint = _get_endpoint_by_id(endpoint_id)

    assert endpoint

    response = client.delete(f"/api/projects/{project}/model-endpoints/{endpoint_id}")

    assert response.status_code == 200

    endpoint = _get_endpoint_by_id(endpoint_id)

    assert not endpoint


@pytest.mark.skipif(
    is_env_params_exist(),
    reason="Either V3IO_ACCESS_KEY or V3IO_API environment params not found",
)
def test_list_endpoints(db: Session, client: TestClient):
    for _ in range(5):
        endpoint_details = get_random_endpoint_details()
        response = client.post(
            f"/api/projects/{endpoint_details['project']}/model-endpoints",
            json=endpoint_details,
        )
        assert response.status_code == 200

    response = client.get("/api/projects/test/model-endpoints")
    body = json.loads(response.text)["endpoints"]
    assert len(body) > 0


@pytest.mark.skipif(
    is_env_params_exist(),
    reason="Either V3IO_ACCESS_KEY or V3IO_API environment params not found",
)
def test_list_endpoints_filter(db: Session, client: TestClient):
    for i in range(5):
        endpoint_details = get_random_endpoint_details()

        if i < 1:
            endpoint_details["model"] = "filterme"

        if i < 2:
            endpoint_details["function"] = "filterme"

        if i < 3:
            endpoint_details["tag"] = "filterme"

        if i < 4:
            endpoint_details["labels"] = ["filtermex==1", "filtermey==2"]

        response = client.post(
            f"/api/projects/test/model-endpoints",
            json=endpoint_details,
        )

        assert response.status_code == 200

    filter_model = json.loads(
        client.get("/api/projects/test/model-endpoints/?model=filterme").text
    )["endpoints"]
    assert len(filter_model) == 1

    filter_function = json.loads(
        client.get("/api/projects/test/model-endpoints/?function=filterme").text
    )["endpoints"]
    assert len(filter_function) == 2

    filter_tag = json.loads(
        client.get("/api/projects/test/model-endpoints/?tag=filterme").text
    )["endpoints"]
    assert len(filter_tag) == 3

    filter_labels = json.loads(
        client.get("/api/projects/test/model-endpoints/?label=filtermex==1").text
    )["endpoints"]
    assert len(filter_labels) == 4

    filter_labels = json.loads(
        client.get(
            "/api/projects/test/model-endpoints/?label=filtermex==1&label=filtermey==2"
        ).text
    )["endpoints"]
    assert len(filter_labels) == 4

    filter_labels = json.loads(
        client.get("/api/projects/test/model-endpoints/?label=filtermey==2").text
    )["endpoints"]
    assert len(filter_labels) == 4

    filter_labels = json.loads(
        client.get(
            "/api/projects/test/model-endpoints/?label=filtermex==1&label=filtermey==2"
        ).text
    )["endpoints"]
    assert len(filter_labels) == 4

    filter_labels = json.loads(
        client.get("/api/projects/test/model-endpoints/?label=filtermex==1").text
    )["endpoints"]
    assert len(filter_labels) == 4

    filter_labels = json.loads(
        client.get("/api/projects/test/model-endpoints/?label=filtermey==2").text
    )["endpoints"]
    assert len(filter_labels) == 4


@pytest.fixture(autouse=True)
def cleanup_endpoints(db: Session, client: TestClient):
    response = client.get("/api/projects/test/model-endpoints")
    for endpoint in json.loads(response.text).get("endpoints", []):
        endpoint_id = get_model_endpoint_id(
            endpoint["project"],
            endpoint["model"],
            endpoint["function"],
            endpoint["tag"],
        )

        resp = get_v3io_client().kv.delete(
            container=config.model_endpoint_monitoring_container,
            table_path=ENDPOINTS_KV_TABLE,
            key=endpoint_id,
        )


def get_random_endpoint_details() -> Dict[str, Any]:
    return {
        "project": "test",
        "model": f"model_{randint(0,100)}",
        "function": f"function_{randint(0,100)}",
        "tag": f"v{randint(0,100)}",
        "model_class": "classifier",
        "labels": [
            f"{choice(string.ascii_letters)}=={randint(0,100)}" for _ in range(1, 5)
        ],
    }
