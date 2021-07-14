import unittest.mock
from http import HTTPStatus

import kubernetes.client.rest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.api.crud
import mlrun.api.schemas
import mlrun.api.utils.singletons.db
import mlrun.api.utils.singletons.k8s
import mlrun.artifacts.dataset
import mlrun.artifacts.model
import mlrun.errors


def test_build_status_pod_not_found(db: Session, client: TestClient):
    function = {
        "kind": "job",
        "metadata": {
            "name": "function-name",
            "project": "project-name",
            "tag": "latest",
        },
        "status": {"build_pod": "some-pod-name"},
    }
    response = client.post(
        f"/api/func/{function['metadata']['project']}/{function['metadata']['name']}",
        json=function,
    )
    assert response.status_code == HTTPStatus.OK.value

    mlrun.api.utils.singletons.k8s.get_k8s().v1api = unittest.mock.Mock()
    mlrun.api.utils.singletons.k8s.get_k8s().v1api.read_namespaced_pod = unittest.mock.Mock(
        side_effect=kubernetes.client.rest.ApiException(
            status=HTTPStatus.NOT_FOUND.value
        )
    )
    response = client.get(
        "/api/build/status",
        params={
            "project": function["metadata"]["project"],
            "name": function["metadata"]["name"],
            "tag": function["metadata"]["tag"],
        },
    )
    assert response.status_code == HTTPStatus.NOT_FOUND.value
