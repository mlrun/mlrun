import unittest.mock
from http import HTTPStatus

import kubernetes.client.rest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.api.api.endpoints.functions
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


def test_build_function_with_mlrun_bool(db: Session, client: TestClient):
    function_dict = {
        "kind": "job",
        "metadata": {
            "name": "function-name",
            "project": "project-name",
            "tag": "latest",
        },
    }
    original_build_function = mlrun.api.api.endpoints.functions._build_function
    for with_mlrun in [True, False]:
        request_body = {
            "function": function_dict,
            "with_mlrun": with_mlrun,
        }
        function = mlrun.new_function(runtime=function_dict)
        mlrun.api.api.endpoints.functions._build_function = unittest.mock.Mock(
            return_value=(function, True)
        )
        response = client.post("/api/build/function", json=request_body,)
        assert response.status_code == HTTPStatus.OK.value
        assert (
            mlrun.api.api.endpoints.functions._build_function.call_args[0][3]
            == with_mlrun
        )
    mlrun.api.api.endpoints.functions._build_function = original_build_function
