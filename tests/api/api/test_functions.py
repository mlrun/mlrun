import asyncio
import unittest.mock
from http import HTTPStatus

import httpx
import kubernetes.client.rest
import pytest
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
import tests.conftest


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
        f"func/{function['metadata']['project']}/{function['metadata']['name']}",
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
        "build/status",
        params={
            "project": function["metadata"]["project"],
            "name": function["metadata"]["name"],
            "tag": function["metadata"]["tag"],
        },
    )
    assert response.status_code == HTTPStatus.NOT_FOUND.value


@pytest.mark.asyncio
async def test_multiple_store_function_race_condition(
    db: Session, async_client: httpx.AsyncClient
):
    """
    This is testing the case that the retry_on_conflict decorator is coming to solve, see its docstring for more details
    """
    project = {"metadata": {"name": "project-name"}}
    response = await async_client.post("projects", json=project,)
    assert response.status_code == HTTPStatus.CREATED.value
    # Make the get function method to return None on the first two calls, and then use the original function
    get_function_mock = tests.conftest.MockSpecificCalls(
        mlrun.api.utils.singletons.db.get_db()._get_class_instance_by_uid, [1, 2], None
    ).mock_function
    mlrun.api.utils.singletons.db.get_db()._get_class_instance_by_uid = unittest.mock.Mock(
        side_effect=get_function_mock
    )
    function = {
        "kind": "job",
        "metadata": {
            "name": "function-name",
            "project": "project-name",
            "tag": "latest",
        },
    }

    request1_task = asyncio.create_task(
        async_client.post(
            f"func/{function['metadata']['project']}/{function['metadata']['name']}",
            json=function,
        )
    )
    request2_task = asyncio.create_task(
        async_client.post(
            f"func/{function['metadata']['project']}/{function['metadata']['name']}",
            json=function,
        )
    )
    response1, response2 = await asyncio.gather(request1_task, request2_task,)

    assert response1.status_code == HTTPStatus.OK.value
    assert response2.status_code == HTTPStatus.OK.value
    # 2 times for two store function requests + 1 time on retry for one of them
    assert (
        mlrun.api.utils.singletons.db.get_db()._get_class_instance_by_uid.call_count
        == 3
    )


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
        response = client.post("build/function", json=request_body,)
        assert response.status_code == HTTPStatus.OK.value
        assert (
            mlrun.api.api.endpoints.functions._build_function.call_args[0][3]
            == with_mlrun
        )
    mlrun.api.api.endpoints.functions._build_function = original_build_function
