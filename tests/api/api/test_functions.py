import asyncio
import http
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
import tests.api.api.utils
import tests.conftest

PROJECT = "project-name"


def test_build_status_pod_not_found(db: Session, client: TestClient):
    tests.api.api.utils.create_project(client, PROJECT)
    function = {
        "kind": "job",
        "metadata": {
            "name": "function-name",
            "project": PROJECT,
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
    mlrun.api.utils.singletons.k8s.get_k8s().v1api.read_namespaced_pod = (
        unittest.mock.Mock(
            side_effect=kubernetes.client.rest.ApiException(
                status=HTTPStatus.NOT_FOUND.value
            )
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
    await tests.api.api.utils.create_project_async(async_client, PROJECT)
    # Make the get function method to return None on the first two calls, and then use the original function
    get_function_mock = tests.conftest.MockSpecificCalls(
        mlrun.api.utils.singletons.db.get_db()._get_class_instance_by_uid, [1, 2], None
    ).mock_function
    mlrun.api.utils.singletons.db.get_db()._get_class_instance_by_uid = (
        unittest.mock.Mock(side_effect=get_function_mock)
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
    response1, response2 = await asyncio.gather(
        request1_task,
        request2_task,
    )

    assert response1.status_code == HTTPStatus.OK.value
    assert response2.status_code == HTTPStatus.OK.value
    # 2 times for two store function requests + 1 time on retry for one of them
    assert (
        mlrun.api.utils.singletons.db.get_db()._get_class_instance_by_uid.call_count
        == 3
    )


def test_build_function_with_mlrun_bool(db: Session, client: TestClient):
    tests.api.api.utils.create_project(client, PROJECT)

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
        response = client.post(
            "build/function",
            json=request_body,
        )
        assert response.status_code == HTTPStatus.OK.value
        assert (
            mlrun.api.api.endpoints.functions._build_function.call_args[0][3]
            == with_mlrun
        )
    mlrun.api.api.endpoints.functions._build_function = original_build_function


def test_start_function_succeeded(db: Session, client: TestClient, monkeypatch):
    name = "dask"
    project = "test-dask"
    dask_cluster = mlrun.new_function(name, project=project, kind="dask")
    monkeypatch.setattr(
        mlrun.api.api.endpoints.functions,
        "_parse_start_function_body",
        lambda *args, **kwargs: dask_cluster,
    )
    monkeypatch.setattr(
        mlrun.api.api.endpoints.functions,
        "_start_function",
        lambda *args, **kwargs: unittest.mock.Mock(),
    )
    response = client.post(
        "start/function",
        json=mlrun.utils.generate_object_uri(
            dask_cluster.metadata.project,
            dask_cluster.metadata.name,
        ),
    )
    assert response.status_code == http.HTTPStatus.OK.value
    background_task = mlrun.api.schemas.BackgroundTask(**response.json())
    assert background_task.status.state == mlrun.api.schemas.BackgroundTaskState.running

    response = client.get(
        f"projects/{project}/background-tasks/{background_task.metadata.name}"
    )
    assert response.status_code == http.HTTPStatus.OK.value
    background_task = mlrun.api.schemas.BackgroundTask(**response.json())
    assert (
        background_task.status.state == mlrun.api.schemas.BackgroundTaskState.succeeded
    )


def test_start_function_fails(db: Session, client: TestClient, monkeypatch):
    def failing_func():
        raise mlrun.errors.MLRunRuntimeError()

    name = "dask"
    project = "test-dask"
    dask_cluster = mlrun.new_function(name, project=project, kind="dask")
    monkeypatch.setattr(
        mlrun.api.api.endpoints.functions,
        "_parse_start_function_body",
        lambda *args, **kwargs: dask_cluster,
    )
    monkeypatch.setattr(
        mlrun.api.api.endpoints.functions,
        "_start_function",
        lambda *args, **kwargs: failing_func(),
    )

    response = client.post(
        "start/function",
        json=mlrun.utils.generate_object_uri(
            dask_cluster.metadata.project,
            dask_cluster.metadata.name,
        ),
    )
    assert response.status_code == http.HTTPStatus.OK
    background_task = mlrun.api.schemas.BackgroundTask(**response.json())
    assert background_task.status.state == mlrun.api.schemas.BackgroundTaskState.running
    response = client.get(
        f"projects/{project}/background-tasks/{background_task.metadata.name}"
    )
    assert response.status_code == http.HTTPStatus.OK.value
    background_task = mlrun.api.schemas.BackgroundTask(**response.json())
    assert background_task.status.state == mlrun.api.schemas.BackgroundTaskState.failed


def test_start_function(db: Session, client: TestClient, monkeypatch):
    def failing_func():
        raise mlrun.errors.MLRunRuntimeError()

    name = "dask"
    project = "test-dask"
    for test_case in [
        {
            "_start_function_mock": unittest.mock.Mock,
            "expected_status_result": mlrun.api.schemas.BackgroundTaskState.succeeded,
            "background_timeout_mode": "enabled",
            "dask_timeout": 100,
        },
        {
            "_start_function_mock": failing_func,
            "expected_status_result": mlrun.api.schemas.BackgroundTaskState.failed,
            "background_timeout_mode": "enabled",
            "dask_timeout": None,
        },
        {
            "_start_function_mock": unittest.mock.Mock,
            "expected_status_result": mlrun.api.schemas.BackgroundTaskState.succeeded,
            "background_timeout_mode": "disabled",
            "dask_timeout": 0,
        },
    ]:
        _start_function_mock = test_case.get("_start_function_mock", unittest.mock.Mock)
        expected_status_result = test_case.get(
            "expected_status_result", mlrun.api.schemas.BackgroundTaskState.running
        )
        background_timeout_mode = test_case.get("background_timeout_mode", "enabled")
        dask_timeout = test_case.get("dask_timeout", None)

        mlrun.mlconf.background_tasks.timeout_mode = background_timeout_mode
        mlrun.mlconf.background_tasks.default_timeouts.runtimes.dask = dask_timeout

        dask_cluster = mlrun.new_function(name, project=project, kind="dask")
        monkeypatch.setattr(
            mlrun.api.api.endpoints.functions,
            "_parse_start_function_body",
            lambda *args, **kwargs: dask_cluster,
        )
        monkeypatch.setattr(
            mlrun.api.api.endpoints.functions,
            "_start_function",
            lambda *args, **kwargs: _start_function_mock(),
        )
        response = client.post(
            "start/function",
            json=mlrun.utils.generate_object_uri(
                dask_cluster.metadata.project,
                dask_cluster.metadata.name,
            ),
        )
        assert response.status_code == http.HTTPStatus.OK
        background_task = mlrun.api.schemas.BackgroundTask(**response.json())
        assert (
            background_task.status.state
            == mlrun.api.schemas.BackgroundTaskState.running
        )
        response = client.get(
            f"projects/{project}/background-tasks/{background_task.metadata.name}"
        )
        assert response.status_code == http.HTTPStatus.OK.value
        background_task = mlrun.api.schemas.BackgroundTask(**response.json())
        assert background_task.status.state == expected_status_result
