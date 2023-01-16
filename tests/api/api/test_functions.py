# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import asyncio
import http
import unittest.mock
from http import HTTPStatus

import fastapi.testclient
import httpx
import kubernetes.client.rest
import pytest
import sqlalchemy.orm

import mlrun.api.api.endpoints.functions
import mlrun.api.api.utils
import mlrun.api.crud
import mlrun.api.schemas
import mlrun.api.utils.clients.chief
import mlrun.api.utils.singletons.db
import mlrun.api.utils.singletons.k8s
import mlrun.artifacts.dataset
import mlrun.artifacts.model
import mlrun.errors
import tests.api.api.utils
import tests.conftest

PROJECT = "project-name"
ORIGINAL_VERSIONED_API_PREFIX = mlrun.api.main.BASE_VERSIONED_API_PREFIX


def test_build_status_pod_not_found(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
):
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
    db: sqlalchemy.orm.Session, async_client: httpx.AsyncClient
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


def test_redirection_from_worker_to_chief_only_if_serving_function_with_track_models(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    httpserver,
    monkeypatch,
):
    mlrun.mlconf.httpdb.clusterization.role = "worker"
    endpoint = f"{ORIGINAL_VERSIONED_API_PREFIX}/build/function"
    tests.api.api.utils.create_project(client, PROJECT)

    function_name = "test-function"
    function = _generate_function(function_name)

    handler_mock = mlrun.api.utils.clients.chief.Client()
    handler_mock._proxy_request_to_chief = unittest.mock.AsyncMock(
        return_value=fastapi.Response()
    )
    monkeypatch.setattr(
        mlrun.api.utils.clients.chief,
        "Client",
        lambda *args, **kwargs: handler_mock,
    )

    json_body = _generate_build_function_request(function)
    client.post(endpoint, data=json_body)
    # no schedule inside job body, expecting to be run in worker
    assert handler_mock._proxy_request_to_chief.call_count == 0

    function_with_track_models = _generate_function(function_name)
    function_with_track_models.spec.track_models = True
    json_body = _generate_build_function_request(function_with_track_models)
    client.post(endpoint, data=json_body)
    assert handler_mock._proxy_request_to_chief.call_count == 1


def test_redirection_from_worker_to_chief_deploy_serving_function_with_track_models(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient, httpserver
):
    mlrun.mlconf.httpdb.clusterization.role = "worker"
    endpoint = f"{ORIGINAL_VERSIONED_API_PREFIX}/build/function"
    tests.api.api.utils.create_project(client, PROJECT)

    function_name = "test-function"
    function_with_track_models = _generate_function(function_name)
    function_with_track_models.spec.track_models = True

    json_body = _generate_build_function_request(function_with_track_models)

    for test_case in [
        {
            "body": json_body,
            "expected_status": http.HTTPStatus.OK.value,
            "expected_body": {
                "data": function_with_track_models.to_dict(),
                "ready": True,
            },
        },
        {
            "body": json_body,
            "expected_status": http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
            "expected_body": {"detail": {"reason": "Unknown error"}},
        },
    ]:
        expected_status = test_case.get("expected_status")
        expected_response = test_case.get("expected_body")
        body = test_case.get("body")

        httpserver.expect_ordered_request(endpoint, method="POST").respond_with_json(
            expected_response, status=expected_status
        )
        url = httpserver.url_for("")
        mlrun.mlconf.httpdb.clusterization.chief.url = url
        response = client.post(endpoint, data=body)
        assert response.status_code == expected_status
        assert response.json() == expected_response


def test_build_function_with_mlrun_bool(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
):
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


def test_start_function_succeeded(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient, monkeypatch
):
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


def test_start_function_fails(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient, monkeypatch
):
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


def test_start_function(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient, monkeypatch
):
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


def _generate_function(
    function_name: str, project: str = PROJECT, function_tag: str = "latest"
):
    return mlrun.new_function(
        name=function_name,
        project=project,
        tag=function_tag,
        kind="serving",
        image="mlrun/mlrun",
    )


def _generate_build_function_request(
    func, with_mlrun: bool = True, skip_deployed: bool = False, to_json: bool = True
):

    request = {
        "function": func.to_dict(),
        "with_mlrun": "yes" if with_mlrun else "false",
        "skip_deployed": skip_deployed,
    }
    if not to_json:
        return request
    return mlrun.utils.dict_to_json(request)
