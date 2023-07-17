# Copyright 2023 Iguazio
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
from types import ModuleType

import deepdiff
import fastapi.testclient
import httpx
import kubernetes.client.rest
import nuclio
import pytest
import sqlalchemy.orm

import mlrun.api.api.endpoints.functions
import mlrun.api.api.utils
import mlrun.api.crud
import mlrun.api.main
import mlrun.api.utils.builder
import mlrun.api.utils.clients.chief
import mlrun.api.utils.singletons.db
import mlrun.api.utils.singletons.k8s
import mlrun.artifacts.dataset
import mlrun.artifacts.model
import mlrun.common.model_monitoring.helpers
import mlrun.common.schemas
import mlrun.errors
import mlrun.model_monitoring.tracking_policy
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

    mlrun.api.utils.singletons.k8s.get_k8s_helper().v1api = unittest.mock.Mock()
    mlrun.api.utils.singletons.k8s.get_k8s_helper().v1api.read_namespaced_pod = (
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
async def test_list_functions_with_hash_key_versioned(
    db: sqlalchemy.orm.Session, async_client: httpx.AsyncClient
):
    await tests.api.api.utils.create_project_async(async_client, PROJECT)

    function_tag = "function-tag"
    function_project = "project-name"
    function_name = "function-name"

    function = {
        "kind": "job",
        "metadata": {
            "name": function_name,
            "project": function_project,
            "tag": function_tag,
        },
        "spec": {"image": "mlrun/mlrun"},
    }

    another_tag = "another-tag"
    function2 = {
        "kind": "job",
        "metadata": {
            "name": function_name,
            "project": function_project,
            "tag": "another-tag",
        },
    }

    post_function1_response = await async_client.post(
        f"projects/{function_project}/functions/{function_name}?tag={function_tag}&versioned={True}",
        json=function,
    )

    assert post_function1_response.status_code == HTTPStatus.OK.value
    hash_key = post_function1_response.json()["hash_key"]

    # Store another function with the same project and name but different tag and hash key
    post_function2_response = await async_client.post(
        f"projects/{function_project}/functions/"
        f"{function_name}?tag={another_tag}&versioned={True}",
        json=function2,
    )
    assert post_function2_response.status_code == HTTPStatus.OK.value

    list_functions_by_hash_key_response = await async_client.get(
        f"projects/{function_project}/functions?name={function_name}&hash_key={hash_key}"
    )

    list_functions_results = list_functions_by_hash_key_response.json()["funcs"]
    assert len(list_functions_results) == 1
    assert list_functions_results[0]["metadata"]["hash"] == hash_key


@pytest.mark.parametrize("post_schedule", [True, False])
def test_delete_function_with_schedule(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    post_schedule,
):
    # create project and function
    tests.api.api.utils.create_project(client, PROJECT)

    function_tag = "function-tag"
    function_name = "function-name"
    project_name = "project-name"

    function = {
        "kind": "job",
        "metadata": {
            "name": function_name,
            "project": project_name,
            "tag": function_tag,
        },
        "spec": {"image": "mlrun/mlrun"},
    }

    function_endpoint = f"projects/{PROJECT}/functions/{function_name}"
    function = client.post(function_endpoint, data=mlrun.utils.dict_to_json(function))
    assert function.status_code == HTTPStatus.OK.value
    hash_key = function.json()["hash_key"]

    endpoint = f"projects/{PROJECT}/schedules"
    if post_schedule:
        # generate schedule object that matches to the function and create it
        scheduled_object = {
            "task": {
                "spec": {
                    "function": f"{PROJECT}/{function_name}@{hash_key}",
                    "handler": "handler",
                },
                "metadata": {"name": "my-task", "project": f"{PROJECT}"},
            }
        }
        schedule_cron_trigger = mlrun.common.schemas.ScheduleCronTrigger(minute=1)

        schedule = mlrun.common.schemas.ScheduleInput(
            name=function_name,
            kind=mlrun.common.schemas.ScheduleKinds.job,
            scheduled_object=scheduled_object,
            cron_trigger=schedule_cron_trigger,
        )

        endpoint = f"projects/{PROJECT}/schedules"
        response = client.post(endpoint, data=mlrun.utils.dict_to_json(schedule.dict()))
        assert response.status_code == HTTPStatus.CREATED.value

        response = client.get(endpoint)
        assert (
            response.status_code == HTTPStatus.OK.value
            and response.json()["schedules"][0]["name"] == function_name
        )

    # delete the function and assert that it has been removed, as has its schedule if created
    response = client.delete(function_endpoint)
    assert response.status_code == HTTPStatus.NO_CONTENT.value

    response = client.get(function_endpoint)
    assert response.status_code == HTTPStatus.NOT_FOUND.value

    if post_schedule:
        response = client.get(endpoint)
        assert (
            response.status_code == HTTPStatus.OK.value
            and not response.json()["schedules"]
        )


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


def test_tracking_on_serving(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    httpserver,
    monkeypatch,
):
    """Validate that the `mlrun.common.schemas.model_monitoring.tracking_policy.TrackingPolicy` configurations are
    generated as expected when the user applies model monitoring on a serving function
    """

    # Generate a test project
    tests.api.api.utils.create_project(client, PROJECT)

    # Generate a basic serving function and apply model monitoring
    function_name = "test-function"
    function = _generate_function(function_name)
    function.set_tracking()

    # Mock the client and unnecessary functions for this test
    handler_mock = mlrun.api.utils.clients.chief.Client()
    handler_mock._proxy_request_to_chief = unittest.mock.AsyncMock(
        return_value=fastapi.Response()
    )

    functions_to_monkeypatch = {
        mlrun.api.api.utils: ["apply_enrichment_and_validation_on_function"],
        mlrun.api.api.endpoints.functions: [
            "_process_model_monitoring_secret",
            "_create_model_monitoring_stream",
        ],
        mlrun.api.crud: ["ModelEndpoints"],
        nuclio.deploy: ["deploy_config"],
        mlrun.api.crud.model_monitoring: ["get_stream_path"],
    }

    for package in functions_to_monkeypatch:
        _function_to_monkeypatch(
            monkeypatch=monkeypatch,
            package=package,
            list_of_functions=functions_to_monkeypatch[package],
        )

    # Adjust the required request endpoint and body
    endpoint = f"{ORIGINAL_VERSIONED_API_PREFIX}/build/function"
    json_body = _generate_build_function_request(function)
    response = client.post(endpoint, data=json_body)

    assert response.status_code == 200

    # Validate that the default configurations were set as expected
    function_from_db = mlrun.api.crud.Functions().get_function(
        db_session=db, project=PROJECT, name=function_name, tag="latest"
    )

    assert function_from_db["spec"]["track_models"]

    tracking_policy_default = (
        mlrun.model_monitoring.tracking_policy.TrackingPolicy().to_dict()
    )
    assert (
        deepdiff.DeepDiff(
            tracking_policy_default,
            function_from_db["spec"]["tracking_policy"],
            ignore_order=True,
        )
        == {}
    )


def _function_to_monkeypatch(monkeypatch, package: ModuleType, list_of_functions: list):
    """Monkey patching a provided list of functions. Each function will be converted into `unittest.mock.Mock()`"""
    for function in list_of_functions:
        monkeypatch.setattr(
            package,
            function,
            lambda *args, **kwargs: unittest.mock.Mock(),
        )


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


@pytest.mark.parametrize(
    "source, load_source_on_run",
    [
        ("./", False),
        (".", False),
        ("./", True),
        (".", True),
    ],
)
def test_build_function_with_project_repo(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    source,
    load_source_on_run,
):
    git_repo = "git://github.com/mlrun/test.git"
    tests.api.api.utils.create_project(
        client, PROJECT, source=git_repo, load_source_on_run=load_source_on_run
    )
    function_dict = {
        "kind": "job",
        "metadata": {
            "name": "function-name",
            "project": "project-name",
            "tag": "latest",
        },
        "spec": {
            "build": {
                "source": source,
            },
        },
    }
    original_build_runtime = mlrun.api.utils.builder.build_image
    mlrun.api.utils.builder.build_image = unittest.mock.Mock(return_value="success")
    response = client.post(
        "build/function",
        json={"function": function_dict},
    )
    assert response.status_code == HTTPStatus.OK.value
    function = mlrun.new_function(runtime=response.json()["data"])
    assert function.spec.build.source == git_repo
    assert function.spec.build.load_source_on_run == load_source_on_run

    mlrun.api.utils.builder.build_image = original_build_runtime


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
    background_task = mlrun.common.schemas.BackgroundTask(**response.json())
    assert (
        background_task.status.state == mlrun.common.schemas.BackgroundTaskState.running
    )

    response = client.get(
        f"projects/{project}/background-tasks/{background_task.metadata.name}"
    )
    assert response.status_code == http.HTTPStatus.OK.value
    background_task = mlrun.common.schemas.BackgroundTask(**response.json())
    assert (
        background_task.status.state
        == mlrun.common.schemas.BackgroundTaskState.succeeded
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
    background_task = mlrun.common.schemas.BackgroundTask(**response.json())
    assert (
        background_task.status.state == mlrun.common.schemas.BackgroundTaskState.running
    )
    response = client.get(
        f"projects/{project}/background-tasks/{background_task.metadata.name}"
    )
    assert response.status_code == http.HTTPStatus.OK.value
    background_task = mlrun.common.schemas.BackgroundTask(**response.json())
    assert (
        background_task.status.state == mlrun.common.schemas.BackgroundTaskState.failed
    )


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
            "expected_status_result": mlrun.common.schemas.BackgroundTaskState.succeeded,
            "background_timeout_mode": "enabled",
            "dask_timeout": 100,
        },
        {
            "_start_function_mock": failing_func,
            "expected_status_result": mlrun.common.schemas.BackgroundTaskState.failed,
            "background_timeout_mode": "enabled",
            "dask_timeout": None,
        },
        {
            "_start_function_mock": unittest.mock.Mock,
            "expected_status_result": mlrun.common.schemas.BackgroundTaskState.succeeded,
            "background_timeout_mode": "disabled",
            "dask_timeout": 0,
        },
    ]:
        _start_function_mock = test_case.get("_start_function_mock", unittest.mock.Mock)
        expected_status_result = test_case.get(
            "expected_status_result", mlrun.common.schemas.BackgroundTaskState.running
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
        background_task = mlrun.common.schemas.BackgroundTask(**response.json())
        assert (
            background_task.status.state
            == mlrun.common.schemas.BackgroundTaskState.running
        )
        response = client.get(
            f"projects/{project}/background-tasks/{background_task.metadata.name}"
        )
        assert response.status_code == http.HTTPStatus.OK.value
        background_task = mlrun.common.schemas.BackgroundTask(**response.json())
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
