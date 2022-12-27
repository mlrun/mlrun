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
import http
import json
import typing
import unittest.mock
from http import HTTPStatus

import fastapi.testclient
import pandas as pd
import pytest
import sqlalchemy.orm
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun
import mlrun.api.api.utils
import mlrun.api.main
import mlrun.api.utils.auth.verifier
import mlrun.api.utils.clients.chief
import mlrun.api.utils.clients.iguazio
import tests.api.api.utils
from mlrun.api.schemas import AuthInfo
from mlrun.api.utils.singletons.k8s import get_k8s
from mlrun.config import config as mlconf
from tests.api.conftest import K8sSecretsMock

ORIGINAL_VERSIONED_API_PREFIX = mlrun.api.main.BASE_VERSIONED_API_PREFIX
DEFAULT_FUNCTION_OUTPUT_PATH = "/some/fictive/path/to/make/everybody/happy"


def test_submit_job_failure_function_not_found(db: Session, client: TestClient) -> None:
    project = "project-name"
    tests.api.api.utils.create_project(client, project)

    function_reference = (
        "cat-and-dog-servers/aggregate@b145b6d958a7b4d84f12821a06459e31ea422308"
    )
    body = {
        "task": {
            "metadata": {"name": "task-name", "project": project},
            "spec": {"function": function_reference},
        },
    }
    resp = client.post("submit_job", json=body)
    assert resp.status_code == HTTPStatus.NOT_FOUND.value
    assert f"Function not found {function_reference}" in resp.json()["detail"]["reason"]


username = "voldemort"
access_key = "12345"


@pytest.fixture()
def pod_create_mock():
    create_pod_orig_function = get_k8s().create_pod
    _get_project_secrets_raw_data_orig_function = (
        get_k8s()._get_project_secrets_raw_data
    )
    get_k8s().create_pod = unittest.mock.Mock(return_value=("pod-name", "namespace"))
    get_k8s()._get_project_secrets_raw_data = unittest.mock.Mock(return_value={})

    update_run_state_orig_function = (
        mlrun.runtimes.kubejob.KubejobRuntime._update_run_state
    )
    mlrun.runtimes.kubejob.KubejobRuntime._update_run_state = unittest.mock.Mock()

    mock_run_object = mlrun.RunObject()
    mock_run_object.metadata.uid = "1234567890"
    mock_run_object.metadata.project = "project-name"

    wrap_run_result_orig_function = mlrun.runtimes.base.BaseRuntime._wrap_run_result
    mlrun.runtimes.base.BaseRuntime._wrap_run_result = unittest.mock.Mock(
        return_value=mock_run_object
    )

    auth_info_mock = AuthInfo(
        username=username, session="some-session", data_session=access_key
    )

    authenticate_request_orig_function = (
        mlrun.api.utils.auth.verifier.AuthVerifier().authenticate_request
    )
    mlrun.api.utils.auth.verifier.AuthVerifier().authenticate_request = (
        unittest.mock.Mock(return_value=auth_info_mock)
    )

    yield get_k8s().create_pod

    # Have to revert the mocks, otherwise other tests are failing
    get_k8s().create_pod = create_pod_orig_function
    get_k8s()._get_project_secrets_raw_data = (
        _get_project_secrets_raw_data_orig_function
    )
    mlrun.runtimes.kubejob.KubejobRuntime._update_run_state = (
        update_run_state_orig_function
    )
    mlrun.runtimes.base.BaseRuntime._wrap_run_result = wrap_run_result_orig_function
    mlrun.api.utils.auth.verifier.AuthVerifier().authenticate_request = (
        authenticate_request_orig_function
    )


def test_submit_job_auto_mount(
    db: Session, client: TestClient, pod_create_mock, k8s_secrets_mock
) -> None:
    mlconf.storage.auto_mount_type = "v3io_credentials"
    api_url = "https://api/url"
    # Set different auto-mount-params, to ensure the auth info is overridden
    mlconf.storage.auto_mount_params = (
        f"api={api_url},user=invalid-user,access_key=invalid-access-key"
    )
    project = "my-proj1"
    tests.api.api.utils.create_project(client, project)

    function_name = "test-function"
    function_tag = "latest"
    function = mlrun.new_function(
        name=function_name,
        project=project,
        tag=function_tag,
        kind="job",
        image="mlrun/mlrun",
    )
    submit_job_body = _create_submit_job_body(function, project)

    resp = client.post("submit_job", json=submit_job_body)
    assert resp
    secret_name = k8s_secrets_mock.get_auth_secret_name(username, access_key)
    expected_env_vars = {
        "V3IO_API": api_url,
        "V3IO_USERNAME": username,
        "V3IO_ACCESS_KEY": (
            secret_name,
            mlrun.api.schemas.AuthSecretData.get_field_secret_key("access_key"),
        ),
    }
    _assert_pod_env_vars(pod_create_mock, expected_env_vars)


def test_submit_job_ensure_function_has_auth_set(
    db: Session, client: TestClient, pod_create_mock, k8s_secrets_mock
) -> None:
    mlrun.mlconf.httpdb.authentication.mode = "iguazio"
    project = "my-proj1"
    tests.api.api.utils.create_project(client, project)

    function = mlrun.new_function(
        name="test-function",
        project=project,
        tag="latest",
        kind="job",
        image="mlrun/mlrun",
    )
    access_key = "some-access-key"
    function.metadata.credentials.access_key = access_key
    submit_job_body = _create_submit_job_body(function, project)
    resp = client.post("submit_job", json=submit_job_body)
    assert resp

    secret_name = k8s_secrets_mock.get_auth_secret_name(username, access_key)
    expected_env_vars = {
        mlrun.runtimes.constants.FunctionEnvironmentVariables.auth_session: (
            secret_name,
            mlrun.api.schemas.AuthSecretData.get_field_secret_key("access_key"),
        ),
    }
    _assert_pod_env_vars(pod_create_mock, expected_env_vars)


def test_submit_job_with_output_path_enrichment(
    db: Session, client: TestClient, pod_create_mock, k8s_secrets_mock
) -> None:
    project_name = "proj-with-artifact-path"
    project_artifact_path = f"/{project_name}"
    tests.api.api.utils.create_project(
        client, project_name, artifact_path=project_artifact_path
    )
    function = mlrun.new_function(
        name="test-function",
        project=project_name,
        tag="latest",
        kind="job",
        image="mlrun/mlrun",
    )
    # set default artifact path
    mlconf.artifact_path = "/some-path"
    submit_job_body = _create_submit_job_body(
        function, project_name, with_output_path=False
    )

    resp = client.post("submit_job", json=submit_job_body)
    assert resp.status_code == http.HTTPStatus.OK.value

    # expected to get enriched with the project artifact path
    _assert_pod_output_path(pod_create_mock, project_artifact_path)

    # create project without default artifact path
    project_name = "proj-without-artifact-path"
    tests.api.api.utils.create_project(client, project_name)
    function = mlrun.new_function(
        name="test-function",
        project=project_name,
        tag="latest",
        kind="job",
        image="mlrun/mlrun",
    )
    # set default artifact path
    mlconf.artifact_path = "/some-path"
    submit_job_body = _create_submit_job_body(
        function, project_name, with_output_path=False
    )
    resp = client.post("submit_job", json=submit_job_body)
    assert resp.status_code == http.HTTPStatus.OK.value

    # project doesn't have default artifact path, expected to get enriched with the default artifact path
    _assert_pod_output_path(pod_create_mock, mlconf.artifact_path)

    function = mlrun.new_function(
        name="test-function-with-output-path",
        project=project_name,
        tag="latest",
        kind="job",
        image="mlrun/mlrun",
    )
    # create task with output_path, expected to be used
    submit_job_body = _create_submit_job_body(
        function, project_name, with_output_path=True
    )
    resp = client.post("submit_job", json=submit_job_body)
    assert resp.status_code == http.HTTPStatus.OK.value

    _assert_pod_output_path(pod_create_mock, DEFAULT_FUNCTION_OUTPUT_PATH)


def test_submit_job_service_accounts(
    db: Session, client: TestClient, pod_create_mock, k8s_secrets_mock: K8sSecretsMock
):
    project = "my-proj1"
    tests.api.api.utils.create_project(client, project)

    # must set the default project since new_function creates the function object and ignores the project parameter.
    # Instead, the function always gets the default project name. This may be a bug, need to check.
    mlconf.default_project = project

    function_name = "test-function"
    function_tag = "latest"

    k8s_secrets_mock.set_service_account_keys(project, "sa1", ["sa1", "sa2"])

    function = mlrun.new_function(
        name=function_name,
        project=project,
        tag=function_tag,
        kind="job",
        image="mlrun/mlrun",
    )
    submit_job_body = _create_submit_job_body(function, project)

    resp = client.post("submit_job", json=submit_job_body)
    assert resp
    _assert_pod_service_account(pod_create_mock, "sa1")

    pod_create_mock.reset_mock()
    function.spec.service_account = "sa2"
    submit_job_body = _create_submit_job_body(function, project)

    resp = client.post("submit_job", json=submit_job_body)
    assert resp
    _assert_pod_service_account(pod_create_mock, "sa2")

    # Invalid service-account
    pod_create_mock.reset_mock()
    function.spec.service_account = "sa3"
    submit_job_body = _create_submit_job_body(function, project)
    resp = client.post("submit_job", json=submit_job_body)
    assert resp.status_code == HTTPStatus.BAD_REQUEST.value

    # Validate that without setting the secrets, any SA is allowed
    k8s_secrets_mock.delete_project_secrets(project, None)
    pod_create_mock.reset_mock()
    resp = client.post("submit_job", json=submit_job_body)
    assert resp
    _assert_pod_service_account(pod_create_mock, "sa3")

    # Validate that a global service account works as expected
    pod_create_mock.reset_mock()
    mlconf.function.spec.service_account.default = "some-sa"
    function.spec.service_account = None
    submit_job_body = _create_submit_job_body(function, project)
    resp = client.post("submit_job", json=submit_job_body)
    assert resp
    _assert_pod_service_account(pod_create_mock, "some-sa")
    mlconf.function.spec.service_account.default = None


class _MockDataItem:
    def as_df(self):
        return pd.DataFrame({"key1": [0], "key2": [1]})


def test_submit_job_with_hyper_params_file(
    db: Session,
    client: TestClient,
    pod_create_mock,
    k8s_secrets_mock: K8sSecretsMock,
    monkeypatch,
):
    project_name = "proj-with-hyper-params"
    project_artifact_path = f"/{project_name}"
    tests.api.api.utils.create_project(
        client, project_name, artifact_path=project_artifact_path
    )
    function = mlrun.new_function(
        name="test-function",
        project=project_name,
        tag="latest",
        kind="job",
        image="mlrun/mlrun",
    )
    # set default artifact path
    mlconf.artifact_path = "/some-path"
    submit_job_body = _create_submit_job_body(
        function, project_name, with_output_path=False
    )

    # Create test-specific mocks
    auth_info = mlrun.api.schemas.AuthInfo(username="user", access_key=access_key)
    monkeypatch.setattr(
        mlrun.api.utils.auth.verifier.AuthVerifier(),
        "authenticate_request",
        lambda *args, **kwargs: auth_info,
    )
    orig_get_dataitem = mlrun.MLClientCtx.get_dataitem
    mlrun.MLClientCtx.get_dataitem = unittest.mock.Mock(return_value=_MockDataItem())

    project_secrets = {"SECRET1": "VALUE1"}
    k8s_secrets_mock.store_project_secrets(project_name, project_secrets)

    # Configure hyper-param related values
    task_spec = submit_job_body["task"]["spec"]
    task_spec["param_file"] = "v3io://users/user1"
    task_spec["selector"] = "max.loss"
    task_spec["strategy"] = "list"

    resp = client.post("submit_job", json=submit_job_body)
    assert resp.status_code == http.HTTPStatus.OK.value

    # Validate that secrets were properly passed to get_dataitem
    project_secrets.update({"V3IO_ACCESS_KEY": access_key})
    mlrun.MLClientCtx.get_dataitem.assert_called_once_with(
        task_spec["param_file"], secrets=project_secrets
    )

    mlrun.MLClientCtx.get_dataitem = orig_get_dataitem


def test_redirection_from_worker_to_chief_only_if_schedules_in_job(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    httpserver,
    monkeypatch,
):
    mlrun.mlconf.httpdb.clusterization.role = "worker"

    project = "test-project"
    function_name = "test-function"
    function_tag = "latest"

    tests.api.api.utils.create_project(client, project)
    function = mlrun.new_function(
        name=function_name,
        project=project,
        tag=function_tag,
        kind="job",
        image="mlrun/mlrun",
    )

    handler_mock = mlrun.api.utils.clients.chief.Client()
    handler_mock._proxy_request_to_chief = unittest.mock.Mock(
        return_value=fastapi.Response()
    )
    monkeypatch.setattr(
        mlrun.api.utils.clients.chief,
        "Client",
        lambda *args, **kwargs: handler_mock,
    )

    submit_job_body = _create_submit_job_body_with_schedule(function, project)
    json_body = mlrun.utils.dict_to_json(submit_job_body)
    client.post("submit_job", data=json_body)
    assert handler_mock._proxy_request_to_chief.call_count == 1

    handler_mock._proxy_request_to_chief.reset_mock()

    submit_job_body = _create_submit_job_body(function, project)
    json_body = mlrun.utils.dict_to_json(submit_job_body)
    client.post("submit_job", data=json_body)
    # no schedule inside job body, expecting to be run in worker
    assert handler_mock._proxy_request_to_chief.call_count == 0


def test_redirection_from_worker_to_chief_submit_job_with_schedule(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient, httpserver
):
    mlrun.mlconf.httpdb.clusterization.role = "worker"
    endpoint = f"{ORIGINAL_VERSIONED_API_PREFIX}/submit_job"
    project = "test-project"

    function_name = "test-function"
    function_tag = "latest"
    function = mlrun.new_function(
        name=function_name,
        project=project,
        tag=function_tag,
        kind="job",
        image="mlrun/mlrun",
    )

    tests.api.api.utils.create_project(client, project)
    submit_job_body = _create_submit_job_body_with_schedule(function, project)

    for test_case in [
        {
            "body": submit_job_body,
            "expected_status": http.HTTPStatus.OK.value,
            "expected_body": {
                "data": {
                    "schedule": submit_job_body["schedule"],
                    "project": project,
                    "name": function_name,
                }
            },
        },
        {
            "body": submit_job_body,
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
        json_body = mlrun.utils.dict_to_json(body)
        response = client.post(endpoint, data=json_body)
        assert response.status_code == expected_status
        assert response.json() == expected_response


@pytest.mark.parametrize(
    "task_name,parameters,hyperparameters",
    [
        ("param-pos", {"x": 2**63 + 1}, None),
        ("param-neg", {"x": -(2**63 + 1)}, None),
        ("hyperparam-pos", None, {"x": [1, 2**63 + 1]}),
        ("hyperparam-neg", None, {"x": [1, -(2**63 + 1)]}),
    ],
)
def test_submit_job_failure_params_exceed_int64(
    db: Session,
    client: TestClient,
    pod_create_mock,
    task_name: str,
    parameters: typing.Dict[str, int],
    hyperparameters: typing.Dict[str, typing.List[int]],
) -> None:
    project_name = "params-exceed-int64"
    project_artifact_path = f"/{project_name}"
    tests.api.api.utils.create_project(
        client, project_name, artifact_path=project_artifact_path
    )
    function = mlrun.new_function(
        name="test-function",
        project=project_name,
        tag="latest",
        kind="job",
        image="mlrun/mlrun",
    )
    submit_job_body = _create_submit_job_body(function, project_name)
    submit_job_body["task"]["metadata"]["name"] = task_name
    if parameters:
        submit_job_body["task"]["spec"]["parameters"] = parameters
    if hyperparameters:
        submit_job_body["task"]["spec"]["hyperparams"] = hyperparameters
    resp = client.post("submit_job", json=submit_job_body)

    assert resp.status_code == HTTPStatus.BAD_REQUEST.value
    assert "exceeds int64" in resp.json()["detail"]["reason"]

    resp = client.get("runs", params={"project": project_name})
    # assert the run wasn't saved to the DB
    assert len(resp.json()["runs"]) == 0


def _create_submit_job_body(function, project, with_output_path=True):
    body = {
        "task": {
            "spec": {},
            "metadata": {"name": "task1", "project": project},
        },
        "function": function.to_dict(),
    }
    if with_output_path:
        body["task"]["spec"]["output_path"] = DEFAULT_FUNCTION_OUTPUT_PATH

    return body


def _create_submit_job_body_with_schedule(function, project):
    job_body = _create_submit_job_body(function, project)
    job_body["schedule"] = mlrun.api.schemas.ScheduleCronTrigger(year=1999).dict()
    return job_body


def _assert_pod_env_vars(pod_create_mock, expected_env_vars):
    pod_env_dict = _get_pod_env_vars_as_dict(pod_create_mock)
    for key, value in expected_env_vars.items():
        assert pod_env_dict[key] == value


def _get_pod_env_vars_as_dict(pod_create_mock, assert_called=True):
    if assert_called:
        pod_create_mock.assert_called_once()
    args, _ = pod_create_mock.call_args
    pod_env = args[0].spec.containers[0].env
    pod_env_dict = {}
    for env_item in pod_env:
        name = mlrun.runtimes.utils.get_item_name(env_item)
        value = mlrun.runtimes.utils.get_item_name(env_item, "value")
        value_from = mlrun.runtimes.utils.get_item_name(env_item, "value_from")
        if value:
            pod_env_dict[name] = value
        else:
            pod_env_dict[name] = (
                value_from.secret_key_ref.name,
                value_from.secret_key_ref.key,
            )
    return pod_env_dict


def _assert_pod_output_path(pod_create_mock, expected_output_path):
    pod_env_dict = _get_pod_env_vars_as_dict(pod_create_mock, assert_called=False)
    pod_mlrun_exec_config: dict = json.loads(
        pod_env_dict["MLRUN_EXEC_CONFIG"],
    )
    assert pod_mlrun_exec_config["spec"]["output_path"] == expected_output_path


def _assert_pod_service_account(pod_create_mock, expected_service_account):
    pod_create_mock.assert_called_once()
    args, _ = pod_create_mock.call_args
    pod_spec = args[0].spec
    assert pod_spec.service_account == expected_service_account
