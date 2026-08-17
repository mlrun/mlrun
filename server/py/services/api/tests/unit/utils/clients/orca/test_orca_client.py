# Copyright 2026 Iguazio
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

import uuid

import pytest
import requests_mock as requests_mock_package

import mlrun.common.schemas
import mlrun.config
import mlrun.errors

import framework.utils.clients.orca.client


@pytest.fixture()
def api_url() -> str:
    api_url = "http://orca-api-url:8080"
    mlrun.mlconf.orca_api_url = api_url
    return api_url


@pytest.fixture()
def fast_poll(monkeypatch):
    monkeypatch.setattr(
        mlrun.mlconf.httpdb.projects, "orca_project_states_poll_interval", "0 seconds"
    )
    monkeypatch.setattr(
        mlrun.mlconf.httpdb.projects, "orca_project_states_poll_timeout", "2 seconds"
    )


@pytest.fixture()
def orca_client(api_url: str, fast_poll) -> framework.utils.clients.orca.client.Client:
    return framework.utils.clients.orca.client.Client()


@pytest.fixture()
def auth_info() -> mlrun.common.schemas.AuthInfo:
    return mlrun.common.schemas.AuthInfo(
        request_headers={"authorization": "Bearer some-user-jwt"}
    )


def _generate_project(name="project-name") -> mlrun.common.schemas.Project:
    return mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=name),
        spec=mlrun.common.schemas.ProjectSpec(owner="some-owner", description="desc"),
    )


def _project_state_body(name: str, op_id: str, state: str) -> dict:
    return {
        "metadata": {"name": name, "labels": {}, "annotations": {}},
        "spec": {"owner": "some-owner", "description": "desc"},
        "status": {
            "state": state,
            "op_id": op_id,
            "updated_at": "2026-08-14T00:00:00+00:00",
        },
    }


def test_create_project_polls_until_online(
    api_url: str,
    orca_client: framework.utils.clients.orca.client.Client,
    auth_info: mlrun.common.schemas.AuthInfo,
    requests_mock: requests_mock_package.Mocker,
):
    project = _generate_project()
    op_id = str(uuid.uuid4())
    requests_mock.post(
        f"{api_url}/api/v1/projects",
        json={"status": {"op_id": op_id}},
        status_code=202,
    )
    requests_mock.get(
        f"{api_url}/api/v1/project-states/{project.metadata.name}",
        [
            {"json": _project_state_body(project.metadata.name, op_id, "creating")},
            {"json": _project_state_body(project.metadata.name, op_id, "online")},
        ],
    )

    is_running_in_background = orca_client.create_project(
        "unused-session", project, auth_info, wait_for_completion=True
    )

    assert is_running_in_background is False
    post_request = requests_mock.request_history[0]
    assert post_request.headers["authorization"] == "Bearer some-user-jwt"


def test_create_project_does_not_assert_leader_role(
    api_url: str,
    orca_client: framework.utils.clients.orca.client.Client,
    auth_info: mlrun.common.schemas.AuthInfo,
    requests_mock: requests_mock_package.Mocker,
):
    project = _generate_project()
    op_id = str(uuid.uuid4())
    requests_mock.post(
        f"{api_url}/api/v1/projects",
        json={"status": {"op_id": op_id}},
        status_code=202,
    )

    orca_client.create_project(
        "unused-session", project, auth_info, wait_for_completion=False
    )

    # a pure identity relay must never assert mlrun's own leader-role identity - only the user's
    post_request = requests_mock.request_history[0]
    assert "x-projects-role" not in post_request.headers


def test_create_project_async_returns_immediately(
    api_url: str,
    orca_client: framework.utils.clients.orca.client.Client,
    auth_info: mlrun.common.schemas.AuthInfo,
    requests_mock: requests_mock_package.Mocker,
):
    project = _generate_project()
    op_id = str(uuid.uuid4())
    requests_mock.post(
        f"{api_url}/api/v1/projects",
        json={"status": {"op_id": op_id}},
        status_code=202,
    )

    is_running_in_background = orca_client.create_project(
        "unused-session", project, auth_info, wait_for_completion=False
    )

    assert is_running_in_background is True
    # no polling should have happened
    assert len(requests_mock.request_history) == 1


def test_create_project_poll_timeout_raises(
    api_url: str,
    orca_client: framework.utils.clients.orca.client.Client,
    auth_info: mlrun.common.schemas.AuthInfo,
    requests_mock: requests_mock_package.Mocker,
):
    project = _generate_project()
    op_id = str(uuid.uuid4())
    requests_mock.post(
        f"{api_url}/api/v1/projects",
        json={"status": {"op_id": op_id}},
        status_code=202,
    )
    # never reaches a terminal state
    requests_mock.get(
        f"{api_url}/api/v1/project-states/{project.metadata.name}",
        json=_project_state_body(project.metadata.name, op_id, "creating"),
    )

    with pytest.raises(mlrun.errors.MLRunRetryExhaustedError):
        orca_client.create_project(
            "unused-session", project, auth_info, wait_for_completion=True
        )


def test_update_project_uses_current_op_id_and_polls(
    api_url: str,
    orca_client: framework.utils.clients.orca.client.Client,
    auth_info: mlrun.common.schemas.AuthInfo,
    requests_mock: requests_mock_package.Mocker,
):
    project = _generate_project()
    previous_op_id = str(uuid.uuid4())
    new_op_id = str(uuid.uuid4())
    project.status.op_id = previous_op_id

    requests_mock.put(
        f"{api_url}/api/v1/projects/{project.metadata.name}",
        json={"status": {"op_id": new_op_id}},
        status_code=202,
    )
    requests_mock.get(
        f"{api_url}/api/v1/project-states/{project.metadata.name}",
        json=_project_state_body(project.metadata.name, new_op_id, "online"),
    )

    orca_client.update_project(
        "unused-session", project.metadata.name, project, auth_info
    )

    put_request = requests_mock.request_history[0]
    assert put_request.json()["current_op_id"] == previous_op_id


def test_delete_project_polls_until_absent(
    api_url: str,
    orca_client: framework.utils.clients.orca.client.Client,
    auth_info: mlrun.common.schemas.AuthInfo,
    requests_mock: requests_mock_package.Mocker,
):
    project = _generate_project()
    op_id = str(uuid.uuid4())
    requests_mock.delete(
        f"{api_url}/api/v1/projects/{project.metadata.name}",
        json={"status": {"op_id": op_id}},
        status_code=202,
    )
    requests_mock.get(
        f"{api_url}/api/v1/project-states/{project.metadata.name}",
        status_code=404,
    )

    is_running_in_background = orca_client.delete_project(
        "unused-session", project.metadata.name, auth_info, wait_for_completion=True
    )

    assert is_running_in_background is False


def test_list_projects_not_implemented(
    orca_client: framework.utils.clients.orca.client.Client,
    auth_info: mlrun.common.schemas.AuthInfo,
):
    with pytest.raises(NotImplementedError):
        orca_client.list_projects("unused-session", auth_info)


def test_get_project_owner_not_implemented(
    orca_client: framework.utils.clients.orca.client.Client,
    auth_info: mlrun.common.schemas.AuthInfo,
):
    with pytest.raises(NotImplementedError):
        orca_client.get_project_owner("unused-session", "project-name", auth_info)


def test_format_as_leader_project_not_implemented(
    orca_client: framework.utils.clients.orca.client.Client,
):
    with pytest.raises(NotImplementedError):
        orca_client.format_as_leader_project(_generate_project())
