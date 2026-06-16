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

import collections.abc
import unittest.mock

import pytest

import mlrun
import mlrun.common.schemas
import mlrun.errors
import mlrun.utils.singleton

import services.api.crud.projects as projects_crud


@pytest.fixture
def reset_projects_singleton() -> collections.abc.Iterator[None]:
    """Drop the Projects singleton so __init__ re-runs with the current mlconf."""
    mlrun.utils.singleton.Singleton._instances.pop(projects_crud.Projects, None)
    yield
    mlrun.utils.singleton.Singleton._instances.pop(projects_crud.Projects, None)


@pytest.fixture
def patched_nuclio_deletion_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> unittest.mock.MagicMock:
    """Stub the externals ``_wait_for_nuclio_project_deletion`` touches: Nuclio
    reports the project already gone, no function pods remain, and the service
    account client returns sentinel headers. Returns the Nuclio client mock."""
    mlrun.mlconf.nuclio_dashboard_url = "http://nuclio-dashboard:8070"

    nuclio_client_mock = unittest.mock.MagicMock()
    nuclio_client_mock.get_project.side_effect = mlrun.errors.MLRunNotFoundError(
        "not found"
    )
    monkeypatch.setattr(
        "framework.utils.clients.nuclio.Client", lambda: nuclio_client_mock
    )

    k8s_helper_mock = unittest.mock.MagicMock()
    k8s_helper_mock.list_pods.return_value = []
    monkeypatch.setattr(
        "framework.utils.singletons.k8s.get_k8s_helper", lambda: k8s_helper_mock
    )

    monkeypatch.setattr(
        projects_crud.service_account_token.Client,
        "escalate_request_headers",
        lambda self, headers: {"Authorization": "Bearer sa-token"},
    )
    return nuclio_client_mock


def test_wait_for_nuclio_project_deletion_polls_with_service_account_in_iguazio_v4(
    reset_projects_singleton: None,
    patched_nuclio_deletion_dependencies: unittest.mock.MagicMock,
    monkeypatch: pytest.MonkeyPatch,
):
    """In IG4 the deletion poll uses the service account token, not the user's
    expiring one."""
    monkeypatch.setattr(type(mlrun.mlconf), "is_iguazio_v4_mode", lambda self: True)
    nuclio_client_mock = patched_nuclio_deletion_dependencies
    user_auth_info = mlrun.common.schemas.AuthInfo(
        request_headers={"Authorization": "Bearer user-token-that-will-expire"},
    )

    projects_crud.Projects()._wait_for_nuclio_project_deletion(
        project_name="some-project",
        session=unittest.mock.MagicMock(),
        auth_info=user_auth_info,
    )

    polled_auth_info = nuclio_client_mock.get_project.call_args.kwargs["auth_info"]
    assert polled_auth_info.request_headers == {"Authorization": "Bearer sa-token"}
    # the caller's auth_info must remain untouched for downstream use
    assert user_auth_info.request_headers == {
        "Authorization": "Bearer user-token-that-will-expire"
    }


def test_wait_for_nuclio_project_deletion_keeps_user_auth_when_not_iguazio_v4(
    reset_projects_singleton: None,
    patched_nuclio_deletion_dependencies: unittest.mock.MagicMock,
    monkeypatch: pytest.MonkeyPatch,
):
    """Outside IG4 there is no service account escalation; the poll keeps using
    the supplied auth_info unchanged."""
    monkeypatch.setattr(type(mlrun.mlconf), "is_iguazio_v4_mode", lambda self: False)
    nuclio_client_mock = patched_nuclio_deletion_dependencies
    user_auth_info = mlrun.common.schemas.AuthInfo(session="user-session")

    projects_crud.Projects()._wait_for_nuclio_project_deletion(
        project_name="some-project",
        session=unittest.mock.MagicMock(),
        auth_info=user_auth_info,
    )

    polled_auth_info = nuclio_client_mock.get_project.call_args.kwargs["auth_info"]
    assert polled_auth_info is user_auth_info
