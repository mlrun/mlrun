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

import unittest.mock
from contextlib import nullcontext as does_not_raise

import pytest
from fastapi.testclient import TestClient

import mlrun.common.schemas
import mlrun.launcher.base
import mlrun.launcher.factory
import server.api.launcher
import server.api.utils.clients.iguazio
import tests.api.api.utils


@pytest.mark.parametrize(
    "is_remote, local, expectation",
    [
        (True, False, does_not_raise()),
        (False, False, does_not_raise()),
        # local run is not allowed when running as API
        (True, True, pytest.raises(mlrun.errors.MLRunPreconditionFailedError)),
        (False, True, pytest.raises(mlrun.errors.MLRunPreconditionFailedError)),
    ],
)
def test_create_server_side_launcher(is_remote, local, expectation):
    """Test that the server side launcher is created when we are running as API"""
    with expectation:
        launcher = mlrun.launcher.factory.LauncherFactory().create_launcher(
            is_remote,
            local=local,
        )
        assert isinstance(launcher, server.api.launcher.ServerSideLauncher)


def test_enrich_runtime_with_auth_info(
    monkeypatch, k8s_secrets_mock, client: TestClient
):
    mlrun.mlconf.httpdb.authentication.mode = "iguazio"
    monkeypatch.setattr(
        server.api.utils.clients.iguazio,
        "AsyncClient",
        lambda *args, **kwargs: unittest.mock.AsyncMock(),
    )
    auth_info = mlrun.common.schemas.auth.AuthInfo(
        access_key="access_key",
        username="username",
    )
    tests.api.api.utils.create_project(client, mlrun.mlconf.default_project)

    launcher_kwargs = {"auth_info": auth_info}
    launcher = mlrun.launcher.factory.LauncherFactory().create_launcher(
        is_remote=True,
        **launcher_kwargs,
    )

    assert launcher._auth_info == auth_info
    function = mlrun.new_function(
        name="launcher-test",
        kind="job",
    )
    function.metadata.credentials.access_key = (
        mlrun.model.Credentials.generate_access_key
    )

    launcher.enrich_runtime(function)
    assert (
        function.get_env("MLRUN_AUTH_SESSION").secret_key_ref.name
        == "secret-ref-username-access_key"
    )


def test_validate_state_thresholds_success():
    server.api.launcher.ServerSideLauncher._validate_state_thresholds(
        state_thresholds={
            "pending_scheduled": "-1",
            "executing": "1000s",
            "image_pull_backoff": "3m",
        }
    )


@pytest.mark.parametrize(
    "state_thresholds, expected_error",
    [
        (
            {
                "pending_scheduled": "-1",
                "executing": "1000s",
                "image_pull_backoff": "3mm",
            },
            "Threshold '3mm' for state 'image_pull_backoff' is not a valid timelength string. "
            "Error: Input TimeLength \"3mm\" contains an invalid value: ['mm']",
        ),
        (
            {
                "pending_scheduled": -1,
            },
            "Threshold '-1' for state 'pending_scheduled' must be a string",
        ),
        (
            {
                "unknown_state": "10s",
            },
            f"Invalid state unknown_state for state threshold, must be one of "
            f"{mlrun.common.runtimes.constants.ThresholdStates.all()}",
        ),
        (
            {
                "executing": "10",
            },
            "Threshold '10' for state 'executing' is not a valid timelength string. "
            'Error: Input TimeLength "10" contains no valid Value and Scale pairs.',
        ),
    ],
)
def test_validate_state_thresholds_failure(state_thresholds, expected_error):
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc:
        server.api.launcher.ServerSideLauncher._validate_state_thresholds(
            state_thresholds=state_thresholds
        )
    assert expected_error in str(exc.value)
