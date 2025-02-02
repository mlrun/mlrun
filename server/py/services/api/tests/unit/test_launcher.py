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
import pathlib
import unittest.mock
from contextlib import nullcontext as does_not_raise

import pytest
from fastapi.testclient import TestClient

import mlrun.common.runtimes.constants
import mlrun.common.schemas
import mlrun.launcher.base
import mlrun.launcher.factory

import framework.utils.clients.iguazio
import services.api.launcher
import services.api.tests.unit.api.utils


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
        assert isinstance(launcher, services.api.launcher.ServerSideLauncher)


def test_enrich_runtime_with_auth_info(
    monkeypatch, k8s_secrets_mock, client: TestClient
):
    mlrun.mlconf.httpdb.authentication.mode = "iguazio"
    monkeypatch.setattr(
        framework.utils.clients.iguazio,
        "AsyncClient",
        lambda *args, **kwargs: unittest.mock.AsyncMock(),
    )
    auth_info = mlrun.common.schemas.auth.AuthInfo(
        access_key="access_key",
        username="username",
    )
    services.api.tests.unit.api.utils.create_project(
        client, mlrun.mlconf.default_project
    )

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
    services.api.launcher.ServerSideLauncher._validate_state_thresholds(
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
        services.api.launcher.ServerSideLauncher._validate_state_thresholds(
            state_thresholds=state_thresholds
        )
    assert expected_error in str(exc.value)


def test_new_function_args_with_default_image_pull_secret(rundb_mock):
    assets_path = pathlib.Path(__file__).parent / "assets"
    func_path = assets_path / "sample_function.py"
    handler = "hello_word"

    mlrun.mlconf.function.spec.image_pull_secret = "adam-docker-registry-auth"
    launcher = services.api.launcher.ServerSideLauncher(
        auth_info=mlrun.common.schemas.AuthInfo()
    )
    runtime = mlrun.code_to_function(
        name="test",
        kind="job",
        filename=str(func_path),
        handler=handler,
        image="mlrun/mlrun",
    )
    uid = "123"
    run = mlrun.run.RunObject(
        metadata=mlrun.model.RunMetadata(uid=uid),
    )
    rundb_mock.store_run(run, uid)
    run = launcher._create_run_object(run)

    run = launcher._enrich_run(
        runtime,
        run=run,
    )
    assert run.spec.image_pull_secret == mlrun.mlconf.function.spec.image_pull_secret


@pytest.mark.parametrize(
    "end_time, run_state, should_update",
    [
        (None, mlrun.common.runtimes.constants.RunStates.completed, True),
        (None, mlrun.common.runtimes.constants.RunStates.error, True),
        (
            "2024-01-28T12:00:00Z",
            mlrun.common.runtimes.constants.RunStates.completed,
            False,
        ),
        (
            "2024-01-28T12:00:00Z",
            mlrun.common.runtimes.constants.RunStates.error,
            False,
        ),
        (None, mlrun.common.runtimes.constants.RunStates.running, False),
        (
            "2024-01-28T12:00:00Z",
            mlrun.common.runtimes.constants.RunStates.running,
            False,
        ),
    ],
)
def test_update_end_time_if_terminal_state(end_time, run_state, should_update):
    runtime = unittest.mock.MagicMock()
    runtime._get_db.return_value = unittest.mock.MagicMock()

    uid = "123"
    run = mlrun.run.RunObject(metadata=mlrun.model.RunMetadata(uid=uid))
    run.status.state = run_state
    run.status.end_time = end_time

    services.api.launcher.ServerSideLauncher._update_end_time_if_terminal_state(
        runtime, run
    )

    if should_update:
        db = runtime._get_db()
        db.update_run.assert_called_once()
        updates = db.update_run.call_args[0][0]
        assert "status.end_time" in updates
        assert updates["status.end_time"] is not None
    else:
        runtime._get_db().update_run.assert_not_called()
