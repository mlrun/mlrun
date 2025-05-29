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

import pathlib
import re
import unittest.mock
from contextlib import nullcontext as does_not_raise

import pytest
import sqlalchemy.orm
from fastapi.testclient import TestClient

import mlrun.common.runtimes.constants
import mlrun.common.schemas
import mlrun.launcher.base
import mlrun.launcher.factory
from mlrun.config import Config

import framework.utils.clients.iguazio
import services.api.launcher
import services.api.tests.unit.api.utils

assets_path = pathlib.Path(__file__).parent / "assets"
func_path = assets_path / "sample_function.py"
handler = "hello_word"


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
    project = "some-project"
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
    services.api.tests.unit.api.utils.create_project(client, project)

    launcher_kwargs = {"auth_info": auth_info}
    launcher = mlrun.launcher.factory.LauncherFactory().create_launcher(
        is_remote=True,
        **launcher_kwargs,
    )

    assert launcher._auth_info == auth_info
    function = mlrun.new_function(
        name="launcher-test",
        kind="job",
        project=project,
    )
    function.metadata.credentials.access_key = (
        mlrun.model.Credentials.generate_access_key
    )

    launcher.enrich_runtime(function, project)
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


def test_new_function_args_with_default_image_pull_secret(
    db: sqlalchemy.orm.Session, client: TestClient
):
    project = "some-project"
    services.api.tests.unit.api.utils.create_project(client, project)

    mlrun.mlconf.function.spec.image_pull_secret = Config(
        {"default": "adam-docker-registry-auth"}
    )
    launcher = services.api.launcher.ServerSideLauncher(
        auth_info=mlrun.common.schemas.AuthInfo()
    )
    runtime = mlrun.code_to_function(
        name="test",
        kind="job",
        filename=str(func_path),
        handler=handler,
        image="mlrun/mlrun",
        project=project,
    )
    uid = "123"
    run = {
        "metadata": {
            "uid": uid,
            "name": "test",
        },
    }
    rundb = mlrun.get_run_db()
    rundb.store_run(run, uid, project)
    run = launcher._create_run_object(run)

    run = launcher._enrich_run(
        runtime,
        run=run,
    )
    assert (
        run.spec.image_pull_secret
        == mlrun.mlconf.function.spec.image_pull_secret.default
    )
    launcher.enrich_runtime(runtime, project, full=True)
    assert (
        runtime.spec.image_pull_secret
        == mlrun.mlconf.function.spec.image_pull_secret.default
    )


@pytest.mark.parametrize(
    "count, base_delay, default_base_delay, min_base_delay, expectation",
    [
        (None, None, "30s", "30s", does_not_raise()),
        (
            1,
            "29s",
            "30s",
            "30s",
            pytest.raises(
                mlrun.errors.MLRunInvalidArgumentError,
                match="Retry backoff base_delay must be at least 30s, got 29s",
            ),
        ),
        (
            1,
            "31s",
            "30s",
            "5m",
            pytest.raises(
                mlrun.errors.MLRunInvalidArgumentError,
                match="Retry backoff base_delay must be at least 5m, got 31s",
            ),
        ),
        (3, None, "30s", "30s", does_not_raise()),
        (3, "1 min", "30s", "30s", does_not_raise()),
        (
            -1,
            None,
            "30s",
            "30s",
            pytest.raises(
                mlrun.errors.MLRunInvalidArgumentError,
                match="Retry count must be at least 0, got -1",
            ),
        ),
    ],
)
def test_validate_run_retry(
    count, base_delay, default_base_delay, min_base_delay, expectation
):
    mlrun.mlconf.function.spec.retry.backoff.default_base_delay = default_base_delay
    mlrun.mlconf.function.spec.retry.backoff.min_base_delay = min_base_delay
    launcher = services.api.launcher.ServerSideLauncher(
        auth_info=mlrun.common.schemas.AuthInfo()
    )
    runtime = mlrun.code_to_function(
        name="test", kind="job", filename=str(func_path), handler=handler
    )

    retry = None
    if count or base_delay:
        retry = {}
        if count is not None:
            retry["count"] = count

        if base_delay is not None:
            retry["backoff"] = {
                "base_delay": base_delay,
            }

    run = mlrun.run.RunObject(
        spec=mlrun.model.RunSpec(
            retry=retry,
        ),
    )
    assert run.spec.retry.count == (count if count else None)

    if count:
        assert run.spec.retry.backoff.base_delay == (
            base_delay if base_delay is not None else default_base_delay
        )
    else:
        assert run.spec.retry.backoff is None
    with (
        expectation,
    ):
        launcher._validate_retry(runtime.kind, run.spec.retry)


def test_validate_run_retry_runtime_kind():
    launcher = services.api.launcher.ServerSideLauncher(
        auth_info=mlrun.common.schemas.AuthInfo()
    )
    runtime = mlrun.code_to_function(
        name="test", kind="mpijob", filename=str(func_path), handler=handler
    )

    retry = {
        "count": 3,
    }
    run = mlrun.run.RunObject(
        spec=mlrun.model.RunSpec(
            retry=retry,
        ),
    )
    with (
        pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match=re.escape(
                f"Retry is not supported for runtime kind mpijob, supported kinds are: "
                f"{mlrun.runtimes.RuntimeKinds.retriable_runtimes()}"
            ),
        ),
    ):
        launcher._validate_run(runtime, run)
