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
import uuid
from contextlib import nullcontext as does_not_raise

import pytest
import sqlalchemy.orm
from fastapi.testclient import TestClient

import mlrun.common.constants
import mlrun.common.runtimes.constants
import mlrun.common.schemas
import mlrun.errors
import mlrun.launcher.base
import mlrun.launcher.factory
from mlrun.common.types import AuthenticationMode
from mlrun.config import Config

import services.api.launcher
import services.api.tests.unit.api.utils
import services.api.utils.helpers

assets_path = pathlib.Path(__file__).parent / "assets"
func_path = assets_path / "sample_function.py"
handler = "hello_word"


@pytest.fixture
def random_project_name():
    return f"some-project-{uuid.uuid4().hex[:8]}"


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
    monkeypatch, k8s_secrets_mock, client: TestClient, random_project_name: str
):
    project = random_project_name
    mlrun.mlconf.httpdb.authentication.mode = AuthenticationMode.IGUAZIO

    services.api.tests.unit.api.utils.setup_iguazio_v3_async_client_mock(monkeypatch)
    auth_info = mlrun.common.schemas.auth.AuthInfo(
        access_key="access_key",
        username="username",
    )
    services.api.tests.unit.api.utils.create_project(client, project_name=project)

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
    db: sqlalchemy.orm.Session, client: TestClient, random_project_name: str
):
    project = random_project_name
    services.api.tests.unit.api.utils.create_project(client, project_name=project)

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
        (
            10,
            "7 days",
            None,
            None,
            pytest.raises(
                mlrun.errors.MLRunInvalidArgumentError,
                match=re.escape(
                    "Retry backoff base_delay 7 days * retry count 10 must be less than 259200 seconds, "
                    "got 6048000 seconds"
                ),
            ),
        ),
    ],
)
def test_validate_run_retry(
    count, base_delay, default_base_delay, min_base_delay, expectation
):
    if default_base_delay:
        mlrun.mlconf.function.spec.retry.backoff.default_base_delay = default_base_delay
    if min_base_delay:
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


def test_run_status_retry_updates():
    """
    Test that the run status is updated when a retry is triggered.
    The test simulates a run that is in the pending_retry state and checks that the retry count is incremented
    and the state is updated to running when the run is enriched again.
    """
    runtime = mlrun.code_to_function(
        name="test", kind="job", filename=str(func_path), handler="raise_func"
    )
    run = mlrun.run.RunObject(
        spec=mlrun.model.RunSpec(
            retry={
                "count": 10,
            },
        ),
        status=mlrun.model.RunStatus(
            state=mlrun.common.runtimes.constants.RunStates.pending_retry,
        ),
    )

    launcher = services.api.launcher.ServerSideLauncher()
    enriched_run = launcher._enrich_run(runtime=runtime, run=run)
    assert (
        enriched_run.status.state == mlrun.common.runtimes.constants.RunStates.running
    )
    assert enriched_run.status.start_time is None
    assert enriched_run.status.retry_count == 1, "Expected retry count to be 1"
    assert run.metadata.labels[mlrun.common.constants.MLRunInternalLabels.retry] == str(
        enriched_run.status.retry_count
    )
    assert enriched_run.status.retries is not None
    assert len(enriched_run.status.retries) == 1
    assert enriched_run.status.retries[0]["attempt"] == 0

    enriched_run.status.state = mlrun.common.runtimes.constants.RunStates.pending_retry
    enriched_run_2 = launcher._enrich_run(runtime=runtime, run=enriched_run)
    assert (
        enriched_run_2.status.state == mlrun.common.runtimes.constants.RunStates.running
    )
    assert enriched_run_2.status.start_time is None
    assert enriched_run_2.status.retry_count == 2, "Expected retry count to be 2"
    assert run.metadata.labels[mlrun.common.constants.MLRunInternalLabels.retry] == str(
        enriched_run_2.status.retry_count
    )
    assert enriched_run_2.status.retries is not None
    assert len(enriched_run_2.status.retries) == 2
    assert enriched_run_2.status.retries[1]["attempt"] == 1


@pytest.mark.parametrize(
    "initial_state, db_state, db_deleted, expected_should_skip",
    [
        # Not pending_retry, should not skip
        (mlrun.common.runtimes.constants.RunStates.running, None, False, False),
        # Deleted run in DB, should skip
        (mlrun.common.runtimes.constants.RunStates.pending_retry, None, True, True),
        # Aborted in DB, should skip
        (
            mlrun.common.runtimes.constants.RunStates.pending_retry,
            mlrun.common.runtimes.constants.RunStates.aborted,
            False,
            True,
        ),
        # Not aborted in DB, should not skip
        (
            mlrun.common.runtimes.constants.RunStates.pending_retry,
            mlrun.common.runtimes.constants.RunStates.running,
            False,
            False,
        ),
    ],
)
def test_should_skip_run(initial_state, db_state, db_deleted, expected_should_skip):
    # Verify the `_should_skip_run` method correctly determines whether to skip retried runs based on their current
    # state and the latest status in the database (including deleted or aborted runs).
    run = mlrun.run.RunObject(
        status=mlrun.model.RunStatus(state=initial_state),
        spec=mlrun.model.RunSpec(
            retry={
                "count": 10,
            },
        ),
    )
    launcher = services.api.launcher.ServerSideLauncher()

    with (
        unittest.mock.patch("framework.utils.singletons.db.get_db") as get_db_mock,
        unittest.mock.patch(
            "framework.db.session.run_function_with_new_db_session"
        ) as run_with_session_mock,
    ):
        get_db_mock.return_value = unittest.mock.Mock()

        if db_deleted:
            run_with_session_mock.side_effect = mlrun.errors.MLRunNotFoundError()
        elif db_state:
            run_with_session_mock.return_value = {"status": {"state": db_state}}

        should_skip = launcher._should_skip_run(run)
        assert should_skip is expected_should_skip


def test_launcher_skips_aborted_or_deleted_run(monkeypatch):
    """
    Verify that the launcher skips running a function when `_should_skip_run` returns True,
    meaning the run was aborted or deleted after being scheduled for retry.
    """
    runtime = mlrun.code_to_function(
        name="test", kind="job", filename=str(func_path), handler=handler
    )
    run = mlrun.run.RunObject(
        status=mlrun.model.RunStatus(
            state=mlrun.common.runtimes.constants.RunStates.pending_retry
        ),
        spec=mlrun.model.RunSpec(
            retry={
                "count": 10,
            },
        ),
    )
    launcher = services.api.launcher.ServerSideLauncher()

    # Force `_should_skip_run` to return True to simulate aborted/deleted run
    monkeypatch.setattr(launcher, "_should_skip_run", lambda x: True)

    # Mock runtime handler to validate that it is not called
    runtime_handler_mock = unittest.mock.Mock()
    monkeypatch.setattr(
        services.api.runtime_handlers,
        "get_runtime_handler",
        lambda kind: unittest.mock.Mock(run=runtime_handler_mock),
    )

    # Mock execution object
    mock_execution = unittest.mock.Mock()

    try:
        # Simulate the same logic that exists in the launcher
        if launcher._should_skip_run(run):
            run.status.state = mlrun.common.runtimes.constants.RunStates.aborted
        else:
            runtime_handler = services.api.runtime_handlers.get_runtime_handler(
                runtime.kind
            )
            runtime_handler.run(runtime, run, mock_execution)
    except mlrun.runtimes.utils.RunError:
        pass

    # Validate result
    assert run.status.state == mlrun.common.runtimes.constants.RunStates.aborted
    assert not runtime_handler_mock.called


def test_enrich_and_validate_auth_token_name_noop_without_v4_mode():
    """Test that auth is not modified when not in iguazio v4 mode."""
    launcher = services.api.launcher.ServerSideLauncher(
        auth_info=mlrun.common.schemas.AuthInfo()
    )
    initial_auth = {"token_name": "custom-token"}
    run = mlrun.run.RunObject(
        spec=mlrun.model.RunSpec(auth=initial_auth),
    )

    launcher.enrich_and_validate_auth_token_name(run)

    # auth should not be modified when not in v4 mode
    assert run.spec.auth == initial_auth


@pytest.fixture
def iguazio_v4_mode():
    """Fixture that sets up iguazio v4 authentication mode."""
    mlrun.mlconf.httpdb.authentication.mode = AuthenticationMode.IGUAZIO_V4


@pytest.mark.parametrize(
    "initial_auth,expected_token_name",
    [
        # No token provided → resolved token
        (None, "resolved-token"),
        # Explicit token → preserved as-is
        ({"token_name": "custom-token"}, "custom-token"),
    ],
)
def test_enrich_and_validate_auth_token_name_iguazio_v4_resolution(
    monkeypatch, iguazio_v4_mode, initial_auth, expected_token_name
):
    """Test token resolution in iguazio v4 mode."""
    mock_resolve = unittest.mock.Mock(
        side_effect=lambda user_id, provided_token_name: (
            provided_token_name if provided_token_name else "resolved-token"
        )
    )
    monkeypatch.setattr(
        services.api.utils.helpers,
        "resolve_auth_token_name",
        mock_resolve,
    )

    launcher = services.api.launcher.ServerSideLauncher(
        auth_info=mlrun.common.schemas.AuthInfo(user_id="1234")
    )
    run = mlrun.run.RunObject(
        spec=mlrun.model.RunSpec(auth=initial_auth),
    )

    launcher.enrich_and_validate_auth_token_name(run)

    assert run.spec.auth["token_name"] == expected_token_name
    mock_resolve.assert_called_once_with(
        user_id="1234",
        provided_token_name=initial_auth.get("token_name") if initial_auth else None,
    )


def test_enrich_and_validate_auth_token_name_iguazio_v4_token_not_found(
    monkeypatch, iguazio_v4_mode
):
    """Test that MLRunNotFoundError is raised when token resolution fails."""
    monkeypatch.setattr(
        services.api.utils.helpers,
        "resolve_auth_token_name",
        unittest.mock.Mock(
            side_effect=mlrun.errors.MLRunNotFoundError("No valid tokens found")
        ),
    )

    launcher = services.api.launcher.ServerSideLauncher(
        auth_info=mlrun.common.schemas.AuthInfo(user_id="1234")
    )
    run = mlrun.run.RunObject(
        spec=mlrun.model.RunSpec(auth=None),
    )

    with pytest.raises(mlrun.errors.MLRunNotFoundError, match="No valid tokens found"):
        launcher.enrich_and_validate_auth_token_name(run)
