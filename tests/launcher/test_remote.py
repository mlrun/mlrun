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
import unittest.mock

import pytest

import mlrun
import mlrun.common.constants as mlrun_constants
import mlrun.config
import mlrun.launcher.remote
import mlrun.runtimes.utils
from mlrun import Client, Credentials

assets_path = pathlib.Path(__file__).parent / "assets"
func_path = assets_path / "sample_function.py"
handler = "hello_word"


def test_launch_remote_job(rundb_mock):
    launcher = mlrun.launcher.remote.ClientRemoteLauncher()
    mlrun.mlconf.artifact_path = "v3io:///users/admin/mlrun"
    runtime = mlrun.code_to_function(
        name="test",
        kind="job",
        filename=str(func_path),
        handler=handler,
        image="mlrun/mlrun",
    )

    # store the run is done by the API so we need to mock it
    uid = "123"
    run = mlrun.run.RunObject(
        metadata=mlrun.model.RunMetadata(uid=uid),
    )
    rundb_mock.store_run(run, uid)
    result = launcher.launch(runtime, run)
    assert result.status.state == "completed"


def test_launch_remote_job_no_watch(rundb_mock):
    launcher = mlrun.launcher.remote.ClientRemoteLauncher()
    mlrun.mlconf.artifact_path = "v3io:///users/admin/mlrun"
    runtime = mlrun.code_to_function(
        name="test",
        kind="job",
        filename=str(func_path),
        handler=handler,
        image="mlrun/mlrun",
    )
    result = launcher.launch(runtime, watch=False)
    assert result.status.state == "created"


def test_validate_inputs():
    launcher = mlrun.launcher.remote.ClientRemoteLauncher()
    runtime = mlrun.code_to_function(
        name="test", kind="job", filename=str(func_path), handler=handler
    )
    run = mlrun.run.RunObject(spec=mlrun.model.RunSpec(inputs={"input1": 1}))
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentTypeError) as exc:
        launcher._validate_run(runtime, run)
    assert "'Inputs' should be of type Dict[str, Union[str,list,dict]]." in str(
        exc.value
    )


def test_validate_run_success():
    launcher = mlrun.launcher.remote.ClientRemoteLauncher()
    runtime = mlrun.code_to_function(
        name="test", kind="local", filename=str(func_path), handler=handler
    )
    run = mlrun.run.RunObject(
        spec=mlrun.model.RunSpec(inputs={"input1": ""}, output_path="./some_path")
    )
    launcher._validate_run(runtime, run)


@pytest.mark.parametrize(
    "kind, requirements, expected_base_image, expected_image",
    [
        ("job", [], None, "mlrun/mlrun"),
        ("job", ["pandas"], "mlrun/mlrun", ""),
        ("nuclio", ["pandas"], None, "mlrun/mlrun"),
        ("serving", ["pandas"], None, "mlrun/mlrun"),
    ],
)
def test_prepare_image_for_deploy(
    kind, requirements, expected_base_image, expected_image
):
    launcher = mlrun.launcher.remote.ClientRemoteLauncher()
    runtime = mlrun.code_to_function(
        name="test",
        kind=kind,
        filename=str(func_path),
        handler=handler,
        image="mlrun/mlrun",
        requirements=requirements,
    )
    launcher.prepare_image_for_deploy(runtime)
    assert runtime.spec.build.base_image == expected_base_image
    assert runtime.spec.image == expected_image


def test_run_error_status(rundb_mock):
    launcher = mlrun.launcher.remote.ClientRemoteLauncher()
    mlrun.mlconf.artifact_path = "v3io:///users/admin/mlrun"
    runtime = mlrun.code_to_function(
        name="test",
        kind="job",
        filename=str(func_path),
        handler=handler,
        image="mlrun/mlrun",
    )

    # store the run is done by the API so we need to mock it
    uid = "123"
    run = mlrun.run.RunObject(
        metadata=mlrun.model.RunMetadata(uid=uid),
    )
    rundb_mock.store_run(run, uid)

    result = mlrun.run.RunObject(
        metadata=mlrun.model.RunMetadata(uid=uid),
        status=mlrun.model.RunStatus(state="error", reason="some error"),
    )
    runtime._get_db_run = unittest.mock.MagicMock(return_value=result.to_dict())

    with pytest.raises(mlrun.runtimes.utils.RunError) as exc:
        launcher.launch(runtime, run, watch=True)
    assert "some error" in str(exc.value)


def test_store_function_set_token_name():
    launcher = mlrun.launcher.remote.ClientRemoteLauncher()
    runtime = mlrun.code_to_function(
        name="test",
        kind="job",
        filename=str(func_path),
        handler=handler,
    )
    runtime.kind = "handler"
    db = mlrun.get_run_db()
    db.token_provider = unittest.mock.MagicMock(token_name="provider-run-token")
    run = mlrun.run.RunObject(spec=mlrun.model.RunSpec())

    launcher._store_function(runtime, run)
    assert run.spec.auth["token_name"] == "provider-run-token"

    with mlrun.RuntimeConfigurationContext(auth_token_name="context-run-token"):
        launcher._store_function(runtime, run)
        assert run.spec.auth["token_name"] == "context-run-token"


# ---------------------------------------------------------------------------
# _enrich_run_labels_with_v3io_user lives on ClientBaseLauncher; testing via
# ClientRemoteLauncher covers both subclasses through inheritance.
# ---------------------------------------------------------------------------


def _make_run() -> mlrun.run.RunObject:
    run = mlrun.run.RunObject()
    run.metadata.name = "r"
    run.metadata.uid = "u"
    return run


def test_enrich_run_labels_with_v3io_user_skips_inside_session(monkeypatch):
    monkeypatch.setattr(mlrun.mlconf, "dbpath", "https://mock-server")
    monkeypatch.setenv("V3IO_USERNAME", "process-user")
    run = _make_run()
    client = Client(credentials=Credentials(token="t"))
    with client.session():
        mlrun.launcher.remote.ClientRemoteLauncher._enrich_run_labels_with_v3io_user(
            run
        )
    assert mlrun_constants.MLRunInternalLabels.v3io_user not in run.metadata.labels, (
        f"v3io_user label stamped from env inside session: {run.metadata.labels!r}"
    )


def test_enrich_run_labels_with_v3io_user_stamps_outside_session(monkeypatch):
    monkeypatch.setenv("V3IO_USERNAME", "process-user")
    run = _make_run()
    mlrun.launcher.remote.ClientRemoteLauncher._enrich_run_labels_with_v3io_user(run)
    assert (
        run.metadata.labels.get(mlrun_constants.MLRunInternalLabels.v3io_user)
        == "process-user"
    )


def test_enrich_run_labels_with_v3io_user_preserves_existing_label(monkeypatch):
    """A workflow runner (or any caller) that pre-set v3io_user must not be
    overwritten by the env value."""
    monkeypatch.setenv("V3IO_USERNAME", "process-user")
    run = _make_run()
    run.metadata.labels[mlrun_constants.MLRunInternalLabels.v3io_user] = "preset-user"
    mlrun.launcher.remote.ClientRemoteLauncher._enrich_run_labels_with_v3io_user(run)
    assert (
        run.metadata.labels[mlrun_constants.MLRunInternalLabels.v3io_user]
        == "preset-user"
    )


def test_enrich_run_labels_with_v3io_user_no_op_without_env(monkeypatch):
    monkeypatch.delenv("V3IO_USERNAME", raising=False)
    run = _make_run()
    mlrun.launcher.remote.ClientRemoteLauncher._enrich_run_labels_with_v3io_user(run)
    assert mlrun_constants.MLRunInternalLabels.v3io_user not in run.metadata.labels
