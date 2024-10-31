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

import pytest

import mlrun.config
import mlrun.launcher.remote

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
        launcher._validate_runtime(runtime, run)
    assert "'Inputs' should be of type Dict[str, str]" in str(exc.value)


def test_validate_runtime_success():
    launcher = mlrun.launcher.remote.ClientRemoteLauncher()
    runtime = mlrun.code_to_function(
        name="test", kind="local", filename=str(func_path), handler=handler
    )
    run = mlrun.run.RunObject(
        spec=mlrun.model.RunSpec(inputs={"input1": ""}, output_path="./some_path")
    )
    launcher._validate_runtime(runtime, run)


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
