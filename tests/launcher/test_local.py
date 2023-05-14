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
import pathlib

import pytest

import mlrun.launcher.local

assets_path = pathlib.Path(__file__).parent / "assets"
func_path = assets_path / "sample_function.py"
handler = "test_func"


def test_launch_local():
    launcher = mlrun.launcher.local.ClientLocalLauncher(local=True)
    runtime = mlrun.code_to_function(
        name="test", kind="job", filename=str(func_path), handler=handler
    )
    result = launcher.launch(runtime)
    assert result.status.state == "completed"
    assert result.status.results.get("return") == "hello world"


def test_launch_remote_job_locally():
    launcher = mlrun.launcher.local.ClientLocalLauncher(local=False)
    runtime = mlrun.code_to_function(
        name="test", kind="job", filename=str(func_path), handler=handler
    )
    with pytest.raises(mlrun.errors.MLRunRuntimeError) as exc:
        launcher.launch(runtime)
    assert "Remote function cannot be executed locally" in str(exc.value)
