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
import sys

import pytest

import mlrun.launcher.local

assets_path = pathlib.Path(__file__).parent / "assets"
func_path = assets_path / "sample_function.py"
handler = "hello_world"


def test_launch_local():
    launcher = mlrun.launcher.local.ClientLocalLauncher(local=True)
    runtime = mlrun.code_to_function(
        name="test", kind="job", filename=str(func_path), handler=handler
    )
    result = launcher.launch(runtime)
    assert result.status.state == "completed"
    assert result.status.results.get("return") == "hello world"


def test_override_handler():
    launcher = mlrun.launcher.local.ClientLocalLauncher(local=True)
    runtime = mlrun.code_to_function(
        name="test", kind="job", filename=str(func_path), handler=handler
    )
    result = launcher.launch(runtime, handler="handler_v2")
    assert result.status.state == "completed"
    assert result.status.results.get("return") == "hello world v2"


def test_launch_remote_job_locally():
    launcher = mlrun.launcher.local.ClientLocalLauncher(local=False)
    runtime = mlrun.code_to_function(
        name="test", kind="job", filename=str(func_path), handler=handler
    )
    with pytest.raises(mlrun.errors.MLRunRuntimeError) as exc:
        launcher.launch(runtime)
    assert "Remote function cannot be executed locally" in str(exc.value)


def test_create_local_function_for_execution():
    launcher = mlrun.launcher.local.ClientLocalLauncher(local=False)
    runtime = mlrun.code_to_function(
        name="test", kind="job", filename=str(func_path), handler=handler
    )
    run = mlrun.run.RunObject()
    runtime = launcher._create_local_function_for_execution(
        runtime=runtime,
        run=run,
    )
    assert runtime.metadata.project == "default"
    assert runtime.metadata.name == "test"
    assert run.spec.handler == handler
    assert runtime.kind == "local"
    assert runtime._is_run_local


def test_create_local_function_for_execution_with_enrichment():
    launcher = mlrun.launcher.local.ClientLocalLauncher(local=False)
    runtime = mlrun.code_to_function(
        name="test", kind="job", filename=str(func_path), handler=handler
    )
    runtime.spec.allow_empty_resources = True
    run = mlrun.run.RunObject()
    runtime = launcher._create_local_function_for_execution(
        runtime=runtime,
        run=run,
        local_code_path="some_path.py",
        project="some_project",
        name="other_name",
        workdir="some_workdir",
        handler="handler_v2",
    )
    assert runtime.spec.command == "some_path.py"
    assert runtime.metadata.project == "some_project"
    assert runtime.metadata.name == "other_name"
    assert runtime.spec.workdir == "some_workdir"
    assert run.spec.handler == "handler_v2"
    assert runtime.kind == "local"
    assert runtime._is_run_local
    assert runtime.spec.allow_empty_resources


def test_validate_inputs():
    launcher = mlrun.launcher.local.ClientLocalLauncher(local=False)
    runtime = mlrun.code_to_function(
        name="test", kind="job", filename=str(func_path), handler=handler
    )
    run = mlrun.run.RunObject(spec=mlrun.model.RunSpec(inputs={"input1": 1}))
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentTypeError) as exc:
        launcher._validate_runtime(runtime, run)
    assert "'Inputs' should be of type Dict[str, str]" in str(exc.value)


def test_validate_runtime_success():
    launcher = mlrun.launcher.local.ClientLocalLauncher(local=False)
    runtime = mlrun.code_to_function(
        name="test", kind="local", filename=str(func_path), handler=handler
    )
    run = mlrun.run.RunObject(
        spec=mlrun.model.RunSpec(inputs={"input1": ""}, output_path="./some_path")
    )
    launcher._validate_runtime(runtime, run)


def test_launch_local_reload_module(tmp_path):
    """This test ensures that the function code is updated when running a relative handler in local
    mode when the code changes during execution"""
    sys.path.append(str(tmp_path.parent))
    dir_name = tmp_path.name
    file_path = f"{tmp_path}/temp_function.py"

    function_code = '''def func():
    return "dummy value"'''

    with open(file_path, mode="w+") as file:
        file.write(function_code)

    project = mlrun.new_project("some-project")
    project.set_function(name="func", handler=f"{dir_name}.temp_function.func")
    run = project.run_function("func", local=True)
    assert run.output("return") == "dummy value"

    # change the function's return value in the file
    function_code = '''def func():
    return "dummy value updated"'''

    with open(file_path, mode="w+") as file:
        file.write(function_code)

    run = project.run_function("func", local=True, reset_on_run=True)
    assert run.output("return") == "dummy value updated"


def test_launch_local_reload_module_depends_on_another_changed_module(tmp_path):
    """This test ensures that the function code is updated when running a relative handler in local mode
    when the code module depends on another module and the other module has changed during execution."""
    sys.path.append(str(tmp_path.parent))
    dir_name = tmp_path.name

    # creating the temp_a file
    function_code = '''def func_a():
    return "dummy value"'''

    with open(f"{tmp_path}/temp_a.py", mode="w+") as file:
        file.write(function_code)

    # creating the temp_b file, which depends on temp_a
    function_code = f"""import {dir_name}.temp_a as tmp_file
def func_b():
    return tmp_file.func_a()"""

    with open(f"{tmp_path}/temp_b.py", mode="w+") as file:
        file.write(function_code)

    # running temp_b with temp_a dependency
    project = mlrun.new_project("some-project")
    project.set_function(name="func", handler=f"{dir_name}.temp_b.func_b")
    run = project.run_function("func", local=True)
    assert run.output("return") == "dummy value"

    # changing the code in temp_a
    function_code = '''def func_a():
    return "dummy value updated"'''

    with open(f"{tmp_path}/temp_a.py", mode="w+") as file:
        file.write(function_code)

    # rerunning temp_b with temp_a dependence and verifying with the updated temp_a code
    run = project.run_function("func", local=True, reset_on_run=True)
    assert run.output("return") == "dummy value updated"
