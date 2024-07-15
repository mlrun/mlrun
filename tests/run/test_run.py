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
import contextlib
import io
import pathlib
import sys
from unittest.mock import MagicMock, Mock

import pytest

import mlrun
import mlrun.errors
import mlrun.launcher.factory
from mlrun import new_function, new_task
from tests.conftest import (
    examples_path,
    has_secrets,
    out_path,
    tag_test,
    tests_root_directory,
    verify_state,
)

from .common import my_func

function_path = str(pathlib.Path(__file__).parent / "assets" / "handler.py")
base_spec = new_task(params={"p1": 8}, out_path=out_path)
input_file_path = str(
    pathlib.Path(__file__).parent / "assets" / "test_run_input_file.txt"
)
base_spec.spec.inputs = {"infile.txt": str(input_file_path)}

s3_spec = base_spec.copy().with_secrets("file", "secrets.txt")
s3_spec.spec.inputs = {"infile.txt": "s3://yarons-tests/infile.txt"}
assets_path = str(pathlib.Path(__file__).parent / "assets")


@contextlib.contextmanager
def captured_output():
    new_out, new_err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def test_noparams(rundb_mock):
    mlrun.get_or_create_project("default", allow_cross_project=True)
    # Since we're executing the function without inputs, it will try to use the input name as the file path
    result = new_function().run(
        params={"input_name": str(input_file_path)}, handler=my_func
    )

    assert result.output("accuracy") == 2, "failed to run"
    assert result.status.artifacts[0]["metadata"].get("key") == "chart", "failed to run"

    # verify the DF artifact was created and stored
    df = result.artifact("mydf").as_df()
    df.shape


def test_failed_schedule_not_creating_run():
    function = new_function()
    # mock we're with remote api (only there schedule is relevant)
    function._is_remote = True
    # mock failure in submit job (failed schedule)
    db = MagicMock()
    function.set_db_connection(db)
    db.submit_job.side_effect = RuntimeError("Explode!")
    function.store_run = Mock()
    function.run(handler=my_func, schedule="* * * * *")
    assert 0 == function.store_run.call_count


def test_schedule_with_local_exploding():
    function = new_function()
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as excinfo:
        function.run(local=True, schedule="* * * * *")
    assert (
        "Unexpected schedule='* * * * *' parameter for local function execution"
        in str(excinfo.value)
    )
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as excinfo:
        function.run(schedule="* * * * *")
    assert (
        "Unexpected schedule='* * * * *' parameter for local function execution"
        in str(excinfo.value)
    )


def test_invalid_name():
    with pytest.raises(ValueError) as excinfo:
        # name cannot have / in it
        new_function().run(name="asd/asd", handler=my_func)
    assert (
        "Field 'run.metadata.name' is malformed. 'asd/asd' does not match required pattern"
        in str(excinfo.value)
    )


def test_with_params(rundb_mock):
    mlrun.get_or_create_project("default", allow_cross_project=True)
    spec = tag_test(base_spec, "test_with_params")
    result = new_function().run(spec, handler=my_func)

    assert result.output("accuracy") == 16, "failed to run"
    assert result.status.artifacts[0]["metadata"].get("key") == "chart", "failed to run"
    assert result.artifact("chart").url, "failed to return artifact data item"


@pytest.mark.skipif(not has_secrets(), reason="no secrets")
def test_with_params_s3():
    spec = tag_test(s3_spec, "test_with_params")
    result = new_function().run(spec, handler=my_func)

    assert result.output("accuracy") == 16, "failed to run"
    assert result.status.artifacts[0].get("key") == "chart", "failed to run"


def test_handler_project():
    spec = tag_test(base_spec, "test_handler_project")
    spec.metadata.project = "myproj"
    spec.metadata.labels = {"owner": "yaronh"}
    result = new_function().run(spec, handler=my_func)
    print(result)
    assert result.output("accuracy") == 16, "failed to run"
    verify_state(result)


def test_local_runtime():
    spec = tag_test(base_spec, "test_local_runtime")
    result = new_function(command=f"{examples_path}/training.py").run(spec)
    verify_state(result)


def test_local_runtime_failure_before_executing_the_function_code(rundb_mock):
    function = new_function(command=f"{assets_path}/fail.py")
    with pytest.raises(mlrun.runtimes.utils.RunError) as exc:
        function.run(local=True, handler="handler")
    assert "Failed on pre-loading" in str(exc.value)


@pytest.mark.parametrize(
    "handler_name,params,kwargs,expected_kwargs",
    [
        ("func", {"x": 2}, {"y": 3, "z": 4}, {"y": 3, "z": 4}),
        ("func", {"x": 2}, {}, {}),
        ("func_with_default", {}, {"y": 3, "z": 4}, {"y": 3, "z": 4}),
    ],
)
def test_local_runtime_with_kwargs(
    rundb_mock, handler_name, params, kwargs, expected_kwargs
):
    params.update(kwargs)
    function = new_function(command=f"{assets_path}/kwargs.py")
    result = function.run(local=True, params=params, handler=handler_name)
    verify_state(result)
    assert result.outputs.get("return", {}) == expected_kwargs


def test_local_runtime_with_kwargs_with_code_to_function(rundb_mock):
    mlrun.get_or_create_project("default", allow_cross_project=True)
    function = mlrun.code_to_function(
        "kwarg",
        filename=f"{assets_path}/kwargs.py",
        image="mlrun/mlrun",
        kind="job",
        handler="func",
    )
    kwargs = {"y": 3, "z": 4}
    params = {"x": 2}
    params.update(kwargs)
    result = function.run(local=True, params=params)
    assert result.outputs["return"] == kwargs


def test_local_runtime_hyper():
    spec = tag_test(base_spec, "test_local_runtime_hyper")
    spec.with_hyper_params({"p1": [1, 5, 3]}, selector="max.accuracy")
    result = new_function(command=f"{examples_path}/training.py").run(spec)
    verify_state(result)


def test_local_handler():
    spec = tag_test(base_spec, "test_local_runtime")
    result = new_function(command=f"{examples_path}/handler.py").run(
        spec, handler="my_func"
    )
    verify_state(result)


@pytest.mark.parametrize(
    "kind,watch,expected_watch_count",
    [
        ("", True, 0),
        ("", True, 0),
        ("local", False, 0),
        ("local", False, 0),
        ("dask", True, 0),
        ("dask", False, 0),
        ("job", True, 1),
        ("job", False, 0),
    ],
)
def test_is_watchable(rundb_mock, kind, watch, expected_watch_count):
    mlrun.RunObject.logs = Mock()
    spec = tag_test(base_spec, "test_is_watchable")
    func = new_function(
        command=f"{examples_path}/handler.py",
        kind=kind,
    )

    if kind == "dask":
        # don't start dask cluster
        func.spec.remote = False
    elif kind == "job":
        # mark as deployed
        func.spec.image = "some-image"

    result = func.run(
        spec,
        handler="my_func",
        watch=watch,
    )

    # rundb_mock mocks the job submission when kind is job
    # therefore, if we watch we get an empty result as the run was not created (it is mocked)
    # else, the state will not be 'completed'
    if kind != "job":
        verify_state(result)

    assert mlrun.RunObject.logs.call_count == expected_watch_count


@pytest.mark.asyncio
async def test_local_args(rundb_mock):
    spec = tag_test(base_spec, "test_local_no_context")
    spec.spec.parameters = {"xyz": "789"}

    function = new_function(command=f"{tests_root_directory}/no_ctx.py --xyz {{xyz}}")
    with captured_output() as (out, err):
        result = function.run(spec)

    output = out.getvalue().strip()

    verify_state(result)

    assert output.find(", --xyz, 789") != -1, "params not detected in argv"


def test_run_class_code():
    cases = [
        ({"y": 3}, {"rx": 0, "ry": 3, "ra1": 1}),
        ({"_init_args": {"a1": 9}, "y": 5}, {"rx": 0, "ry": 5, "ra1": 9}),
    ]
    fn = mlrun.code_to_function("mytst", filename=function_path, kind="local")
    for params, results in cases:
        run = mlrun.run_function(fn, handler="MyCls::mtd", params=params)
        assert run.status.results == results


def test_run_class_file():
    cases = [
        ({"x": 7}, {"rx": 7, "ry": 0, "ra1": 1}),
        ({"_init_args": {"a1": 9}, "y": 5}, {"rx": 0, "ry": 5, "ra1": 9}),
    ]
    fn = mlrun.new_function("mytst", command=function_path, kind="job")
    for params, results in cases:
        run = fn.run(handler="MyCls::mtd", params=params, local=True)
        assert run.status.results == results


def test_run_from_module():
    fn = mlrun.new_function("mytst", kind="job")
    run = fn.run(handler="json.dumps", params={"obj": {"x": 99}}, local=True)
    assert run.output("return") == '{"x": 99}'


def test_args_integrity():
    spec = tag_test(base_spec, "test_local_no_context")
    spec.spec.parameters = {"xyz": "789"}
    function = new_function(
        command=f"{tests_root_directory}/no_ctx.py",
        args=["It's", "a", "nice", "day!"],
    )

    with captured_output() as (out, err):
        result = function.run(spec)

    output = out.getvalue().strip()
    verify_state(result)

    assert output.find("It's, a, nice, day!") != -1, "params not detected in argv"


def test_get_or_create_ctx_run_kind():
    # varify the default run kind is local
    context = mlrun.get_or_create_ctx("ctx")
    assert context.labels.get("kind") == "local"
    assert context.state == "running"
    context.commit(completed=True)
    assert context.state == "completed"


def test_get_or_create_ctx_run_kind_local_from_function():
    project = mlrun.get_or_create_project("dummy-project")
    project.set_function(
        name="func",
        func=f"{assets_path}/simple.py",
        handler="get_ctx_kind_label",
        image="mlrun/mlrun",
    )
    run = project.run_function(
        "func",
        local=True,
    )
    assert run.state() == "completed"
    assert run.output("return") == "local"


def test_get_or_create_ctx_run_kind_exists_in_mlrun_exec_config(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(
        "MLRUN_EXEC_CONFIG",
        '{"spec":{},"metadata":{"uid":"123411", "name":"tst", "labels": {"kind": "spark"}}}',
    )
    context = mlrun.get_or_create_ctx("ctx")
    assert context.labels.get("kind") == "spark"


def test_verify_tag_exists_in_run_output_uri():
    project = mlrun.get_or_create_project("dummy-project")
    project.set_function(
        func=function_path, handler="myhandler", name="test", image="mlrun/mlrun"
    )
    run = project.run_function("test", params={"tag": "v1"}, local=True)
    uri = run.output("file_result")

    # Verify that the tag exists in the URI
    assert ":v1" in uri
