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
import pathlib
from unittest.mock import MagicMock, Mock

import pytest

import mlrun
import mlrun.errors
from mlrun import get_run_db, new_function, new_task
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


def test_noparams():
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
    function._use_remote_api = Mock(return_value=True)
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
    assert "local and schedule cannot be used together" in str(excinfo.value)


def test_invalid_name():
    with pytest.raises(ValueError) as excinfo:
        # name cannot have / in it
        new_function().run(name="asd/asd", handler=my_func)
    assert (
        "Field 'run.metadata.name' is malformed. Does not match required pattern"
        in str(excinfo.value)
    )


def test_with_params():
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


def test_local_runtime_failure_before_executing_the_function_code():
    function = new_function(command=f"{assets_path}/fail.py")
    with pytest.raises(mlrun.runtimes.utils.RunError) as exc:
        function.run(local=True, handler="handler")
    assert "failed on pre-loading" in str(exc.value)


def test_local_runtime_hyper():
    spec = tag_test(base_spec, "test_local_runtime_hyper")
    spec.with_hyper_params({"p1": [1, 5, 3]}, selector="max.accuracy")
    result = new_function(command=f"{examples_path}/training.py").run(spec)
    verify_state(result)


def test_local_handler():
    mlrun.RunObject.logs = Mock()

    spec = tag_test(base_spec, "test_local_runtime")
    result = new_function(command=f"{examples_path}/handler.py").run(
        spec, handler="my_func"
    )
    verify_state(result)

    assert mlrun.RunObject.logs.call_count == 1


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


def test_local_args():
    spec = tag_test(base_spec, "test_local_no_context")
    spec.spec.parameters = {"xyz": "789"}
    result = new_function(
        command=f"{tests_root_directory}/no_ctx.py --xyz {{xyz}}"
    ).run(spec)
    verify_state(result)

    db = get_run_db()
    state, log = db.get_log(result.metadata.uid)
    log = str(log)
    print(state)
    print(log)
    assert log.find(", --xyz, 789") != -1, "params not detected in argv"


def test_local_context():
    project_name = "xtst"
    mlrun.mlconf.artifact_path = out_path
    context = mlrun.get_or_create_ctx("xx", project=project_name, upload_artifacts=True)
    db = mlrun.get_run_db()
    run = db.read_run(context._uid, project=project_name)
    assert run["status"]["state"] == "running", "run status not updated in db"

    with context:
        context.log_artifact("xx", body="123", local_path="a.txt")
        context.log_model("mdl", body="456", model_file="mdl.pkl", artifact_path="+/mm")

        artifact = context.get_cached_artifact("xx")
        artifact.format = "z"
        context.update_artifact(artifact)

    assert context._state == "completed", "task did not complete"

    run = db.read_run(context._uid, project=project_name)
    assert run["status"]["state"] == "running", "run status was updated in db"
    assert (
        run["status"]["artifacts"][0]["metadata"]["key"] == "xx"
    ), "artifact not updated in db"
    assert (
        run["status"]["artifacts"][0]["spec"]["format"] == "z"
    ), "run/artifact attribute not updated in db"
    assert run["status"]["artifacts"][1]["spec"]["target_path"].startswith(
        out_path
    ), "artifact not uploaded to subpath"

    db_artifact = db.read_artifact(artifact.db_key, project=project_name)
    assert db_artifact["spec"]["format"] == "z", "artifact attribute not updated in db"


def test_run_class_code():
    cases = [
        ({"y": 3}, {"rx": 0, "ry": 3, "ra1": 1}),
        ({"_init_args": {"a1": 9}, "y": 5}, {"rx": 0, "ry": 5, "ra1": 9}),
    ]
    fn = mlrun.code_to_function("mytst", filename=function_path, kind="local")
    for params, results in cases:
        run = mlrun.run_function(fn, handler="mycls::mtd", params=params)
        assert run.status.results == results


def test_run_class_file():
    cases = [
        ({"x": 7}, {"rx": 7, "ry": 0, "ra1": 1}),
        ({"_init_args": {"a1": 9}, "y": 5}, {"rx": 0, "ry": 5, "ra1": 9}),
    ]
    fn = mlrun.new_function("mytst", command=function_path, kind="job")
    for params, results in cases:
        run = fn.run(handler="mycls::mtd", params=params, local=True)
        assert run.status.results == results


def test_run_from_module():
    fn = mlrun.new_function("mytst", kind="job")
    run = fn.run(handler="json.dumps", params={"obj": {"x": 99}}, local=True)
    assert run.output("return") == '{"x": 99}'


def test_args_integrity():
    spec = tag_test(base_spec, "test_local_no_context")
    spec.spec.parameters = {"xyz": "789"}
    result = new_function(
        command=f"{tests_root_directory}/no_ctx.py",
        args=["It's", "a", "nice", "day!"],
    ).run(spec)
    verify_state(result)

    db = get_run_db()
    state, log = db.get_log(result.metadata.uid)
    log = str(log)
    print(state)
    print(log)
    assert log.find("It's, a, nice, day!") != -1, "params not detected in argv"
