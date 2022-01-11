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
from unittest.mock import Mock

import pandas as pd
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


def my_func(context, p1=1, p2="a-string", input_name="infile.txt"):
    print(f"Run: {context.name} (uid={context.uid})")
    print(f"Params: p1={p1}, p2={p2}\n")
    input_str = context.get_input(input_name).get()
    print(f"file\n{input_str}\n")

    context.log_result("accuracy", p1 * 2)
    context.logger.info("some info")
    context.logger.debug("debug info")
    context.log_metric("loss", 7)
    context.log_artifact("chart", body="abc")

    raw_data = {
        "first_name": ["Jason", "Molly", "Tina", "Jake", "Amy"],
        "last_name": ["Miller", "Jacobson", "Ali", "Milner", "Cooze"],
        "age": [42, 52, 36, 24, 73],
        "postTestScore": [25, 94, 57, 62, 70],
    }
    df = pd.DataFrame(
        raw_data, columns=["first_name", "last_name", "age", "postTestScore"]
    )
    context.log_dataset("mydf", df=df)


def hyper_func(context, p1, p2, p3):
    print(f"p2={p2}, p3={p3}")
    context.log_result("r1", p2 * p3)


base_spec = new_task(params={"p1": 8}, out_path=out_path)

repo_root = pathlib.Path(__file__).resolve().absolute().parent.parent
input_file_path = repo_root / "tests" / "assets" / "test_run_input_file.txt"
base_spec.spec.inputs = {"infile.txt": str(input_file_path)}

s3_spec = base_spec.copy().with_secrets("file", "secrets.txt")
s3_spec.spec.inputs = {"infile.txt": "s3://yarons-tests/infile.txt"}


def test_noparams():
    # Since we're executing the function without inputs, it will try to use the input name as the file path
    result = new_function().run(
        params={"input_name": str(input_file_path)}, handler=my_func
    )

    assert result.output("accuracy") == 2, "failed to run"
    assert result.status.artifacts[0].get("key") == "chart", "failed to run"

    # verify the DF artifact was created and stored
    df = result.artifact("mydf").as_df()
    df.shape


def test_failed_schedule_not_creating_run():
    function = new_function()
    # mock we're with remote api (only there schedule is relevant)
    function._use_remote_api = Mock(return_value=True)
    # mock failure in submit job (failed schedule)
    db = Mock()
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
    assert result.status.artifacts[0].get("key") == "chart", "failed to run"
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


def test_handler_hyper():
    run_spec = tag_test(base_spec, "test_handler_hyper")
    run_spec.with_hyper_params({"p1": [1, 5, 3]}, selector="max.accuracy")
    result = new_function().run(run_spec, handler=my_func)
    print(result)
    assert len(result.status.iterations) == 3 + 1, "hyper parameters test failed"
    assert (
        result.status.results["best_iteration"] == 2
    ), "failed to select best iteration"
    verify_state(result)


def test_handler_hyperlist():
    run_spec = tag_test(base_spec, "test_handler_hyperlist")
    run_spec.spec.param_file = f"{tests_root_directory}/param_file.csv"
    result = new_function().run(run_spec, handler=my_func)
    print(result)
    assert len(result.status.iterations) == 3 + 1, "hyper parameters test failed"
    verify_state(result)


def test_hyper_grid():
    grid_params = '{"p2": [2,1,3], "p3": [10,20]}'
    mlrun.datastore.set_in_memory_item("params.json", grid_params)

    run_spec = tag_test(base_spec, "test_hyper_grid")
    run_spec.with_param_file("memory://params.json", selector="r1", strategy="grid")
    run = new_function().run(run_spec, handler=hyper_func)

    verify_state(run)
    # 3 x p2, 2 x p3 = 6 iterations + 1 header line
    assert len(run.status.iterations) == 1 + 2 * 3, "wrong number of iterations"

    results = [line[5] for line in run.status.iterations[1:]]
    assert results == [20, 10, 30, 40, 20, 60], "unexpected results"
    assert run.output("best_iteration") == 6, "wrong best iteration"


def test_hyper_grid_parallel():
    grid_params = '{"p2": [2,1,3], "p3": [10,20]}'
    mlrun.datastore.set_in_memory_item("params.json", grid_params)

    run_spec = tag_test(base_spec, "test_hyper_grid")
    run_spec.with_param_file(
        "memory://params.json", selector="r1", strategy="grid", parallel_runs=2
    )
    run = new_function().run(run_spec, handler=hyper_func)

    verify_state(run)
    # 3 x p2, 2 x p3 = 6 iterations + 1 header line
    assert len(run.status.iterations) == 1 + 2 * 3, "wrong number of iterations"


def test_hyper_list():
    list_params = '{"p2": [2,3,1], "p3": [10,30,20]}'
    mlrun.datastore.set_in_memory_item("params.json", list_params)

    run_spec = tag_test(base_spec, "test_hyper_list")
    run_spec.with_param_file("memory://params.json", selector="r1", strategy="list")
    run = new_function().run(run_spec, handler=hyper_func)

    verify_state(run)
    assert len(run.status.iterations) == 1 + 3, "wrong number of iterations"

    results = [line[5] for line in run.status.iterations[1:]]
    assert results == [20, 90, 20], "unexpected results"
    assert run.output("best_iteration") == 2, "wrong best iteration"


def test_hyper_list_with_stop():
    list_params = '{"p2": [2,3,7,4,5], "p3": [10,10,10,10,10]}'
    mlrun.datastore.set_in_memory_item("params.json", list_params)

    run_spec = tag_test(base_spec, "test_hyper_list_with_stop")
    run_spec.with_param_file(
        "memory://params.json",
        selector="max.r1",
        strategy="list",
        stop_condition="r1>=70",
    )
    run = new_function().run(run_spec, handler=hyper_func)

    verify_state(run)
    # result: r1 = p2 * p3, r1 >= 70 lead to stop on third run
    assert len(run.status.iterations) == 1 + 3, "wrong number of iterations"
    assert run.output("best_iteration") == 3, "wrong best iteration"


def test_hyper_parallel_with_stop():
    list_params = '{"p2": [2,3,7,4,5], "p3": [10,10,10,10,10]}'
    mlrun.datastore.set_in_memory_item("params.json", list_params)

    run_spec = mlrun.new_task(params={"p1": 1})
    run_spec.with_hyper_params(
        {"p2": [2, 3, 7, 4, 5], "p3": [10, 10, 10, 10, 10]},
        parallel_runs=2,
        selector="max.r1",
        strategy="list",
        stop_condition="r1>=70",
    )
    run = new_function().run(run_spec, handler=hyper_func)

    verify_state(run)
    # result: r1 = p2 * p3, r1 >= 70 lead to stop on third run
    # may have one extra iterations in flight so checking both 4 or 5
    assert len(run.status.iterations) in [4, 5], "wrong number of iterations"
    assert run.output("best_iteration") == 3, "wrong best iteration"


def test_hyper_random():
    grid_params = {"p2": [2, 1, 3], "p3": [10, 20, 30]}
    run_spec = tag_test(base_spec, "test_hyper_random")
    run_spec.with_hyper_params(
        grid_params, selector="r1", strategy="random", max_iterations=5
    )
    run = new_function().run(run_spec, handler=hyper_func)

    verify_state(run)
    assert len(run.status.iterations) == 1 + 5, "wrong number of iterations"


def custom_hyper_func(context: mlrun.MLClientCtx):
    best_accuracy = 0
    for param in [1, 2, 4, 3]:
        with context.get_child_context(myparam=param) as child:
            accuracy = child.get_param("myparam")
            child.log_result("accuracy", accuracy)
            if accuracy > best_accuracy:
                child.mark_as_best()
                best_accuracy = accuracy


def test_hyper_custom():
    run_spec = tag_test(base_spec, "test_hyper_custom")
    run = new_function().run(run_spec, handler=custom_hyper_func)
    verify_state(run)
    assert len(run.status.iterations) == 1 + 4, "wrong number of iterations"

    results = [line[3] for line in run.status.iterations[1:]]
    print(results)
    assert run.output("best_iteration") == 3, "wrong best iteration"


def test_local_runtime():
    spec = tag_test(base_spec, "test_local_runtime")
    result = new_function(command=f"{examples_path}/training.py").run(spec)
    verify_state(result)


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
    assert log.find(", '--xyz', '789']") != -1, "params not detected in argv"


def test_local_context():
    project_name = "xtst"
    mlrun.mlconf.artifact_path = out_path
    context = mlrun.get_or_create_ctx("xx", project=project_name, upload_artifacts=True)
    with context:
        context.log_artifact("xx", body="123", local_path="a.txt")
        context.log_model("mdl", body="456", model_file="mdl.pkl", artifact_path="+/mm")

        artifact = context.get_cached_artifact("xx")
        artifact.format = "z"
        context.update_artifact(artifact)

    assert context._state == "completed", "task did not complete"

    db = mlrun.get_run_db()
    run = db.read_run(context._uid, project=project_name)
    assert run["status"]["state"] == "completed", "run status not updated in db"
    assert run["status"]["artifacts"][0]["key"] == "xx", "artifact not updated in db"
    assert (
        run["status"]["artifacts"][0]["format"] == "z"
    ), "run/artifact attribute not updated in db"
    assert (
        run["status"]["artifacts"][1]["target_path"] == out_path + "/mm/"
    ), "artifact not uploaded to subpath"

    db_artifact = db.read_artifact(artifact.db_key, project=project_name)
    assert db_artifact["format"] == "z", "artifact attribute not updated in db"
