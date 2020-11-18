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

import pytest
import pathlib
import pandas as pd
from tests.conftest import (
    examples_path,
    has_secrets,
    tests_root_directory,
    out_path,
    tag_test,
    verify_state,
)
from unittest.mock import Mock
from mlrun import new_task, get_run_db, new_function


def my_func(context, p1=1, p2="a-string", input_name="infile.txt"):
    print(f"Run: {context.name} (uid={context.uid})")
    print(f"Params: p1={p1}, p2={p2}\n")
    print("file\n{}\n".format(context.get_input(input_name).get()))

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


def test_invalid_name():
    with pytest.raises(ValueError) as excinfo:
        # name cannot have / in it
        new_function().run(name="asd/asd", handler=my_func)
    assert (
        "Field run.metadata.name is malformed. Does not match required pattern"
        in str(excinfo.value)
    )


def test_with_params():
    spec = tag_test(base_spec, "test_with_params")
    result = new_function().run(spec, handler=my_func)

    assert result.output("accuracy") == 16, "failed to run"
    assert result.status.artifacts[0].get("key") == "chart", "failed to run"


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
    run_spec.spec.param_file = "{}/param_file.csv".format(tests_root_directory)
    result = new_function().run(run_spec, handler=my_func)
    print(result)
    assert len(result.status.iterations) == 3 + 1, "hyper parameters test failed"
    verify_state(result)


def test_local_runtime():
    spec = tag_test(base_spec, "test_local_runtime")
    result = new_function(command="{}/training.py".format(examples_path)).run(spec)
    verify_state(result)


def test_local_runtime_hyper():
    spec = tag_test(base_spec, "test_local_runtime_hyper")
    spec.with_hyper_params({"p1": [1, 5, 3]}, selector="max.accuracy")
    result = new_function(command="{}/training.py".format(examples_path)).run(spec)
    verify_state(result)


def test_local_handler():
    spec = tag_test(base_spec, "test_local_runtime")
    result = new_function(command="{}/handler.py".format(examples_path)).run(
        spec, handler="my_func"
    )
    verify_state(result)


def test_local_no_context():
    spec = tag_test(base_spec, "test_local_no_context")
    spec.spec.parameters = {"xyz": "789"}
    result = new_function(
        command="{}/no_ctx.py".format(tests_root_directory), mode="noctx"
    ).run(spec)
    verify_state(result)

    db = get_run_db().connect()
    state, log = db.get_log(result.metadata.uid)
    log = str(log)
    print(state)
    print(log)
    assert log.find(", '--xyz', '789']") != -1, "params not detected in noctx"
