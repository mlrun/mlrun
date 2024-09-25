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
import datetime
import unittest.mock

import numpy as np
import pandas as pd
import pytest
from mlrun_pipelines.models import PipelineRun

import mlrun
import mlrun.artifacts
import mlrun.common.constants as mlrun_constants
import mlrun.errors
from mlrun import new_task
from tests.conftest import out_path, tag_test, verify_state


def my_func(context):
    print(f"Run: {context.name} (uid={context.uid})")

    context.log_result("float", 1.5)
    context.log_result("np-float32", np.float32(1.5))
    context.log_result("date", datetime.datetime(2018, 1, 1))
    context.log_result("np-date", np.datetime64("2018-01-01"))
    context.log_result("np-nan", np.nan)
    context.log_result("np-list", [1.5, np.nan, np.inf])
    context.log_result("dict", {"x": -1.3, "y": np.float32(1.5), "z": "ab"})
    context.log_result(
        "array", np.array([1, 2, 3.2, np.nan, np.datetime64("2018-01-01")])
    )

    raw_data = {
        "first_name": ["Jason", "Molly", "Tina", "Jake", "Amy"],
        "last_name": ["Miller", "Jacobson", "Ali", "Milner", "Cooze"],
        "x": np.array([1, 2, 3.2, np.nan, 5.5]),
        "y": [25, 94, 0.1, 57, datetime.datetime(2018, 1, 1)],
    }
    df = pd.DataFrame(raw_data, columns=["first_name", "last_name", "x", "y"])
    context.log_dataset("df1", df=df, format="csv")

    date_rng = pd.date_range("2018-01-01", periods=4, freq="H")
    df = pd.DataFrame(date_rng, columns=["date"])
    df["data"] = np.random.rand(4)
    df["nan"] = np.nan
    df["datetime"] = pd.to_datetime(df["date"])
    df["text"] = "x"
    df = df.set_index("datetime")
    context.log_dataset("df2", df=df)

    return np.nan


def test_log_serialization():
    function = mlrun.new_function(name="test_serialization", kind="job")
    base_task = new_task(artifact_path=out_path, handler=my_func)
    task = tag_test(base_task, "test_serialization")
    result = function.run(task, local=True)
    verify_state(result)


def test_local_context(rundb_mock):
    project_name = "xtst"
    mlrun.mlconf.artifact_path = out_path
    context = mlrun.get_or_create_ctx("xx", project=project_name, upload_artifacts=True)
    db = mlrun.get_run_db()
    run = db.read_run(context._uid, project=project_name)
    assert run["status"]["state"] == "running", "run status not updated in db"

    # calls __exit__ and commits the context
    with context:
        context.log_artifact("xx", body="123", local_path="a.txt")
        context.log_model("mdl", body="456", model_file="mdl.pkl", artifact_path="+/mm")
        context.get_param("p1", 1)
        context.get_param("p2", "a string")
        context.log_result("accuracy", 16)
        context.set_label("label-key", "label-value")
        context.set_annotation("annotation-key", "annotation-value")
        context._set_input("input-key", "input-url")

        artifact = context.get_cached_artifact("xx")
        artifact.format = "z"
        context.update_artifact(artifact)

    assert context._state == "completed", "task did not complete"

    run = db.read_run(context._uid, project=project_name)

    # run state should be updated by the context for local run
    assert run["status"]["state"] == "completed", "run status was not updated in db"
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

    assert run["spec"]["parameters"]["p1"] == 1, "param not updated in db"
    assert run["spec"]["parameters"]["p2"] == "a string", "param not updated in db"
    assert run["status"]["results"]["accuracy"] == 16, "result not updated in db"
    assert run["metadata"]["labels"]["label-key"] == "label-value", "label not updated"
    assert (
        run["metadata"]["annotations"]["annotation-key"] == "annotation-value"
    ), "annotation not updated"

    assert run["spec"]["inputs"]["input-key"] == "input-url", "input not updated"


def test_context_from_dict_when_start_time_is_string():
    context = mlrun.get_or_create_ctx("ctx")
    context_dict = context.to_dict()
    context = mlrun.MLClientCtx.from_dict(context_dict)
    assert isinstance(context._start_time, datetime.datetime)


@pytest.mark.parametrize(
    "is_api",
    [True, False],
)
def test_context_from_run_dict(is_api):
    with unittest.mock.patch("mlrun.config.is_running_as_api", return_value=is_api):
        run_dict = _generate_run_dict()

        # create run object from dict and dict again to mock the run serialization
        run = mlrun.run.RunObject.from_dict(run_dict)
        context = mlrun.MLClientCtx.from_dict(PipelineRun(run.to_dict()), is_api=is_api)

        assert context.name == run_dict["metadata"]["name"]
        assert context._project == run_dict["metadata"]["project"]
        assert context._labels == run_dict["metadata"]["labels"]
        assert context._annotations == run_dict["metadata"]["annotations"]
        assert context.get_param("p1") == run_dict["spec"]["parameters"]["p1"]
        assert context.get_param("p2") == run_dict["spec"]["parameters"]["p2"]
        assert (
            context.labels["label-key"] == run_dict["metadata"]["labels"]["label-key"]
        )
        assert (
            context.annotations["annotation-key"]
            == run_dict["metadata"]["annotations"]["annotation-key"]
        )
        assert context.artifact_path == run_dict["spec"]["output_path"]


@pytest.mark.parametrize(
    "state, error, expected_state",
    [
        ("running", None, "completed"),
        ("completed", None, "completed"),
        (None, "error message", "error"),
        (None, "", "error"),
    ],
)
def test_context_set_state(rundb_mock, state, error, expected_state):
    project_name = "test_context_error"
    mlrun.mlconf.artifact_path = out_path
    context = mlrun.get_or_create_ctx("xx", project=project_name, upload_artifacts=True)
    db = mlrun.get_run_db()
    run = db.read_run(context._uid, project=project_name)
    assert run["status"]["state"] == "running", "run status not updated in db"

    # calls __exit__ and commits the context
    with context:
        context.set_state(execution_state=state, error=error, commit=False)

    assert context._state == expected_state, "task state was not set correctly"
    assert context._error == error, "task error was not set"


@pytest.mark.parametrize(
    "is_api",
    [True, False],
)
def test_context_inputs(rundb_mock, is_api):
    with unittest.mock.patch("mlrun.config.is_running_as_api", return_value=is_api):
        run_dict = _generate_run_dict()

        # create run object from dict and dict again to mock the run serialization
        run = mlrun.run.RunObject.from_dict(run_dict)
        context = mlrun.MLClientCtx.from_dict(PipelineRun(run.to_dict()), is_api=is_api)
        assert (
            context.get_input("input-key").artifact_url
            == run_dict["spec"]["inputs"]["input-key"]
        )
        assert context._inputs["input-key"] == run_dict["spec"]["inputs"]["input-key"]

        key = "store-input"
        url = run_dict["spec"]["inputs"][key]
        assert context._inputs[key] == run_dict["spec"]["inputs"][key]

        # 'store-input' is a store artifact, store it in the db before getting it
        artifact = mlrun.artifacts.Artifact(key, b"123")
        rundb_mock.store_artifact(key, artifact.to_dict(), uid="123")
        mlrun.datastore.store_manager.object(
            url,
            key,
            project=run_dict["metadata"]["project"],
            allow_empty_resources=True,
        )
        context._allow_empty_resources = True
        assert context.get_input(key).artifact_url == run_dict["spec"]["inputs"][key]


@pytest.mark.parametrize(
    "host, is_logging_worker", [("test-worker-0", True), ("test-worker-1", False)]
)
def test_is_logging_worker(host: str, is_logging_worker: bool):
    """
    Test the `is_logging_worker` method of the context.

    :param host:              The pod's name where the worker's rank is expected to be.
    :param is_logging_worker: The expected result.
    """
    context = mlrun.execution.MLClientCtx()
    context.set_label(mlrun_constants.MLRunInternalLabels.kind, "mpijob")
    context.set_label(mlrun_constants.MLRunInternalLabels.host, host)
    assert context.is_logging_worker() is is_logging_worker


@pytest.mark.parametrize(
    "owner",
    [
        "some-owner",
        None,
    ],
)
def test_artifact_owner(rundb_mock, owner):
    run_dict = _generate_run_dict()
    if owner:
        run_dict["metadata"]["labels"][mlrun_constants.MLRunInternalLabels.owner] = (
            owner
        )

    run = mlrun.run.RunObject.from_dict(run_dict)
    context = mlrun.MLClientCtx.from_dict(run.to_dict())

    artifact = context.log_artifact("artifact", body="123")
    assert artifact.producer.get("owner") == owner


def _generate_run_dict():
    return {
        "metadata": {
            "name": "test-context-from-run-dict",
            "project": "default",
            "labels": {"label-key": "label-value"},
            "annotations": {"annotation-key": "annotation-value"},
        },
        "spec": {
            "parameters": {"p1": 1, "p2": "a string"},
            "output_path": "test_artifact_path",
            "inputs": {
                "input-key": "input-url",
                "store-input": "store://store-input",
            },
            "allow_empty_resources": True,
        },
    }
