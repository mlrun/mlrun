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
import os
import pathlib
import shutil
import tempfile
import time
import typing
import unittest.mock
from copy import deepcopy
from datetime import datetime
from itertools import product
from types import SimpleNamespace
from typing import Optional, Union

import pandas as pd
import pytest

import mlrun
import mlrun.common.schemas as schemas
from mlrun.artifacts.llm_prompt import LLMPromptArtifact
from mlrun.artifacts.model import ModelArtifact
from mlrun.errors import MLRunInvalidArgumentError, ModelRunnerError
from mlrun.serving import (  # noqa: F401
    LLModel,
    Model,
    ModelRunnerSelector,
    ModelRunnerStep,
    ModelSelector,
    RouterStep,
)
from mlrun.serving.states import GraphError
from mlrun.utils import logger
from tests.conftest import results
from tests.serving.demo_states import (  # noqa: F401
    Chain,
    ChainWithContext,
    Counter,
    Echo,
    EchoError,
    LLModelWithTools,
    ModelClass,
    MyRemoteModel,
    MySelector,
    Raiser,
    Route,
    Tool,
    multiply_input,
)


class _DummyStreamRaiser:
    def push(self, data):
        raise ValueError("DummyStreamRaiser raises an error")


def append_and_return(lst, event):
    body = event.body
    body["timestamp"] = datetime.now()
    lst.append(event.body)
    return lst


def create_mocked_get_store_artifact(
    model_artifact: Union[ModelArtifact, LLMPromptArtifact],
    origin_model: ModelArtifact = None,
):
    _model_artifact = origin_model

    def mocked_get_store_artifact(uri, **kwargs):
        if uri == model_artifact.uri:
            return model_artifact, None
        elif uri == model_artifact.spec.parent_uri:
            return _model_artifact, None
        else:
            raise mlrun.errors.MLRunInvalidArgumentError("Artifact uri not found")

    return mocked_get_store_artifact


def test_async_basic():
    function = mlrun.new_function("tests", kind="serving")
    flow = function.set_topology("flow", engine="async")
    queue = flow.to(name="s1", class_name="ChainWithContext").to(
        "$queue", "q1", path=""
    )

    s2 = queue.to(name="s2", class_name="ChainWithContext", function="some_function")
    s2.to(name="s4", class_name="ChainWithContext")
    s2.to(
        name="s5", class_name="ChainWithContext"
    ).respond()  # this state returns the resp

    queue.to(name="s3", class_name="ChainWithContext", function="some_other_function")

    # plot the graph for test & debug
    flow.plot(f"{results}/serving/async.png")
    server = function.to_mock_server()
    server.context.visits = {}
    logger.info(f"\nAsync Flow:\n{flow.to_yaml()}")
    resp = server.test(body=[])

    server.wait_for_completion()
    assert resp == ["s1", "s2", "s5"], "flow result is incorrect"
    assert server.context.visits == {
        "s1": 1,
        "s2": 1,
        "s4": 1,
        "s3": 1,
        "s5": 1,
    }, "flow didnt visit expected states"


def test_async_error_on_missing_function_parameter():
    function = mlrun.new_function("tests", kind="serving")
    flow = function.set_topology("flow", engine="async")
    queue = flow.to(name="s1", class_name="ChainWithContext").to(
        "$queue", "q1", path=""
    )

    with pytest.raises(
        MLRunInvalidArgumentError,
        match="step 's2' must specify a function, because it follows a queue step",
    ):
        queue.to(name="s2", class_name="ChainWithContext")


def test_async_nested():
    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="async")
    graph.add_step(name="s1", class_name="Echo")
    graph.add_step(name="s2", handler="multiply_input", after="s1")
    graph.add_step(name="s3", class_name="Echo", after="s2")

    router_step = graph.add_step("*", name="ensemble", after="s2")
    router_step.add_route("m1", class_name="ModelClass", model_path=".", multiplier=100)
    router_step.add_route("m2", class_name="ModelClass", model_path=".", multiplier=200)
    router_step.add_route(
        "m3:v1", class_name="ModelClass", model_path=".", multiplier=300
    )

    graph.add_step(name="final", class_name="Echo", after="ensemble").respond()

    server = function.to_mock_server()
    try:
        # plot the graph for test & debug
        graph.plot(f"{results}/serving/nested.png")
        resp = server.test("/v2/models/m2/infer", body={"inputs": [5]})
    finally:
        server.wait_for_completion()
    # resp should be input (5) * multiply_input (2) * m2 multiplier (200)
    assert resp["outputs"] == 5 * 2 * 200, f"wrong health response {resp}"


def test_on_error():
    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="async")
    chain = graph.to("Chain", name="s1")
    chain.to("Raiser").respond().error_handler(
        name="catch", class_name="EchoError", full_event=True
    ).to("Chain", name="s3")

    function.verbose = True
    try:
        server = function.to_mock_server()

        # plot the graph for test & debug
        graph.plot(f"{results}/serving/on_error.png")
        resp = server.test(body=[])
    finally:
        server.wait_for_completion()
    if isinstance(resp, dict):
        assert (
            resp["error"] and resp["origin_state"] == "Raiser"
        ), f"error wasn't caught, resp={resp}"
    else:
        assert (
            resp.error and resp.origin_state == "Raiser"
        ), f"error wasn't caught, resp={resp}"


def test_push_error():
    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="async")
    chain = graph.to("Chain", name="s1")
    chain.to("Raiser")

    function.verbose = True
    server = function.to_mock_server()
    server.error_stream = "dummy:///nothing"
    # Force an error inside push_error itself
    server._error_stream_object = _DummyStreamRaiser()

    server.test(body=[])
    server.wait_for_completion()


def test_batch():
    function = mlrun.new_function("tests", kind="serving", project="x")
    graph = function.set_topology("flow", engine="async")
    graph.to("storey.Batch", "my_batching", max_events=3, flush_after_seconds=1).to(
        "storey.ToDataFrame", "my_to_df", index="my_int"
    ).to("storey.Reduce", initial_value=[], fn=append_and_return, full_event=True)
    # Reduce is used to get a single result in wait_for_completion (termination result in storey)
    server = function.to_mock_server()

    events = [{"my_int": i, "my_string": f"this is {i}"} for i in range(10)]

    for event in events:
        time.sleep(0.1)
        server.test(body=event)
    results = server.wait_for_completion()
    assert len(results) == 4

    prev_ts = pd.Timestamp.min
    for i, df in enumerate(results[:4]):
        if i < 3:
            assert len(df) == 3, f"Batch {i} expected 3 rows, got {len(df)}"
        if i == 3:
            assert len(df) == 1, f"Batch {i} expected 1 rows, got {len(df)}"

            # check all timestamps in the batch are the same
            unique_ts = df["timestamp"].unique()
            assert (
                len(unique_ts) == 1
            ), f"Batch {i} has multiple timestamps: {unique_ts}"
            batch_ts = unique_ts[0]

            # check timestamp order between batches
            assert (
                batch_ts > prev_ts
            ), f"Batch {i} timestamp {batch_ts} not greater than previous {prev_ts}"
            prev_ts = batch_ts


class MyModel(Model):
    def __init__(
        self, inc: int, gpu_number: Optional[int] = None, err: bool = True, **kwargs
    ):
        super().__init__(**kwargs)
        self.inc = inc
        self.gpu_number = gpu_number
        self.err = err

    def predict(self, body):
        try:
            body["n"] += self.inc
        except TypeError:
            if self.err:
                raise
            else:
                body["n"] = 1
        body.pop("models", None)
        if self.gpu_number is not None:
            body["gpu"] = self.gpu_number
        return body

    async def predict_async(self, body):
        body = self.predict(body)
        body["async"] = True
        return body

    def do(self, event):
        return self.predict(event)


class MyLLM(LLModel):
    def predict(self, body, **kwargs):
        body["url"] = self.model_artifact.model_url
        body["default_config"] = self.model_artifact.default_config
        body["invocation_config"] = kwargs.get("invocation_config")
        body["prompt"] = kwargs.get("messages")
        return body


class DummyLLM(LLModel):
    def predict(self, body: typing.Any, **kwargs):
        return body


class DummyAsyncLLM(LLModel):
    async def predict_async(self, body: typing.Any, **kwargs):
        return body


class DummyAsyncLLMWithoutAsyncPredict(LLModel):
    def predict(self, body: typing.Any, **kwargs):
        return body


class MyPklModel(Model):
    def __init__(self, name, artifact_uri, **kwargs):
        super().__init__(
            name=name,
            artifact_uri=artifact_uri,
            **kwargs,
        )
        self.model = None

    def load(self) -> None:
        model_path, _ = self.get_local_model_path()
        with open(model_path) as f:
            data = f.read()
        # Create a simple mock model object with a .predict method
        self.model = SimpleNamespace(predict=lambda x=None: data)

    def predict(self, body):
        body["result"] = self.model.predict(body)
        return body


class ModelWithoutPredict(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def predict_async(self, body: typing.Any, **kwargs) -> typing.Any:
        return body


class ModelWithoutAsyncPredict(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self, body: typing.Any, **kwargs) -> typing.Any:
        return body


def test_model_runner():
    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(name="my_model_runner")
    model_runner_step.add_model(
        model_class="MyModel",
        execution_mechanism="naive",
        endpoint_name="my_model",
        inc=1,
    )
    graph.to(model_runner_step).respond()

    assert "my_model" in graph.model_endpoints_names, "model endpoint name not in graph"

    server = function.to_mock_server()
    try:
        resp = server.test(body={"n": 1})
        assert resp == {"n": 2}
    finally:
        server.wait_for_completion()


@pytest.mark.parametrize("method", ["add_step", "to", "set_flow"])
def test_model_runner_add_model(method: str):
    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(name="my_model_runner")
    model_runner_step.add_model(
        model_class="MyModel",
        execution_mechanism="naive",
        endpoint_name="my_model_1",
        inc=1,
    )
    model_runner_step.add_model(
        model_class="MyModel",
        execution_mechanism="naive",
        endpoint_name="my_model_2",
        inc=2,
    )
    if method == "add_step":
        graph.add_step(model_runner_step).respond()
    elif method == "to":
        graph.to(
            name="echo",
            class_name="Echo",
            model_endpoint_creation_strategy=schemas.ModelEndpointCreationStrategy.SKIP,
        ).to(model_runner_step).respond()
    elif method == "set_flow":
        graph.set_flow([model_runner_step]).respond()
    assert set(graph.model_endpoints_names) == {
        "my_model_1",
        "my_model_2",
    }, "model endpoints name not in graph"

    server = function.to_mock_server()
    try:
        resp = server.test(body={"n": 1})
        assert resp == {"my_model_1": {"n": 2}, "my_model_2": {"n": 3}}
    finally:
        server.wait_for_completion()


@pytest.mark.parametrize("method", ["add_step", "to", "set_flow"])
def test_model_runner_add_model_failure(method: str):
    function = mlrun.new_function("tests", kind="serving")
    function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(name="my_model_runner")
    model_runner_step.add_model(
        model_class="MyModel",
        execution_mechanism="naive",
        endpoint_name="my_model",
        inc=1,
    )
    try:
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            model_runner_step.add_model(
                model_class="MyModel",
                execution_mechanism="naive",
                endpoint_name="my_model",
                inc=2,
            )
    except AssertionError:
        pytest.fail(
            "Expected 'mlrun.errors.MLRunInvalidArgumentError' using the same model name twice in step"
        )

    function_0 = mlrun.new_function("tests_1", kind="serving")
    graph_0 = function_0.set_topology("flow", engine="async")
    model_runner_step_0 = ModelRunnerStep(name="my_model_runner_0")
    model_runner_step_1 = ModelRunnerStep(name="my_model_runner_1")
    model_runner_step_0.add_model(
        model_class="MyModel",
        execution_mechanism="naive",
        endpoint_name="my_model",
        inc=1,
    )
    model_runner_step_1.add_model(
        model_class="MyModel",
        execution_mechanism="naive",
        endpoint_name="my_model",
        inc=2,
    )
    try:
        with pytest.raises(mlrun.serving.states.GraphError):
            if method == "add_step":
                graph_0.add_step(model_runner_step_0)
                graph_0.add_step(model_runner_step_1).respond()
            elif method == "to":
                graph_0.to(name="echo", class_name="Echo").to(model_runner_step_0).to(
                    model_runner_step_1
                ).respond()
            elif method == "set_flow":
                graph_0.set_flow([model_runner_step_0, model_runner_step_1]).respond()
    except AssertionError:
        pytest.fail(
            "Expected 'mlrun.serving.states.GraphError' using the same model name twice in graph"
        )


@pytest.mark.parametrize("model_runner_first", [True, False])
@pytest.mark.parametrize("method", ["add_step", "to"])
def test_model_runner_with_route_failure(model_runner_first: bool, method: str):
    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(name="my_model_runner")
    model_runner_step.add_model(
        model_class="MyModel",
        execution_mechanism="naive",
        endpoint_name="my_model",
        inc=1,
    )
    graph.to(class_name=RouterStep())

    if method == "add_step":
        adding_method = graph.add_step
    elif method == "to":
        adding_method = graph.to
    else:
        return

    if model_runner_first:
        adding_method(model_runner_step)
        try:
            with pytest.raises(mlrun.serving.states.GraphError):
                function.add_model(
                    class_name="MyModel", execution_mechanism="naive", key="my_model"
                )
        except AssertionError:
            pytest.fail(
                "Expected 'mlrun.serving.states.GraphError' using the same model name with router and ModelRunnerStep"
            )
    else:
        function.add_model(
            class_name="MyModel", execution_mechanism="naive", key="my_model"
        )
        try:
            with pytest.raises(mlrun.serving.states.GraphError):
                adding_method(model_runner_step)
        except AssertionError:
            pytest.fail(
                "Expected 'mlrun.serving.states.GraphError' using the same model name with router and ModelRunnerStep"
            )


@pytest.mark.parametrize("model_runner_first", [True, False])
@pytest.mark.parametrize("method", ["add_step", "to"])
def test_model_runner_with_route(model_runner_first: bool, method: str):
    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(name="my_model_runner_with_route")
    model_runner_step.add_model(
        model_class="MyModel",
        execution_mechanism="naive",
        endpoint_name="my_model",
        inc=1,
    )
    graph.to(class_name=RouterStep()).respond()

    if method == "add_step":
        adding_method = graph.add_step
    elif method == "to":
        adding_method = graph.to
    else:
        return

    if model_runner_first:
        adding_method(model_runner_step)

    function.add_model(
        "my_model_1",
        ".",
        class_name="MyModel",
        execution_mechanism="naive",
        name="my_model_1",
        inc=1,
    )

    if not model_runner_first:
        adding_method(model_runner_step)

    server = function.to_mock_server()
    try:
        resp = server.test(body={"n": 1})
        assert resp["n"] == 2
        assert graph.model_endpoints_names == ["my_model"]
        assert graph.model_endpoints_routes_names == ["my_model_1"]
    finally:
        server.wait_for_completion()


@pytest.mark.parametrize("raise_error", (True, False))
@pytest.mark.parametrize("with_error", (True, False))
def test_model_runner_error_raiser(raise_error: bool, with_error: bool):
    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(
        name="my_model_runner", raise_exception=raise_error
    )
    model_runner_step.add_model(
        model_class="MyModel",
        execution_mechanism="naive",
        endpoint_name="my_model",
        raise_error=False,
        inc=1,
    )
    graph.to(model_runner_step).respond()
    _test_model_runner_raise_error_output(function, raise_error, with_error)


@pytest.mark.parametrize("raise_error", (True, False))
@pytest.mark.parametrize("with_error", (True, False))
def test_model_runner_error_raiser_multiple_models(raise_error: bool, with_error: bool):
    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(
        name="my_model_runner", raise_exception=raise_error
    )
    model_runner_step.add_model(
        model_class="MyModel",
        execution_mechanism="naive",
        endpoint_name="my_model_0",
        raise_error=False,
        inc=1,
        err=False,
    )
    model_runner_step.add_model(
        model_class="MyModel",
        execution_mechanism="naive",
        endpoint_name="my_model_1",
        raise_error=False,
        inc=1,
    )
    graph.to(model_runner_step).respond()
    _test_model_runner_raise_error_output(
        function,
        raise_error,
        with_error,
        models=["my_model_0", "my_model_1"],
        models_with_error=["my_model_1"],
    )


@pytest.mark.parametrize("raise_error", (True, False))
@pytest.mark.parametrize("with_error", (True, False))
def test_model_runner_multiple_downstream_steps(raise_error: bool, with_error: bool):
    function = mlrun.new_function("tests-1", kind="serving")
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(
        name="my_model_runner", raise_exception=raise_error
    )
    model_runner_step.add_model(
        model_class="MyModel",
        execution_mechanism="naive",
        endpoint_name="my_model",
        raise_error=False,
        inc=1,
    )
    step = graph.to(model_runner_step)
    for i in range(5):
        echo = step.to(name=f"echo-{i}", class_name="Echo")
        if i == 4:
            echo.respond()

    _test_model_runner_raise_error_output(function, raise_error, with_error)


def _test_model_runner_raise_error_output(
    function, raise_error, with_error, models=None, models_with_error=None
):
    models_with_error = models_with_error or models
    server = function.to_mock_server()
    if with_error:
        if raise_error:
            with pytest.raises(RuntimeError):
                server.test(body={"n": "This should fail"})
        else:
            body = server.test(body={"n": "This should fail"})
            if models is None or len(models) == 1:
                assert "error" in body, f"Expected error field in body got {body}"
            else:
                assert all(
                    "error" in body.get(model) for model in models_with_error
                ), f"Expected error field for each model in body got {body}"
    else:
        if models is None or len(models) == 1:
            assert server.test(body={"n": 1}) == {"n": 2}
        else:
            assert server.test(body={"n": 1}) == {model: {"n": 2} for model in models}
    server.wait_for_completion()


class MyModelSelector(ModelSelector):
    def __init__(self, models: Union[list[str], list[Model]]):
        super().__init__()
        self.models = deepcopy(models)

    def select(
        self, event, available_models: list[Model]
    ) -> Union[list[str], list[Model]]:
        current_models = event.body.get("models")
        if current_models and set(current_models).issubset(set(self.models)):
            return current_models
        return []


class MyModelRunnerSelector(ModelRunnerSelector):
    def __init__(self, models: Union[list[str], list[Model]]):
        super().__init__()
        self.models = deepcopy(models)

    def select_models(
        self, event, available_models: list[Model]
    ) -> Union[list[str], list[Model]]:
        current_models = event.body.get("models")
        if current_models and set(current_models).issubset(set(self.models)):
            return current_models
        return []


@pytest.mark.parametrize(
    ("execution_mechanism", "selector"),
    list(
        product(
            ("process_pool", "dedicated_process", "thread_pool", "asyncio", "naive"),
            ("new", "old"),
        )
    ),
)
def test_model_runner_with_selector(execution_mechanism: str, selector: str):
    m1 = MyModel(
        name="m1",
        execution_mechanism="naive",
        inc=1,
    )
    m2 = MyModel(name="m2", inc=2)

    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="async")
    if selector == "new":
        model_runner_step = ModelRunnerStep(
            name="my_model_runner",
            model_runner_selector=MyModelRunnerSelector(models=["m1", "m2"]),
        )
    else:
        with pytest.warns(FutureWarning, match="model_selector.*deprecated"):
            model_runner_step = ModelRunnerStep(
                name="my_model_runner",
                model_selector=MyModelSelector(models=["m1", "m2"]),
            )
    model_runner_step.add_model(
        endpoint_name=m1.name,
        model_class=m1,
        execution_mechanism="naive",
    )
    model_runner_step.add_model(
        endpoint_name=m2.name,
        model_class=m2,
        execution_mechanism=execution_mechanism,
    )
    graph.to(model_runner_step).respond()

    server = function.to_mock_server()
    try:
        # both models
        resp = server.test(body={"n": 1, "models": ["m1", "m2"]})
        expected = {
            "m1": {"n": 2},
            "m2": {"n": 3},
        }
        if execution_mechanism == "asyncio":
            expected["m2"]["async"] = True
        assert resp == expected

        # only m2
        resp = server.test(body={"n": 1, "models": ["m2"]})
        expected = {"m2": {"n": 3}}
        if execution_mechanism == "asyncio":
            expected["m2"]["async"] = True
        assert resp == expected
    finally:
        server.wait_for_completion()


def test_model_runner_with_gpu_allocation():
    m1 = MyModel(
        name="m1", execution_mechanism="dedicated_process", inc=1, gpu_number=1
    )
    m2 = MyModel(
        name="m2", execution_mechanism="dedicated_process", inc=2, gpu_number=2
    )

    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(
        name="my_model_runner",
    )
    model_runner_step.add_model(
        endpoint_name=m1.name, model_class=m1, execution_mechanism="naive"
    )
    model_runner_step.add_model(
        endpoint_name=m2.name, model_class=m2, execution_mechanism="naive"
    )
    graph.to(model_runner_step).respond()

    server = function.to_mock_server()
    try:
        for n in range(10):
            resp = server.test(body={"n": n})
            assert resp == {"m1": {"n": n + 1, "gpu": 1}, "m2": {"n": n + 2, "gpu": 2}}
    finally:
        server.wait_for_completion()


@pytest.mark.parametrize(
    "execution_mechanism", ("naive", "thread_pool", "process_pool", "dedicated_process")
)
@pytest.mark.parametrize("notebook_usage", (False, True))
def test_model_runner_with_remote_model(execution_mechanism, notebook_usage):
    project = mlrun.new_project("remote-model-project", save=False)
    model_artifact = project.log_model(
        "my_model",
        model_url="http://localhost:8080/v2/models/mymodel/infer",
        default_config={"model_version": "4"},
    )

    if notebook_usage:
        if execution_mechanism in ["process_pool", "dedicated_process"]:
            pytest.skip(
                "ModelRunnerStep with notebook and process_pool / dedicated process is not supported - ML-11340"
            )
        filename = str(pathlib.Path(__file__).parent / "assets" / "remote_model.ipynb")
    else:
        filename = __file__
    function = mlrun.code_to_function("tests", kind="serving", filename=filename)
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(name="my_model_runner")
    model_runner_step.add_model(
        model_class="MyRemoteModel",
        execution_mechanism=execution_mechanism,
        endpoint_name="my_endpoint",
        model_artifact=model_artifact,
    )
    async_model_runner_step = ModelRunnerStep(name="my_async_model_runner")
    async_model_runner_step.add_model(
        model_class="MyRemoteModel",
        execution_mechanism="asyncio",
        endpoint_name="my_async_endpoint",
        model_artifact=model_artifact,
    )

    graph.to(model_runner_step).to(async_model_runner_step).respond()
    assert (
        "my_endpoint" in graph.model_endpoints_names
    ), "model endpoint name not in graph"
    assert (
        "my_async_endpoint" in graph.model_endpoints_names
    ), "async model endpoint name not in graph"
    # Mock needed since no artifact is saved in this test, so retrieval by URI isn't possible.
    # Mocked function used to verify artifact URI is passed correctly.

    with unittest.mock.patch(
        "mlrun.store_manager.get_store_artifact",
        side_effect=create_mocked_get_store_artifact(model_artifact=model_artifact),
    ):
        server = function.to_mock_server()
    try:
        resp = server.test(body={"prompt": "What is the capital of france?"})
        assert resp["default_config"] == {"model_version": "4"}
        assert resp["url"] == "http://localhost:8080/v2/models/mymodel/infer"
        assert resp["prompt"] == "What is the capital of france?"
        assert resp["async_triggered"] == "Async predict was triggered."
    finally:
        server.wait_for_completion()


@pytest.mark.parametrize(
    "execution_mechanism", ("naive", "thread_pool", "process_pool", "dedicated_process")
)
def test_mock_server_invalid_source_path(execution_mechanism):
    project = mlrun.new_project("remote-model-project", save=False)
    model_artifact = project.log_model(
        "my_model",
        model_url="http://localhost:8080/v2/models/mymodel/infer",
        default_config={"model_version": "4"},
    )
    current_temp_dir = tempfile.gettempdir()
    parent_dir = os.path.dirname(current_temp_dir)
    new_temp_dir = os.path.join(parent_dir, "my_custom_mlrun_temp")
    os.makedirs(new_temp_dir, exist_ok=True)
    try:
        file_path = os.path.join(new_temp_dir, "test_script.py")
        with open(file_path, "w") as f:
            f.write('print("Hello from custom temp dir!")\n')

        function = mlrun.code_to_function("tests", kind="serving", filename=file_path)
        graph = function.set_topology("flow", engine="async")
        model_runner_step = ModelRunnerStep(name="my_model_runner")
        model_runner_step.add_model(
            model_class="MyRemoteModel",
            execution_mechanism=execution_mechanism,
            endpoint_name="my_endpoint",
            model_artifact=model_artifact,
        )
        async_model_runner_step = ModelRunnerStep(name="my_async_model_runner")
        async_model_runner_step.add_model(
            model_class="MyRemoteModel",
            execution_mechanism="asyncio",
            endpoint_name="my_async_endpoint",
            model_artifact=model_artifact,
        )

        graph.to(model_runner_step).to(async_model_runner_step).respond()
        try:
            with pytest.raises(
                mlrun.errors.MLRunRuntimeError,
                match="it must be located either under the current working .* or the system temporary directory",
            ):
                server = function.to_mock_server()
        except AssertionError:
            # error was not raised, server was created
            server.wait_for_completion()
    finally:
        # Clean up the custom temp directory
        if os.path.exists(new_temp_dir):
            shutil.rmtree(new_temp_dir)


def test_model_runner_with_remote_shared_model():
    project = mlrun.new_project("remote-model-project", save=False)
    model_artifact = project.log_model(
        "my_model",
        model_url="http://localhost:8080/v2/models/mymodel/infer",
        default_config={"model_version": "4"},
    )
    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="async")
    graph.add_shared_model(
        name="my_model",
        model_class="MyRemoteModel",
        model_artifact=model_artifact,
        execution_mechanism="naive",
    )
    model_runner_step = ModelRunnerStep(name="my_model_runner")
    model_runner_step.add_shared_model_proxy(
        endpoint_name="my_endpoint",
        model_artifact=model_artifact,
        shared_model_name="my_model",
    )
    graph.to(model_runner_step).respond()
    assert (
        "my_endpoint" in graph.model_endpoints_names
    ), "model endpoint name not in graph"
    # Mock needed since no artifact is saved in this test, so retrieval by URI isn't possible.
    # Mocked function used to verify artifact URI is passed correctly.

    with unittest.mock.patch(
        "mlrun.store_manager.get_store_artifact",
        side_effect=create_mocked_get_store_artifact(model_artifact=model_artifact),
    ):
        server = function.to_mock_server()
    try:
        resp = server.test(body={"prompt": "What is the capital of france?"})
        assert resp["default_config"] == {"model_version": "4"}
        assert resp["url"] == "http://localhost:8080/v2/models/mymodel/infer"
        assert resp["prompt"] == "What is the capital of france?"
    finally:
        server.wait_for_completion()


def test_add_model_after_adding_the_mrs_to_the_graph():
    project = mlrun.new_project("remote-model-project", save=False)
    model_artifact = project.log_model(
        "my_model",
        model_url="http://localhost:8080/v2/models/mymodel/infer",
        default_config={"model_version": "4"},
    )
    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="async")
    graph.add_shared_model(
        name="my_model",
        model_class="MyRemoteModel",
        model_artifact=model_artifact,
        execution_mechanism="naive",
    )
    model_runner_step = ModelRunnerStep(name="my_model_runner")
    model_runner_step.add_shared_model_proxy(
        endpoint_name="my_endpoint",
        model_artifact=model_artifact,
        shared_model_name="my_model",
    )
    model_runner_step_2 = graph.to(model_runner_step).respond()
    model_runner_step.add_model(
        endpoint_name="my_endpoint-2",
        model_class="MyRemoteModel",
        model_artifact=model_artifact,
        execution_mechanism="naive",
    )
    assert (
        "my_endpoint" in graph.model_endpoints_names
    ), "model endpoint name not in graph"

    assert (
        "my_endpoint-2" not in graph.model_endpoints_names
    ), "model endpoint name not in graph"

    model_runner_step_2.add_model(
        endpoint_name="my_endpoint-2",
        model_class="MyRemoteModel",
        model_artifact=model_artifact,
        execution_mechanism="naive",
    )

    assert (
        "my_endpoint" in graph.model_endpoints_names
    ), "model endpoint name not in graph"

    assert (
        "my_endpoint-2" in graph.model_endpoints_names
    ), "model endpoint name not in graph"


def test_get_local_model_path():
    project = mlrun.new_project("get-model-path-project", save=False)
    model_dir = str(pathlib.Path(__file__).parent / "assets")
    model_artifact = project.log_model(
        "my_model", target_path=model_dir, model_file="model.pkl", upload=False
    )
    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(name="my_model_runner")
    model_runner_step.add_model(
        model_class="MyPklModel",
        execution_mechanism="naive",
        endpoint_name="my_endpoint",
        model_artifact=model_artifact,
    )
    graph.to(model_runner_step).respond()
    with unittest.mock.patch(
        "mlrun.serving.states.mlrun.store_manager.get_store_artifact",
        side_effect=create_mocked_get_store_artifact(model_artifact=model_artifact),
    ):
        server = function.to_mock_server()
    try:
        resp = server.test(body={})
        assert resp["result"] == "123"
    finally:
        server.wait_for_completion()


@pytest.mark.parametrize("raise_exception", [True, False])
@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("model_uri", [True, False])
@pytest.mark.parametrize("llm", [None, "uri_based", "object_based"])
def test_shared_llm_with_model_runner(raise_exception, shared, model_uri, llm):
    project = mlrun.new_project("get-model-path-project", save=False)
    function = mlrun.new_function("tests", kind="serving")
    model_artifact = project.log_model(
        "my_model",
        model_url="mock://my-model",
        default_config={"model_version": "4"},
    )
    llm_artifact = None
    if llm:
        llm_artifact = project.log_llm_prompt(
            "my_llm",
            prompt_template=[
                {"role": "user", "content": "What is the capital city of {country}?"}
            ],
            model_artifact=model_artifact
            if llm == "object_based"
            else model_artifact.uri,
            prompt_legend={"country": {"field": None, "description": "Great"}},
        )

    with unittest.mock.patch(
        "mlrun.store_manager.get_store_artifact",
        side_effect=create_mocked_get_store_artifact(
            model_artifact=llm_artifact or model_artifact, origin_model=model_artifact
        ),
    ):
        if model_uri:
            model_artifact_param = model_artifact.uri
            llm_artifact_param = llm_artifact.uri if llm_artifact else None
        else:
            model_artifact_param = model_artifact
            llm_artifact_param = llm_artifact

        graph = function.set_topology("flow", engine="async")
        model_runner_step = ModelRunnerStep(
            name="model-runner", raise_exception=raise_exception
        )
        model_class = "LLModel" if llm else "MyRemoteModel"
        if shared:
            graph.add_shared_model(
                name="shared-model",
                execution_mechanism="naive",
                model_class=model_class,
                model_artifact=model_artifact_param,
                result_path="outputs" if llm else None,
            )
            model_runner_step.add_shared_model_proxy(
                endpoint_name="my-model",
                shared_model_name="shared-model",
                model_artifact=llm_artifact_param or model_artifact_param,
            )
        else:
            model_runner_step.add_model(
                model_class=model_class,
                execution_mechanism="naive",
                endpoint_name="my-model",
                model_artifact=llm_artifact_param or model_artifact_param,
                result_path="outputs" if llm else None,
            )

        graph.to(model_runner_step).respond()

        server = function.to_mock_server()
        try:
            resp = server.test(body={"country": "france"})
            if llm:
                assert (
                    resp["outputs"]["answer"]
                    == "You are using a mock model provider, no actual inference is performed."
                )
                assert resp["outputs"]["usage"] == {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                }
            else:
                assert resp["default_config"] == {"model_version": "4"}
                assert resp["url"] == "mock://my-model"
            server.test(body={"country": "france"})
        finally:
            server.wait_for_completion()


@pytest.mark.parametrize(
    "legend",
    (
        None,
        {"country": {"field": None, "description": "Great"}},
        {
            "country": {"field": None, "description": "Great"},
            "Not exists": {"field": "country", "description": "Great"},
        },
        {
            "country": {"field": "country", "description": "Great"},
            "profession": {"field": "profession", "description": "Great"},
            "some_other_ph": {"field": "some_other_ph", "description": "Great"},
        },
        {
            "country": {"field": "not-exist", "description": "Great"},
        },
        {
            "country": {"field": "state", "description": "Great"},
        },
    ),
)
def test_llm_with_missing_legends(legend: dict):
    project = mlrun.new_project("get-model-path-project", save=False)
    function = mlrun.new_function("tests", kind="serving")
    model_artifact = project.log_model(
        "my_model",
        model_url="http://localhost:8080/v2/models/mymodel/infer",
        default_config={"model_version": "4"},
    )
    llm_artifact = project.log_llm_prompt(
        "my_llm",
        prompt_template=[
            {
                "role": "user",
                "content": "What is the capital city of {country} ?{some_other_ph}",
            },
            {"role": "system", "content": "you are answer as {profession}"},
        ],
        model_artifact=model_artifact.uri,
        prompt_legend=legend,
    )
    with unittest.mock.patch(
        "mlrun.store_manager.get_store_artifact",
        side_effect=create_mocked_get_store_artifact(
            model_artifact=llm_artifact, origin_model=model_artifact
        ),
    ):
        graph = function.set_topology("flow", engine="async")
        model_runner_step = ModelRunnerStep(name="model-runner", raise_exception=True)

        model_runner_step.add_model(
            model_class="MyLLM",
            execution_mechanism="naive",
            endpoint_name="my-model",
            model_artifact=llm_artifact,
        )
        graph.to(model_runner_step).respond()
        server = function.to_mock_server()
        if (
            legend is not None
            and "country" in legend
            and legend["country"]["field"] == "state"
        ):
            # If the legend is set to use state, we expect the prompt to use state
            # and not country.
            expected_prompt = [
                {"role": "user", "content": "What is the capital city of Israel ?!"},
                {"role": "system", "content": "you are answer as Data scientist"},
            ]
        else:
            expected_prompt = [
                {"role": "user", "content": "What is the capital city of France ?!"},
                {"role": "system", "content": "you are answer as Data scientist"},
            ]
        resp = server.test(
            body={
                "country": "France",
                "some_other_ph": "!",
                "profession": "Data scientist",
                "state": "Israel",
            }
        )
        server.wait_for_completion()
        assert resp["prompt"] == expected_prompt


def test_llm_with_missing_llm_prompt():
    project = mlrun.new_project("get-model-path-project", save=False)
    function = mlrun.new_function("tests", kind="serving")
    model_artifact = project.log_model(
        "my_model",
        model_url="http://localhost:8080/v2/models/mymodel/infer",
        default_config={"model_version": "4"},
    )
    with unittest.mock.patch(
        "mlrun.store_manager.get_store_artifact",
        side_effect=create_mocked_get_store_artifact(model_artifact=model_artifact),
    ):
        graph = function.set_topology("flow", engine="async")
        model_runner_step = ModelRunnerStep(name="model-runner", raise_exception=True)

        model_runner_step.add_model(
            model_class="MyLLM",
            execution_mechanism="naive",
            endpoint_name="my-model",
            model_artifact=model_artifact,
        )
        graph.to(model_runner_step).respond()
        server = function.to_mock_server()
        expected_prompt = [
            {"role": "user", "content": "What is the capital city of Israel ?!"},
            {"role": "system", "content": "you are answer as Data scientist"},
        ]

        resp = server.test(
            body={
                "messages": expected_prompt,
            }
        )
        server.wait_for_completion()
        assert resp["prompt"] == expected_prompt


@pytest.mark.parametrize("execution_mechanism", ("naive", "thread_pool", "asyncio"))
def test_using_model_without_predict_implementation(execution_mechanism: str):
    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(name="model-runner")
    model_runner_step.add_model(
        model_class="ModelWithoutPredict"
        if execution_mechanism != "asyncio"
        else "ModelWithoutAsyncPredict",
        execution_mechanism=execution_mechanism,
        endpoint_name="model_without_predict",
    )
    graph.to(model_runner_step).respond()

    with pytest.raises(ModelRunnerError) as exc_info:
        function.to_mock_server()
    method_name = "predict()" if execution_mechanism != "asyncio" else "predict_async()"
    expected_msg = (
        f"model_without_predict is running with {execution_mechanism} execution_mechanism but "
        f"{method_name} is not implemented"
    )
    assert expected_msg in str(exc_info.value)


@pytest.mark.parametrize("execution_mechanism", ("naive", "thread_pool", "asyncio"))
def test_shared_using_model_without_predict_implementation(execution_mechanism: str):
    project = mlrun.new_project("model-project", save=False)
    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="async")
    model_artifact = project.log_model(
        "my_model",
        model_url="http://localhost:8080/v2/models/mymodel/infer",
        default_config={"model_version": "4"},
    )
    graph.add_shared_model(
        name="model_without_predict_shared",
        model_class="ModelWithoutPredict"
        if execution_mechanism != "asyncio"
        else "ModelWithoutAsyncPredict",
        execution_mechanism=execution_mechanism,
        model_artifact=model_artifact,
    )
    model_runner_step = ModelRunnerStep(name="model-runner")
    model_runner_step.add_shared_model_proxy(
        endpoint_name="model_without_predict_shared",
        shared_model_name="model_without_predict_shared",
        model_artifact=model_artifact,
    )
    graph.to(model_runner_step).respond()
    with unittest.mock.patch(
        "mlrun.store_manager.get_store_artifact",
        side_effect=create_mocked_get_store_artifact(model_artifact=model_artifact),
    ):
        with pytest.raises(ModelRunnerError) as exc_info:
            function.to_mock_server()
        method_name = (
            "predict()" if execution_mechanism != "asyncio" else "predict_async()"
        )
        expected_msg = (
            f"model_without_predict_shared is running with {execution_mechanism} execution_mechanism but "
            f"{method_name} is not implemented"
        )
        assert expected_msg in str(exc_info.value)


def test_model_runner_add_proxy_model_failure():
    project = mlrun.new_project("remote-model-project", save=False)
    model_artifact = project.log_model(
        "my_model",
        model_url="http://localhost:8080/v2/models/mymodel/infer",
        default_config={"model_version": "4"},
    )
    model_artifact_1 = project.log_model(
        "my_model_1",
        model_url="http://localhost:8080/v2/models/mymodel/infer",
        default_config={"model_version": "4"},
    )
    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(name="my_model_runner")
    with pytest.raises(GraphError, match="Can't find shared model named my_mode"):
        model_runner_step.add_shared_model_proxy(
            endpoint_name="my_endpoint",
            model_artifact=model_artifact,
            shared_model_name="my_model",
        )
        graph.to(model_runner_step).respond()
    graph.add_shared_model(
        name="my_model",
        model_class="MyRemoteModel",
        model_artifact=model_artifact,
        execution_mechanism="naive",
    )
    with pytest.raises(GraphError, match="Can't find shared model named my_model_1"):
        model_runner_step.add_shared_model_proxy(
            endpoint_name="my_endpoint_1",
            model_artifact=model_artifact_1,
            shared_model_name="my_model_1",
        )
        graph.to(model_runner_step).respond()


@pytest.mark.parametrize(
    "concurrency",
    (
        "max_threads",
        "max_processes",
    ),
)
def test_configure_model_runner_step_max_threads_processes(concurrency: str):
    m1 = MyModel(
        name="m1",
        execution_mechanism="naive",
        inc=1,
    )

    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(
        name="my_model_runner",
    )
    model_runner_step.add_model(
        endpoint_name=m1.name,
        model_class=m1,
        execution_mechanism="thread_pool"
        if concurrency == "max_threads"
        else "process_pool",
    )
    if concurrency == "max_threads":
        model_runner_step.configure_pool_resource(max_threads=48)
    elif concurrency == "max_processes":
        model_runner_step.configure_pool_resource(max_processes=32)

    graph.to(model_runner_step).respond()
    server = function.to_mock_server()

    if concurrency == "max_processes":
        assert (
            server.graph["my_model_runner"]._async_object.max_processes == 32
        ), "Max processes not configured properly"
    elif concurrency == "max_threads":
        assert (
            server.graph["my_model_runner"]._async_object.max_threads == 48
        ), "Max threads not configured properly"
    server.test(body={"n": 1})
    server.wait_for_completion()


@pytest.mark.parametrize(
    "model_class, raise_exception",
    [
        (
            "LLModel",
            True,
        ),  #  LLModel should raise error because predict was not overridden
        #  DummyAsyncLLMWithoutAsyncPredict should raise error because async_predict was not overridden:
        ("DummyAsyncLLMWithoutAsyncPredict", True),
        ("DummyLLM", False),
        ("DummyAsyncLLM", False),
    ],
)
def test_llmodel_without_model_artifact(model_class, raise_exception):
    is_async = model_class in ("DummyAsyncLLM", "DummyAsyncLLMWithoutAsyncPredict")
    execution_mechanism = "asyncio" if is_async else "naive"
    predict_function_name = "predict_async" if is_async else "predict"
    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(name="model-runner")
    project = mlrun.new_project("llmodel-without-model-artifact", save=False)
    llm_artifact = project.log_llm_prompt(
        "my_llm",
        prompt_template=[
            {"role": "user", "content": "What is the capital city of {country}?"}
        ],
        prompt_legend={"country": {"field": None, "description": "Great"}},
    )

    model_runner_step.add_model(
        model_class=model_class,
        execution_mechanism=execution_mechanism,
        endpoint_name="my-model",
        model_artifact=llm_artifact,
    )
    graph.to(model_runner_step).respond()
    server = None
    with unittest.mock.patch(
        "mlrun.datastore.datastore.get_store_resource",
        return_value=llm_artifact,
    ):
        try:
            if raise_exception:
                with pytest.raises(
                    mlrun.errors.MLRunRuntimeError,
                    match=f"Model provider could not be determined for model 'my-model', and the"
                    f" {predict_function_name} function was not overridden",
                ):
                    server = function.to_mock_server()
            else:
                server = function.to_mock_server()
                resp = server.test(body={"country": "france"})
                assert resp == {"country": "france"}
        finally:
            if server:
                server.wait_for_completion()


@pytest.mark.parametrize("method", ["add_step", "to"])
def test_cyclic_graph(method):
    function = mlrun.new_function("tests", kind="serving", project="x")
    graph = function.set_topology("flow", engine="async", allow_cyclic=True)

    if method == "to":
        graph.to(name="start", class_name="Echo").to(
            class_name="Counter", name="count"
        ).to(name="route", class_name="Route", cycle_to="count").to(
            name="end", class_name="Echo"
        ).respond()
    else:
        graph.add_step(name="start", class_name="Echo")
        graph.add_step(name="count", class_name="Counter", after="start")
        graph.add_step(
            name="route", class_name="Route", cycle_to="count", after="count"
        )
        graph.add_step(name="end", class_name="Echo", after="route").respond()

    server = function.to_mock_server()
    try:
        resp = server.test(body={"counter": 1})
        assert resp["counter"] == 5
    finally:
        server.wait_for_completion()


@pytest.mark.parametrize("method", ["add_step", "to"])
def test_cyclic_to_first_step(method):
    function = mlrun.new_function("tests", kind="serving", project="x")
    graph = function.set_topology("flow", engine="async", allow_cyclic=True)

    if method == "to":
        graph.to(class_name="Counter", name="count").to(
            name="route", class_name="Route", cycle_to="count"
        ).to(name="end", class_name="Echo").respond()
    else:
        graph.add_step(name="count", class_name="Counter")
        graph.add_step(
            name="route", class_name="Route", cycle_to="count", after="count"
        )
        graph.add_step(name="end", class_name="Echo", after="route").respond()

    server = function.to_mock_server()
    try:
        resp = server.test(body={"counter": 1})
        assert resp["counter"] == 5
    finally:
        server.wait_for_completion()


@pytest.mark.parametrize("method", ["add_step", "to"])
@pytest.mark.parametrize("max_iter", ["local", "global"])
def test_max_iter_of_cyclic_graph(method, max_iter):
    function = mlrun.new_function("tests", kind="serving", project="x")
    graph = function.set_topology(
        "flow",
        engine="async",
        allow_cyclic=True,
        max_iterations=1 if max_iter == "global" else 10,
    )
    if method == "to":
        graph.to(name="start", class_name="Echo").to(
            class_name="Counter", name="count"
        ).to(
            name="route",
            class_name="Route",
            cycle_to="count",
            max_iterations=1 if max_iter == "local" else None,
        ).to(name="end", class_name="Echo").respond()
    else:
        graph.add_step(name="start", class_name="Echo")
        graph.add_step(name="count", class_name="Counter", after="start")
        graph.add_step(
            name="route",
            class_name="Route",
            cycle_to="count",
            after="count",
            max_iterations=1,
        )
        graph.add_step(name="end", class_name="Echo", after="route").respond()
    if max_iter == "local":
        expected_error = r"Max iterations exceeded in step 'route'"
    else:
        expected_error = r"Max iterations exceeded in step 'count'"

    server = function.to_mock_server()
    try:
        with pytest.raises(RuntimeError, match=rf"{expected_error}"):
            server.test(body={"counter": 1})
    finally:
        server.wait_for_completion()


def test_default_max_iter_of_cyclic_graph():
    function = mlrun.new_function("tests", kind="serving", project="x")
    graph = function.set_topology(
        "flow",
        engine="async",
        allow_cyclic=True,
    )
    graph.to(name="start", class_name="Echo").to(class_name="Counter", name="count").to(
        name="route",
        class_name="Route",
        cycle_to="count",
    ).to(name="end", class_name="Echo").respond()

    expected_error = r"Max iterations exceeded in step 'count'"

    server = function.to_mock_server()
    try:
        with pytest.raises(RuntimeError, match=rf"{expected_error}"):
            server.test(body={"counter": -300})
    finally:
        server.wait_for_completion()


def test_mrs_with_tools_routing():
    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="async", allow_cyclic=True)
    model_runner_step = ModelRunnerStep(
        name="my_model_runner", model_runner_selector="MySelector"
    )
    model_runner_step.add_model(
        model_class="LLModelWithTools",
        execution_mechanism="naive",
        endpoint_name="llm_with_tools",
    )
    runner = graph.to(model_runner_step)
    runner.to(name="tool_a", class_name="Tool", cycle_to="my_model_runner")
    runner.to(name="tool_b", class_name="Tool", cycle_to="my_model_runner")
    runner.to(name="end", class_name="Echo").respond()

    server = function.to_mock_server()
    try:
        resp = server.test(body={"counter": 0})
        assert resp["counter"] == 5
        assert resp["tool_a"] == 2
        assert resp["tool_b"] == 2
    finally:
        server.wait_for_completion()


def test_invalid_cyclic_graph_definitions():
    function = mlrun.new_function("tests", kind="serving", project="x")
    graph = function.set_topology("flow", engine="async", allow_cyclic=False)

    with pytest.raises(
        GraphError, match="cyclic graphs are not allowed, enable allow_cyclic"
    ):
        graph.to(name="start", class_name="Echo").to(
            class_name="Counter", name="count"
        ).to(name="route", class_name="Route", cycle_to="count").to(
            name="end", class_name="Echo"
        ).respond()

    function_sync = mlrun.new_function("tests-sync", kind="serving", project="x")
    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError,
        match=r"Cyclic graphs are not supported with sync engine, please use async engine",
    ):
        function_sync.set_topology("flow", engine="sync", allow_cyclic=True)

    graph = function_sync.set_topology("flow", engine="sync")
    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError,
        match=r"Cyclic graphs are not supported with sync engine, please use async engine",
    ):
        graph.allow_cyclic = True
