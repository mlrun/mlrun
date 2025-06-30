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
from types import SimpleNamespace
from typing import Optional

import pytest

import mlrun
import mlrun.common.schemas as schemas
from mlrun.errors import MLRunInvalidArgumentError
from mlrun.serving import Model, ModelRunnerStep, ModelSelector, RouterStep
from mlrun.utils import logger
from tests.conftest import results

from .demo_states import *  # noqa


class _DummyStreamRaiser:
    def push(self, data):
        raise ValueError("DummyStreamRaiser raises an error")


def create_mocked_get_store_resource(model_artifact):
    def mocked_get_store_resource(uri, **kwargs):
        if uri == model_artifact.uri:
            return model_artifact
        else:
            raise mlrun.errors.MLRunInvalidArgumentError("Artifact uri not found")

    return mocked_get_store_resource


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

    # plot the graph for test & debug
    graph.plot(f"{results}/serving/nested.png")
    resp = server.test("/v2/models/m2/infer", body={"inputs": [5]})
    server.wait_for_completion()
    # resp should be input (5) * multiply_input (2) * m2 multiplier (200)
    assert resp["outputs"] == 5 * 2 * 200, f"wrong health response {resp}"


def test_on_error():
    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="async")
    chain = graph.to("Chain", name="s1")
    chain.to("Raiser").error_handler(
        name="catch", class_name="EchoError", full_event=True
    ).to("Chain", name="s3")

    function.verbose = True
    server = function.to_mock_server()

    # plot the graph for test & debug
    graph.plot(f"{results}/serving/on_error.png")
    resp = server.test(body=[])
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


class MyModel(Model):
    def __init__(self, inc: int, gpu_number: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.inc = inc
        self.gpu_number = gpu_number

    def predict(self, body):
        body["n"] += self.inc
        body.pop("models", None)
        if self.gpu_number is not None:
            body["gpu"] = self.gpu_number
        return body

    async def predict_async(self, body):
        return self.predict(body)

    def do(self, event):
        return self.predict(event)


class MyRemoteModel(Model):
    def __init__(self, name, raise_exception, artifact_uri, **kwargs):
        super().__init__(
            name=name,
            raise_exception=raise_exception,
            artifact_uri=artifact_uri,
            **kwargs,
        )
        self.artifact = None

    def predict(self, body):
        body["url"] = self.artifact.model_url
        body["default_config"] = self.artifact.default_config
        return body

    def load(self):
        self.artifact = self._get_artifact_object()


class MyPklModel(Model):
    def __init__(self, name, raise_exception, artifact_uri, **kwargs):
        super().__init__(
            name=name,
            raise_exception=raise_exception,
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
        function, raise_error, with_error, models=["my_model_0", "my_model_1"]
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
    function, raise_error, with_error, models=None
):
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
                    "error" in body.get(model) for model in models
                ), f"Expected error field for each model in body got {body}"
    else:
        if models is None or len(models) == 1:
            assert server.test(body={"n": 1}) == {"n": 2}
        else:
            assert server.test(body={"n": 1}) == {model: {"n": 2} for model in models}
    server.wait_for_completion()


class MyModelSelector(ModelSelector):
    def select(self, event, available_models: list[Model]) -> Optional[list[str]]:
        return event.body.get("models")


@pytest.mark.parametrize(
    "execution_mechanism",
    ("process_pool", "dedicated_process", "thread_pool", "asyncio", "naive"),
)
def test_model_runner_with_selector(execution_mechanism: str):
    m1 = MyModel(
        name="m1",
        execution_mechanism="naive",
        inc=1,
    )
    m2 = MyModel(name="m2", execution_mechanism=execution_mechanism, inc=2)

    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(
        name="my_model_runner",
        model_selector="MyModelSelector",
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
        # both models
        resp = server.test(body={"n": 1})
        assert resp == {"m1": {"n": 2}, "m2": {"n": 3}}

        # only m2
        resp = server.test(body={"n": 1, "models": ["m2"]})
        assert resp == {"m2": {"n": 3}}
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


def test_model_runner_with_remote_model():
    project = mlrun.new_project("remote-model-project", save=False)
    model_artifact = project.log_model(
        "my_model",
        model_url="http://localhost:8080/v2/models/mymodel/infer",
        default_config={"model_version": "4"},
    )
    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(name="my_model_runner")
    model_runner_step.add_model(
        model_class="MyRemoteModel",
        execution_mechanism="naive",
        endpoint_name="my_endpoint",
        model_artifact=model_artifact,
    )
    graph.to(model_runner_step).respond()
    assert (
        "my_endpoint" in graph.model_endpoints_names
    ), "model endpoint name not in graph"
    # Mock needed since no artifact is saved in this test, so retrieval by URI isn't possible.
    # Mocked function used to verify artifact URI is passed correctly.

    with unittest.mock.patch(
        "mlrun.serving.states.get_store_resource",
        side_effect=create_mocked_get_store_resource(model_artifact=model_artifact),
    ):
        server = function.to_mock_server()
    try:
        resp = server.test(body={"prompt": "What is the capital of france?"})
        assert resp["default_config"] == {"model_version": "4"}
        assert resp["url"] == "http://localhost:8080/v2/models/mymodel/infer"
        assert resp["prompt"] == "What is the capital of france?"
    finally:
        server.wait_for_completion()


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
        "mlrun.serving.states.get_store_resource",
        side_effect=create_mocked_get_store_resource(model_artifact=model_artifact),
    ):
        server = function.to_mock_server()
    try:
        resp = server.test(body={})
        assert resp["result"] == "123"
    finally:
        server.wait_for_completion()
