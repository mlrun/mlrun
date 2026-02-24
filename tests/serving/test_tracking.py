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

import json
import math
import pickle
import time
import typing
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from time import sleep
from typing import Union, cast

import numpy as np
import pandas as pd
import pytest
import storey

import mlrun
import mlrun.common.schemas.model_monitoring.constants as mm_constants
from mlrun.common.schemas import ModelEndpointCreationStrategy
from mlrun.datastore.datastore_profile import (
    DatastoreProfile,
    DatastoreProfileKafkaStream,
    DatastoreProfileRedis,
    DatastoreProfileV3io,
    register_temporary_client_datastore_profile,
    remove_temporary_client_datastore_profile,
)
from mlrun.platforms.iguazio import KafkaOutputStream, OutputStream
from mlrun.runtimes import ServingRuntime
from mlrun.serving import Model, ModelRunnerStep, ModelSelector
from mlrun.serving.states import RootFlowStep, RouterStep, StepKinds
from mlrun.serving.system_steps import MonitoringPreProcessor
from tests.serving.test_serving import _log_model

from .demo_states import (  # noqa: F401
    Counter,
    Echo,
    LLModelWithTools,
    MySelector,
    Route,
    Tool,
)

assets_path = str(Path(__file__).parent / "assets")
testdata = '{"inputs": [[5, 6]]}'


class ModelTestingClass(mlrun.serving.V2ModelServer):
    def load(self):
        self.context.logger.info(f"loading model {self.name}")

    def predict(self, request):
        print("predict:", request)
        multiplier = self.get_param("multiplier", 1)
        outputs = [value[0] * multiplier for value in request["inputs"]]
        return np.array(outputs)  # complex result type to check serialization


class ModelTestingCustomTrack(ModelTestingClass):
    def logged_results(self, request: dict, response: dict, op: str):
        return [[1]], [self.get_param("multiplier", 1)]


class BatchedModel(Model):
    def __init__(self, model_path: str, return_as_dict: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.model = None
        self.return_as_dict = return_as_dict

    def load(self) -> None:
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, body, **kwargs):
        invocation_body = body.get("input")
        if isinstance(invocation_body, dict):
            # example of single invocation
            x = pd.DataFrame([invocation_body])
        elif isinstance(invocation_body, list):
            x = pd.DataFrame(invocation_body)
        else:
            x = invocation_body
        predictions = self.model.predict(x).tolist()
        prediction_results = [round(v, 6) for v in predictions]
        return (
            {"output": {"results": prediction_results}}
            if self.return_as_dict
            else prediction_results
        )

    @staticmethod
    def format_batch(body: typing.Any):
        batched_body = {"input": []}
        for item in body:
            if isinstance(item, dict):
                batched_body["input"].append(item.get("input", item))
            elif isinstance(item, list):
                # for example: [[1,2],[3,4]]
                row = {"x1": item[0], "x2": item[1]}
                batched_body["input"].append(row)
        return batched_body


class StringBatchedModel(Model):
    def __init__(self, suffix: str, return_as_dict=False, **kwargs):
        super().__init__(**kwargs)
        self.suffix = suffix
        self.return_as_dict = return_as_dict

    def load(self) -> None:
        # No loading needed for this simple model
        pass

    def predict(self, body, **kwargs):
        invocation_body = body.get("input")

        # Handle different input formats
        if isinstance(invocation_body, list):
            # List of strings or list of dicts
            if invocation_body and isinstance(invocation_body[0], dict):
                # List of dicts - extract "text" field from each
                prediction_results = [
                    item["text"] + self.suffix for item in invocation_body
                ]
            else:
                # List of strings
                prediction_results = [item + self.suffix for item in invocation_body]
        elif isinstance(invocation_body, dict):
            # Single dict invocation
            prediction_results = [invocation_body.get("text", "") + self.suffix]
        elif isinstance(invocation_body, str):
            # Single string
            prediction_results = [invocation_body + self.suffix]
        else:
            raise ValueError(f"Unsupported input type: {type(invocation_body)}")

        return (
            {"output": {"results": prediction_results}}
            if self.return_as_dict
            else prediction_results
        )

    @staticmethod
    def format_batch(body: typing.Any):
        # Reformats the batched list into the expected {"input": [...]} structure
        batched_body = {"input": []}
        for item in body:
            if isinstance(item, dict):
                batched_body["input"].append(item.get("input", item))
            elif isinstance(item, list):
                for sub_item in item:
                    batched_body["input"].append(sub_item)
            else:
                batched_body["input"].append(item)
        return batched_body


class BatchedGraphModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self) -> None:
        # No loading needed for this simple model
        pass

    def predict(self, body, **kwargs):
        # Handle list of dicts with "input" key
        # Input: [{"input": [1,2,3]}, {"input": [10,20,30]}, {"input": [100,200,300]}]
        # Output: [{"input": [1,2,3], "output": 6}, {"input": [10,20,30], "output": 60}, ...]
        if isinstance(body, list):
            for item in body:
                input_data = item["input"]
                # Simple sum as output (you can change this logic)
                if isinstance(input_data, list):
                    output_value = sum(input_data)
                else:
                    output_value = input_data
                item["output"] = output_value
        return body


class BatchedGraphModel2(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self) -> None:
        # No loading needed for this simple model
        pass

    def predict(self, body, **kwargs):
        # Handle list of dicts with "input" key
        # Input: [{"input": [1,2,3]}, {"input": [10,20,30]}, {"input": [100,200,300]}]
        # Output: [{"input": [1,2,3], "output": 6}, {"input": [10,20,30], "output": 60}, ...]
        if isinstance(body, list):
            for item in body:
                input_data = item["input"]
                # Simple sum as output (you can change this logic)
                if isinstance(input_data, list):
                    output_value = sum(input_data) + 1
                else:
                    output_value = input_data
                item["output_2"] = output_value
        return body


def test_tracking(rundb_mock):
    # test that predict() was tracked properly in the stream
    fn = mlrun.new_function("tests", kind="serving")
    fn.add_model(
        "my",
        ".",
        class_name=ModelTestingClass(multiplier=2, model_endpoint_uid="my-uid"),
    )
    fn.set_tracking("v3io://fake", stream_args={"mock": True, "access_key": "x"})

    server = fn.to_mock_server()
    server.test("/v2/models/my/infer", testdata)

    fake_stream = server.context.stream.output_stream._mock_queue
    assert len(fake_stream) == 1
    assert rec_to_data(fake_stream[0]) == ("my", "ModelTestingClass", [[5, 6]], [10])


def test_custom_tracking(rundb_mock):
    # test custom values tracking (using the logged_results() hook)
    fn = mlrun.new_function("tests", kind="serving")
    fn.add_model(
        "my",
        ".",
        class_name=ModelTestingCustomTrack(multiplier=2, model_endpoint_uid="my-uid"),
    )
    fn.set_tracking("v3io://fake", stream_args={"mock": True, "access_key": "x"})

    server = fn.to_mock_server()
    server.test("/v2/models/my/infer", testdata)

    fake_stream = server.context.stream.output_stream._mock_queue
    assert len(fake_stream) == 1
    assert rec_to_data(fake_stream[0]) == ("my", "ModelTestingCustomTrack", [[1]], [2])


def test_ensemble_tracking(rundb_mock):
    # test proper tracking of an ensemble (router + models are logged)
    fn = mlrun.new_function("tests", kind="serving")
    fn.set_topology(
        "router",
        mlrun.serving.VotingEnsemble(
            vote_type="regression", model_endpoint_uid="VotingEnsemble-uid"
        ),
    )
    fn.add_model(
        "1",
        ".",
        class_name=ModelTestingClass(multiplier=2, model_endpoint_uid="my-uid-1"),
    )
    fn.add_model(
        "2",
        ".",
        class_name=ModelTestingClass(multiplier=3, model_endpoint_uid="my-uid-2"),
    )
    fn.set_tracking("v3io://fake", stream_args={"mock": True, "access_key": "x"})

    server = fn.to_mock_server()
    resp = server.test("/v2/models/infer", testdata)

    fake_stream = server.context.stream.output_stream._mock_queue
    assert len(fake_stream) == 3
    print(resp)
    results = {}
    for rec in fake_stream:
        model, cls, inputs, outputs = rec_to_data(rec)
        results[model] = [cls, inputs, outputs]

    assert results == {
        "1": ["ModelTestingClass", [[5, 6]], [10]],
        "2": ["ModelTestingClass", [[5, 6]], [15]],
        "VotingEnsemble": ["VotingEnsemble", [[5, 6]], [12.5]],
    }


@pytest.mark.parametrize("enable_tracking", [True, False])
def test_tracked_function(rundb_mock, enable_tracking):
    project = mlrun.new_project("test-pro", save=False)
    fn = mlrun.new_function("test-fn", kind="serving", project=project.name)
    model_uri = _log_model(project)
    fn.add_model(
        "m1",
        model_uri,
        "ModelTestingClass",
        multiplier=5,
        model_endpoint_uid="my-uid",
        model_endpoint_creation_strategy=ModelEndpointCreationStrategy.ARCHIVE,
    )
    fn.set_tracking("dummy://", enable_tracking=enable_tracking)
    server = fn.to_mock_server()
    server.test("/v2/models/m1/infer", testdata)
    dummy_stream = server.context.stream.output_stream
    if enable_tracking:
        assert len(dummy_stream.event_list) == 1, "expected stream to get one message"
    else:
        assert len(dummy_stream.event_list) == 0, "expected stream to be empty"


@pytest.mark.parametrize("track_before_creating_child", [True, False])
@pytest.mark.parametrize("enable_tracking", [True, False])
@pytest.mark.parametrize("topology", ["flow", "router"])
def test_child_function_tracking(
    rundb_mock, track_before_creating_child, enable_tracking, topology
):
    project = mlrun.new_project("test-child", save=False)
    fn = mlrun.new_function("test-fn", kind="serving", project=project.name)
    if topology == "flow":
        graph = fn.set_topology("flow")
        graph.to(class_name=RouterStep())
    fn.add_model(
        "model1",
        ".",
        class_name=ModelTestingClass(multiplier=7, model_endpoint_uid="model1-uid"),
    )
    if track_before_creating_child:
        fn.set_tracking("dummy://", enable_tracking=enable_tracking)
        child = fn.add_child_function(
            "child", f"{assets_path}/child_function.py", r"mlrun\mlrun"
        )
        child.set_topology(topology)
    else:
        child = fn.add_child_function(
            "child", f"{assets_path}/child_function.py", r"mlrun\mlrun"
        )
        child.set_topology(topology)
        fn.set_tracking("dummy://", enable_tracking=enable_tracking)
    server = fn.to_mock_server()
    for name, ref in fn.spec.function_refs.items():
        assert ref._function.spec.track_models == enable_tracking, (
            f"{name} wrong track models value for child function expected to be "
            f"equal to {enable_tracking}"
        )
        if topology == "flow":
            server.wait_for_completion()
            assert ref._function.spec.graph.track_models == enable_tracking, (
                f"{name} wrong track models value for child function RootFlowStep expected to be "
                f"equal to {enable_tracking}"
            )


def test_child_function_tracking_with_model_runner(rundb_mock):
    project = mlrun.new_project("test-child", save=False)
    fn = mlrun.new_function("test-fn", kind="serving", project=project.name)
    graph = fn.set_topology("flow")
    model_runner_step = ModelRunnerStep(name="my_model_runner_0", raise_exception=True)
    model_runner_step.add_model(
        model_class="MyModel",
        execution_mechanism="naive",
        endpoint_name="my_model_0",
        input_path="n",
        result_path="n",
        raise_error=False,
        inc=1,
    )
    graph.to(">>", name="in", path="dummy://in").to(
        model_runner_step, function="c1"
    ).to(">>", name="out", path="dummy://out")
    fn.set_tracking()
    fn.add_child_function("c1", f"{assets_path}/child_function.py", "mlrun/mlrun")
    server = fn.to_mock_server()
    server.test("/", {"n": 1})
    server.wait_for_completion()

    assert server.graph.steps["my_model_runner_0_error_raise"].function == "c1"
    assert server.graph.steps["my_model_runner_0"].function == "c1"

    dummy_stream = server.context.stream.output_stream
    assert len(dummy_stream.event_list) == 1, "expected stream to get one message"
    assert dummy_stream.event_list[0].get("resp", {}).get("outputs") == [2]
    assert dummy_stream.event_list[0].get("request", {}).get("inputs") == [1]

    output_stream = server.graph.steps["out"].async_object
    assert len(output_stream.event_list) == 1


def rec_to_data(rec):
    data = json.loads(rec["data"])
    inputs = data["request"]["inputs"]
    outputs = data["resp"]["outputs"]
    return data["model"], data["class"], inputs, outputs


@pytest.fixture
def project() -> mlrun.MlrunProject:
    return mlrun.get_or_create_project("test-tracking", allow_cross_project=True)


@pytest.fixture
def serving_output_stream(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Iterator[Union[type[OutputStream], type[KafkaOutputStream]]]:
    """Register the serving stream"""
    stream_profile_name = "special-stream"
    monkeypatch.setenv(
        mm_constants.ProjectSecretKeys.STREAM_PROFILE_NAME, stream_profile_name
    )

    if request.param == "v3io":
        profile = DatastoreProfileV3io(
            name=stream_profile_name, v3io_access_key="v3io-key"
        )
        expected_stream_type = OutputStream
    elif request.param == "kafka":
        profile = DatastoreProfileKafkaStream(
            name=stream_profile_name,
            brokers=["localhost"],
            topics=[],
            kwargs_public={"api_version": (3, 9)},
        )
        expected_stream_type = KafkaOutputStream
    else:
        raise ValueError(f"Unsupported stream type {request.param}")

    register_temporary_client_datastore_profile(profile)
    yield expected_stream_type
    remove_temporary_client_datastore_profile(stream_profile_name)


@pytest.mark.usefixtures("rundb_mock")
@pytest.mark.parametrize("serving_output_stream", ["v3io", "kafka"], indirect=True)
def test_tracking_datastore_profile(
    project: mlrun.MlrunProject,
    serving_output_stream: Union[type[OutputStream], type[KafkaOutputStream]],
) -> None:
    fn = cast(
        ServingRuntime,
        project.set_function(
            name="test-tracking-from-profile", kind=ServingRuntime.kind
        ),
    )
    fn.add_model(
        "model1",
        ".",
        class_name=ModelTestingClass(multiplier=7, model_endpoint_uid="model1-uid"),
    )
    fn.set_tracking(stream_args={"mock": True})

    server = fn.to_mock_server()
    server.test("/v2/models/model1/predict", body=json.dumps({"inputs": [[-5.2, 0.6]]}))
    server.test(
        "/v2/models/model1/predict", body=json.dumps({"inputs": [[0, -0.1], [0.4, 0]]})
    )

    output_stream = server.context.stream.output_stream
    assert isinstance(
        output_stream, serving_output_stream
    ), f"The output stream is of unexpected type {type(output_stream)}"
    mocked_stream = output_stream._mock_queue
    assert len(mocked_stream) == 2

    if isinstance(output_stream, KafkaOutputStream):
        event = mocked_stream[1]
    else:
        # V3IO OutputStream
        event = json.loads(mocked_stream[1]["data"])

    assert event["class"] == "ModelTestingClass"
    assert event["model"] == "model1"
    assert event["effective_sample_count"] == 2
    assert np.array_equal(event["request"]["inputs"], np.array([[0, -0.1], [0.4, 0]]))
    assert np.array_equal(event["resp"]["outputs"], np.array([0.0, 0.4 * 7]))


class MyModelSelector(ModelSelector):
    def select(
        self, event, available_models: list[Model]
    ) -> Union[list[str], list[Model]]:
        return ["my_dict_model"]


class MyModel(Model):
    def __init__(self, *args, inc: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.inc = inc

    def predict(self, body, **kwargs):
        body["n"] += self.inc
        body.pop("models", None)
        return body

    async def predict_async(self, body, **kwargs):
        return self.predict(body, **kwargs)


def handle_error(event):
    return event


class DictOutputModel(Model):
    def predict(self, body, **kwargs):
        body["outputs"] = {}
        for key, value in body["inputs"][self.name].items():
            if not isinstance(value, list) and not isinstance(value, str):
                body["outputs"][key.replace("f", "o")] = value + 1
            elif not isinstance(value, list) and isinstance(value, str):
                body["outputs"][key.replace("f", "o")] = value + "_output"
            elif isinstance(value, list):
                out_value = []
                for v in value:
                    if isinstance(v, int):
                        out_value.append(v + 1)
                    elif isinstance(v, str):
                        out_value.append(v + "_output")
                body["outputs"][key.replace("f", "o")] = out_value
        return body

    async def predict_async(self, body, **kwargs):
        return self.predict(body, **kwargs)


class StrDictOutputModel(Model):
    def predict(self, body, **kwargs):
        body["outputs"] = {}
        for key, value in body["inputs"][self.name].items():
            body["outputs"][key.replace("f", "o")] = (
                value + "_output"
                if not isinstance(value, list)
                else [v + "_output" for v in value]
            )
        return body


class SubDictOutputModel(Model):
    def predict(self, body, **kwargs):
        body["outputs"] = {}
        for key, value in body["inputs"][self.name].items():
            if isinstance(value, list):
                body["outputs"][key.replace("f", "o")] = []
                for single_value in value:
                    body["outputs"][key.replace("f", "o")].append(
                        {f"key_{single_value}": f"value_{single_value}"}
                    )
            else:
                body["outputs"][key.replace("f", "o")] = {
                    f"key_{value}": f"value_{value}"
                }
        return body


def _test_monitoring_system_steps_structure(
    server_graph: RootFlowStep,
    spec_graph: RootFlowStep,
    model_runners_names: list[str],
    streaming_enabled: bool = False,
):
    # When streaming is enabled, Collector steps are inserted between MRS and MM pipeline
    if streaming_enabled:
        source_names = [f"{name}_collector" for name in model_runners_names]
    else:
        source_names = model_runners_names
    system_steps = {
        "background_task_status_step": source_names,
        "filter_none": ["background_task_status_step"],
        "monitoring_pre_processor_step": ["filter_none"],
        "flatten_events": ["monitoring_pre_processor_step"],
        "sampling_step": ["flatten_events"],
        "filter_none_sampling": ["sampling_step"],
        "model_monitoring_stream": [
            "filter_none_sampling"
        ],  # mock creates a dummy pusher and not target
    }
    for step in server_graph.steps.values():
        if step.name in system_steps:
            assert step.after == system_steps[step.name]
    for step in system_steps.keys():
        assert (
            step not in spec_graph.steps.keys()
        ), f"spec graph should not contain system step {step}"


def _test_graph_structure(
    server_graph: RootFlowStep, spec_graph: RootFlowStep, tracked: bool
):
    """Expects server graph contains system steps and function graph does not contain system steps."""
    model_runners = []
    for step in server_graph.steps.values():
        if isinstance(step, ModelRunnerStep):
            model_runners.append(step.name)
        elif model_runners and step.name == f"{model_runners[-1]}_error_raise":
            assert model_runners[-1] in step.after or model_runners[-1] in step.after
    for model_runner in model_runners:
        assert (
            f"{model_runner}_error_raise" not in spec_graph.steps.keys()
        ), "spec graph should not contain error raise steps"
    if tracked:
        _test_monitoring_system_steps_structure(server_graph, spec_graph, model_runners)


@pytest.mark.parametrize("enable_tracking", [True, False])
def test_tracked_model_runner(rundb_mock, enable_tracking: bool):
    function = mlrun.new_function("tests-1", kind="serving")
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(name="my_model_runner")
    model_runner_step.add_model(
        model_class="MyModel",
        execution_mechanism="naive",
        endpoint_name="my_model",
        input_path="n",
        result_path="n",
        raise_error=False,
        inc=1,
    )
    graph.to(model_runner_step).respond()
    function.set_tracking("dummy://", enable_tracking=enable_tracking)
    server = function.to_mock_server()
    server.test("/", {"n": 1})
    server.wait_for_completion()

    dummy_stream = server.context.stream.output_stream
    if enable_tracking:
        assert len(dummy_stream.event_list) == 1, "expected stream to get one message"
        assert dummy_stream.event_list[0].get("resp", {}).get("outputs") == [2]
        assert dummy_stream.event_list[0].get("request", {}).get("inputs") == [1]
    else:
        assert len(dummy_stream.event_list) == 0, "expected stream to be empty"

    _test_graph_structure(server.graph, function.spec.graph, enable_tracking)


@pytest.mark.parametrize("enable_tracking", [True, False])
def test_tracked_model_runner_with_tools(rundb_mock, enable_tracking: bool):
    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="async", allow_cyclic=True)
    model_runner_step = ModelRunnerStep(
        name="my_model_runner", model_runner_selector="MySelector"
    )
    model_runner_step.add_model(
        model_class="LLModelWithTools",
        execution_mechanism="naive",
        endpoint_name="llm_with_tools",
        input_path="counter",
        result_path="counter",
    )
    runner = graph.to(model_runner_step)
    runner.to(name="tool_a", class_name="Tool", cycle_to="my_model_runner")
    runner.to(name="tool_b", class_name="Tool", cycle_to="my_model_runner")
    runner.to(name="end", class_name="Echo").respond()
    function.set_tracking("dummy://", enable_tracking=enable_tracking)
    server = function.to_mock_server()
    server.test("/", {"counter": 0})
    server.wait_for_completion()

    dummy_stream = server.context.stream.output_stream
    if enable_tracking:
        assert len(dummy_stream.event_list) == 5, "expected stream to get 5 messages"
        assert dummy_stream.event_list[0].get("request", {}).get("inputs") == [0]
        assert dummy_stream.event_list[0].get("resp", {}).get("outputs") == [1]
        assert dummy_stream.event_list[4].get("request", {}).get("inputs") == [4]
        assert dummy_stream.event_list[4].get("resp", {}).get("outputs") == [5]
    else:
        assert len(dummy_stream.event_list) == 0, "expected stream to be empty"

    _test_graph_structure(server.graph, function.spec.graph, enable_tracking)


@pytest.mark.parametrize("with_schema", [True, False])
def test_tracked_model_runner_dict(rundb_mock, with_schema):
    function = mlrun.new_function("tests-1", kind="serving")
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(name="my_model_runner", raise_exception=True)
    model_runner_step.add_model(
        model_class="DictOutputModel",
        execution_mechanism="naive",
        endpoint_name="dict_model",
        input_path="inputs.dict_model",
        result_path="outputs",
        inputs=["f1", "f2", "f3", "f4"] if with_schema else None,
        outputs=["o1", "o2", "o3", "o4"] if with_schema else None,
        raise_error=False,
    )
    model_runner_step.add_model(
        model_class="DictOutputModel",
        execution_mechanism="naive",
        endpoint_name="dict_model_2",
        input_path="inputs.dict_model_2",
        result_path="outputs",
        inputs=["f1"] if with_schema else None,
        outputs=["o1"] if with_schema else None,
        raise_error=False,
    )
    model_runner_step.add_model(
        model_class="DictOutputModel",
        execution_mechanism="naive",
        endpoint_name="dict_model_single_event",
        input_path="inputs.dict_model_single_event",
        result_path="outputs",
        raise_error=False,
    )
    model_runner_step.add_model(
        model_class="DictOutputModel",
        execution_mechanism="naive",
        endpoint_name="dict_model_single_event_wrapped",
        input_path="inputs.dict_model_single_event_wrapped",
        result_path="outputs",
        inputs=["f1", "f2", "f3", "f4"] if with_schema else None,
        outputs=["o1", "o2", "o3", "o4"] if with_schema else None,
        raise_error=False,
    )
    model_runner_step.add_model(
        model_class="DictOutputModel",
        execution_mechanism="naive",
        endpoint_name="dict_model_scalar",
        input_path="inputs.dict_model_scalar",
        result_path="outputs",
        inputs=["f1"] if with_schema else None,
        outputs=["o1"] if with_schema else None,
        raise_error=False,
    )
    graph.to(model_runner_step).respond()

    function.set_tracking()
    server = function.to_mock_server()
    inputs_model = (
        {"f1": [1, 2], "f2": ["hi", "bye"], "f3": [3, 4], "f4": [4, 5]}
        if not with_schema
        else {"f4": [4, 5], "f2": ["hi", "bye"], "f1": [1, 2], "f3": [3, 4]}
    )
    server.test(
        "/",
        {
            "inputs": {
                "dict_model": inputs_model,
                "dict_model_2": {"f1": [1, 2]},
                "dict_model_single_event": {"f1": 1, "f2": "hi", "f3": 3, "f4": 4},
                "dict_model_single_event_wrapped": {
                    "f1": [1],
                    "f2": ["hi"],
                    "f3": [3],
                    "f4": [4],
                },
                "dict_model_scalar": {"f1": 1},
            }
        },
    )
    server.wait_for_completion()

    dummy_stream = server.context.stream.output_stream
    assert len(dummy_stream.event_list) == 5, "expected stream to get one message"
    assert dummy_stream.event_list[0].get("request", {}).get("inputs") == [
        [1, "hi", 3, 4],
        [2, "bye", 4, 5],
    ]
    assert dummy_stream.event_list[0].get("resp", {}).get("outputs") == [
        [2, "hi_output", 4, 5],
        [3, "bye_output", 5, 6],
    ]
    assert dummy_stream.event_list[1].get("request", {}).get("inputs") == [1, 2]
    assert dummy_stream.event_list[1].get("resp", {}).get("outputs") == [2, 3]
    assert dummy_stream.event_list[2].get("request", {}).get("inputs") == [
        [1, "hi", 3, 4]
    ]
    assert dummy_stream.event_list[2].get("resp", {}).get("outputs") == [
        [2, "hi_output", 4, 5]
    ]
    assert dummy_stream.event_list[3].get("request", {}).get("inputs") == [
        [1, "hi", 3, 4]
    ]
    assert dummy_stream.event_list[3].get("resp", {}).get("outputs") == [
        [2, "hi_output", 4, 5]
    ]
    assert dummy_stream.event_list[4].get("request", {}).get("inputs") == [1]
    assert dummy_stream.event_list[4].get("resp", {}).get("outputs") == [2]


@pytest.mark.parametrize("with_schema", [True, False])
def test_tracked_model_runner_str_dict(rundb_mock, with_schema):
    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(name="my_model_runner", raise_exception=True)
    model_runner_step.add_model(
        model_class="StrDictOutputModel",
        execution_mechanism="naive",
        endpoint_name="dict_model",
        input_path="inputs.dict_model",
        result_path="outputs",
        inputs=["f1", "f2", "f3", "f4"] if with_schema else None,
        outputs=["o1", "o2", "o3", "o4"] if with_schema else None,
        raise_error=False,
    )
    model_runner_step.add_model(
        model_class="StrDictOutputModel",
        execution_mechanism="naive",
        endpoint_name="dict_model_2",
        input_path="inputs.dict_model_2",
        result_path="outputs",
        inputs=["f1"] if with_schema else None,
        outputs=["o1"] if with_schema else None,
        raise_error=False,
    )
    model_runner_step.add_model(
        model_class="StrDictOutputModel",
        execution_mechanism="naive",
        endpoint_name="dict_model_single_event",
        input_path="inputs.dict_model_single_event",
        result_path="outputs",
        raise_error=False,
    )
    model_runner_step.add_model(
        model_class="StrDictOutputModel",
        execution_mechanism="naive",
        endpoint_name="dict_model_single_event_wrapped",
        input_path="inputs.dict_model_single_event_wrapped",
        result_path="outputs",
        inputs=["f1", "f2", "f3", "f4"] if with_schema else None,
        outputs=["o1", "o2", "o3", "o4"] if with_schema else None,
        raise_error=False,
    )
    model_runner_step.add_model(
        model_class="StrDictOutputModel",
        execution_mechanism="naive",
        endpoint_name="dict_model_scalar",
        input_path="inputs.dict_model_scalar",
        result_path="outputs",
        inputs=["f1"] if with_schema else None,
        outputs=["o1"] if with_schema else None,
        raise_error=False,
    )
    graph.to(model_runner_step).respond()

    function.set_tracking()
    server = function.to_mock_server()
    inputs_model = (
        {"f1": ["1", "2"], "f2": ["2", "3"], "f3": ["3", "4"], "f4": ["4", "5"]}
        if not with_schema
        else {"f4": ["4", "5"], "f2": ["2", "3"], "f1": ["1", "2"], "f3": ["3", "4"]}
    )
    server.test(
        "/",
        {
            "inputs": {
                "dict_model": inputs_model,
                "dict_model_2": {"f1": ["1", "2"]},
                "dict_model_single_event": {"f1": "1", "f2": "2", "f3": "3", "f4": "4"},
                "dict_model_single_event_wrapped": {
                    "f1": ["1"],
                    "f2": ["2"],
                    "f3": ["3"],
                    "f4": ["4"],
                },
                "dict_model_scalar": {"f1": "1"},
            }
        },
    )
    server.wait_for_completion()

    dummy_stream = server.context.stream.output_stream
    assert dummy_stream.event_list[0].get("request", {}).get("inputs") == [
        ["1", "2", "3", "4"],
        ["2", "3", "4", "5"],
    ]
    assert len(dummy_stream.event_list) == 5, "expected stream to get one message"
    assert dummy_stream.event_list[0].get("resp", {}).get("outputs") == [
        ["1_output", "2_output", "3_output", "4_output"],
        ["2_output", "3_output", "4_output", "5_output"],
    ]
    assert dummy_stream.event_list[0].get("request", {}).get("input_schema") == [
        "f1",
        "f2",
        "f3",
        "f4",
    ]
    assert dummy_stream.event_list[0].get("resp", {}).get("output_schema") == [
        "o1",
        "o2",
        "o3",
        "o4",
    ]

    assert dummy_stream.event_list[1].get("request", {}).get("inputs") == ["1", "2"]
    assert dummy_stream.event_list[1].get("resp", {}).get("outputs") == [
        "1_output",
        "2_output",
    ]
    assert dummy_stream.event_list[2].get("request", {}).get("inputs") == [
        ["1", "2", "3", "4"]
    ]
    assert dummy_stream.event_list[2].get("resp", {}).get("outputs") == [
        ["1_output", "2_output", "3_output", "4_output"]
    ]
    assert dummy_stream.event_list[3].get("request", {}).get("inputs") == [
        ["1", "2", "3", "4"]
    ]
    assert dummy_stream.event_list[3].get("resp", {}).get("outputs") == [
        ["1_output", "2_output", "3_output", "4_output"]
    ]
    assert dummy_stream.event_list[4].get("request", {}).get("inputs") == ["1"]
    assert dummy_stream.event_list[4].get("resp", {}).get("outputs") == ["1_output"]
    assert dummy_stream.event_list[4].get("request", {}).get("input_schema") == ["f1"]
    assert dummy_stream.event_list[4].get("resp", {}).get("output_schema") == ["o1"]


@pytest.mark.parametrize("with_schema", [True, False])
def test_tracked_subdict(rundb_mock, with_schema):
    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(name="my_model_runner", raise_exception=True)
    model_runner_step.add_model(
        model_class="SubDictOutputModel",
        execution_mechanism="naive",
        endpoint_name="dict_model",
        input_path="inputs.dict_model",
        result_path="outputs",
        inputs=["f1", "f2", "f3", "f4"] if with_schema else None,
        outputs=["o1", "o2", "o3", "o4"] if with_schema else None,
        raise_error=False,
    )
    graph.to(model_runner_step).respond()
    function.set_tracking()
    server = function.to_mock_server()
    inputs_model = (
        {"f1": ["1", "2"], "f2": ["2", "3"], "f3": ["3", "4"], "f4": ["4", "5"]}
        if not with_schema
        else {"f4": ["4", "5"], "f2": ["2", "3"], "f1": ["1", "2"], "f3": ["3", "4"]}
    )
    server.test(
        "/",
        {
            "inputs": {
                "dict_model": inputs_model,
                "dict_model_2": {"f1": ["1", "2"]},
                "dict_model_single_event": {"f1": "1", "f2": "2", "f3": "3", "f4": "4"},
                "dict_model_single_event_wrapped": {
                    "f1": ["1"],
                    "f2": ["2"],
                    "f3": ["3"],
                    "f4": ["4"],
                },
                "dict_model_scalar": {"f1": "1"},
            }
        },
    )
    server.wait_for_completion()
    dummy_stream = server.context.stream.output_stream
    assert dummy_stream.event_list[0].get("request", {}).get("inputs") == [
        ["1", "2", "3", "4"],
        ["2", "3", "4", "5"],
    ]
    assert len(dummy_stream.event_list) == 1, "expected stream to get one message"
    assert dummy_stream.event_list[0].get("resp", {}).get("outputs")[0] == [
        {"key_1": "value_1"},
        {"key_2": "value_2"},
        {"key_3": "value_3"},
        {"key_4": "value_4"},
    ]
    assert dummy_stream.event_list[0].get("resp", {}).get("outputs")[1] == [
        {"key_2": "value_2"},
        {"key_3": "value_3"},
        {"key_4": "value_4"},
        {"key_5": "value_5"},
    ]


def test_tracked_model_runner_multiple_steps(rundb_mock):
    function = mlrun.new_function("tests-1", kind="serving")
    graph = function.set_topology("flow", engine="async")
    model_runner_step_0 = ModelRunnerStep(
        name="my_model_runner_0", raise_exception=True
    )
    model_runner_step_1 = ModelRunnerStep(
        name="my_model_runner_1", raise_exception=True
    )
    model_runner_step_0.add_model(
        model_class="MyModel",
        execution_mechanism="naive",
        endpoint_name="my_model_0",
        input_path="n",
        result_path="n",
        raise_error=False,
        inc=1,
    )
    model_runner_step_1.add_model(
        model_class="MyModel",
        execution_mechanism="naive",
        endpoint_name="my_model_1",
        input_path="n",
        result_path="n",
        raise_error=False,
        inc=2,
    )
    graph.to(model_runner_step_0).respond()
    graph.to(model_runner_step_1)

    function.set_tracking()
    server = function.to_mock_server()
    server.test("/", {"n": 1})
    server.wait_for_completion()

    dummy_stream = server.context.stream.output_stream

    assert len(dummy_stream.event_list) == 2, "expected stream to get two messages"
    assert dummy_stream.event_list[0].get("resp", {}).get("outputs") == [2]
    assert dummy_stream.event_list[0].get("request", {}).get("inputs") == [1]
    assert dummy_stream.event_list[1].get("resp", {}).get("outputs") == [3]
    assert dummy_stream.event_list[1].get("request", {}).get("inputs") == [1]


def test_tracked_model_runner_multiple_models(rundb_mock):
    function = mlrun.new_function("tests-1", kind="serving")
    graph = function.set_topology("flow", engine="async")
    model_runner_step_0 = ModelRunnerStep(
        name="my_model_runner_0", raise_exception=True
    )
    model_runner_step_1 = ModelRunnerStep(
        name="my_model_runner_1", raise_exception=True
    )
    models = []
    for i in range(4):
        model_name_0 = f"runner_0_my_model_{i}"
        model_name_1 = f"runner_1_my_model_{i}"
        model_runner_step_0.add_model(
            model_class="MyModel",
            execution_mechanism="naive",
            endpoint_name=model_name_0,
            input_path="n",
            result_path="n",
            raise_error=False,
            inc=1,
        )
        model_runner_step_1.add_model(
            model_class="MyModel",
            execution_mechanism="naive",
            endpoint_name=model_name_1,
            input_path="n",
            result_path="n",
            raise_error=False,
            inc=2,
        )
        models.extend([model_name_0, model_name_1])

    graph.to(model_runner_step_0).respond()
    graph.to(model_runner_step_1)
    function.set_tracking()
    server = function.to_mock_server()
    server.test("/", {"n": 1})
    server.wait_for_completion()

    dummy_stream = server.context.stream.output_stream

    assert len(dummy_stream.event_list) == 8, "expected stream to get eight messages"
    output_models = [event["model"] for event in dummy_stream.event_list]
    models.sort()
    output_models.sort()
    assert output_models == models, "expected models to be the same"
    _test_graph_structure(server.graph, function.spec.graph, True)


def test_set_untracked_with_model_runner(rundb_mock):
    function = mlrun.new_function("tests-1", kind="serving")
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(name="my_model_runner", raise_exception=True)
    model_runner_step.add_model(
        model_class="MyModel",
        execution_mechanism="naive",
        endpoint_name="test_model",
        input_path="n",
        result_path="n",
        raise_error=False,
        inc=1,
    )
    graph.to(model_runner_step).respond()
    function.set_tracking()

    server = function.to_mock_server()
    server.test("/", {"n": 1})
    server.wait_for_completion()

    dummy_stream = server.context.stream.output_stream
    _test_graph_structure(server.graph, function.spec.graph, True)
    assert len(dummy_stream.event_list) == 1, "expected stream to get one message"
    function.set_tracking("dummy://", enable_tracking=False)
    _test_graph_structure(graph, function.spec.graph, False)
    server = function.to_mock_server()
    server.test("/", {"n": 1})
    server.wait_for_completion()
    assert (
        len(dummy_stream.event_list) == 1
    ), "expected stream to still have single message"


def test_tracked_multiple_to_mock_with_model_runner(rundb_mock):
    function = mlrun.new_function("tests-1", kind="serving")
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(
        name="my_model_runner",
        raise_exception=True,
        model_selector="MyModelSelector",
    )
    model_runner_step.add_model(
        model_class="DictOutputModel",
        execution_mechanism="naive",
        endpoint_name="my_dict_model",
        input_path="inputs.my_dict_model",
        result_path="outputs",
        outputs=["o1", "o2", "o3", "o4"],
        raise_error=False,
    )
    graph.to(model_runner_step).respond()

    function.set_tracking()
    server = function.to_mock_server()
    server.wait_for_completion()
    model_runner_step_1 = ModelRunnerStep(
        name="my_model_runner_1", raise_exception=True
    )
    model_runner_step_1.add_model(
        model_class="DictOutputModel",
        execution_mechanism="naive",
        endpoint_name="my_dict_model_1",
        input_path="inputs.my_dict_model_1",
        result_path="outputs",
        outputs=["o1", "o2", "o3", "o4"],
        raise_error=False,
    )
    graph.to(model_runner_step_1)
    server = function.to_mock_server()
    server.test(
        "/",
        {
            "inputs": {
                "my_dict_model_1": {"f1": 1, "f2": 2, "f3": 3, "f4": 4},
                "my_dict_model": {"f1": 1, "f2": 2, "f3": 3, "f4": 4},
            }
        },
    )
    server.wait_for_completion()
    dummy_stream = server.context.stream.output_stream
    assert len(dummy_stream.event_list) == 2, "expected stream to get one message"


@pytest.mark.parametrize("sampling_percentage", [100.0, 50.0, 20.0])
def test_sampling_model_runner(rundb_mock, sampling_percentage: float):
    function = mlrun.new_function("tests-sampling", kind="serving")
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(name="my_model_runner", raise_exception=True)
    model_runner_step.add_model(
        model_class="DictOutputModel",
        execution_mechanism="naive",
        endpoint_name="dict_model_1",
        input_path="inputs.dict_model_1",
        result_path="outputs",
        outputs=["o1", "o2", "o3", "o4"],
        raise_error=False,
    )
    graph.to(model_runner_step).respond()

    function.set_tracking(
        "dummy://", enable_tracking=True, sampling_percentage=sampling_percentage
    )
    server = function.to_mock_server()
    server.test(
        "/",
        {
            "inputs": {
                "dict_model_1": {
                    "f1": [1, 4, 8, 12] * 1000,
                    "f2": [2, 5, 9, 13] * 1000,
                    "f3": [3, 6, 10, 14] * 1000,
                    "f4": [4, 7, 11, 15] * 1000,
                }
            }
        },
    )
    server.wait_for_completion()

    dummy_stream = server.context.stream.output_stream

    _test_graph_structure(server.graph, function.spec.graph, True)

    if sampling_percentage == 100.0:
        assert len(dummy_stream.event_list) == 1, "expected stream to get one message"
        assert len(dummy_stream.event_list[0]["resp"]["outputs"]) == 4000
    else:
        if len(dummy_stream.event_list) == 1:
            assert len(dummy_stream.event_list[0]["resp"]["outputs"]) < 4000, (
                f"expected sampling will remove"
                f" some outputs with sampling_percentage"
                f" = {sampling_percentage} "
            )


@pytest.mark.parametrize("enable_tracking", [True, False])
def test_tracked_model_runner_shared(rundb_mock, enable_tracking: bool):
    project = mlrun.new_project("remote-model-project", save=False)
    model_artifact = project.log_model(
        "my_model",
        model_url="http://localhost:8080/v2/models/mymodel/infer",
        default_config={"model_version": "4"},
    )
    function = mlrun.new_function("tests-1", kind="serving")
    graph = function.set_topology("flow", engine="async")
    graph.add_shared_model(
        model_class=MyModel(name="shared-model", raise_exception=False, inc=1),
        name="shared-model",
        execution_mechanism="naive",
        model_artifact=model_artifact,
        input_path="n",
        result_path="n",
    )
    model_runner_step = ModelRunnerStep(name="my_model_runner", raise_exception=True)
    model_runner_step.add_shared_model_proxy(
        endpoint_name="my_model",
        shared_model_name="shared-model",
        model_artifact=model_artifact,
    )
    model_runner_step.add_shared_model_proxy(
        endpoint_name="my_model-2",
        model_artifact=model_artifact,
    )
    graph.to(model_runner_step).respond()

    function.set_tracking("dummy://", enable_tracking=enable_tracking)
    server = function.to_mock_server()
    res = server.test("/", {"n": 1})
    server.wait_for_completion()

    assert "my_model" in res, "expected response to contain model name 'my_model'"
    assert "my_model-2" in res, "expected response to contain model name 'my_model-2'"
    assert (
        "shared-model" not in res
    ), "expected response to not contain model name 'shared_model'"

    dummy_stream = server.context.stream.output_stream
    if enable_tracking:
        assert len(dummy_stream.event_list) == 2, "expected stream to get one message"
        assert dummy_stream.event_list[0].get("resp", {}).get("outputs") == [2]
        assert dummy_stream.event_list[0].get("request", {}).get("inputs") == [1]
        assert dummy_stream.event_list[0].get("model") == "my_model"
        assert dummy_stream.event_list[1].get("resp", {}).get("outputs") == [2]
        assert dummy_stream.event_list[1].get("request", {}).get("inputs") == [1]
        assert dummy_stream.event_list[1].get("model") == "my_model-2"
    else:
        assert len(dummy_stream.event_list) == 0, "expected stream to be empty"

    _test_graph_structure(server.graph, function.spec.graph, enable_tracking)


def test_shared_model_invalid_usage():
    project = mlrun.new_project("remote-model-project", save=False)
    model_artifact = project.log_model(
        "my_model",
        model_url="http://localhost:8080/v2/models/mymodel/infer",
        default_config={"model_version": "4"},
    )
    model_artifact_2 = project.log_model(
        "my_model-2",
        model_url="http://localhost:8080/v2/models/mymodel/infer",
        default_config={"model_version": "4"},
    )
    function = mlrun.new_function("tests-1", kind="serving")
    graph = function.set_topology("flow", engine="async")
    graph.add_shared_model(
        model_class=MyModel(name="shared-model", raise_exception=False, inc=1),
        name="shared-model",
        execution_mechanism="naive",
        model_artifact=model_artifact,
        input_path="n",
        result_path="n",
    )
    model_runner_step = ModelRunnerStep(name="my_model_runner", raise_exception=True)
    model_runner_step.add_shared_model_proxy(
        endpoint_name="my_model",
        shared_model_name="shared-model-2",
        model_artifact=model_artifact,
    )
    with pytest.raises(mlrun.serving.states.GraphError):
        graph.to(model_runner_step).respond()

    model_runner_step.add_shared_model_proxy(
        endpoint_name="my_model-2",
        model_artifact=model_artifact_2,
    )
    with pytest.raises(mlrun.serving.states.GraphError):
        graph.to(model_runner_step).respond()

    model_runner_step_2 = ModelRunnerStep(name="my_model_runner", raise_exception=True)
    model_runner_step_2 = graph.to(model_runner_step_2)
    with pytest.raises(mlrun.serving.states.GraphError):
        model_runner_step_2.add_shared_model_proxy(
            endpoint_name="my_model",
            shared_model_name="shared-model-2",
            model_artifact=model_artifact,
        )

    model_runner_step_2.add_shared_model_proxy(
        endpoint_name="my_model",
        model_artifact=model_artifact,
    )
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        graph.add_shared_model(
            model_class=MyModel(name="shared-model", raise_exception=False, inc=1),
            name="shared-model",
            execution_mechanism="naive",
            model_artifact=model_artifact,
            input_path="n",
            result_path="n",
        )
    graph.add_shared_model(
        model_class=MyModel(name="shared-model", raise_exception=False, inc=1),
        name="shared-model",
        execution_mechanism="naive",
        override=True,
        model_artifact=model_artifact,
        input_path="n",
        result_path="n",
    )


def test_tracked_model_runner_background_task(rundb_mock):
    function = mlrun.new_function("tests-1", kind="serving")
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(name="my_model_runner", raise_exception=True)
    model_runner_step.add_model(
        model_class="MyModel",
        execution_mechanism="naive",
        endpoint_name="my_model",
        input_path="n",
        result_path="n",
        raise_error=False,
        inc=1,
    )
    rundb_mock._get_background_task_calls = 0
    graph.to(model_runner_step).respond()
    function.set_tracking()
    server = function.to_mock_server()
    server.test("/", {"n": 1})
    dummy_stream = server.context.stream.output_stream
    assert len(dummy_stream.event_list) == 0, "expected stream to be empty"
    mlrun.mlconf.model_endpoint_monitoring.model_endpoint_creation_check_period = 1
    sleep(mlrun.mlconf.model_endpoint_monitoring.model_endpoint_creation_check_period)
    server.test("/", {"n": 2})
    server.wait_for_completion()

    assert len(dummy_stream.event_list) == 1, "expected stream to get one message"
    assert dummy_stream.event_list[0].get("resp", {}).get("outputs") == [3]
    assert dummy_stream.event_list[0].get("request", {}).get("inputs") == [2]


@pytest.mark.parametrize("enable_tracking", [True, False])
@pytest.mark.parametrize("raise_exception", [True, False])
@pytest.mark.parametrize("as_responder", [True, False])
@pytest.mark.parametrize("all_graph_handler", [True, False])
def test_tracked_model_runner_with_error_handler(
    rundb_mock,
    enable_tracking: bool,
    raise_exception: bool,
    as_responder: bool,
    all_graph_handler: bool,
):
    function = mlrun.new_function("tests-1", kind="serving")
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(
        name="my_model_runner", raise_exception=raise_exception
    )
    model_runner_step.add_model(
        model_class="MyModel",
        execution_mechanism="naive",
        endpoint_name="my_model",
        input_path="n",
        result_path="n",
        raise_error=False,
        inc=1,
    )
    if as_responder:
        step = graph.to(model_runner_step).respond()
    else:
        step = graph.to(model_runner_step)
    if all_graph_handler:
        graph.error_handler("echo_error", handler="handle_error")
    else:
        step.error_handler("echo_error", handler="handle_error")
    function.set_tracking("dummy://", enable_tracking=enable_tracking)
    server = function.to_mock_server()
    resp = server.test("/", {"n": "1"})
    server.wait_for_completion()

    dummy_stream = server.context.stream.output_stream
    if enable_tracking:
        assert len(dummy_stream.event_list) == 1, "expected stream to get one message"
        assert (
            dummy_stream.event_list[0].get("error")
            == 'TypeError: can only concatenate str (not "int") to str'
        )
        assert dummy_stream.event_list[0].get("request", {}).get("inputs") == ["1"]
    elif not enable_tracking and as_responder:
        assert len(dummy_stream.event_list) == 0, "expected stream to be empty"
        assert resp == {
            "error": 'TypeError: can only concatenate str (not "int") to str'
        }

    _test_graph_structure(server.graph, function.spec.graph, enable_tracking)


def test_transpose_by_key_with_str():
    data = {
        "Price": 30.0,
        "Product": "Keyboard",
        "Stock": 100,
        "extra": 123,
        "time": "2020-01-01T01:00:00Z",
    }
    result, new_schema = MonitoringPreProcessor.transpose_by_key(data)
    expected_result = [[30.0, "Keyboard", 100, 123, "2020-01-01T01:00:00Z"]]

    assert result == expected_result
    assert new_schema == ["Price", "Product", "Stock", "extra", "time"]

    data = {
        "Price": [30.0, 6.0],
        "Product": ["Keyboard", "Mouse"],
        "Stock": [100, 200],
        "extra": [123, 80],
        "time": ["2020-01-01T01:00:00Z", "2020-01-01T02:00:00Z"],
    }
    result, new_schema = MonitoringPreProcessor.transpose_by_key(data)

    expected_result = [
        [30.0, "Keyboard", 100, 123, "2020-01-01T01:00:00Z"],
        [6.0, "Mouse", 200, 80, "2020-01-01T02:00:00Z"],
    ]
    assert result == expected_result
    assert new_schema == ["Price", "Product", "Stock", "extra", "time"]


def test_negative_schema_with_dict_model(rundb_mock):
    function = mlrun.new_function("tests-1", kind="serving")
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(name="my_model_runner", raise_exception=True)
    model_runner_step.add_model(
        model_class="DictOutputModel",
        execution_mechanism="naive",
        endpoint_name="my_dict_model",
        input_path="inputs.my_dict_model",
        result_path="outputs",
        inputs=["f1", "f2", "f3", "f4"],
        raise_error=False,
    )
    graph.to(model_runner_step).respond()

    function.set_tracking()
    server = function.to_mock_server()
    # bad key right length
    server.test(
        "/",
        {
            "inputs": {
                "my_dict_model": {"f0": 1, "f2": 2, "f3": 3, "f4": 4},
            }
        },
    )
    # missing keys
    server.test(
        "/",
        {
            "inputs": {
                "my_dict_model": {"f0": 1, "f1": 2, "f2": 3},
            }
        },
    )
    # wrong lengthes
    server.test(
        "/",
        {
            "inputs": {
                "my_dict_model": {"f0": [1, 2], "f1": 2, "f2": 3, "f4": 4},
            }
        },
    )

    server.wait_for_completion()

    dummy_stream = server.context.stream.output_stream
    assert len(dummy_stream.event_list) == 0, "expected stream to get zero messages"


@pytest.fixture
def serving_fn(tmp_path: Path) -> ServingRuntime:
    project = mlrun.get_or_create_project(
        "test-auto-mock", save=False, context=str(tmp_path)
    )
    fn = cast(
        ServingRuntime, project.set_function(name="test-fn", kind=ServingRuntime.kind)
    )
    graph = fn.set_topology(StepKinds.flow)
    model_runner_step = ModelRunnerStep(name="my_model_runner_0", raise_exception=True)
    model_runner_step.add_model(
        model_class="MyModel",
        execution_mechanism="naive",
        endpoint_name="my_model_0",
        input_path="n",
        result_path="n",
        raise_error=False,
        inc=1,
    )
    graph.to(model_runner_step).respond()
    return fn


def test_stream_is_set(serving_fn: ServingRuntime) -> None:
    """Test that a dummy stream is set automatically"""
    serving_fn.set_tracking()  # Without any custom arguments
    server = serving_fn.to_mock_server()
    server.test("/", {"n": 1})
    server.wait_for_completion()


@pytest.mark.parametrize(
    ("stream_profile", "expectation"),
    [
        (
            DatastoreProfileKafkaStream(
                name="kafka-profile",
                brokers=["localhost"],
                topics=[],
            ),
            does_not_raise(KafkaOutputStream),
        ),
        (DatastoreProfileV3io(name="v3io-profile"), does_not_raise(OutputStream)),
        (
            DatastoreProfileRedis(
                name="redis-profile", endpoint_url="redis://localhost:6379"
            ),
            pytest.raises(
                mlrun.errors.MLRunValueError,
                match="Expects `DatastoreProfileV3io` or `DatastoreProfileKafkaStream`",
            ),
        ),
    ],
)
def test_serving_stream_profile(
    serving_fn: ServingRuntime,
    stream_profile: DatastoreProfile,
    expectation: AbstractContextManager,
) -> None:
    """Test directly passing stream profile to `to_mock_server`"""
    serving_fn.set_tracking(stream_args={"mock": True})
    with expectation as output_stream_type:
        server = serving_fn.to_mock_server(stream_profile=stream_profile)
        assert isinstance(server.context.stream.output_stream, output_stream_type)
        server.test("/", {"n": 1})
        server.wait_for_completion()


#  test batch
# Helper functions for batch step tests
def verify_batch_step_tracking_events(
    dummy_stream,
    events,
    expected_responses,
    batch_size,
    multiple_models,
    model_names,
    model_class,
    input_schema=None,
    output_schema=None,
):
    """
    Verify tracking events for batch step tests.

    Args:
        dummy_stream: The dummy stream with event_list
        events: Original input events
        expected_responses: Expected responses for each event
        batch_size: Batch size used
        multiple_models: Whether multiple models are used
        model_names: List of model names to check (e.g., ["my_model", "my_model_2"])
        model_class: Expected model class name
        input_schema: Expected input schema (None for lists/strings)
        output_schema: Expected output schema (None for lists/strings)
    """
    num_models = len(model_names)
    num_batches = math.ceil(len(events) / batch_size)
    expected_tracking_events = num_batches * num_models

    assert (
        len(dummy_stream.event_list) == expected_tracking_events
    ), f"Expected {expected_tracking_events} tracking events, got {len(dummy_stream.event_list)}"

    # Group events by model
    model_events = {name: [] for name in model_names}
    for event in dummy_stream.event_list:
        if event["model"] in model_events:
            model_events[event["model"]].append(event)

    # Verify events for each model
    for model_name in model_names:
        model_specific_events = model_events[model_name]
        assert len(model_specific_events) == num_batches

        for i, event in enumerate(model_specific_events):
            # Iterate over batches
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(events))
            batch_events = events[start_idx:end_idx]
            expected_count = len(batch_events)

            # Extract expected inputs and outputs
            expected_inputs = batch_events
            if multiple_models:
                expected_outputs = [
                    expected_responses[j][model_name] for j in range(start_idx, end_idx)
                ]
            else:
                expected_outputs = expected_responses[start_idx:end_idx]

            assert event["effective_sample_count"] == expected_count
            assert event["model"] == model_name
            assert event["model_class"] == model_class
            assert event["error"] is None
            assert event["request"]["inputs"] == expected_inputs
            assert event["request"]["input_schema"] == input_schema
            assert event["resp"]["outputs"] == expected_outputs
            assert event["resp"]["output_schema"] == output_schema


def _verify_batch_step_error_tracking(
    dummy_stream, events, multiple_models, error_substring
):
    """
    Verify error tracking events for batch step tests when batch fails.

    Args:
        dummy_stream: The dummy stream with event_list
        events: Original input events that caused the error
        multiple_models: Whether multiple models are used
        error_substring: Substring expected in error message
    """
    num_models = 2 if multiple_models else 1
    assert len(dummy_stream.event_list) == num_models

    for event in dummy_stream.event_list:
        assert event["error"] is not None
        assert error_substring in event["error"]
        assert event["effective_sample_count"] == len(events)


def _verify_error_in_response_body(
    responses, events, multiple_models, error_substring, model_names=None
):
    """
    Verify that error is returned in response body (not raised as exception).
    """

    assert len(responses) == len(events)

    if model_names is None:
        model_names = ["my_model", "my_model_2"]

    # Verify response body contains error field
    for response in responses:
        if multiple_models:
            # Multiple models: {"model1": {"error": "..."}, "model2": {"error": "..."}}
            assert isinstance(response, dict)
            assert all(
                "error" in response.get(model, {}) for model in model_names
            ), f"Expected error field for each model in response, got {response}"
            # Verify error message content
            for model in model_names:
                assert (
                    error_substring in response[model]["error"]
                    or "can only concatenate str" in response[model]["error"]
                )
        else:
            # Single model: {"error": "..."}
            assert isinstance(response, dict)
            assert "error" in response, f"Expected error field in body, got {response}"
            assert (
                error_substring in response["error"]
                or "can only concatenate str" in response["error"]
            )


def _generate_batch_string_responses(strings, suffixes, model_names=None):
    """
    Generate expected string responses for multiple models.

    Args:
        strings: List of input strings
        suffixes: List of suffixes for each model (or single suffix for single model)
        model_names: List of model names (default: ["my_model_1", "my_model_2"])

    Returns:
        List of expected responses (dicts if multiple models, strings if single model)
    """
    if isinstance(suffixes, list):
        # Multiple models - return list of dicts
        if model_names is None:
            model_names = ["my_model_1", "my_model_2"]
        return [
            {model_names[i]: s + suffix for i, suffix in enumerate(suffixes)}
            for s in strings
        ]
    else:
        # Single model - return list of strings
        return [s + suffixes for s in strings]


@pytest.mark.parametrize("multiple_models", (True, False))
@pytest.mark.parametrize("raise_exception", (True, False))
@pytest.mark.parametrize("return_as_dict", (True, False))
@pytest.mark.parametrize("batching_format", ("raw_list", "input_list", "list_of_lists"))
def test_mrs_direct_batch_input(
    multiple_models, raise_exception, return_as_dict, batching_format, rundb_mock
):
    function = mlrun.new_function("tests", kind="serving")
    function.set_tracking("dummy://", enable_tracking=True)
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(name="my_model_runner")
    if batching_format == "raw_list":
        if raise_exception:
            inputs = [{"z": 1}, {"z": 2}, {"z": 3}, {"z": 4}, {"z": 5}]
        else:
            inputs = [
                {"x1": 1, "x2": 0},
                {"x1": 2, "x2": 1},
                {"x1": 4, "x2": 3},
            ]
    elif batching_format == "input_list":
        if raise_exception:
            inputs = [
                {"input": {"z": 1}},
                {"input": {"z": 2}},
                {"input": {"z": 3}},
                {"input": {"z": 4}},
                {"input": {"z": 5}},
            ]
        else:
            inputs = [
                {"input": {"x1": 1, "x2": 0}},
                {"input": {"x1": 2, "x2": 1}},
                {"input": {"x1": 4, "x2": 3}},
            ]
    else:  # list_of_lists
        if raise_exception:
            inputs = [[1, 2, 3], []]
        else:
            inputs = [[1, 0], [2, 1], [4, 3]]
    model_path = str(Path(__file__).parent / "assets" / "linear_model.pkl")
    model_path2 = str(Path(__file__).parent / "assets" / "linear_model2.pkl")
    endpoint_name = "my_model_1"
    endpoint_name2 = "my_model_2"
    model_runner_step.add_model(
        model_class="BatchedModel",
        execution_mechanism="naive",
        endpoint_name=endpoint_name,
        model_path=model_path,
        return_as_dict=return_as_dict,
        input_path="input" if batching_format == "input_list" else None,
        result_path="output.results" if return_as_dict else None,
    )

    if multiple_models:
        model_runner_step.add_model(
            model_class="BatchedModel",
            endpoint_name=endpoint_name2,
            execution_mechanism="naive",
            model_path=model_path2,
            return_as_dict=return_as_dict,
            input_path="input" if batching_format == "input_list" else None,
            result_path="output.results" if return_as_dict else None,
        )
    graph.to(model_runner_step).respond()
    server = function.to_mock_server()

    try:
        if raise_exception:
            error_regex = (
                "list index out of range"
                if batching_format == "list_of_lists"
                else ".*The feature names should match those that were passed during fit.*"
            )
            with pytest.raises(
                RuntimeError,
                match=error_regex,
            ):
                server.test(body=inputs)
        else:
            resp = server.test(body=inputs)
            if multiple_models:
                if return_as_dict:
                    assert resp == {
                        "my_model_1": {"output": {"results": [3.0, 8.0, 18.0]}},
                        "my_model_2": {"output": {"results": [7.0, 12.0, 22.0]}},
                    }
                else:
                    assert resp == {
                        endpoint_name: [3.0, 8.0, 18.0],
                        endpoint_name2: [7.0, 12.0, 22.0],
                    }
            elif return_as_dict:
                assert resp == {"output": {"results": [3.0, 8.0, 18.0]}}
            else:
                assert resp == [3.0, 8.0, 18.0]
    finally:
        server.wait_for_completion()
    if not raise_exception:
        # Both raw_list and list_of_lists are now transposed to [[1, 0], [2, 1], [4, 3]]
        # raw_list (list of dicts with same keys) gets transposed by key
        # list_of_lists is already in that format
        expected_inputs = [[1, 0], [2, 1], [4, 3]]
        # list_of_lists has no schema (no dict keys), raw_list has schema from dict keys
        expected_input_schema = (
            None if batching_format == "list_of_lists" else ["x1", "x2"]
        )

        dummy_stream = server.context.stream.output_stream
        event = dummy_stream.event_list[0]
        assert event["effective_sample_count"] == 3
        assert event["labels"] == {}
        assert event["request"]["input_schema"] == expected_input_schema
        assert event["request"]["inputs"] == expected_inputs
        assert event["resp"]["output_schema"] is None
        assert event["resp"]["outputs"] == [3.0, 8.0, 18.0]
        assert event["error"] is None
        assert event["model"] == endpoint_name
        assert event["metrics"] is None
        if multiple_models:
            event = dummy_stream.event_list[1]
            assert event["effective_sample_count"] == 3
            assert event["labels"] == {}
            assert event["request"]["input_schema"] == expected_input_schema
            assert event["request"]["inputs"] == expected_inputs
            assert event["resp"]["output_schema"] is None
            assert event["resp"]["outputs"] == [7.0, 12.0, 22.0]
            assert event["error"] is None
            assert event["model"] == endpoint_name2
            assert event["metrics"] is None


@pytest.mark.parametrize("multiple_models", (True, False))
@pytest.mark.parametrize("raise_exception", (True, False))
@pytest.mark.parametrize("return_as_dict", (True, False))
@pytest.mark.parametrize("batching_format", ("raw_list", "input_list", "list_of_lists"))
def test_mrs_direct_batch_str(
    multiple_models, raise_exception, return_as_dict, batching_format, rundb_mock
):
    function = mlrun.new_function("tests", kind="serving")
    function.set_tracking("dummy://", enable_tracking=True)
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(name="my_model_runner")
    if batching_format == "raw_list":
        if raise_exception:
            # Missing "text" field will cause empty strings
            inputs = [{"z": 1}, {"z": 2}, {"z": 3}]
        else:
            inputs = ["hello", "world", "test", "mlrun", "data", "science"]
    elif batching_format == "input_list":
        if raise_exception:
            inputs = [
                {"input": {"z": 1}},
                {"input": {"z": 2}},
                {"input": {"z": 3}},
            ]
        else:
            inputs = [
                {"input": {"text": "hello"}},
                {"input": {"text": "world"}},
                {"input": {"text": "test"}},
                {"input": {"text": "mlrun"}},
                {"input": {"text": "data"}},
                {"input": {"text": "science"}},
            ]
    else:  # list_of_lists (treated as list of strings)
        if raise_exception:
            # Non-string items will cause errors
            inputs = [[123, 456, 789], [111, 222, 333]]
        else:
            inputs = [["hello", "world", "test"], ["mlrun", "data", "science"]]

    suffix1 = "_model1"
    suffix2 = "_model2"
    endpoint_name = "my_model_1"
    endpoint_name2 = "my_model_2"
    model_runner_step.add_model(
        model_class="StringBatchedModel",
        execution_mechanism="naive",
        endpoint_name=endpoint_name,
        suffix=suffix1,
        return_as_dict=return_as_dict,
        input_path="input.text" if batching_format == "input_list" else None,
        result_path="output.results" if return_as_dict else None,
    )

    if multiple_models:
        model_runner_step.add_model(
            model_class="StringBatchedModel",
            endpoint_name=endpoint_name2,
            execution_mechanism="naive",
            suffix=suffix2,
            return_as_dict=return_as_dict,
            input_path="input.text" if batching_format == "input_list" else None,
            result_path="output.results" if return_as_dict else None,
        )
    graph.to(model_runner_step).respond()
    server = function.to_mock_server()

    try:
        if raise_exception:
            if batching_format == "list_of_lists":
                error_regex = (
                    r".*unsupported operand type\(s\) for \+: 'int' and 'str'.*"
                )
            else:
                error_regex = ".*KeyError: 'text'"
            with pytest.raises(
                RuntimeError,
                match=error_regex,
            ):
                server.test(body=inputs)
        else:
            resp = server.test(body=inputs)
            # Generate expected outputs using helper function
            strings = ["hello", "world", "test", "mlrun", "data", "science"]
            expected_output_1 = _generate_batch_string_responses(strings, "_model1")
            expected_output_2 = _generate_batch_string_responses(strings, "_model2")

            if multiple_models:
                if return_as_dict:
                    assert resp == {
                        "my_model_1": {"output": {"results": expected_output_1}},
                        "my_model_2": {"output": {"results": expected_output_2}},
                    }
                else:
                    assert resp == {
                        endpoint_name: expected_output_1,
                        endpoint_name2: expected_output_2,
                    }
            elif return_as_dict:
                assert resp == {"output": {"results": expected_output_1}}
            else:
                assert resp == expected_output_1
    finally:
        server.wait_for_completion()
    if not raise_exception:
        # Generate expected outputs using helper function
        strings = ["hello", "world", "test", "mlrun", "data", "science"]
        expected_output_1 = _generate_batch_string_responses(strings, "_model1")
        expected_output_2 = _generate_batch_string_responses(strings, "_model2")

        expected_inputs = (
            [["hello", "world", "test"], ["mlrun", "data", "science"]]
            if batching_format == "list_of_lists"
            else ["hello", "world", "test", "mlrun", "data", "science"]
        )

        effective_sample_count = 2 if batching_format == "list_of_lists" else 6
        dummy_stream = server.context.stream.output_stream
        event = dummy_stream.event_list[0]
        assert event["effective_sample_count"] == effective_sample_count
        assert event["labels"] == {}
        assert event["request"]["inputs"] == expected_inputs
        assert event["resp"]["outputs"] == expected_output_1
        assert event["error"] is None
        assert event["model"] == endpoint_name
        assert event["metrics"] is None
        if multiple_models:
            event = dummy_stream.event_list[1]
            assert event["effective_sample_count"] == effective_sample_count
            assert event["labels"] == {}
            assert event["request"]["inputs"] == expected_inputs
            assert event["resp"]["outputs"] == expected_output_2
            assert event["error"] is None
            assert event["model"] == endpoint_name2
            assert event["metrics"] is None


@pytest.mark.parametrize("multiple_models", [True, False])
def test_batch_step_with_mrs(rundb_mock, multiple_models):
    number_of_events = 7
    batch_size = 2
    function = mlrun.new_function("test-batch-mrs", kind="serving")
    function.set_tracking("dummy://", enable_tracking=True)
    graph = function.set_topology("flow", engine="async")

    # Batch step: accumulate up to 2 events or flush after 1 second
    graph = graph.to(
        "storey.Batch",
        "batching",
        max_events=batch_size,
        flush_after_seconds=1,
        full_event=True,
    )

    # ModelRunnerStep: process batches through the model(s)
    model_runner_step = ModelRunnerStep(name="model_runner", raise_exception=True)
    model_runner_step.add_model(
        model_class="BatchedGraphModel",
        execution_mechanism="naive",
        endpoint_name="my_model",
        input_path="input",
        result_path="output",
    )

    if multiple_models:
        model_runner_step.add_model(
            model_class="BatchedGraphModel2",
            execution_mechanism="naive",
            endpoint_name="my_model_2",
            input_path="input",
            result_path="output_2",
        )

    step = graph.to(model_runner_step)
    step = step.to("storey.FlatMap", _fn="(event.body)", full_event=True)
    step.respond()
    server = function.to_mock_server()

    try:
        events = [
            {"input": [10 + i, 20 + i, 30 + i]} for i in range(0, number_of_events)
        ]

        def send_event(event, delay):
            time.sleep(delay)  # Stagger the sends
            return server.test(body=event)

        # Send events in thread pool with 0.1s between sends using submit
        with ThreadPoolExecutor(max_workers=number_of_events) as executor:
            futures = [
                executor.submit(send_event, event, i * 0.1)
                for i, event in enumerate(events)
            ]
            responses = [future.result() for future in futures]
    finally:
        server.wait_for_completion()

    # Verify we got all responses
    assert (
        len(responses) == number_of_events
    ), f"Expected {number_of_events} responses, got {len(responses)}"
    assert all(r is not None for r in responses)

    # Verify each response has correct input/output
    if multiple_models:
        expected_responses = [
            {
                "my_model": {"input": [10, 20, 30], "output": 60},
                "my_model_2": {"input": [10, 20, 30], "output_2": 61},
            },
            {
                "my_model": {"input": [11, 21, 31], "output": 63},
                "my_model_2": {"input": [11, 21, 31], "output_2": 64},
            },
            {
                "my_model": {"input": [12, 22, 32], "output": 66},
                "my_model_2": {"input": [12, 22, 32], "output_2": 67},
            },
            {
                "my_model": {"input": [13, 23, 33], "output": 69},
                "my_model_2": {"input": [13, 23, 33], "output_2": 70},
            },
            {
                "my_model": {"input": [14, 24, 34], "output": 72},
                "my_model_2": {"input": [14, 24, 34], "output_2": 73},
            },
            {
                "my_model": {"input": [15, 25, 35], "output": 75},
                "my_model_2": {"input": [15, 25, 35], "output_2": 76},
            },
            {
                "my_model": {"input": [16, 26, 36], "output": 78},
                "my_model_2": {"input": [16, 26, 36], "output_2": 79},
            },
        ]
    else:
        expected_responses = [
            {"input": [10, 20, 30], "output": 60},
            {"input": [11, 21, 31], "output": 63},
            {"input": [12, 22, 32], "output": 66},
            {"input": [13, 23, 33], "output": 69},
            {"input": [14, 24, 34], "output": 72},
            {"input": [15, 25, 35], "output": 75},
            {"input": [16, 26, 36], "output": 78},
        ]
    assert responses == expected_responses

    # Verify tracking events
    # Single model: 4 batches (2+2+2+1 = 4 total batches)
    # Multiple models: 8 batches (4 batches × 2 models)
    dummy_stream = server.context.stream.output_stream
    num_models = 2 if multiple_models else 1
    num_batches = math.ceil(number_of_events / batch_size)
    expected_tracking_events = num_batches * num_models

    assert (
        len(dummy_stream.event_list) == expected_tracking_events
    ), f"Expected {expected_tracking_events} tracking events, got {len(dummy_stream.event_list)}"

    # Group events by model
    model_events = {"my_model": [], "my_model_2": []}
    for event in dummy_stream.event_list:
        model_events[event["model"]].append(event)

    # Verify events for each model
    models_to_check = ["my_model", "my_model_2"] if multiple_models else ["my_model"]

    for model_name in models_to_check:
        events = model_events[model_name]
        assert len(events) == num_batches

        base_id = None
        for i, event in enumerate(events):
            # iterate over batches
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(expected_responses))
            batch_items = expected_responses[start_idx:end_idx]

            expected_count = len(batch_items)

            # Extract inputs and outputs based on response structure
            if multiple_models:
                expected_inputs = [item[model_name]["input"] for item in batch_items]
                if model_name == "my_model_2":
                    expected_outputs = [
                        item[model_name]["output_2"] for item in batch_items
                    ]
                else:
                    expected_outputs = [
                        item[model_name]["output"] for item in batch_items
                    ]
            else:
                expected_inputs = [item["input"] for item in batch_items]
                expected_outputs = [item["output"] for item in batch_items]

            assert event["effective_sample_count"] == expected_count
            assert event["model"] == model_name
            assert (
                event["model_class"] == "BatchedGraphModel"
                if model_name == "my_model"
                else "BatchedGraphModel2"
            )
            assert event["error"] is None
            assert event["request"]["inputs"] == expected_inputs
            assert event["resp"]["outputs"] == expected_outputs

            request_id = event["request"]["id"]
            if not base_id:
                base_id = request_id.split("-")[0]
            assert request_id == f"{base_id}-{i:04d}"


class SimpleTestModel(Model):
    """Simple model for testing that doesn't require external files."""

    def predict(self, body, **kwargs):
        return {"result": "ok"}


class FailingStreamingModel(Model):
    """A streaming model that raises an error after yielding some chunks."""

    def predict(self, body, **kwargs):
        yield "chunk_0"
        raise RuntimeError("stream failure mid-generation")


class TestMonitoringPreProcessorStreamingAggregation:
    """Tests for MonitoringPreProcessor streaming chunk aggregation."""

    def test_aggregate_string_chunks(self):
        preprocessor = MonitoringPreProcessor()

        chunks = ["Hello", " ", "world", "!"]
        result = preprocessor._aggregate_collected_chunks(chunks)
        assert result == "Hello world!"

    def test_aggregate_dict_chunks_with_string_outputs(self):
        """Test aggregation of dict chunks with string outputs."""
        preprocessor = MonitoringPreProcessor()

        chunks = [
            {"output": "Hello ", "metrics": {"tokens": 2}},
            {"output": "world", "metrics": {"tokens": 1}},
            {"output": "!", "metrics": {"tokens": 1}},
        ]
        result = preprocessor._aggregate_collected_chunks(chunks)

        assert result["output"] == "Hello world!"
        assert result["metrics"]["tokens"] == 4

    def test_aggregate_flat_dict_chunks(self):
        """Test aggregation of flat dict chunks (no nested metrics).

        Matches the StreamingModel output format: {"output": "...", "token_count": 1}.
        The "output" key is concatenated; other keys take first chunk's value.
        """
        preprocessor = MonitoringPreProcessor()

        chunks = [
            {"output": "test_chunk_0", "token_count": 1},
            {"output": "test_chunk_1", "token_count": 1},
            {"output": "test_chunk_2", "token_count": 1},
        ]
        result = preprocessor._aggregate_collected_chunks(chunks)

        assert result["output"] == "test_chunk_0test_chunk_1test_chunk_2"
        # Non-output, non-metrics keys take value from first chunk
        assert result["token_count"] == 1

    def test_aggregate_dict_chunks_with_list_outputs(self):
        """Test aggregation of dict chunks with list outputs."""
        preprocessor = MonitoringPreProcessor()

        chunks = [
            {"outputs": [1, 2]},
            {"outputs": [3, 4]},
        ]
        result = preprocessor._aggregate_collected_chunks(chunks)

        assert result["outputs"] == [1, 2, 3, 4]

    def test_aggregate_metrics_sums_numeric_values(self):
        """Test that numeric metrics are summed."""
        preprocessor = MonitoringPreProcessor()

        metrics_list = [
            {"tokens": 10, "latency": 0.5},
            {"tokens": 20, "latency": 0.3},
            {"tokens": 5, "latency": 0.2},
        ]
        result = preprocessor._aggregate_metrics(metrics_list)

        assert result["tokens"] == 35
        assert result["latency"] == 1.0

    def test_aggregate_metrics_with_none_values(self):
        """Test metrics aggregation handles None values."""
        preprocessor = MonitoringPreProcessor()

        metrics_list = [None, {"tokens": 10}, None, {"tokens": 5}]
        result = preprocessor._aggregate_metrics(metrics_list)

        assert result["tokens"] == 15

    def test_aggregate_metrics_all_none(self):
        """Test metrics aggregation when all values are None."""
        preprocessor = MonitoringPreProcessor()

        result = preprocessor._aggregate_metrics([None, None])
        assert result is None

    def test_aggregate_outputs_concatenates_strings(self):
        """Test output aggregation concatenates strings."""
        preprocessor = MonitoringPreProcessor()

        outputs = ["Hello", " ", "world"]
        result = preprocessor._aggregate_outputs(outputs)
        assert result == "Hello world"

    def test_aggregate_outputs_flattens_lists(self):
        """Test output aggregation flattens lists."""
        preprocessor = MonitoringPreProcessor()

        outputs = [[1, 2], [3, 4], [5]]
        result = preprocessor._aggregate_outputs(outputs)
        assert result == [1, 2, 3, 4, 5]

    def test_aggregate_error_keeps_first_non_none(self):
        """Test that error aggregation keeps the first non-None error."""
        preprocessor = MonitoringPreProcessor()

        chunks = [
            {"output": "chunk1"},
            {"output": "chunk2", "error": "Something went wrong"},
            {"output": "chunk3"},
        ]
        result = preprocessor._merge_dict_chunks(chunks)

        assert result["error"] == "Something went wrong"

    def test_aggregate_empty_chunks(self):
        """Test aggregation of empty chunk list."""
        preprocessor = MonitoringPreProcessor()

        result = preprocessor._aggregate_collected_chunks([])
        assert result == {}

    def test_aggregate_mixed_types_returns_list(self):
        """Test that mixed types are returned as a list."""
        preprocessor = MonitoringPreProcessor()

        chunks = ["string", {"dict": "value"}, 123]
        result = preprocessor._aggregate_collected_chunks(chunks)
        assert result == chunks

    def test_do_aggregates_when_stream_collected_marker_set(self, rundb_mock):
        """Test that do() aggregates body when event has stream_collected=True."""
        function = mlrun.new_function("test-stream-agg", kind="serving")
        graph = function.set_topology("flow", engine="async")
        model_runner_step = ModelRunnerStep(name="my_runner")
        model_runner_step.add_model(
            model_class="SimpleTestModel",
            execution_mechanism="naive",
            endpoint_name="my_model",
        )
        graph.to(model_runner_step).respond()
        function.set_tracking()
        server = function.to_mock_server()

        try:
            # Build a collected streaming event
            event = storey.Event(
                body=[
                    {"output": "chunk_0", "token_count": 1},
                    {"output": "chunk_1", "token_count": 1},
                ],
            )
            event._metadata = {
                "model_runner_name": "my_runner",
                "when": "2026-01-01 00:00:00.000000+00:00",
                "microsec": 1000,
                "inputs": [[1.0, 2.0]],
            }
            event.stream_collected = True

            preprocessor = server.graph.steps["monitoring_pre_processor_step"]._object
            result = preprocessor.do(event)

            # After aggregation, body should be a list of monitoring events (not the raw chunks)
            assert isinstance(result.body, list)
            assert len(result.body) == 1
            monitoring_event = result.body[0]
            # Outputs should contain the concatenated string
            assert "chunk_0chunk_1" in str(monitoring_event["resp"]["outputs"])
        finally:
            server.wait_for_completion()

    def test_do_does_not_aggregate_without_stream_collected_marker(self, rundb_mock):
        """Test that do() does NOT aggregate body when stream_collected is absent."""
        function = mlrun.new_function("test-no-agg", kind="serving")
        graph = function.set_topology("flow", engine="async")
        model_runner_step = ModelRunnerStep(name="my_runner")
        model_runner_step.add_model(
            model_class="SimpleTestModel",
            execution_mechanism="naive",
            endpoint_name="my_model",
        )
        graph.to(model_runner_step).respond()
        function.set_tracking()
        server = function.to_mock_server()

        try:
            # Build a regular (non-streaming) event with a dict body
            event = storey.Event(
                body={"output": "single_result"},
            )
            event._metadata = {
                "model_runner_name": "my_runner",
                "when": "2026-01-01 00:00:00.000000+00:00",
                "microsec": 1000,
                "inputs": [[1.0, 2.0]],
            }
            # No stream_collected marker

            preprocessor = server.graph.steps["monitoring_pre_processor_step"]._object
            result = preprocessor.do(event)

            assert isinstance(result.body, list)
            assert len(result.body) == 1
            monitoring_event = result.body[0]
            # Should have processed the original dict body directly
            assert "single_result" in str(monitoring_event["resp"]["outputs"])
        finally:
            server.wait_for_completion()

    def test_do_aggregates_string_chunks_with_result_path(self, rundb_mock):
        """Test that stream-collected string chunks work even when result_path is set.

        LLModel through MRS always has result_path='output'. In streaming mode the
        aggregated body is a plain string (not a dict), so result_path must be
        ignored during output extraction.
        """
        function = mlrun.new_function("test-stream-str-rp", kind="serving")
        graph = function.set_topology("flow", engine="async")
        model_runner_step = ModelRunnerStep(name="my_runner")
        model_runner_step.add_model(
            model_class="SimpleTestModel",
            execution_mechanism="naive",
            endpoint_name="my_model",
            result_path="output",
        )
        graph.to(model_runner_step).respond()
        function.set_tracking()
        server = function.to_mock_server()

        try:
            event = storey.Event(
                body=["hello ", "world "],
            )
            event._metadata = {
                "model_runner_name": "my_runner",
                "when": "2026-01-01 00:00:00.000000+00:00",
                "microsec": 1000,
                "inputs": {"question": "test"},
            }
            event.stream_collected = True

            preprocessor = server.graph.steps["monitoring_pre_processor_step"]._object
            result = preprocessor.do(event)

            assert isinstance(result.body, list)
            assert len(result.body) == 1
            monitoring_event = result.body[0]
            assert "hello world " in str(monitoring_event["resp"]["outputs"])
        finally:
            server.wait_for_completion()

    def test_streaming_success_produces_single_output(self, rundb_mock):
        """Verify that a stream-collected scalar body produces a single-element
        outputs list without crashing on result_path extraction."""
        function = mlrun.new_function("test-stream-cols", kind="serving")
        graph = function.set_topology("flow", engine="async")
        model_runner_step = ModelRunnerStep(name="my_runner")
        model_runner_step.add_model(
            model_class="SimpleTestModel",
            execution_mechanism="naive",
            endpoint_name="my_model",
            result_path="output",
            outputs=["answer", "usage"],
        )
        graph.to(model_runner_step).respond()
        function.set_tracking()
        server = function.to_mock_server()

        try:
            preprocessor = server.graph.steps["monitoring_pre_processor_step"]._object

            stream_event = storey.Event(
                body=["Par", "is"],
            )
            stream_event._metadata = {
                "model_runner_name": "my_runner",
                "when": "2026-01-01 00:00:01.000000+00:00",
                "microsec": None,
                "inputs": {"question": "What is the capital of France?"},
            }
            stream_event.stream_collected = True
            stream_mon = preprocessor.do(stream_event).body[0]

            assert stream_mon["resp"]["outputs"] == ["Paris"]
        finally:
            server.wait_for_completion()

    def test_streaming_error_reaches_monitoring_stream(self, rundb_mock):
        """Verify that an error in a streaming model reaches the monitoring stream.

        When a streaming generator raises mid-stream, the Collector should emit
        an event with body={"error": "..."} which MonitoringPreProcessor records
        to the monitoring stream, matching non-streaming error tracking behavior.
        """
        function = mlrun.new_function("test-stream-err-track", kind="serving")
        graph = function.set_topology("flow", engine="async")
        model_runner_step = ModelRunnerStep(name="my_runner")
        model_runner_step.add_model(
            model_class="FailingStreamingModel",
            execution_mechanism="naive",
            endpoint_name="my_model",
        )
        graph.to(model_runner_step).respond()

        function.set_streaming(enabled=True)
        function.set_tracking("dummy://")
        server = function.to_mock_server()

        try:
            server.test("/", {"n": 1})
        finally:
            server.wait_for_completion()

        dummy_stream = server.context.stream.output_stream
        assert (
            len(dummy_stream.event_list) == 1
        ), "expected one tracking event for the streaming error"
        event = dummy_stream.event_list[0]
        assert (
            event.get("error") is not None
        ), "expected 'error' field in tracking event"
        assert "RuntimeError" in event["error"]
        assert "stream failure mid-generation" in event["error"]


def test_collector_step_added_to_monitoring_graph(rundb_mock):
    """Test that Collector steps are added between MRS and monitoring steps when streaming is enabled."""
    function = mlrun.new_function("test-collector", kind="serving")
    graph = function.set_topology("flow", engine="async")

    model_runner_step = ModelRunnerStep(name="my_model_runner")
    model_runner_step.add_model(
        model_class="SimpleTestModel",
        execution_mechanism="naive",
        endpoint_name="my_model",
    )
    graph.to(model_runner_step).respond()

    # Enable streaming - this triggers Collector insertion
    function.set_streaming(enabled=True)
    function.set_tracking()
    server = function.to_mock_server()

    try:
        # Verify the Collector step was added
        assert (
            "my_model_runner_collector" in server.graph.steps
        ), "Collector step should be added after the model runner step"

        # Verify the Collector step is after the model runner
        collector_step = server.graph.steps["my_model_runner_collector"]
        collector_after = collector_step.after
        if isinstance(collector_after, list):
            assert (
                "my_model_runner" in collector_after
            ), "Collector step should come after the model runner step"
        else:
            assert (
                collector_after == "my_model_runner"
            ), "Collector step should come after the model runner step"

        # Verify that a monitoring step receives from the collector (directly or indirectly)
        # The first monitoring step may be background_task_status_step or filter_none
        first_mm_step_name = (
            "background_task_status_step"
            if "background_task_status_step" in server.graph.steps
            else "filter_none"
        )
        if first_mm_step_name in server.graph.steps:
            first_mm_step = server.graph.steps[first_mm_step_name]
            after = first_mm_step.after
            if isinstance(after, list):
                assert (
                    "my_model_runner_collector" in after
                ), f"First MM step {first_mm_step_name} should receive from collector, got {after}"
            else:
                assert (
                    after == "my_model_runner_collector"
                ), f"First MM step {first_mm_step_name} should receive from collector, got {after}"
    finally:
        # Explicitly wait for and close the server to avoid hanging
        server.wait_for_completion()


@pytest.mark.parametrize("multiple_models", (True, False))
@pytest.mark.parametrize(
    "raise_exception, with_error",
    [
        (False, False),
        (True, True),
        (False, True),
    ],
)
def test_batch_step_with_mrs_list(
    multiple_models, raise_exception, with_error, rundb_mock
):
    """
    Test batch step with MRS for:
    - Single vs multiple models
    - Error handling (valid inputs vs mixed valid/invalid)
    - Proper tracking of batch events
    - Error returned as dict when raise_exception=False
    """

    function = mlrun.new_function("tests", kind="serving")
    function.set_tracking("dummy://", enable_tracking=True)
    graph = function.set_topology("flow", engine="async")

    # Batch step: groups events into batches
    batch_size = 2
    graph = graph.to(
        "storey.Batch",
        "batching",
        max_events=batch_size,
        flush_after_seconds=1,
        full_event=True,
    )

    # ModelRunnerStep: process batches through the model(s)
    model_runner_step = ModelRunnerStep(
        name="model_runner", raise_exception=raise_exception
    )

    model_path = str(Path(__file__).parent / "assets" / "linear_model.pkl")
    model_path2 = str(Path(__file__).parent / "assets" / "linear_model2.pkl")

    model_runner_step.add_model(
        model_class="BatchedModel",
        execution_mechanism="naive",
        endpoint_name="my_model",
        model_path=model_path,
    )

    if multiple_models:
        model_runner_step.add_model(
            model_class="BatchedModel",
            execution_mechanism="naive",
            endpoint_name="my_model_2",
            model_path=model_path2,
        )

    step = graph.to(model_runner_step)
    step = step.to("storey.FlatMap", _fn="(event.body)", full_event=True)
    step.respond()
    server = function.to_mock_server()

    try:
        if with_error:
            # Mix valid and invalid inputs - invalid list has wrong length
            events = [
                [1, 0],  # Valid
                [],  # Invalid - empty list causes error
            ]
        else:
            # All valid inputs as simple lists
            events = [
                [1, 0],
                [2, 1],
                [4, 3],
                [5, 4],
                [7, 6],
            ]

        def send_event(event, delay):
            time.sleep(delay)
            return server.test(body=event)

        # Send events in thread pool with staggered delays
        with ThreadPoolExecutor(max_workers=len(events)) as executor:
            futures = [
                executor.submit(send_event, event, i * 0.1)
                for i, event in enumerate(events)
            ]

            if with_error and raise_exception:
                # Expect error to be raised when batch processes
                error_count = 0
                for future in futures:
                    try:
                        future.result()
                    except RuntimeError as e:
                        if "list index out of range" in str(e):
                            error_count += 1
                        else:
                            raise
                # Both events in the batch should fail together
                assert error_count == len(
                    events
                ), f"Expected {len(events)} errors, got {error_count}"
            else:
                responses = [future.result() for future in futures]
    finally:
        server.wait_for_completion()

    if with_error and raise_exception:
        # Error was raised - verify error tracking
        dummy_stream = server.context.stream.output_stream
        _verify_batch_step_error_tracking(
            dummy_stream=dummy_stream,
            events=events,
            multiple_models=multiple_models,
            error_substring="list index out of range",
        )
    elif with_error and not raise_exception:
        # Error should be returned in response dict, not raised
        _verify_error_in_response_body(
            responses=responses,
            events=events,
            multiple_models=multiple_models,
            error_substring="list index out of range",
            model_names=["my_model", "my_model_2"],
        )

        # Verify error tracking in stream
        dummy_stream = server.context.stream.output_stream
        _verify_batch_step_error_tracking(
            dummy_stream=dummy_stream,
            events=events,
            multiple_models=multiple_models,
            error_substring="list index out of range",
        )
    else:
        # No error - verify normal responses
        assert len(responses) == len(events)
        assert all(r is not None for r in responses)

        # Expected responses based on linear model predictions
        if multiple_models:
            expected_responses = [
                {"my_model": 3.0, "my_model_2": 7.0},
                {"my_model": 8.0, "my_model_2": 12.0},
                {"my_model": 18.0, "my_model_2": 22.0},
                {"my_model": 23.0, "my_model_2": 27.0},
                {"my_model": 33.0, "my_model_2": 37.0},
            ]
        else:
            expected_responses = [3.0, 8.0, 18.0, 23.0, 33.0]
        assert responses == expected_responses

        # Verify tracking events
        dummy_stream = server.context.stream.output_stream
        model_names = ["my_model", "my_model_2"] if multiple_models else ["my_model"]
        verify_batch_step_tracking_events(
            dummy_stream=dummy_stream,
            events=events,
            expected_responses=expected_responses,
            batch_size=batch_size,
            multiple_models=multiple_models,
            model_names=model_names,
            model_class="BatchedModel",
            input_schema=None,
            output_schema=None,
        )


@pytest.mark.parametrize("multiple_models", (True, False))
@pytest.mark.parametrize(
    "raise_exception, with_error",
    [
        (False, False),
        (True, True),
        (False, True),
    ],
)
def test_batch_step_with_mrs_string(
    multiple_models, raise_exception, with_error, rundb_mock
):
    """
    Test batch step with MRS for simple string inputs (e.g., "hello", "world").
    - Single vs multiple models
    - Error handling with invalid inputs
    - Proper tracking of batch events
    - Error returned as dict when raise_exception=False
    """
    function = mlrun.new_function("tests", kind="serving")
    function.set_tracking("dummy://", enable_tracking=True)
    graph = function.set_topology("flow", engine="async")

    # Batch step: groups events into batches
    batch_size = 2
    graph = graph.to(
        "storey.Batch",
        "batching",
        max_events=batch_size,
        flush_after_seconds=1,
        full_event=True,
    )

    # ModelRunnerStep: process batches through the string model
    model_runner_step = ModelRunnerStep(
        name="model_runner", raise_exception=raise_exception
    )

    suffix1 = "_model1"
    suffix2 = "_model2"
    endpoint_name = "my_model_1"
    endpoint_name2 = "my_model_2"
    model_runner_step.add_model(
        model_class="StringBatchedModel",
        execution_mechanism="naive",
        endpoint_name=endpoint_name,
        suffix=suffix1,
    )

    if multiple_models:
        model_runner_step.add_model(
            model_class="StringBatchedModel",
            execution_mechanism="naive",
            endpoint_name=endpoint_name2,
            suffix=suffix2,
        )

    step = graph.to(model_runner_step)
    step = step.to("storey.FlatMap", _fn="(event.body)", full_event=True)
    step.respond()
    server = function.to_mock_server()

    try:
        if with_error:
            # Mix valid and invalid inputs - integers will cause error
            events = [
                "hello",  # Valid
                123,  # Invalid - not a string
            ]
        else:
            # All valid string inputs
            events = [
                "hello",
                "world",
                "test",
                "mlrun",
                "batch",
            ]

        def send_event(event, delay):
            time.sleep(delay)
            return server.test(body=event)

        # Send events in thread pool with staggered delays
        with ThreadPoolExecutor(max_workers=len(events)) as executor:
            futures = [
                executor.submit(send_event, event, i * 0.1)
                for i, event in enumerate(events)
            ]

            if with_error and raise_exception:
                # Expect error to be raised when batch processes
                for future in futures:
                    with pytest.raises(
                        RuntimeError, match=r".*unsupported operand type"
                    ):
                        future.result()
            else:
                responses = [future.result() for future in futures]
    finally:
        server.wait_for_completion()

    if with_error and raise_exception:
        # Error was raised - verify error tracking
        dummy_stream = server.context.stream.output_stream
        _verify_batch_step_error_tracking(
            dummy_stream=dummy_stream,
            events=events,
            multiple_models=multiple_models,
            error_substring="unsupported operand type",
        )
    elif with_error and not raise_exception:
        # Error should be returned in response dict, not raised
        _verify_error_in_response_body(
            responses=responses,
            events=events,
            multiple_models=multiple_models,
            error_substring="unsupported operand type",
            model_names=["my_model_1", "my_model_2"],
        )

        # Verify error tracking in stream
        dummy_stream = server.context.stream.output_stream
        _verify_batch_step_error_tracking(
            dummy_stream=dummy_stream,
            events=events,
            multiple_models=multiple_models,
            error_substring="unsupported operand type",
        )
    else:
        # No error - verify normal responses
        assert len(responses) == len(events)
        assert all(r is not None for r in responses)

        # Expected responses based on string concatenation using helper function
        strings = ["hello", "world", "test", "mlrun", "batch"]
        if multiple_models:
            expected_responses = _generate_batch_string_responses(
                strings, ["_model1", "_model2"], ["my_model_1", "my_model_2"]
            )
        else:
            expected_responses = _generate_batch_string_responses(strings, "_model1")
        assert responses == expected_responses

        # Verify tracking events
        dummy_stream = server.context.stream.output_stream
        model_names = (
            ["my_model_1", "my_model_2"] if multiple_models else ["my_model_1"]
        )
        verify_batch_step_tracking_events(
            dummy_stream=dummy_stream,
            events=events,
            expected_responses=expected_responses,
            batch_size=batch_size,
            multiple_models=multiple_models,
            model_names=model_names,
            model_class="StringBatchedModel",
            input_schema=None,
            output_schema=None,
        )
