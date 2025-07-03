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
import pathlib
from collections.abc import Iterator
from typing import Union, cast
from unittest.mock import patch

import numpy as np
import pytest

import mlrun
import mlrun.common.schemas.model_monitoring.constants as mm_constants
from mlrun.common.schemas import ModelEndpointCreationStrategy
from mlrun.datastore.datastore_profile import (
    DatastoreProfileKafkaSource,
    register_temporary_client_datastore_profile,
    remove_temporary_client_datastore_profile,
)
from mlrun.platforms.iguazio import KafkaOutputStream
from mlrun.runtimes import ServingRuntime
from mlrun.serving import Model, ModelRunnerStep, ModelSelector
from mlrun.serving.states import RootFlowStep, RouterStep
from tests.serving.test_serving import _log_model

assets_path = str(pathlib.Path(__file__).parent / "assets")
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
    with patch("mlrun.get_run_db", return_value=rundb_mock):
        project = mlrun.new_project("test-pro", save=False)
        fn = mlrun.new_function("test-fn", kind="serving", project=project.name)
        model_uri = _log_model(project)
        fn.add_model(
            "m1",
            model_uri,
            "ModelTestingClass",
            multiplier=5,
            model_endpoint_uid="my-uid",
            creation_strategy=ModelEndpointCreationStrategy.ARCHIVE,
        )
        fn.set_tracking("dummy://", enable_tracking=enable_tracking)
        server = fn.to_mock_server()
        server.test("/v2/models/m1/infer", testdata)
        dummy_stream = server.context.stream.output_stream
        if enable_tracking:
            assert (
                len(dummy_stream.event_list) == 1
            ), "expected stream to get one message"
        else:
            assert len(dummy_stream.event_list) == 0, "expected stream to be empty"


@pytest.mark.parametrize("track_before_creating_child", [True, False])
@pytest.mark.parametrize("enable_tracking", [True, False])
@pytest.mark.parametrize("topology", ["flow", "router"])
def test_child_function_tracking(
    rundb_mock, track_before_creating_child, enable_tracking, topology
):
    with patch("mlrun.get_run_db", return_value=rundb_mock):
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


def rec_to_data(rec):
    data = json.loads(rec["data"])
    inputs = data["request"]["inputs"]
    outputs = data["resp"]["outputs"]
    return data["model"], data["class"], inputs, outputs


@pytest.fixture
def project() -> mlrun.MlrunProject:
    return mlrun.get_or_create_project("test-tracking", allow_cross_project=True)


@pytest.fixture
def _register_stream_profile(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    stream_profile_name = "special-stream"
    monkeypatch.setenv(
        mm_constants.ProjectSecretKeys.STREAM_PROFILE_NAME, stream_profile_name
    )
    profile = DatastoreProfileKafkaSource(
        name=stream_profile_name,
        brokers=["localhost"],
        topics=[],
        kwargs_public={"api_version": (3, 9)},
    )
    register_temporary_client_datastore_profile(profile)
    yield
    remove_temporary_client_datastore_profile(stream_profile_name)


@pytest.mark.usefixtures("rundb_mock", "_register_stream_profile")
def test_tracking_datastore_profile(project: mlrun.MlrunProject) -> None:
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

    output_stream = cast(KafkaOutputStream, server.context.stream.output_stream)
    mocked_stream = output_stream._mock_queue
    assert len(mocked_stream) == 2

    event = mocked_stream[1]
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

    def predict(self, body):
        body["n"] += self.inc
        body.pop("models", None)
        return body

    async def predict_async(self, body):
        return self.predict(body)


class DictOutputModel(Model):
    def predict(self, body):
        body["outputs"] = {}
        for key, value in body["inputs"].items():
            body["outputs"][key.replace("f", "o")] = (
                value + 1 if not isinstance(value, list) else [v + 1 for v in value]
            )
        return body

    async def predict_async(self, body):
        return self.predict(body)


def _test_monitoring_system_steps_structure(
    graph: RootFlowStep, model_runners_names: list[str]
):
    system_steps = {
        "background_task_status_step": model_runners_names,
        "filter_none": ["background_task_status_step"],
        "monitoring_pre_processor_step": ["filter_none"],
        "flatten_events": ["monitoring_pre_processor_step"],
        "sampling_step": ["flatten_events"],
        "filter_none_sampling": ["sampling_step"],
        "model_monitoring_stream": [
            "filter_none_sampling"
        ],  # mock creates a dummy pusher and not target
    }
    for step in graph.steps.values():
        if step.name in system_steps:
            assert step.after == system_steps[step.name]


def _test_graph_structure(graph: RootFlowStep, tracked: bool):
    """Expects server graph contains system steps"""
    model_runners = []
    for step in graph.steps.values():
        if isinstance(step, ModelRunnerStep):
            model_runners.append(step.name)
        elif model_runners and step.name == f"{model_runners[-1]}_error_raise":
            assert model_runners[-1] in step.after or model_runners[-1] in step.after
    if tracked:
        _test_monitoring_system_steps_structure(graph, model_runners)


@pytest.mark.parametrize("enable_tracking", [True, False])
def test_tracked_model_runner(enable_tracking: bool):
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
    graph.to(model_runner_step).respond()
    function.set_tracking(stream_args={"mock": True})

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

    _test_graph_structure(server.graph, enable_tracking)


def test_tracked_model_runner_dict():
    function = mlrun.new_function("tests-1", kind="serving")
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(name="my_model_runner", raise_exception=True)
    model_runner_step.add_model(
        model_class="DictOutputModel",
        execution_mechanism="naive",
        endpoint_name="dict_model",
        input_path="inputs",
        result_path="outputs",
        outputs=["o1", "o2", "o3", "o4"],
        raise_error=False,
    )
    graph.to(model_runner_step).respond()

    function.set_tracking("dummy://", enable_tracking=True)
    server = function.to_mock_server()
    server.test("/", {"inputs": {"f1": 1, "f2": 2, "f3": 3, "f4": 4}})
    server.wait_for_completion()

    dummy_stream = server.context.stream.output_stream
    assert len(dummy_stream.event_list) == 1, "expected stream to get one message"
    assert dummy_stream.event_list[0].get("resp", {}).get("outputs") == [[2, 3, 4, 5]]
    assert dummy_stream.event_list[0].get("request", {}).get("inputs") == [[1, 2, 3, 4]]


def test_tracked_model_runner_multiple_steps():
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

    function.set_tracking(
        "dummy://",
    )
    server = function.to_mock_server()
    server.test("/", {"n": 1})
    server.wait_for_completion()

    dummy_stream = server.context.stream.output_stream

    assert len(dummy_stream.event_list) == 2, "expected stream to get two messages"
    assert dummy_stream.event_list[0].get("resp", {}).get("outputs") == [2]
    assert dummy_stream.event_list[0].get("request", {}).get("inputs") == [1]
    assert dummy_stream.event_list[1].get("resp", {}).get("outputs") == [3]
    assert dummy_stream.event_list[1].get("request", {}).get("inputs") == [1]


def test_tracked_model_runner_multiple_models():
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
    function.set_tracking(stream_args={"mock": True})

    function.set_tracking(
        "dummy://",
    )
    server = function.to_mock_server()
    server.test("/", {"n": 1})
    server.wait_for_completion()

    dummy_stream = server.context.stream.output_stream

    assert len(dummy_stream.event_list) == 8, "expected stream to get eight messages"
    output_models = [event["model"] for event in dummy_stream.event_list]
    models.sort()
    output_models.sort()
    assert output_models == models, "expected models to be the same"
    _test_graph_structure(server.graph, True)


def test_set_untracked_with_model_runner():
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
    function.set_tracking(stream_args={"mock": True})

    function.set_tracking("dummy://", enable_tracking=True)
    server = function.to_mock_server()
    server.test("/", {"n": 1})
    server.wait_for_completion()

    dummy_stream = server.context.stream.output_stream
    _test_graph_structure(server.graph, True)
    assert len(dummy_stream.event_list) == 1, "expected stream to get one message"
    function.set_tracking("dummy://", enable_tracking=False)
    _test_graph_structure(graph, False)
    server = function.to_mock_server()
    server.test("/", {"n": 1})
    server.wait_for_completion()
    assert (
        len(dummy_stream.event_list) == 1
    ), "expected stream to still have single message"


def test_tracked_multiple_to_mock_with_model_runner():
    function = mlrun.new_function("tests-1", kind="serving")
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(
        name="my_model_runner", raise_exception=True, model_selector="MyModelSelector"
    )
    model_runner_step.add_model(
        model_class="DictOutputModel",
        execution_mechanism="naive",
        endpoint_name="my_dict_model",
        input_path="inputs",
        result_path="outputs",
        outputs=["o1", "o2", "o3", "o4"],
        raise_error=False,
    )
    graph.to(model_runner_step).respond()

    function.set_tracking("dummy://", enable_tracking=True)
    server = function.to_mock_server()
    server.wait_for_completion()
    model_runner_step_1 = ModelRunnerStep(
        name="my_model_runner_1", raise_exception=True
    )
    model_runner_step_1.add_model(
        model_class="DictOutputModel",
        execution_mechanism="naive",
        endpoint_name="my_dict_model_1",
        input_path="inputs",
        result_path="outputs",
        outputs=["o1", "o2", "o3", "o4"],
        raise_error=False,
    )
    graph.to(model_runner_step_1)
    server = function.to_mock_server()
    server.test("/", {"inputs": {"f1": 1, "f2": 2, "f3": 3, "f4": 4}})
    server.wait_for_completion()
    dummy_stream = server.context.stream.output_stream
    assert len(dummy_stream.event_list) == 2, "expected stream to get one message"


@pytest.mark.parametrize("sampling_percentage", [100.0, 50.0, 20.0])
def test_sampling_model_runner(sampling_percentage: float):
    function = mlrun.new_function("tests-sampling", kind="serving")
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(name="my_model_runner", raise_exception=True)
    model_runner_step.add_model(
        model_class="DictOutputModel",
        execution_mechanism="naive",
        endpoint_name="dict_model_1",
        input_path="inputs",
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
                "f1": [1, 4, 8, 12] * 1000,
                "f2": [2, 5, 9, 13] * 1000,
                "f3": [3, 6, 10, 14] * 1000,
                "f4": [4, 7, 11, 15] * 1000,
            }
        },
    )
    server.wait_for_completion()

    dummy_stream = server.context.stream.output_stream

    _test_graph_structure(server.graph, True)

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
