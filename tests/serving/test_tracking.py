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
import json
from pprint import pprint
from unittest.mock import patch

import numpy as np
import pytest

import mlrun
from mlrun.common.schemas import ModelMonitoringMode
from tests.serving.test_serving import _log_model

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


def test_tracking():
    # test that predict() was tracked properly in the stream
    fn = mlrun.new_function("tests", kind="serving")
    fn.add_model("my", ".", class_name=ModelTestingClass(multiplier=2))
    fn.set_tracking("v3io://fake", stream_args={"mock": True, "access_key": "x"})

    server = fn.to_mock_server()
    server.test("/v2/models/my/infer", testdata)

    fake_stream = server.context.stream.output_stream._mock_queue
    assert len(fake_stream) == 1
    assert rec_to_data(fake_stream[0]) == ("my", "ModelTestingClass", [[5, 6]], [10])


def test_custom_tracking():
    # test custom values tracking (using the logged_results() hook)
    fn = mlrun.new_function("tests", kind="serving")
    fn.add_model("my", ".", class_name=ModelTestingCustomTrack(multiplier=2))
    fn.set_tracking("v3io://fake", stream_args={"mock": True, "access_key": "x"})

    server = fn.to_mock_server()
    server.test("/v2/models/my/infer", testdata)

    fake_stream = server.context.stream.output_stream._mock_queue
    assert len(fake_stream) == 1
    assert rec_to_data(fake_stream[0]) == ("my", "ModelTestingCustomTrack", [[1]], [2])


def test_ensemble_tracking():
    # test proper tracking of an ensemble (router + models are logged)
    fn = mlrun.new_function("tests", kind="serving")
    fn.set_topology("router", mlrun.serving.VotingEnsemble(vote_type="regression"))
    fn.add_model("1", ".", class_name=ModelTestingClass(multiplier=2))
    fn.add_model("2", ".", class_name=ModelTestingClass(multiplier=3))
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
    pprint(results)

    assert results == {
        "1": ["ModelTestingClass", [[5, 6]], [10]],
        "2": ["ModelTestingClass", [[5, 6]], [15]],
        "VotingEnsemble": ["VotingEnsemble", [[5, 6]], [12.5]],
    }


@pytest.mark.parametrize("enable_tracking", [True, False])
def test_tracked_function(rundb_mock, enable_tracking):
    with patch("mlrun.get_run_db", return_value=rundb_mock):
        project = mlrun.new_project("test-pro", save=False)
        fn = mlrun.new_function("test-fn", kind="serving")
        model_uri = _log_model(project)
        print(model_uri)
        fn.add_model("m1", model_uri, "ModelTestingClass", multiplier=5)
        fn.set_tracking("dummy://", enable_tracking=enable_tracking)
        server = fn.to_mock_server(track_models=True)
        server.test("/v2/models/m1/infer", testdata)

        dummy_stream = server.context.stream.output_stream
        assert len(dummy_stream.event_list) == 1, "expected stream to get one message"

    rundb_mock.patch_model_endpoint.assert_called_once()
    assert (
        rundb_mock.patch_model_endpoint.call_args.kwargs["attributes"]["model_uri"]
        == model_uri
    ), "model_uri attribute of the model endpoint was not updated as expected"
    if not enable_tracking:
        assert (
            rundb_mock.patch_model_endpoint.call_args.kwargs["attributes"][
                "monitoring_mode"
            ]
            == ModelMonitoringMode.disabled
        ), "model_uri attribute of the model endpoint was not updated as expected"


def rec_to_data(rec):
    data = json.loads(rec["data"])
    inputs = data["request"]["inputs"]
    outputs = data["resp"]["outputs"]
    return data["model"], data["class"], inputs, outputs
