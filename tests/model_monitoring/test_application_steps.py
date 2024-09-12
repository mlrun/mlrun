# Copyright 2024 Iguazio
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
import logging
import typing
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

import mlrun
import mlrun.artifacts
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.model_monitoring.applications.context as mm_context
import mlrun.serving.states
from mlrun.common.schemas.model_monitoring import ResultData
from mlrun.model_monitoring.applications import (
    ModelMonitoringApplicationMetric,
    ModelMonitoringApplicationResult,
)
from mlrun.model_monitoring.applications._application_steps import (
    _PrepareMonitoringEvent,  # noqa: F401
    _PushToMonitoringWriter,
)
from mlrun.utils import Logger, logger

STREAM_PATH = "./test_stream.json"


class TestEventPreparation:
    ENDPOINT_ID = "test-ep-id"
    APPLICATION_NAME = "test-app"

    @staticmethod
    @pytest.fixture
    def mock_load_project(tmp_path: Path) -> typing.Iterator[None]:
        with patch(
            "mlrun.load_project",
            Mock(
                return_value=mlrun.projects.MlrunProject(
                    spec=mlrun.projects.ProjectSpec(artifact_path=str(tmp_path))
                )
            ),
        ):
            yield

    @classmethod
    @pytest.fixture
    def controller_event(cls) -> dict[str, typing.Any]:
        return {
            mm_constants.ApplicationEvent.ENDPOINT_ID: cls.ENDPOINT_ID,
            mm_constants.ApplicationEvent.APPLICATION_NAME: cls.APPLICATION_NAME,
        }

    @classmethod
    @pytest.mark.usefixtures("mock_load_project")
    @patch("mlrun.model_monitoring.applications.context.get_endpoint_record")
    def test_prepare_monitoring_event(
        cls, mock_get_endpoint_record: Mock, controller_event: dict[str, typing.Any]
    ) -> None:
        logger.info("Set up a mock server with a `_PrepareMonitoringEvent` step")

        fn = typing.cast(
            mlrun.runtimes.ServingRuntime,
            mlrun.code_to_function(
                filename=__file__,
                name="model-monitoring-context-preparation",
                kind=mlrun.run.RuntimeKinds.serving,
            ),
        )
        graph = fn.set_topology(mlrun.serving.states.StepKinds.flow)

        graph.to(
            "_PrepareMonitoringEvent", application_name=cls.APPLICATION_NAME
        ).respond()
        server = fn.to_mock_server()
        monitoring_context = typing.cast(
            mm_context.MonitoringApplicationContext,
            server.test(body=controller_event),
        )

        logger.info("Test `monitoring_context` functionality")

        monitoring_context.logger.debug("Checking `get_endpoint_record` was called")
        mock_get_endpoint_record.assert_called_once()

        monitoring_context.logger.debug("Logging an artifact")
        artifact = monitoring_context.log_artifact(
            "my-app-data",
            body=b"Sometimes, context is important.",
            format="txt",
            labels={"framework": "deepeval"},
        )

        monitoring_context.logger.debug("Checking logged artifact labels")
        assert {
            "framework": "deepeval",
            "mlrun/producer-type": "model-monitoring-app",
            "mlrun/app-name": cls.APPLICATION_NAME,
            "mlrun/endpoint-id": cls.ENDPOINT_ID,
        }.items() <= artifact.labels.items()

        server.wait_for_completion()
        monitoring_context.logger.debug("I'm done")


class Pusher:
    def __init__(self, stream_uri):
        self.stream_uri = stream_uri

    def push(self, data: list[dict[str, typing.Any]]):
        data = data[0]
        with open(self.stream_uri, "w") as json_file:
            json.dump(data, json_file)
            json_file.write("\n")


@pytest.fixture
def push_to_monitoring_writer():
    return _PushToMonitoringWriter(
        project="demo-project",
        writer_application_name=mm_constants.MonitoringFunctionNames.WRITER,
        name="PushToMonitoringWriter",
        stream_uri="./test_stream.json",
    )


@pytest.fixture
def monitoring_context() -> Mock:
    mock_monitoring_context = Mock(spec=mm_context.MonitoringApplicationContext)
    mock_monitoring_context.log_stream = Logger(
        name="test_data_drift_app", level=logging.DEBUG
    )
    mock_monitoring_context._artifacts_manager = Mock(
        spec=mlrun.artifacts.manager.ArtifactManager
    )
    mock_monitoring_context.application_name = "test_data_drift_app"
    mock_monitoring_context.endpoint_id = "test_endpoint_id"
    mock_monitoring_context.start_infer_time = pd.Timestamp(
        "2022-01-01 00:00:00.000000"
    )
    mock_monitoring_context.end_infer_time = pd.Timestamp("2022-01-01 00:00:00.000000")
    mock_monitoring_context.sample_df_stats = {}
    return mock_monitoring_context


@patch("mlrun.datastore.get_stream_pusher")
def test_push_result_to_monitoring_writer_stream(
    mock_get_stream_pusher,
    push_to_monitoring_writer: _PushToMonitoringWriter,
    monitoring_context: Mock,
    tmp_path: Path,
):
    """
    Test that the `_PushToMonitoringWriter` step pushes the results to the monitoring writer stream. In addition,
    test that the extra data is not pushed to the stream if it exceeds the maximum size of 998 characters.
    """
    mock_get_stream_pusher.return_value = Pusher(stream_uri=f"{tmp_path}/{STREAM_PATH}")
    results = [
        ModelMonitoringApplicationResult(
            name="res1",
            value=1,
            status=mm_constants.ResultStatusApp.detected,
            extra_data={"extra_data": "extra_data"},
            kind=mm_constants.ResultKindApp.data_drift,
        ),
        ModelMonitoringApplicationResult(
            name="res2",
            value=2,
            status=mm_constants.ResultStatusApp.detected,
            extra_data={"extra_data": "extra_data" * 1000},
            kind=mm_constants.ResultKindApp.data_drift,
        ),
        ModelMonitoringApplicationMetric(
            name="met",
            value=2,
        ),
    ]

    for result in results:
        push_to_monitoring_writer.do(
            (
                [result],
                monitoring_context,
            )
        )

        with open(f"{tmp_path}/{STREAM_PATH}") as json_file:
            for i, line in enumerate(json_file):
                loaded_data = json.loads(line.strip())
            if isinstance(result, ModelMonitoringApplicationResult):
                event_kind = mm_constants.WriterEventKind.RESULT
                result = result.to_dict()
                data_from_file = json.loads(loaded_data["data"])

                result[ResultData.CURRENT_STATS] = "{}"
                if len(result["result_extra_data"]) <= 998:
                    assert (
                        data_from_file[ResultData.RESULT_EXTRA_DATA]
                        == result[ResultData.RESULT_EXTRA_DATA]
                    )
                else:
                    assert (
                        data_from_file[ResultData.RESULT_EXTRA_DATA]
                        != result[ResultData.RESULT_EXTRA_DATA]
                    )
                    result["extra_data"] = "{}"
            else:
                event_kind = mm_constants.WriterEventKind.METRIC
                result = result.to_dict()

            assert loaded_data == {
                "application_name": "test_data_drift_app",
                "endpoint_id": "test_endpoint_id",
                "start_infer_time": "2022-01-01 00:00:00.000000",
                "end_infer_time": "2022-01-01 00:00:00.000000",
                "event_kind": event_kind,
                "data": json.dumps(result),
            }
