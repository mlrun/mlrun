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

import mlrun.artifacts
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.model_monitoring.applications.context as mm_context
from mlrun.model_monitoring.applications import (
    ModelMonitoringApplicationMetric,
    ModelMonitoringApplicationResult,
)
from mlrun.model_monitoring.applications._application_steps import (
    _PushToMonitoringWriter,
)
from mlrun.utils import Logger

STREAM_PATH = "./test_stream.json"


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
    mock_get_stream_pusher.return_value = Pusher(stream_uri=f"{tmp_path}/{STREAM_PATH}")
    results = [
        ModelMonitoringApplicationResult(
            name="res1",
            value=1,
            status=mm_constants.ResultStatusApp.detected,
            extra_data={},
            kind=mm_constants.ResultKindApp.data_drift,
        ),
        ModelMonitoringApplicationResult(
            name="res2",
            value=2,
            status=mm_constants.ResultStatusApp.detected,
            extra_data={},
            kind=mm_constants.ResultKindApp.data_drift,
        ),
        ModelMonitoringApplicationMetric(
            name="met",
            value=2,
        ),
    ]
    push_to_monitoring_writer.do(
        (
            results,
            monitoring_context,
        )
    )
    with open(f"{tmp_path}/{STREAM_PATH}") as json_file:
        for i, line in enumerate(json_file):
            loaded_data = json.loads(line.strip())
            result = results[2 - i].to_dict()
            if isinstance(results[2 - i], ModelMonitoringApplicationResult):
                event_kind = mm_constants.WriterEventKind.RESULT
                result["current_stats"] = {}
            else:
                event_kind = mm_constants.WriterEventKind.METRIC
            assert loaded_data == {
                "application_name": "test_data_drift_app",
                "endpoint_id": "test_endpoint_id",
                "start_infer_time": "2022-01-01 00:00:00.000000",
                "end_infer_time": "2022-01-01 00:00:00.000000",
                "event_kind": event_kind,
                "data": json.dumps(result),
            }
