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
import json
import os
from unittest.mock import Mock

import pytest
import semver
import v3io_frames.client

import mlrun.common.schemas.model_monitoring as mm_constants
import mlrun.model_monitoring
import mlrun.model_monitoring.db.tsdb.v3io
from mlrun.model_monitoring.writer import (
    MetricData,
    ModelMonitoringWriter,
    ResultData,
    WriterEvent,
    _AppResultEvent,
    _Notifier,
    _RawEvent,
    _WriterEventTypeError,
    _WriterEventValueError,
)
from mlrun.utils.notifications.notification_pusher import CustomNotificationPusher

TEST_PROJECT = "test-application-results"
V3IO_TABLE_CONTAINER = f"bigdata/{TEST_PROJECT}"


@pytest.fixture(params=[(0, "1.7.0", "result")])
def event(request: pytest.FixtureRequest) -> _AppResultEvent:
    now = datetime.datetime.now()
    start_infer_time = now - datetime.timedelta(minutes=5)
    if semver.Version.parse("1.7.0") > semver.Version.parse(request.param[1]):
        return _AppResultEvent(
            {
                WriterEvent.ENDPOINT_ID: "some-ep-id",
                WriterEvent.START_INFER_TIME: start_infer_time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                WriterEvent.END_INFER_TIME: now.strftime("%Y-%m-%d %H:%M:%S"),
                WriterEvent.APPLICATION_NAME: "dummy-app",
                ResultData.RESULT_NAME: "data-drift-0",
                ResultData.RESULT_KIND: 0,
                ResultData.RESULT_VALUE: 0.32,
                ResultData.RESULT_STATUS: request.param[0],
                ResultData.RESULT_EXTRA_DATA: "",
                ResultData.CURRENT_STATS: "",
            }
        )
    elif request.param[2] == "result":
        return _AppResultEvent(
            {
                WriterEvent.ENDPOINT_ID: "some-ep-id",
                WriterEvent.START_INFER_TIME: start_infer_time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                WriterEvent.END_INFER_TIME: now.strftime("%Y-%m-%d %H:%M:%S"),
                WriterEvent.APPLICATION_NAME: "dummy-app",
                WriterEvent.EVENT_KIND: request.param[2],
                WriterEvent.DATA: json.dumps(
                    {
                        ResultData.RESULT_NAME: "data-drift-0",
                        ResultData.RESULT_KIND: 0,
                        ResultData.RESULT_VALUE: 0.32,
                        ResultData.RESULT_STATUS: request.param[0],
                        ResultData.RESULT_EXTRA_DATA: "",
                        ResultData.CURRENT_STATS: "",
                    }
                ),
            }
        )
    elif request.param[2] == "metric":
        return _AppResultEvent(
            {
                WriterEvent.ENDPOINT_ID: "some-ep-id",
                WriterEvent.START_INFER_TIME: start_infer_time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                WriterEvent.END_INFER_TIME: now.strftime("%Y-%m-%d %H:%M:%S"),
                WriterEvent.APPLICATION_NAME: "dummy-app",
                WriterEvent.EVENT_KIND: request.param[2],
                WriterEvent.DATA: json.dumps(
                    {
                        MetricData.METRIC_NAME: "metric_1",
                        MetricData.METRIC_VALUE: 0.32,
                    }
                ),
            }
        )


@pytest.fixture
def notification_pusher() -> CustomNotificationPusher:
    return Mock(spec=CustomNotificationPusher)


@pytest.mark.parametrize(
    ("event", "exception"),
    [
        ("key1:val1,key2:val2", _WriterEventTypeError),
        ({WriterEvent.ENDPOINT_ID: "ep2211"}, _WriterEventValueError),
        ({WriterEvent.EVENT_KIND: "special"}, _WriterEventValueError),
    ],
)
def test_reconstruct_event_error(event: _RawEvent, exception: type[Exception]) -> None:
    with pytest.raises(exception):
        ModelMonitoringWriter._reconstruct_event(event)


class TestTSDB:
    """
    In the following test, we will test the TSDB writer functionality with a V3IO TSDB connector.
    It includes:
    - Writing an application result event to the TSDB.
    - Verifying that we don't write the extra data to the TSDB.
    - Verifying that the written record includes all the expected columns.
    - Deleting the resources from the TSDB.
    """

    @staticmethod
    @pytest.fixture
    def tsdb_connector() -> mlrun.model_monitoring.db.tsdb.v3io.V3IOTSDBConnector:
        tsdb_connector = mlrun.model_monitoring.db.tsdb.v3io.V3IOTSDBConnector(
            project=TEST_PROJECT
        )

        # Generate dummy tables for the test
        tsdb_connector._frames_client = v3io_frames.client.ClientBase = (
            tsdb_connector._get_v3io_frames_client(V3IO_TABLE_CONTAINER)
        )
        tsdb_connector.tables = {
            mm_constants.V3IOTSDBTables.APP_RESULTS: mm_constants.V3IOTSDBTables.APP_RESULTS,
            mm_constants.V3IOTSDBTables.METRICS: mm_constants.V3IOTSDBTables.METRICS,
        }
        tsdb_connector.create_tsdb_application_tables()

        return tsdb_connector

    @staticmethod
    @pytest.fixture
    def writer(
        tsdb_connector: tsdb_connector,
    ) -> ModelMonitoringWriter:
        writer = Mock(spec=ModelMonitoringWriter)
        writer.project = TEST_PROJECT
        writer._tsdb_connector = tsdb_connector
        return writer

    @staticmethod
    @pytest.mark.parametrize(
        ("event"),
        [(0, "1.6.0", "result"), (0, "1.7.0", "result")],
        indirect=["event"],
    )
    @pytest.mark.skipif(
        os.getenv("V3IO_FRAMESD") is None or os.getenv("V3IO_ACCESS_KEY") is None,
        reason="Configure Framsed to access V3IO store targets",
    )
    def test_tsdb_writer(
        event: _AppResultEvent,
        writer: ModelMonitoringWriter,
    ) -> None:
        event, kind = ModelMonitoringWriter._reconstruct_event(event)
        writer._tsdb_connector.write_application_result(event=event.copy(), kind=kind)
        record_from_tsdb = writer._tsdb_connector.get_records(
            table=mm_constants.V3IOTSDBTables.APP_RESULTS,
            filter_query=f"endpoint_id=='{event[WriterEvent.ENDPOINT_ID]}'",
            start="now-1d",
            end="now+1d",
        )

        actual_columns = list(record_from_tsdb.columns)
        assert (
            ResultData.RESULT_EXTRA_DATA not in actual_columns
        ), "The extra data should not be written to the TSDB"

        expected_columns = WriterEvent.list() + ResultData.list()
        expected_columns.remove(ResultData.RESULT_EXTRA_DATA)
        expected_columns.remove(WriterEvent.END_INFER_TIME)
        expected_columns.remove(WriterEvent.DATA)
        expected_columns.remove(WriterEvent.EVENT_KIND)

        # Assert that the record includes all the expected columns
        assert sorted(expected_columns) == sorted(actual_columns)

        # Cleanup the resources and verify that the data was deleted
        writer._tsdb_connector.delete_tsdb_resources()

        with pytest.raises(v3io_frames.errors.ReadError):
            writer._tsdb_connector.get_records(
                table=mm_constants.V3IOTSDBTables.APP_RESULTS,
                filter_query=f"endpoint_id=='{event[WriterEvent.ENDPOINT_ID]}'",
                start="now-1d",
                end="now+1d",
            )

    @staticmethod
    @pytest.mark.parametrize(
        ("event", "expected_notification_call"),
        [
            ((2, "1.6.0", "result"), True),
            ((1, "1.6.0", "result"), False),
            ((0, "1.6.0", "result"), False),
            ((2, "1.7.0", "result"), True),
            ((1, "1.7.0", "result"), False),
            ((0, "1.7.0", "result"), False),
        ],
        indirect=["event"],
    )
    def test_notifier(
        event: _AppResultEvent,
        expected_notification_call: bool,
        notification_pusher: Mock,
    ) -> None:
        event, kind = ModelMonitoringWriter._reconstruct_event(event)
        _Notifier(event=event, notification_pusher=notification_pusher).notify()
        assert notification_pusher.push.call_count == expected_notification_call

    @staticmethod
    @pytest.mark.parametrize(
        ("event"),
        [(0, "1.7.0", "metric")],
        indirect=["event"],
    )
    def test_metric_write(
        event: _AppResultEvent,
        writer: ModelMonitoringWriter,
    ) -> None:
        event, kind = ModelMonitoringWriter._reconstruct_event(event)
        writer._tsdb_connector.write_application_result(event, kind)
