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
import os
from functools import partial
from unittest.mock import Mock, patch

import pytest
import v3io_frames.client
from _pytest.fixtures import FixtureRequest

import mlrun.common.schemas.model_monitoring as mm_constants
import mlrun.model_monitoring
import mlrun.model_monitoring.db.tsdb.v3io
from mlrun.model_monitoring.writer import (
    ModelMonitoringWriter,
    WriterEvent,
    _AppResultEvent,
    _Notifier,
    _RawEvent,
    _WriterEventTypeError,
    _WriterEventValueError,
)
from mlrun.utils.notifications.notification_pusher import CustomNotificationPusher

TEST_PROJECT = "test-application-results-v2"
V3IO_TABLE_CONTAINER = f"bigdata/{TEST_PROJECT}"


@pytest.fixture(params=[0])
def event(request: FixtureRequest) -> _AppResultEvent:
    now = datetime.datetime.now()
    start_infer_time = now - datetime.timedelta(minutes=5)
    return _AppResultEvent(
        {
            WriterEvent.ENDPOINT_ID: "some-ep-id",
            WriterEvent.START_INFER_TIME: start_infer_time.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            WriterEvent.END_INFER_TIME: now.strftime("%Y-%m-%d %H:%M:%S"),
            WriterEvent.APPLICATION_NAME: "dummy-app",
            WriterEvent.RESULT_NAME: "data-drift-0",
            WriterEvent.RESULT_KIND: 0,
            WriterEvent.RESULT_VALUE: 0.32,
            WriterEvent.RESULT_STATUS: request.param,
            WriterEvent.RESULT_EXTRA_DATA: "",
            WriterEvent.CURRENT_STATS: "",
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
    ],
)
def test_reconstruct_event_error(event: _RawEvent, exception: type[Exception]) -> None:
    with pytest.raises(exception):
        ModelMonitoringWriter._reconstruct_event(event)


@pytest.mark.parametrize(
    ("event", "expected_notification_call"),
    [(2, True), (1, False), (0, False)],
    indirect=["event"],
)
def test_notifier(
    event: _AppResultEvent,
    expected_notification_call: bool,
    notification_pusher: Mock,
) -> None:
    _Notifier(event=event, notification_pusher=notification_pusher).notify()
    assert notification_pusher.push.call_count == expected_notification_call


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
    def writer() -> ModelMonitoringWriter:
        writer = Mock(spec=ModelMonitoringWriter)
        writer._update_tsdb = partial(ModelMonitoringWriter._update_tsdb, writer)
        writer.project = TEST_PROJECT
        return writer

    @staticmethod
    @pytest.mark.skipif(
        os.getenv("V3IO_FRAMESD") is None or os.getenv("V3IO_ACCESS_KEY") is None,
        reason="Configure Framsed to access V3IO store targets",
    )
    def test_tsdb_writer(
        event: _AppResultEvent,
        writer: ModelMonitoringWriter,
        tsdb_connector: tsdb_connector,
    ) -> None:
        with patch(
            "mlrun.model_monitoring.get_tsdb_connector", return_value=tsdb_connector
        ):
            writer._update_tsdb(event)

            # Compare stored TSDB record and provided event
            record_from_tsdb = tsdb_connector.get_records(
                table=mm_constants.V3IOTSDBTables.APP_RESULTS,
                filter_query=f"endpoint_id=='{event[WriterEvent.ENDPOINT_ID]}'",
                start="now-1d",
                end="now+1d",
            )
            actual_columns = list(record_from_tsdb.columns)

        assert (
            WriterEvent.RESULT_EXTRA_DATA not in actual_columns
        ), "The extra data should not be written to the TSDB"

        expected_columns = WriterEvent.list()
        expected_columns.remove(WriterEvent.RESULT_EXTRA_DATA)
        expected_columns.remove(WriterEvent.END_INFER_TIME)

        # Assert that the record includes all the expected columns
        assert sorted(expected_columns) == sorted(actual_columns)

        # Cleanup the resources and verify that the data was deleted
        tsdb_connector.delete_tsdb_resources()

        with pytest.raises(v3io_frames.errors.ReadError):
            tsdb_connector.get_records(
                table=mm_constants.V3IOTSDBTables.APP_RESULTS,
                filter_query=f"endpoint_id=='{event[WriterEvent.ENDPOINT_ID]}'",
                start="now-1d",
                end="now+1d",
            )
