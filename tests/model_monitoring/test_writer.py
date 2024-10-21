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
from collections.abc import Iterator
from unittest.mock import Mock, patch

import pytest
import semver
import v3io.dataplane.kv
import v3io_frames.client

import mlrun.common.schemas.model_monitoring as mm_schemas
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
from mlrun.utils.v3io_clients import V3IOClient

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
    else:
        raise ValueError


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


class TestHistogramGeneralDriftResultEvent:
    @staticmethod
    @pytest.fixture
    def mock_v3io_client() -> V3IOClient:
        mock_client = Mock(spec=V3IOClient)
        mock_client.kv = Mock(spec=v3io.dataplane.kv.Model)
        return mock_client

    @staticmethod
    @pytest.fixture
    def general_drift_event() -> _RawEvent:
        return _RawEvent(
            {
                "application_name": "histogram-data-drift",
                "end_infer_time": "2024-05-12 14:53:26.000000+00:00",
                "endpoint_id": "fc96fcd22935343586ce32c5d4614f5dca816405",
                "current_stats": {},  # Don't rely on current stats
                "result_extra_data": '{"current_stats": "{\\"sepal_length_cm\\": {\\"count\\": 150.0, \\"mean\\": 5.843333333333334, \\"std\\": 0.828066127977863, \\"min\\": 4.3, \\"25%\\": 5.1, \\"50%\\": 5.8, \\"75%\\": 6.4, \\"max\\": 7.9, \\"hist\\": [[0, 4, 5, 7, 16, 9, 5, 13, 14, 10, 6, 10, 16, 7, 11, 4, 2, 4, 1, 5, 1, 0], [-1.7976931348623157e+308, 4.3, 4.4799999999999995, 4.66, 4.84, 5.02, 5.2, 5.38, 5.5600000000000005, 5.74, 5.92, 6.1, 6.28, 6.46, 6.640000000000001, 6.82, 7.0, 7.18, 7.36, 7.54, 7.720000000000001, 7.9, 1.7976931348623157e+308]]}, \\"sepal_width_cm\\": {\\"count\\": 150.0, \\"mean\\": 3.0573333333333337, \\"std\\": 0.4358662849366982, \\"min\\": 2.0, \\"25%\\": 2.8, \\"50%\\": 3.0, \\"75%\\": 3.3, \\"max\\": 4.4, \\"hist\\": [[0, 1, 3, 4, 3, 8, 14, 14, 10, 26, 11, 19, 12, 6, 4, 9, 2, 1, 1, 1, 1, 0], [-1.7976931348623157e+308, 2.0, 2.12, 2.24, 2.3600000000000003, 2.48, 2.6, 2.72, 2.8400000000000003, 2.96, 3.08, 3.2, 3.3200000000000003, 3.4400000000000004, 3.5600000000000005, 3.6800000000000006, 3.8000000000000003, 3.9200000000000004, 4.040000000000001, 4.16, 4.28, 4.4, 1.7976931348623157e+308]]}, \\"petal_length_cm\\": {\\"count\\": 150.0, \\"mean\\": 3.7580000000000005, \\"std\\": 1.7652982332594662, \\"min\\": 1.0, \\"25%\\": 1.6, \\"50%\\": 4.35, \\"75%\\": 5.1, \\"max\\": 6.9, \\"hist\\": [[0, 4, 33, 11, 2, 0, 0, 1, 2, 3, 5, 12, 14, 12, 17, 6, 12, 7, 4, 2, 3, 0], [-1.7976931348623157e+308, 1.0, 1.295, 1.59, 1.8850000000000002, 2.18, 2.475, 2.7700000000000005, 3.0650000000000004, 3.3600000000000003, 3.6550000000000002, 3.95, 4.245000000000001, 4.540000000000001, 4.835000000000001, 5.130000000000001, 5.425000000000001, 5.720000000000001, 6.015000000000001, 6.3100000000000005, 6.605, 6.9, 1.7976931348623157e+308]]}, \\"petal_width_cm\\": {\\"count\\": 150.0, \\"mean\\": 1.1993333333333336, \\"std\\": 0.7622376689603465, \\"min\\": 0.1, \\"25%\\": 0.3, \\"50%\\": 1.3, \\"75%\\": 1.8, \\"max\\": 2.5, \\"hist\\": [[0, 34, 7, 7, 1, 1, 0, 0, 7, 3, 5, 21, 12, 4, 2, 12, 11, 6, 3, 8, 6, 0], [-1.7976931348623157e+308, 0.1, 0.22, 0.33999999999999997, 0.45999999999999996, 0.58, 0.7, 0.82, 0.94, 1.06, 1.1800000000000002, 1.3, 1.42, 1.54, 1.6600000000000001, 1.78, 1.9, 2.02, 2.14, 2.2600000000000002, 2.38, 2.5, 1.7976931348623157e+308]]}}", "drift_measures": "{\\"sepal_length_cm\\":{\\"hellinger\\":0.0,\\"kld\\":0.0,\\"tvd\\":0.0},\\"sepal_width_cm\\":{\\"hellinger\\":0.0,\\"kld\\":0.0,\\"tvd\\":0.0},\\"petal_length_cm\\":{\\"hellinger\\":0.0000000105,\\"kld\\":0.0,\\"tvd\\":0.0},\\"petal_width_cm\\":{\\"hellinger\\":0.0,\\"kld\\":0.0,\\"tvd\\":0.0}}", "drift_status": 0}',  # noqa: E501
                "result_kind": 0,
                "result_name": "general_drift",
                "result_status": 0,
                "result_value": 1.3170890159654386e-9,
                "start_infer_time": "2024-05-12 14:52:26.000000+00:00",
            }
        )

    @staticmethod
    @pytest.fixture(autouse=True)
    def writer(mock_v3io_client: V3IOClient) -> Iterator[None]:
        with patch(
            "mlrun.utils.v3io_clients.get_v3io_client",
            Mock(return_value=mock_v3io_client),
        ):
            with patch(
                "mlrun.model_monitoring.get_store_object",
                return_value=mlrun.model_monitoring.get_store_object(
                    store_connection_string="v3io", project=TEST_PROJECT
                ),
            ):
                with patch(
                    "mlrun.model_monitoring.get_tsdb_connector",
                    return_value=Mock(
                        spec=mlrun.model_monitoring.db.tsdb.v3io.V3IOTSDBConnector
                    ),
                ):
                    writer = ModelMonitoringWriter(project=TEST_PROJECT)
                    writer._app_result_store = mlrun.model_monitoring.get_store_object(
                        store_connection_string="v3io", project=TEST_PROJECT
                    )
                    yield writer

    @staticmethod
    def test_update_model_endpoint(
        general_drift_event: _RawEvent,
        mock_v3io_client: V3IOClient,
        writer: ModelMonitoringWriter,
    ) -> None:
        writer.do(event=general_drift_event)
        update_mock = mock_v3io_client.kv.update
        assert update_mock.call_count == 2, (
            "Expects two update calls - one for the results KV, "
            "and one for the model endpoint"
        )
        update_mock.assert_called_with(
            container="users",
            table_path="pipelines/test-application-results/model-endpoints/endpoints/",
            key="fc96fcd22935343586ce32c5d4614f5dca816405",
            attributes={
                "current_stats": b'{"sepal_length_cm": {"count": 150.0, "mean": 5.843333333333334, "std": 0.828066127977863, "min": 4.3, "25%": 5.1, "50%": 5.8, "75%": 6.4, "max": 7.9, "hist": [[0, 4, 5, 7, 16, 9, 5, 13, 14, 10, 6, 10, 16, 7, 11, 4, 2, 4, 1, 5, 1, 0], [-1.7976931348623157e+308, 4.3, 4.4799999999999995, 4.66, 4.84, 5.02, 5.2, 5.38, 5.5600000000000005, 5.74, 5.92, 6.1, 6.28, 6.46, 6.640000000000001, 6.82, 7.0, 7.18, 7.36, 7.54, 7.720000000000001, 7.9, 1.7976931348623157e+308]]}, "sepal_width_cm": {"count": 150.0, "mean": 3.0573333333333337, "std": 0.4358662849366982, "min": 2.0, "25%": 2.8, "50%": 3.0, "75%": 3.3, "max": 4.4, "hist": [[0, 1, 3, 4, 3, 8, 14, 14, 10, 26, 11, 19, 12, 6, 4, 9, 2, 1, 1, 1, 1, 0], [-1.7976931348623157e+308, 2.0, 2.12, 2.24, 2.3600000000000003, 2.48, 2.6, 2.72, 2.8400000000000003, 2.96, 3.08, 3.2, 3.3200000000000003, 3.4400000000000004, 3.5600000000000005, 3.6800000000000006, 3.8000000000000003, 3.9200000000000004, 4.040000000000001, 4.16, 4.28, 4.4, 1.7976931348623157e+308]]}, "petal_length_cm": {"count": 150.0, "mean": 3.7580000000000005, "std": 1.7652982332594662, "min": 1.0, "25%": 1.6, "50%": 4.35, "75%": 5.1, "max": 6.9, "hist": [[0, 4, 33, 11, 2, 0, 0, 1, 2, 3, 5, 12, 14, 12, 17, 6, 12, 7, 4, 2, 3, 0], [-1.7976931348623157e+308, 1.0, 1.295, 1.59, 1.8850000000000002, 2.18, 2.475, 2.7700000000000005, 3.0650000000000004, 3.3600000000000003, 3.6550000000000002, 3.95, 4.245000000000001, 4.540000000000001, 4.835000000000001, 5.130000000000001, 5.425000000000001, 5.720000000000001, 6.015000000000001, 6.3100000000000005, 6.605, 6.9, 1.7976931348623157e+308]]}, "petal_width_cm": {"count": 150.0, "mean": 1.1993333333333336, "std": 0.7622376689603465, "min": 0.1, "25%": 0.3, "50%": 1.3, "75%": 1.8, "max": 2.5, "hist": [[0, 34, 7, 7, 1, 1, 0, 0, 7, 3, 5, 21, 12, 4, 2, 12, 11, 6, 3, 8, 6, 0], [-1.7976931348623157e+308, 0.1, 0.22, 0.33999999999999997, 0.45999999999999996, 0.58, 0.7, 0.82, 0.94, 1.06, 1.1800000000000002, 1.3, 1.42, 1.54, 1.6600000000000001, 1.78, 1.9, 2.02, 2.14, 2.2600000000000002, 2.38, 2.5, 1.7976931348623157e+308]]}}',  # noqa: E501
                "drift_measures": '{"sepal_length_cm":{"hellinger":0.0,"kld":0.0,"tvd":0.0},"sepal_width_cm":{"hellinger":0.0,"kld":0.0,"tvd":0.0},"petal_length_cm":{"hellinger":0.0000000105,"kld":0.0,"tvd":0.0},"petal_width_cm":{"hellinger":0.0,"kld":0.0,"tvd":0.0}}',  # noqa: E501
                "drift_status": "0",
            },
        )


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
            mm_schemas.V3IOTSDBTables.APP_RESULTS: mm_schemas.V3IOTSDBTables.APP_RESULTS,
            mm_schemas.V3IOTSDBTables.METRICS: mm_schemas.V3IOTSDBTables.METRICS,
        }
        tsdb_connector.create_tables()

        return tsdb_connector

    @staticmethod
    @pytest.fixture
    def writer(
        tsdb_connector: mlrun.model_monitoring.db.tsdb.v3io.V3IOTSDBConnector,
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
        writer._tsdb_connector.write_application_event(event=event.copy(), kind=kind)
        record_from_tsdb = writer._tsdb_connector._get_records(
            table=mm_schemas.V3IOTSDBTables.APP_RESULTS,
            filter_query=f"endpoint_id=='{event[WriterEvent.ENDPOINT_ID]}'",
            start="now-1d",
            end="now+1d",
        )

        actual_columns = list(record_from_tsdb.columns)

        assert (
            ResultData.CURRENT_STATS not in actual_columns
        ), "Current stats should not be written to the TSDB"

        # TODO: Remove this assertion after the extra data is supported in TSDB (ML-7460)
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
            writer._tsdb_connector._get_records(
                table=mm_schemas.V3IOTSDBTables.APP_RESULTS,
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
    @pytest.mark.skipif(
        os.getenv("V3IO_FRAMESD") is None or os.getenv("V3IO_ACCESS_KEY") is None,
        reason="Configure Framsed to access V3IO store targets",
    )
    def test_metric_write(
        event: _AppResultEvent,
        writer: ModelMonitoringWriter,
    ) -> None:
        event, kind = ModelMonitoringWriter._reconstruct_event(event)
        writer._tsdb_connector.write_application_event(event, kind)
