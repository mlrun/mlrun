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
from functools import partial
from unittest.mock import Mock

import pytest
import semver

from mlrun.model_monitoring.writer import (
    MetricData,
    ModelMonitoringWriter,
    ResultData,
    V3IOFramesClient,
    WriterEvent,
    _AppResultEvent,
    _Notifier,
    _RawEvent,
    _WriterEventTypeError,
    _WriterEventValueError,
)
from mlrun.utils.notifications.notification_pusher import CustomNotificationPusher


@pytest.fixture(params=[(0, "1.7.0", "result")])
def event(request: pytest.FixtureRequest) -> _AppResultEvent:
    if semver.Version.parse("1.7.0") > semver.Version.parse(request.param[1]):
        return _AppResultEvent(
            {
                WriterEvent.ENDPOINT_ID: "some-ep-id",
                WriterEvent.START_INFER_TIME: "2023-09-19 14:26:06.501084",
                WriterEvent.END_INFER_TIME: "2023-09-19 16:26:06.501084",
                WriterEvent.APPLICATION_NAME: "dummy-app",
                ResultData.RESULT_NAME: "data-drift-0",
                ResultData.RESULT_KIND: 0,
                ResultData.RESULT_VALUE: 0.32,
                ResultData.RESULT_STATUS: request.param[0],
                ResultData.RESULT_EXTRA_DATA: "",
                WriterEvent.CURRENT_STATS: "",
            }
        )
    elif request.param[2] == "result":
        return _AppResultEvent(
            {
                WriterEvent.ENDPOINT_ID: "some-ep-id",
                WriterEvent.START_INFER_TIME: "2023-09-19 14:26:06.501084",
                WriterEvent.END_INFER_TIME: "2023-09-19 16:26:06.501084",
                WriterEvent.APPLICATION_NAME: "dummy-app",
                WriterEvent.CURRENT_STATS: "",
                WriterEvent.EVENT_KIND: request.param[2],
                WriterEvent.DATA: json.dumps(
                    {
                        ResultData.RESULT_NAME: "data-drift-0",
                        ResultData.RESULT_KIND: 0,
                        ResultData.RESULT_VALUE: 0.32,
                        ResultData.RESULT_STATUS: request.param[0],
                        ResultData.RESULT_EXTRA_DATA: "",
                    }
                ),
            }
        )
    elif request.param[2] == "metric":
        return _AppResultEvent(
            {
                WriterEvent.ENDPOINT_ID: "some-ep-id",
                WriterEvent.START_INFER_TIME: "2023-09-19 14:26:06.501084",
                WriterEvent.END_INFER_TIME: "2023-09-19 16:26:06.501084",
                WriterEvent.APPLICATION_NAME: "dummy-app",
                WriterEvent.CURRENT_STATS: "",
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
    @staticmethod
    @pytest.fixture
    def tsdb_client() -> V3IOFramesClient:
        return Mock(spec=V3IOFramesClient)

    @staticmethod
    @pytest.fixture
    def writer(tsdb_client: V3IOFramesClient) -> ModelMonitoringWriter:
        writer = Mock(spec=ModelMonitoringWriter)
        writer._tsdb_client = tsdb_client
        writer._update_tsdb = partial(ModelMonitoringWriter._update_tsdb, writer)
        return writer

    @staticmethod
    @pytest.mark.parametrize(
        ("event"),
        [(0, "1.6.0", "result"), (0, "1.7.0", "result")],
        indirect=["event"],
    )
    def test_no_extra(
        event: _AppResultEvent,
        tsdb_client: V3IOFramesClient,
        writer: ModelMonitoringWriter,
    ) -> None:
        event, kind = ModelMonitoringWriter._reconstruct_event(event)
        writer._update_tsdb(event, kind)
        tsdb_client.write.assert_called()
        assert (
            ResultData.RESULT_EXTRA_DATA
            not in tsdb_client.write.call_args.kwargs["dfs"].columns
        ), "The extra data should not be written to the TSDB"

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
        tsdb_client: V3IOFramesClient,
        writer: ModelMonitoringWriter,
    ) -> None:
        event, kind = ModelMonitoringWriter._reconstruct_event(event)
        writer._update_tsdb(event, kind)
        tsdb_client.write.assert_not_called()
