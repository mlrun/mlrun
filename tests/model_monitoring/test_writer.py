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

from functools import partial
from unittest.mock import Mock

import pytest

from mlrun.model_monitoring.writer import (
    ModelMonitoringWriter,
    V3IOFramesClient,
    WriterEvent,
    _AppResultEvent,
    _Notifier,
    _RawEvent,
    _WriterEventTypeError,
    _WriterEventValueError,
)
from mlrun.utils.notifications.notification_pusher import CustomNotificationPusher


@pytest.fixture(params=[0])
def event(request: pytest.FixtureRequest) -> _AppResultEvent:
    return _AppResultEvent(
        {
            WriterEvent.ENDPOINT_ID: "some-ep-id",
            WriterEvent.START_INFER_TIME: "2023-09-19 14:26:06.501084",
            WriterEvent.END_INFER_TIME: "2023-09-19 16:26:06.501084",
            WriterEvent.APPLICATION_NAME: "dummy-app",
            WriterEvent.RESULT_NAME: "data-drift-0",
            WriterEvent.RESULT_KIND: 0,
            WriterEvent.RESULT_VALUE: 0.32,
            WriterEvent.RESULT_STATUS: request.param,
            WriterEvent.RESULT_EXTRA_DATA: "",
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
    def test_no_extra(
        event: _AppResultEvent,
        tsdb_client: V3IOFramesClient,
        writer: ModelMonitoringWriter,
    ) -> None:
        writer._update_tsdb(event)
        tsdb_client.write.assert_called()
        assert (
            WriterEvent.RESULT_EXTRA_DATA
            not in tsdb_client.write.call_args.kwargs["dfs"].columns
        ), "The extra data should not be written to the TSDB"
