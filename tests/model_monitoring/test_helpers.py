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
import http
from collections.abc import Iterator
from typing import NamedTuple, Optional
from unittest.mock import Mock, patch

import nuclio
import pytest
from v3io.dataplane.response import HttpResponseError

import mlrun
from mlrun.common.model_monitoring.helpers import (
    _MAX_FLOAT,
    FeatureStats,
    Histogram,
    pad_features_hist,
    pad_hist,
)
from mlrun.common.schemas.model_monitoring.constants import EventFieldType
from mlrun.db.nopdb import NopDB
from mlrun.model_monitoring.controller import (
    _BatchWindow,
    _BatchWindowGenerator,
    _Interval,
)
from mlrun.model_monitoring.helpers import (
    _get_monitoring_time_window_from_controller_run,
    get_invocations_fqn,
    update_model_endpoint_last_request,
)
from mlrun.model_monitoring.model_endpoint import ModelEndpoint
from mlrun.utils import datetime_now

TEST_PROJECT = "my-project"


class _HistLen(NamedTuple):
    counts_len: int
    edges_len: int


class TemplateFunction(mlrun.runtimes.ServingRuntime):
    def __init__(self):
        super().__init__()
        self.add_trigger(
            "cron_interval",
            spec=nuclio.CronTrigger(interval=f"{1}m"),
        )


@pytest.fixture
def feature_stats() -> FeatureStats:
    return FeatureStats(
        {
            "feat0": {"hist": [[0, 1], [1.1, 2.2, 3.3]]},
            "feat1": {
                "key": "val",
                "hist": [[4, 2, 0, 2, 0], [-5, 5, 6, 100, 101, 222]],
            },
        }
    )


@pytest.fixture
def histogram() -> Histogram:
    return Histogram([[0, 1], [1.1, 2.2, 3.3]])


@pytest.fixture
def padded_histogram(histogram: Histogram) -> Histogram:
    pad_hist(histogram)
    return histogram


@pytest.fixture
def orig_feature_stats_hist_data(feature_stats: FeatureStats) -> dict[str, _HistLen]:
    data: dict[str, _HistLen] = {}
    for feat_name, feat in feature_stats.items():
        hist = feat["hist"]
        data[feat_name] = _HistLen(counts_len=len(hist[0]), edges_len=len(hist[1]))
    return data


def _check_padded_hist_spec(hist: Histogram, orig_data: _HistLen) -> None:
    counts = hist[0]
    edges = hist[1]
    edges_len = len(edges)
    counts_len = len(counts)
    assert edges_len == counts_len + 1
    assert counts_len == orig_data.counts_len + 2
    assert edges_len == orig_data.edges_len + 2
    assert counts[0] == counts[-1] == 0
    assert (-edges[0]) == edges[-1] == _MAX_FLOAT


def test_pad_hist(histogram: Histogram) -> None:
    orig_data = _HistLen(
        counts_len=len(histogram[0]),
        edges_len=len(histogram[1]),
    )
    pad_hist(histogram)
    _check_padded_hist_spec(histogram, orig_data)


def test_padded_hist_unchanged(padded_histogram: Histogram) -> None:
    orig_hist = padded_histogram.copy()
    pad_hist(padded_histogram)
    assert padded_histogram == orig_hist, "A padded histogram should not be changed"


def test_pad_features_hist(
    feature_stats: FeatureStats,
    orig_feature_stats_hist_data: dict[str, _HistLen],
) -> None:
    pad_features_hist(feature_stats)
    for feat_name, feat in feature_stats.items():
        _check_padded_hist_spec(feat["hist"], orig_feature_stats_hist_data[feat_name])


class TestBatchInterval:
    @staticmethod
    @pytest.fixture(autouse=True)
    def set_mlconf():
        mlrun.mlconf.model_endpoint_monitoring.endpoint_store_connection = "v3io"
        mlrun.mlconf.model_endpoint_monitoring.tsdb_connection = "v3io"
        yield

    @staticmethod
    @pytest.fixture
    def timedelta_seconds(request: pytest.FixtureRequest) -> int:
        if marker := request.node.get_closest_marker(
            TestBatchInterval.timedelta_seconds.__name__
        ):
            return marker.args[0]
        return int(datetime.timedelta(minutes=6).total_seconds())

    @staticmethod
    @pytest.fixture
    def first_request(request: pytest.FixtureRequest) -> int:
        if marker := request.node.get_closest_marker(
            TestBatchInterval.first_request.__name__
        ):
            return marker.args[0]
        return int(
            datetime.datetime(
                2021, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
            ).timestamp()
        )

    @staticmethod
    @pytest.fixture
    def last_updated(request: pytest.FixtureRequest) -> int:
        if marker := request.node.get_closest_marker(
            TestBatchInterval.last_updated.__name__
        ):
            return marker.args[0]
        return int(
            datetime.datetime(
                2021, 1, 1, 13, 1, 0, tzinfo=datetime.timezone.utc
            ).timestamp()
        )

    @staticmethod
    @pytest.fixture(autouse=True)
    def mock_kv() -> Iterator[None]:
        mock = Mock(spec=["kv"])
        mock.kv.get = Mock(
            side_effect=HttpResponseError(status_code=http.HTTPStatus.NOT_FOUND)
        )
        with patch(
            "mlrun.utils.v3io_clients.get_v3io_client",
            return_value=mock,
        ):
            with patch(
                "mlrun.model_monitoring.get_store_object",
                return_value=mlrun.model_monitoring.get_store_object(
                    store_connection_string="v3io", project=TEST_PROJECT
                ),
            ):
                yield

    @staticmethod
    @pytest.fixture
    def intervals(
        timedelta_seconds: int,
        first_request: int,
        last_updated: int,
    ) -> list[_Interval]:
        return list(
            _BatchWindow(
                project="project",
                endpoint="ep",
                application="app",
                timedelta_seconds=timedelta_seconds,
                first_request=first_request,
                last_updated=last_updated,
            ).get_intervals()
        )

    @staticmethod
    @pytest.fixture
    def expected_intervals() -> list[_Interval]:
        def dt(hour: int, minute: int) -> datetime.datetime:
            return datetime.datetime(
                2021, 1, 1, hour, minute, tzinfo=datetime.timezone.utc
            )

        def interval(start: tuple[int, int], end: tuple[int, int]) -> _Interval:
            return _Interval(dt(*start), dt(*end))

        return [
            interval((12, 0), (12, 6)),
            interval((12, 6), (12, 12)),
            interval((12, 12), (12, 18)),
            interval((12, 18), (12, 24)),
            interval((12, 24), (12, 30)),
            interval((12, 30), (12, 36)),
            interval((12, 36), (12, 42)),
            interval((12, 42), (12, 48)),
            interval((12, 48), (12, 54)),
            interval((12, 54), (13, 0)),
        ]

    @staticmethod
    def test_touching_intervals(intervals: list[_Interval]) -> None:
        assert len(intervals) > 1, "There should be more than one interval"
        for prev, curr in zip(intervals[:-1], intervals[1:]):
            assert prev[1] == curr[0], "The intervals should be touching"

    @staticmethod
    def test_intervals(
        intervals: list[_Interval], expected_intervals: list[_Interval]
    ) -> None:
        assert len(intervals) == len(
            expected_intervals
        ), "The number of intervals is not as expected"
        assert intervals == expected_intervals, "The intervals are not as expected"

    @staticmethod
    def test_last_interval_does_not_overflow(
        intervals: list[_Interval], last_updated: int
    ) -> None:
        assert (
            intervals[-1][1].timestamp() <= last_updated
        ), "The last interval should be after last_updated"

    @staticmethod
    @pytest.mark.parametrize(
        (
            "timedelta_seconds",
            "first_request",
            "last_updated",
            "expected_last_analyzed",
        ),
        [
            (60, 100, 300, 100),
            (60, 100, 110, 100),
            (60, 0, 0, 0),
            (60, None, None, None),
        ],
    )
    def test_get_last_analyzed(
        timedelta_seconds: int,
        last_updated: Optional[int],
        first_request: Optional[int],
        expected_last_analyzed: Optional[int],
    ) -> None:
        assert (
            _BatchWindow(
                project=TEST_PROJECT,
                endpoint="some-endpoint",
                application="special-app",
                timedelta_seconds=timedelta_seconds,
                first_request=first_request,
                last_updated=last_updated,
            )._get_last_analyzed()
            == expected_last_analyzed
        ), "The last analyzed time is not as expected"

    @staticmethod
    @pytest.mark.timedelta_seconds(int(datetime.timedelta(days=6).total_seconds()))
    @pytest.mark.first_request(
        int(
            datetime.datetime(
                2020, 12, 25, 23, 0, 0, tzinfo=datetime.timezone.utc
            ).timestamp()
        )
    )
    @pytest.mark.last_updated(
        int(
            datetime.datetime(
                2021, 1, 1, 3, 1, 0, tzinfo=datetime.timezone.utc
            ).timestamp()
        )
    )
    def test_large_base_period(
        timedelta_seconds: int, intervals: list[_Interval]
    ) -> None:
        assert len(intervals) == 1, "There should be exactly one interval"
        assert timedelta_seconds == datetime.datetime.timestamp(
            intervals[0][1]
        ) - datetime.datetime.timestamp(
            intervals[0][0]
        ), "The time slot should be equal to timedelta_seconds (6 days)"


class TestBatchWindowGenerator:
    @staticmethod
    @pytest.mark.parametrize(
        ("first_request", "expected"),
        [("", None), (None, None), ("2023-11-09 09:25:59.554971+00:00", 1699521959)],
    )
    def test_normalize_first_request(
        first_request: Optional[str], expected: Optional[int]
    ) -> None:
        assert (
            _BatchWindowGenerator._normalize_first_request(
                first_request=first_request, endpoint=""
            )
            == expected
        )

    @staticmethod
    def test_last_updated_is_in_the_past() -> None:
        last_request = datetime.datetime(2023, 11, 16, 12, 0, 0)
        last_updated = _BatchWindowGenerator._get_last_updated_time(
            last_request=last_request.isoformat(), has_stream=True
        )
        assert last_updated
        assert (
            last_updated < last_request.timestamp()
        ), "The last updated time should be before the last request"


class TestBumpModelEndpointLastRequest:
    @staticmethod
    @pytest.fixture
    def project() -> str:
        return "project"

    @staticmethod
    @pytest.fixture
    def db() -> NopDB:
        return NopDB()

    @staticmethod
    @pytest.fixture
    def empty_model_endpoint() -> ModelEndpoint:
        return ModelEndpoint()

    @staticmethod
    @pytest.fixture
    def last_request() -> str:
        return "2023-12-05 18:17:50.255143"

    @staticmethod
    @pytest.fixture
    def model_endpoint(
        empty_model_endpoint: ModelEndpoint, last_request: str
    ) -> ModelEndpoint:
        empty_model_endpoint.status.last_request = last_request
        return empty_model_endpoint

    @staticmethod
    @pytest.fixture
    def function() -> mlrun.runtimes.ServingRuntime:
        return TemplateFunction()

    @staticmethod
    def test_update_last_request(
        project: str,
        model_endpoint: ModelEndpoint,
        db: NopDB,
        last_request: str,
        function: mlrun.runtimes.ServingRuntime,
    ) -> None:
        model_endpoint.spec.stream_path = "stream"
        with patch.object(db, "patch_model_endpoint") as patch_patch_model_endpoint:
            with patch.object(db, "get_function", return_value=function):
                update_model_endpoint_last_request(
                    project=project,
                    model_endpoint=model_endpoint,
                    current_request=datetime.datetime.fromisoformat(last_request),
                    db=db,
                )
        patch_patch_model_endpoint.assert_called_once()
        assert datetime.datetime.fromisoformat(
            patch_patch_model_endpoint.call_args.kwargs["attributes"][
                EventFieldType.LAST_REQUEST
            ]
        ) == datetime.datetime.fromisoformat(last_request)
        model_endpoint.spec.stream_path = ""

        with patch.object(db, "patch_model_endpoint") as patch_patch_model_endpoint:
            with patch.object(db, "get_function", return_value=function):
                update_model_endpoint_last_request(
                    project=project,
                    model_endpoint=model_endpoint,
                    current_request=datetime.datetime.fromisoformat(last_request),
                    db=db,
                )
        patch_patch_model_endpoint.assert_called_once()
        assert datetime.datetime.fromisoformat(
            patch_patch_model_endpoint.call_args.kwargs["attributes"][
                EventFieldType.LAST_REQUEST
            ]
        ) == datetime.datetime.fromisoformat(last_request) + datetime.timedelta(
            minutes=1
        ) + datetime.timedelta(
            seconds=mlrun.mlconf.model_endpoint_monitoring.parquet_batching_timeout_secs
        ), "The patched last request time should be bumped by the given delta"

    @staticmethod
    def test_no_bump(
        project: str,
        model_endpoint: ModelEndpoint,
        db: NopDB,
    ) -> None:
        with patch.object(db, "patch_model_endpoint") as patch_patch_model_endpoint:
            with patch.object(
                db, "get_function", side_effect=mlrun.errors.MLRunNotFoundError
            ):
                update_model_endpoint_last_request(
                    project=project,
                    model_endpoint=model_endpoint,
                    current_request=datetime_now(),
                    db=db,
                )
        patch_patch_model_endpoint.assert_not_called()

    @staticmethod
    def test_get_monitoring_time_window_from_controller_run(
        project: str,
        db: NopDB,
        function: mlrun.runtimes.ServingRuntime,
    ) -> None:
        with patch.object(db, "get_function", return_value=function):
            assert _get_monitoring_time_window_from_controller_run(
                project=project,
                db=db,
            ) == datetime.timedelta(minutes=1), "The window is different than expected"


def test_get_invocations_fqn() -> None:
    assert get_invocations_fqn("project") == "project.mlrun-infra.metric.invocations"
