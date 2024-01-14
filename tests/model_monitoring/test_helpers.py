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
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import NamedTuple, Optional, Tuple
from unittest.mock import Mock, patch

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
from mlrun.errors import MLRunValueError
from mlrun.model_monitoring.controller import _BatchWindow, _BatchWindowGenerator
from mlrun.model_monitoring.helpers import (
    _get_monitoring_time_window_from_controller_run,
    bump_model_endpoint_last_request,
)
from mlrun.model_monitoring.model_endpoint import ModelEndpoint


class _HistLen(NamedTuple):
    counts_len: int
    edges_len: int


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
    @pytest.fixture
    def intervals(
        timedelta_seconds: int = int(datetime.timedelta(minutes=6).total_seconds()),
        first_request: int = int(datetime.datetime(2021, 1, 1, 12, 0, 0).timestamp()),
        last_updated: int = int(datetime.datetime(2021, 1, 1, 13, 1, 0).timestamp()),
    ) -> list[Tuple[datetime.datetime, datetime.datetime]]:
        mock = Mock(spec=["kv"])
        mock.kv.get = Mock(side_effect=HttpResponseError)
        with patch(
            "mlrun.model_monitoring.controller.get_v3io_client",
            return_value=mock,
        ):
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
    def test_touching_intervals(
        intervals: list[Tuple[datetime.datetime, datetime.datetime]],
    ) -> None:
        assert len(intervals) > 1, "There should be more than one interval"
        for prev, curr in zip(intervals[:-1], intervals[1:]):
            assert prev[1] == curr[0], "The intervals should be touching"


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
            last_request=last_request.isoformat(),
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
    def runs() -> list[dict]:
        return [
            {
                "kind": "run",
                "metadata": {
                    "name": "model-monitoring-controller",
                    "uid": "3a88d8aef52f4a90a12b681a87d9dc51",
                    "iteration": 0,
                    "project": "test-mm-1",
                    "labels": {
                        "mlrun/schedule-name": "model-monitoring-controller",
                        "kind": "job",
                        "v3io_user": "pipelines",
                        "host": "model-monitoring-controller-cbvs4",
                    },
                    "annotations": {},
                },
                "spec": {
                    "function": "test-mm-1/model-monitoring-controller@8056f87c8e5b11408d9d990fc0381f7a0fca83cf",
                    "log_level": "info",
                    "parameters": {
                        "batch_intervals_dict": {"minutes": 1, "hours": 0, "days": 0}
                    },
                    "handler": "handler",
                    "outputs": [],
                    "output_path": "v3io:///projects/test-mm-1/artifacts",
                    "inputs": {},
                    "notifications": [],
                    "state_thresholds": {
                        "pending_scheduled": "1h",
                        "pending_not_scheduled": "-1",
                        "image_pull_backoff": "1h",
                        "executing": "24h",
                    },
                    "hyperparams": {},
                    "hyper_param_options": {},
                    "data_stores": [],
                },
                "status": {
                    "results": {},
                    "start_time": "2024-01-14T15:01:03.639771+00:00",
                    "last_update": "2024-01-14T15:01:04.049320+00:00",
                    "state": "completed",
                    "artifacts": [],
                },
            }
        ]

    @staticmethod
    def test_empty_last_request(
        project: str, empty_model_endpoint: ModelEndpoint, db: NopDB
    ) -> None:
        with pytest.raises(
            MLRunValueError, match="Model endpoint last request time is empty"
        ):
            bump_model_endpoint_last_request(
                project=project,
                model_endpoint=empty_model_endpoint,
                db=db,
            )

    @staticmethod
    def test_bump(
        project: str,
        model_endpoint: ModelEndpoint,
        db: NopDB,
        last_request: str,
        runs: list[dict],
    ) -> None:
        with patch.object(db, "patch_model_endpoint") as patch_patch_model_endpoint:
            with patch.object(db, "list_runs", return_value=runs):
                bump_model_endpoint_last_request(
                    project=project,
                    model_endpoint=model_endpoint,
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
    @pytest.mark.parametrize(
        ("runs", "error_context", "expected_window"),
        [
            (
                [],
                pytest.raises(
                    MLRunValueError, match="No model-monitoring-controller runs"
                ),
                None,
            ),
            (
                [{"kind": "run", "spec": {"parameters": {}}}],
                pytest.raises(
                    MLRunValueError,
                    match="Could not find `batch_intervals_dict` in model-monitoring-controller run",
                ),
                None,
            ),
            (
                [
                    {
                        "kind": "run",
                        "spec": {
                            "parameters": {
                                "batch_intervals_dict": {
                                    "minutes": 1,
                                    "hours": 0,
                                    "days": 0,
                                }
                            }
                        },
                    },
                    {
                        "kind": "run",
                        "spec": {
                            "parameters": {
                                "batch_intervals_dict": {
                                    "minutes": 1,
                                    "hours": 2,
                                    "days": 3,
                                }
                            }
                        },
                    },
                ],
                does_not_raise(),
                datetime.timedelta(minutes=1),
            ),
        ],
    )
    def test_get_monitoring_time_window_from_controller_run(
        project: str,
        db: NopDB,
        runs: list[dict],
        error_context: AbstractContextManager,
        expected_window: Optional[datetime.timedelta],
    ) -> None:
        with patch.object(db, "list_runs", return_value=runs):
            with error_context:
                assert (
                    _get_monitoring_time_window_from_controller_run(
                        project=project,
                        db=db,
                    )
                    == expected_window
                ), "The window is different than expected"
