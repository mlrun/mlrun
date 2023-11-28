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
import typing
from typing import Optional, Tuple
from unittest.mock import Mock, patch

import pytest
from v3io.dataplane.response import HttpResponseError

from mlrun.common.model_monitoring.helpers import (
    _MAX_FLOAT,
    FeatureStats,
    Histogram,
    pad_features_hist,
)
from mlrun.model_monitoring.controller import _BatchWindow, _BatchWindowGenerator


class _HistLen(typing.NamedTuple):
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
        intervals: list[Tuple[datetime.datetime, datetime.datetime]]
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
        time = datetime.datetime(2023, 11, 16, 12, 0, 0).timestamp()
        last_updated = _BatchWindowGenerator._get_last_updated_time(
            now_func=lambda: time
        )
        assert last_updated < time, "The last updated time should be in the past"
