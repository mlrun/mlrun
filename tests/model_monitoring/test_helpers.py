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
from typing import Callable, Tuple

import pytest

from mlrun.common.model_monitoring.helpers import (
    _MAX_FLOAT,
    FeatureStats,
    Histogram,
    pad_features_hist,
)
from mlrun.common.schemas.model_monitoring import EventFieldType
from mlrun.model_monitoring.batch_application import BatchApplicationProcessor


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
    interval_range = BatchApplicationProcessor._get_interval_range

    @staticmethod
    def _fake_now_func_factory(
        delta: datetime.timedelta,
        base_time: datetime.datetime = datetime.datetime(2021, 1, 1, 12, 0, 0),
    ) -> Callable[[], datetime.datetime]:
        def fake_now_func() -> datetime.datetime:
            nonlocal base_time
            current_time = base_time
            base_time += delta
            return current_time

        return fake_now_func

    @classmethod
    @pytest.fixture
    def intervals(
        cls, minutes_delta: int = 6
    ) -> list[Tuple[datetime.datetime, datetime.datetime]]:
        now_func = cls._fake_now_func_factory(
            delta=datetime.timedelta(minutes=minutes_delta)
        )
        return [
            BatchApplicationProcessor._get_interval_range(
                batch_dict={
                    EventFieldType.MINUTES: minutes_delta,
                    EventFieldType.HOURS: 0,
                    EventFieldType.DAYS: 0,
                },
                now_func=now_func,
            )
            for _ in range(5)
        ]

    @staticmethod
    def test_touching_interval(
        intervals: list[Tuple[datetime.datetime, datetime.datetime]]
    ) -> None:
        for prev, curr in zip(intervals[:-1], intervals[1:]):
            assert prev[1] == curr[0], "The intervals should be touching"

    @staticmethod
    def test_end_time_is_in_the_past() -> None:
        time = datetime.datetime(2023, 11, 16, 12, 0, 0)
        _, end_time = BatchApplicationProcessor._get_interval_range(
            batch_dict={
                EventFieldType.MINUTES: 10,
                EventFieldType.HOURS: 0,
                EventFieldType.DAYS: 0,
            },
            now_func=lambda: time,
        )
        assert end_time < time, "End time should be in the past"
