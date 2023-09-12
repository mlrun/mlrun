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

import typing

import pytest

from mlrun.common.model_monitoring.helpers import (
    _MAX_FLOAT,
    FeatureStats,
    Histogram,
    pad_features_hist,
)


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
