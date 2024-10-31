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

import os
from typing import Union

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from mlrun.model_monitoring.metrics.histogram_distance import (
    HellingerDistance,
    HistogramDistanceMetric,
    KullbackLeiblerDivergence,
    TotalVarianceDistance,
)


@pytest.mark.parametrize(
    ("distrib_u", "distrib_t", "metric_class", "expected_result"),
    [
        (
            np.array(
                [
                    0.07101131,
                    0.25138296,
                    0.03700347,
                    0.06843929,
                    0.11482246,
                    0.1339393,
                    0.0503426,
                    0.05639414,
                    0.13406714,
                    0.08259733,
                ]
            ),
            np.array(
                [
                    0.17719221,
                    0.0949423,
                    0.02982267,
                    0.19927063,
                    0.19288318,
                    0.00137802,
                    0.07398598,
                    0.05829106,
                    0.06771774,
                    0.10451622,
                ]
            ),
            class_,
            result,
        )
        for class_, result in [
            (TotalVarianceDistance, 0.3625321),
            (HellingerDistance, 0.338189),
            (KullbackLeiblerDivergence, 1.097619),
        ]
    ]
    + [
        (
            np.array([0.0, 0.16666667, 0.33333333, 0.5]),
            np.array([0.5, 0.33333333, 0.16666667, 0.0]),
            class_,
            result,
        )
        for class_, result in [
            (TotalVarianceDistance, 0.666666),
            (HellingerDistance, 0.727045),
            (KullbackLeiblerDivergence, 8.748242),
        ]
    ],
)
def test_histogram_metric_calculation(
    metric_class: type[HistogramDistanceMetric],
    distrib_u: np.ndarray,
    distrib_t: np.ndarray,
    expected_result: float,
    atol: float = 1e-8,
) -> None:
    assert np.isclose(
        metric_class(distrib_t=distrib_t, distrib_u=distrib_u).compute(),
        expected_result,
        atol=atol,
    )


def _norm_arr(arr: np.ndarray) -> np.ndarray:
    """
    Normalize a nonnegative array to sum 1.
    If a zeros array, put 1 at the first index.
    """
    arr_sum = arr.sum()
    if not arr_sum:
        new_arr = np.zeros_like(arr)
        new_arr[0] = 1
        return new_arr
    return arr / arr_sum.sum()


_max_value = 100 if os.getenv("CI") == "true" else 500
_length_strategy = st.integers(min_value=1, max_value=_max_value)


def distribution_strategy(
    length: Union[int, st.SearchStrategy[int]] = _length_strategy,
) -> st.SearchStrategy[np.ndarray]:
    return st.builds(
        _norm_arr,
        arrays(
            dtype=np.float64,
            elements=st.floats(min_value=0, max_value=1),
            shape=length,
        ),
    )


@st.composite
def two_distributions_strategy(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray]:
    """Two distributions of the same shape"""
    length = draw(_length_strategy)
    arr_st = distribution_strategy(length)
    return draw(arr_st), draw(arr_st)


@pytest.mark.parametrize(
    "metric_class",
    HistogramDistanceMetric.__subclasses__(),
)
@pytest.mark.filterwarnings("error")
class TestMetricProperties:
    @staticmethod
    @given(distrib=distribution_strategy())
    def test_same_distrib_gives_zero_distance(
        metric_class: type[HistogramDistanceMetric], distrib: np.ndarray
    ) -> None:
        return test_histogram_metric_calculation(
            metric_class=metric_class,
            distrib_t=distrib,
            distrib_u=distrib,
            expected_result=0,
            atol=1e-7,
        )

    @staticmethod
    @given(two_distributions=two_distributions_strategy())
    def test_symmetric(
        metric_class: type[HistogramDistanceMetric],
        two_distributions: tuple[np.ndarray, np.ndarray],
    ) -> None:
        distrib_t, distrib_u = two_distributions
        assert np.isclose(
            metric_class(distrib_t=distrib_t, distrib_u=distrib_u).compute(),
            metric_class(distrib_t=distrib_u, distrib_u=distrib_t).compute(),
            atol=1e-8,
        )
