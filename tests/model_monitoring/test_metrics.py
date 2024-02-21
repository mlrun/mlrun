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

import itertools

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from mlrun.model_monitoring.batch import (
    HellingerDistance,
    HistogramDistanceMetric,
    KullbackLeiblerDivergence,
    TotalVarianceDistance,
)


@pytest.mark.parametrize(
    ("distrib_u", "distrib_t", "metric_class", "expected_result"),
    itertools.chain(
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
        ],
        [
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
    ),
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


@pytest.mark.parametrize(
    "metric_class",
    HistogramDistanceMetric.__subclasses__(),
)
@given(
    distrib=st.builds(
        _norm_arr,
        arrays(
            dtype=np.float64,
            elements=st.floats(min_value=0, max_value=1),
            shape=st.integers(min_value=1, max_value=500),
        ),
    )
)
@pytest.mark.filterwarnings("error")
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
