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
from typing import Type

import numpy as np
import pytest

from mlrun.model_monitoring.model_monitoring_batch import (
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
                ],
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
                ],
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
                [0.0, 0.16666667, 0.33333333, 0.5],
                [0.5, 0.33333333, 0.16666667, 0.0],
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
    metric_class: Type[HistogramDistanceMetric],
    distrib_u: list[float],
    distrib_t: list[float],
    expected_result: float,
) -> None:
    distrib_u, distrib_t = np.asarray(distrib_u), np.asarray(distrib_t)
    assert np.isclose(
        metric_class(distrib_t=distrib_t, distrib_u=distrib_u).compute(),
        expected_result,
    )


@pytest.mark.parametrize(
    ("distrib", "metric_class"),
    itertools.product(
        [[1], [0.5, 0.5], [0.2, 0, 0.1, 0.09, 0, 0.61]],
        HistogramDistanceMetric.__subclasses__(),
    ),
)
def test_same_distrib_gives_zero_distance(
    distrib: list[float], metric_class: Type[HistogramDistanceMetric]
) -> None:
    return test_histogram_metric_calculation(
        metric_class=metric_class,
        distrib_t=distrib,
        distrib_u=distrib,
        expected_result=0,
    )
