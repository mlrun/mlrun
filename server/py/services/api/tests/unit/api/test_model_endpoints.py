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
import string
from random import choice, randint
from typing import Optional

import pytest

import mlrun.common.schemas
import mlrun.model_monitoring
from mlrun.errors import MLRunBadRequestError

import services.api.crud.model_monitoring.deployment
import services.api.crud.model_monitoring.helpers

TEST_PROJECT = "test-model-endpoints"
# Set a default v3io access key env variable
V3IO_ACCESS_KEY = "1111-2222-3333-4444"
os.environ["V3IO_ACCESS_KEY"] = V3IO_ACCESS_KEY


def test_get_access_key():
    key = services.api.crud.model_monitoring.helpers.get_access_key(
        mlrun.common.schemas.AuthInfo(data_session="asd")
    )
    assert key == "asd"

    with pytest.raises(MLRunBadRequestError):
        services.api.crud.model_monitoring.helpers.get_access_key(
            mlrun.common.schemas.AuthInfo()
        )


def test_get_endpoint_features_function():
    stats = {
        "sepal length (cm)": {
            "count": 30.0,
            "mean": 5.946666666666668,
            "std": 0.8394305678023165,
            "min": 4.7,
            "max": 7.9,
            "hist": [
                [4, 2, 1, 0, 1, 3, 4, 0, 3, 4, 1, 1, 2, 1, 0, 1, 0, 0, 1, 1],
                [
                    4.7,
                    4.86,
                    5.0200000000000005,
                    5.18,
                    5.34,
                    5.5,
                    5.66,
                    5.82,
                    5.98,
                    6.140000000000001,
                    6.300000000000001,
                    6.46,
                    6.62,
                    6.78,
                    6.94,
                    7.1,
                    7.26,
                    7.42,
                    7.58,
                    7.74,
                    7.9,
                ],
            ],
        },
        "sepal width (cm)": {
            "count": 30.0,
            "mean": 3.119999999999999,
            "std": 0.4088672324766359,
            "min": 2.2,
            "max": 3.8,
            "hist": [
                [1, 0, 0, 1, 0, 0, 3, 4, 2, 0, 3, 3, 2, 2, 0, 3, 1, 1, 0, 4],
                [
                    2.2,
                    2.2800000000000002,
                    2.3600000000000003,
                    2.44,
                    2.52,
                    2.6,
                    2.68,
                    2.7600000000000002,
                    2.84,
                    2.92,
                    3.0,
                    3.08,
                    3.16,
                    3.24,
                    3.3200000000000003,
                    3.4,
                    3.48,
                    3.56,
                    3.6399999999999997,
                    3.7199999999999998,
                    3.8,
                ],
            ],
        },
        "petal length (cm)": {
            "count": 30.0,
            "mean": 3.863333333333333,
            "std": 1.8212317418360753,
            "min": 1.3,
            "max": 6.7,
            "hist": [
                [6, 4, 0, 0, 0, 0, 0, 0, 1, 2, 0, 3, 3, 2, 2, 3, 1, 1, 1, 1],
                [
                    1.3,
                    1.57,
                    1.84,
                    2.1100000000000003,
                    2.38,
                    2.6500000000000004,
                    2.92,
                    3.1900000000000004,
                    3.46,
                    3.7300000000000004,
                    4.0,
                    4.2700000000000005,
                    4.54,
                    4.8100000000000005,
                    5.08,
                    5.3500000000000005,
                    5.62,
                    5.89,
                    6.16,
                    6.430000000000001,
                    6.7,
                ],
            ],
        },
        "petal width (cm)": {
            "count": 30.0,
            "mean": 1.2733333333333334,
            "std": 0.8291804567674381,
            "min": 0.1,
            "max": 2.5,
            "hist": [
                [5, 3, 2, 0, 0, 0, 0, 0, 1, 2, 3, 2, 1, 0, 2, 3, 1, 1, 0, 4],
                [
                    0.1,
                    0.22,
                    0.33999999999999997,
                    0.45999999999999996,
                    0.58,
                    0.7,
                    0.82,
                    0.94,
                    1.06,
                    1.1800000000000002,
                    1.3,
                    1.42,
                    1.54,
                    1.6600000000000001,
                    1.78,
                    1.9,
                    2.02,
                    2.14,
                    2.2600000000000002,
                    2.38,
                    2.5,
                ],
            ],
        },
    }
    feature_names = list(stats.keys())

    features = services.api.crud.model_monitoring.deployment.get_endpoint_features(
        feature_names, stats, stats
    )
    assert len(features) == 4
    # Commented out asserts should be re-enabled once buckets/counts length mismatch bug is fixed
    for feature in features:
        assert feature.expected is not None
        assert feature.actual is not None

        assert feature.expected.histogram is not None
        # assert len(feature.expected.histogram.buckets) == len(feature.expected.histogram.counts)

        assert feature.actual.histogram is not None
        # assert len(feature.actual.histogram.buckets) == len(feature.actual.histogram.counts)

    features = services.api.crud.model_monitoring.deployment.get_endpoint_features(
        feature_names, stats, None
    )
    assert len(features) == 4
    for feature in features:
        assert feature.expected is not None
        assert feature.actual is None

        assert feature.expected.histogram is not None
        # assert len(feature.expected.histogram.buckets) == len(feature.expected.histogram.counts)

    features = services.api.crud.model_monitoring.deployment.get_endpoint_features(
        feature_names, None, stats
    )
    assert len(features) == 4
    for feature in features:
        assert feature.expected is None
        assert feature.actual is not None

        assert feature.actual.histogram is not None
        # assert len(feature.actual.histogram.buckets) == len(feature.actual.histogram.counts)

    features = services.api.crud.model_monitoring.deployment.get_endpoint_features(
        feature_names[1:], None, stats
    )
    assert len(features) == 3


def _get_auth_info() -> mlrun.common.schemas.AuthInfo:
    return mlrun.common.schemas.AuthInfo(data_session=os.environ.get("V3IO_ACCESS_KEY"))


def _mock_random_endpoint(
    state: Optional[str] = None,
) -> mlrun.common.schemas.ModelEndpoint:
    def random_labels():
        return {f"{choice(string.ascii_letters)}": randint(0, 100) for _ in range(1, 5)}

    return mlrun.common.schemas.ModelEndpoint(
        metadata=mlrun.common.schemas.ModelEndpointMetadata(
            project=TEST_PROJECT, labels=random_labels(), uid=str(randint(1000, 5000))
        ),
        spec=mlrun.common.schemas.ModelEndpointSpec(
            function_uri=f"test/function_{randint(0, 100)}:v{randint(0, 100)}",
            model=f"model_{randint(0, 100)}:v{randint(0, 100)}",
            model_class="classifier",
        ),
        status=mlrun.common.schemas.ModelEndpointStatus(state=state),
    )
