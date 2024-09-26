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
import typing
from random import choice, randint
from typing import Optional

import deepdiff
import pytest

import mlrun.common.schemas
import mlrun.model_monitoring
import server.api.crud.model_monitoring.deployment
import server.api.crud.model_monitoring.helpers
from mlrun.common.schemas.model_monitoring.constants import ModelMonitoringStoreKinds
from mlrun.errors import MLRunBadRequestError, MLRunInvalidArgumentError
from mlrun.model_monitoring.db.stores import StoreBase

TEST_PROJECT = "test-model-endpoints"
# Set a default v3io access key env variable
V3IO_ACCESS_KEY = "1111-2222-3333-4444"
os.environ["V3IO_ACCESS_KEY"] = V3IO_ACCESS_KEY

# Bound a typing variable for StoreBase
KVmodelType = typing.TypeVar("KVmodelType", bound="StoreBase")


def test_build_kv_cursor_filter_expression():
    """Validate that the filter expression format converter for the KV cursor works as expected."""

    # Initialize endpoint store target object
    store_type_object = mlrun.model_monitoring.db.ObjectStoreFactory(value="v3io-nosql")

    endpoint_store: KVmodelType = store_type_object.to_object_store(
        project=TEST_PROJECT,
    )

    with pytest.raises(MLRunInvalidArgumentError):
        endpoint_store._build_kv_cursor_filter_expression("")

    filter_expression = endpoint_store._build_kv_cursor_filter_expression(
        project=TEST_PROJECT
    )
    assert filter_expression == f"project=='{TEST_PROJECT}'"

    filter_expression = endpoint_store._build_kv_cursor_filter_expression(
        project=TEST_PROJECT,
        function="test_function",
        model="test_model",
    )
    expected = (
        f"project=='{TEST_PROJECT}' "
        f"AND function_uri=='{TEST_PROJECT}/test_function' AND model=='test_model:latest'"
    )
    assert filter_expression == expected


def test_get_access_key():
    key = server.api.crud.model_monitoring.helpers.get_access_key(
        mlrun.common.schemas.AuthInfo(data_session="asd")
    )
    assert key == "asd"

    with pytest.raises(MLRunBadRequestError):
        server.api.crud.model_monitoring.helpers.get_access_key(
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

    features = server.api.crud.model_monitoring.deployment.get_endpoint_features(
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

    features = server.api.crud.model_monitoring.deployment.get_endpoint_features(
        feature_names, stats, None
    )
    assert len(features) == 4
    for feature in features:
        assert feature.expected is not None
        assert feature.actual is None

        assert feature.expected.histogram is not None
        # assert len(feature.expected.histogram.buckets) == len(feature.expected.histogram.counts)

    features = server.api.crud.model_monitoring.deployment.get_endpoint_features(
        feature_names, None, stats
    )
    assert len(features) == 4
    for feature in features:
        assert feature.expected is None
        assert feature.actual is not None

        assert feature.actual.histogram is not None
        # assert len(feature.actual.histogram.buckets) == len(feature.actual.histogram.counts)

    features = server.api.crud.model_monitoring.deployment.get_endpoint_features(
        feature_names[1:], None, stats
    )
    assert len(features) == 3


def test_generating_tsdb_paths():
    """Validate that the TSDB paths for the KVStoreBase object are created as expected. These paths are
    usually important when the user call the delete project API and as a result the TSDB resources should be deleted
    """

    # Initialize endpoint store target object
    store_type_object = mlrun.model_monitoring.db.ObjectStoreFactory(value="v3io-nosql")
    endpoint_store: KVmodelType = store_type_object.to_object_store(
        project=TEST_PROJECT,
    )

    # Generating the required tsdb paths
    tsdb_path, filtered_path = endpoint_store._generate_tsdb_paths()

    # Validate the expected results based on the full path to the TSDB events directory
    full_path = mlrun.mlconf.model_endpoint_monitoring.store_prefixes.default.format(
        project=TEST_PROJECT,
        kind=ModelMonitoringStoreKinds.EVENTS,
    )

    # TSDB short path that should point to the main directory
    assert tsdb_path == full_path[: len(tsdb_path)]

    # Filtered path that should point to the events directory without container and schema
    assert filtered_path == full_path[-len(filtered_path) + 1 :] + "/"


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


def test_validate_model_endpoints_schema():
    # Validate that both model endpoint basemodel schema and model endpoint ModelObj schema have similar keys
    model_endpoint_basemodel = mlrun.common.schemas.ModelEndpoint(
        metadata=mlrun.common.schemas.ModelEndpointMetadata(
            project=TEST_PROJECT, uid="a-12_"
        )
    )
    model_endpoint_modelobj = mlrun.model_monitoring.ModelEndpoint()

    # Compare status
    base_model_status = model_endpoint_basemodel.status.__dict__
    model_object_status = model_endpoint_modelobj.status.__dict__
    assert (
        deepdiff.DeepDiff(
            base_model_status,
            model_object_status,
            ignore_order=True,
        )
    ) == {}

    # Compare spec
    base_model_status = model_endpoint_basemodel.status.__dict__
    model_object_status = model_endpoint_modelobj.status.__dict__
    assert (
        deepdiff.DeepDiff(
            base_model_status,
            model_object_status,
            ignore_order=True,
        )
    ) == {}
