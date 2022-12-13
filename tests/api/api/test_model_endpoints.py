# Copyright 2018 Iguazio
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
#
import os
import string
from random import choice, randint
from typing import Optional

import pytest
import sqlalchemy.exc

import mlrun.api.crud
import mlrun.api.schemas
from mlrun.api.schemas import (
    ModelEndpoint,
    ModelEndpointMetadata,
    ModelEndpointSpec,
    ModelEndpointStatus,
)
from mlrun.errors import MLRunBadRequestError, MLRunInvalidArgumentError

TEST_PROJECT = "test_model_endpoints"
CONNECTION_STRING = "sqlite:///test.db"
# Set a default v3io access key env variable
V3IO_ACCESS_KEY = "1111-2222-3333-4444"
os.environ["V3IO_ACCESS_KEY"] = V3IO_ACCESS_KEY


def test_build_kv_cursor_filter_expression():
    """Validate that the filter expression format converter for the KV cursor works as expected."""

    # Initialize endpoint store target object
    store_type_object = mlrun.api.crud.model_monitoring.ModelEndpointStoreType(
        value="kv"
    )

    endpoint_target = store_type_object.to_endpoint_target(
        project=TEST_PROJECT, access_key=V3IO_ACCESS_KEY
    )

    with pytest.raises(MLRunInvalidArgumentError):
        endpoint_target.build_kv_cursor_filter_expression("")

    filter_expression = endpoint_target.build_kv_cursor_filter_expression(
        project=TEST_PROJECT
    )
    assert filter_expression == f"project=='{TEST_PROJECT}'"

    filter_expression = endpoint_target.build_kv_cursor_filter_expression(
        project=TEST_PROJECT, function="test_function", model="test_model"
    )
    expected = f"project=='{TEST_PROJECT}' AND function=='test_function' AND model=='test_model'"
    assert filter_expression == expected

    filter_expression = endpoint_target.build_kv_cursor_filter_expression(
        project=TEST_PROJECT, labels=["lbl1", "lbl2"]
    )
    assert (
        filter_expression
        == f"project=='{TEST_PROJECT}' AND exists(_lbl1) AND exists(_lbl2)"
    )

    filter_expression = endpoint_target.build_kv_cursor_filter_expression(
        project=TEST_PROJECT, labels=["lbl1=1", "lbl2=2"]
    )
    assert (
        filter_expression == f"project=='{TEST_PROJECT}' AND _lbl1=='1' AND _lbl2=='2'"
    )


def test_get_access_key():
    key = mlrun.api.crud.ModelEndpoints().get_access_key(
        mlrun.api.schemas.AuthInfo(data_session="asd")
    )
    assert key == "asd"

    with pytest.raises(MLRunBadRequestError):
        mlrun.api.crud.ModelEndpoints().get_access_key(mlrun.api.schemas.AuthInfo())


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

    # Initialize endpoint store target object
    store_type_object = mlrun.api.crud.model_monitoring.ModelEndpointStoreType(
        value="kv"
    )

    endpoint_target = store_type_object.to_endpoint_target(
        project=TEST_PROJECT, access_key=V3IO_ACCESS_KEY
    )

    features = endpoint_target.get_endpoint_features(feature_names, stats, stats)
    assert len(features) == 4
    # Commented out asserts should be re-enabled once buckets/counts length mismatch bug is fixed
    for feature in features:
        assert feature.expected is not None
        assert feature.actual is not None

        assert feature.expected.histogram is not None
        # assert len(feature.expected.histogram.buckets) == len(feature.expected.histogram.counts)

        assert feature.actual.histogram is not None
        # assert len(feature.actual.histogram.buckets) == len(feature.actual.histogram.counts)

    features = endpoint_target.get_endpoint_features(feature_names, stats, None)
    assert len(features) == 4
    for feature in features:
        assert feature.expected is not None
        assert feature.actual is None

        assert feature.expected.histogram is not None
        # assert len(feature.expected.histogram.buckets) == len(feature.expected.histogram.counts)

    features = endpoint_target.get_endpoint_features(feature_names, None, stats)
    assert len(features) == 4
    for feature in features:
        assert feature.expected is None
        assert feature.actual is not None

        assert feature.actual.histogram is not None
        # assert len(feature.actual.histogram.buckets) == len(feature.actual.histogram.counts)

    features = endpoint_target.get_endpoint_features(feature_names[1:], None, stats)
    assert len(features) == 3


def _get_auth_info() -> mlrun.api.schemas.AuthInfo:
    return mlrun.api.schemas.AuthInfo(data_session=os.environ.get("V3IO_ACCESS_KEY"))


def _mock_random_endpoint(state: Optional[str] = None) -> ModelEndpoint:
    def random_labels():
        return {f"{choice(string.ascii_letters)}": randint(0, 100) for _ in range(1, 5)}

    return ModelEndpoint(
        metadata=ModelEndpointMetadata(project=TEST_PROJECT, labels=random_labels()),
        spec=ModelEndpointSpec(
            function_uri=f"test/function_{randint(0, 100)}:v{randint(0, 100)}",
            model=f"model_{randint(0, 100)}:v{randint(0, 100)}",
            model_class="classifier",
        ),
        status=ModelEndpointStatus(state=state),
    )


def test_sql_target_list_model_endpoints():
    """Testing list model endpoint using _ModelEndpointSQLStore object. In the following test
    we create two model endpoints and list these endpoints. In addition, this test validates the
    filter optional operation within the list model endpoints API. At the end of this test, we validate
    that the model endpoints are deleted from the DB.
    """

    # Generate model endpoint target
    store_type_object = mlrun.api.crud.model_monitoring.ModelEndpointStoreType(
        value="sql"
    )
    endpoint_target = store_type_object.to_endpoint_target(
        project=TEST_PROJECT, connection_string=CONNECTION_STRING
    )

    # First, validate that there are no model endpoints records at the moment
    try:
        list_of_endpoints = endpoint_target.list_model_endpoints()
        endpoint_target.delete_model_endpoints_resources(endpoints=list_of_endpoints)
        list_of_endpoints = endpoint_target.list_model_endpoints()
        assert len(list_of_endpoints.endpoints) == 0

    except sqlalchemy.exc.NoSuchTableError:
        # Model endpoints table was yet to be created
        # This table will be created automatically in the first model endpoint recording
        pass

    # Generate and write the 1st model endpoint into the DB table
    mock_endpoint_1 = _mock_random_endpoint()
    endpoint_target.write_model_endpoint(endpoint=mock_endpoint_1)

    # Validate that there is a single model endpoint
    list_of_endpoints = endpoint_target.list_model_endpoints()
    assert len(list_of_endpoints.endpoints) == 1

    # Generate and write the 2nd model endpoint into the DB table
    mock_endpoint_2 = _mock_random_endpoint()
    mock_endpoint_2.spec.model = "test_model"
    endpoint_target.write_model_endpoint(endpoint=mock_endpoint_2)

    # Validate that there are exactly two model endpoints within the DB
    list_of_endpoints = endpoint_target.list_model_endpoints()
    assert len(list_of_endpoints.endpoints) == 2

    # List only the model endpoint that has the model test_model
    filtered_list_of_endpoints = endpoint_target.list_model_endpoints(
        model="test_model"
    )
    assert len(filtered_list_of_endpoints.endpoints) == 1

    # Clean model endpoints from DB
    endpoint_target.delete_model_endpoints_resources(endpoints=list_of_endpoints)
    list_of_endpoints = endpoint_target.list_model_endpoints()
    assert (len(list_of_endpoints.endpoints)) == 0


def test_sql_target_patch_endpoint():
    """Testing the update of a model endpoint using _ModelEndpointSQLStore object. In the following
    test we update attributes within the model endpoint spec and status and then validate that there
    attributes were actually updated.
    """

    # Generate model endpoint target
    store_type_object = mlrun.api.crud.model_monitoring.ModelEndpointStoreType(
        value="sql"
    )
    endpoint_target = store_type_object.to_endpoint_target(
        project=TEST_PROJECT, connection_string=CONNECTION_STRING
    )

    # First, validate that there are no model endpoints records at the moment
    try:
        list_of_endpoints = endpoint_target.list_model_endpoints()
        endpoint_target.delete_model_endpoints_resources(endpoints=list_of_endpoints)
        list_of_endpoints = endpoint_target.list_model_endpoints()
        assert len(list_of_endpoints.endpoints) == 0

    except sqlalchemy.exc.NoSuchTableError:
        # Model endpoints table was yet to be created
        # This table will be created automatically in the first model endpoint recording
        pass

    # Generate and write the model endpoint into the DB table
    mock_endpoint = _mock_random_endpoint()
    mock_endpoint.metadata.uid = "1234"
    endpoint_target.write_model_endpoint(mock_endpoint)

    # Generate dictionary of attributes and update the model endpoint
    updated_attributes = {"model": "test_model", "latency_avg_1h": 5.2}
    endpoint_target.update_model_endpoint(
        endpoint_id=mock_endpoint.metadata.uid, attributes=updated_attributes
    )

    # Validate that these attributes were actually updated
    endpoint = endpoint_target.get_model_endpoint(
        endpoint_id=mock_endpoint.metadata.uid
    )
    assert endpoint.spec.model == "test_model"
    assert endpoint.status.latency_avg_1h == 5.2

    # Clear model endpoint from DB
    endpoint_target.delete_model_endpoint(endpoint_id=mock_endpoint.metadata.uid)
