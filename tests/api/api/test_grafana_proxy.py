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
import unittest.mock
from datetime import datetime, timedelta
from random import randint
from typing import Optional

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pytest import fail
from sqlalchemy.orm import Session
from v3io.dataplane import RaiseForStatus
from v3io_frames import CreateError
from v3io_frames import frames_pb2 as fpb2

import mlrun
import mlrun.api.crud
import mlrun.api.schemas
import mlrun.api.utils.clients.iguazio
from mlrun.api.api.endpoints.grafana_proxy import (
    _parse_query_parameters,
    _validate_query_parameters,
)
from mlrun.config import config
from mlrun.errors import MLRunBadRequestError
from mlrun.utils.model_monitoring import parse_model_endpoint_store_prefix
from mlrun.utils.v3io_clients import get_frames_client, get_v3io_client
from tests.api.api.test_model_endpoints import _mock_random_endpoint

ENV_PARAMS = {"V3IO_ACCESS_KEY", "V3IO_API", "V3IO_FRAMESD"}
TEST_PROJECT = "test3"


def _build_skip_message():
    return f"One of the required environment params is not initialized ({', '.join(ENV_PARAMS)})"


def _is_env_params_dont_exist() -> bool:
    return not all((os.environ.get(r, False) for r in ENV_PARAMS))


def test_grafana_proxy_model_endpoints_check_connection(
    db: Session, client: TestClient
):
    mlrun.mlconf.httpdb.authentication.mode = "iguazio"
    mlrun.api.utils.clients.iguazio.AsyncClient().verify_request_session = (
        unittest.mock.AsyncMock(
            return_value=(
                mlrun.api.schemas.AuthInfo(
                    username=None,
                    session="some-session",
                    data_session="some-session",
                    user_id=None,
                    user_unix_id=0,
                    user_group_ids=[],
                )
            )
        )
    )
    response = client.get(
        url="grafana-proxy/model-endpoints",
    )
    assert response.status_code == 200


@pytest.mark.skipif(
    _is_env_params_dont_exist(),
    reason=_build_skip_message(),
)
def test_grafana_list_endpoints(db: Session, client: TestClient):
    endpoints_in = [_mock_random_endpoint("active") for _ in range(5)]

    # Initialize endpoint store target object
    store_type_object = mlrun.api.crud.model_monitoring.ModelEndpointStoreType(
        value="kv"
    )
    endpoint_store = store_type_object.to_endpoint_store(
        project=TEST_PROJECT, access_key=_get_access_key()
    )

    for endpoint in endpoints_in:
        endpoint_store.write_model_endpoint(endpoint)

    response = client.post(
        url="grafana-proxy/model-endpoints/query",
        headers={"X-V3io-Session-Key": _get_access_key()},
        json={
            "targets": [
                {"target": f"project={TEST_PROJECT};target_endpoint=list_endpoints"}
            ]
        },
    )

    response_json = response.json()
    if not response_json:
        fail(f"Empty response, expected list of dictionaries. {response_json}")

    response_json = response_json[0]
    if not response_json:
        fail(
            f"Empty dictionary, expected dictionary with 'columns', 'rows' and 'type' fields. {response_json}"
        )

    if "columns" not in response_json:
        fail(f"Missing 'columns' key in response dictionary. {response_json}")

    if "rows" not in response_json:
        fail(f"Missing 'rows' key in response dictionary. {response_json}")

    if "type" not in response_json:
        fail(f"Missing 'type' key in response dictionary. {response_json}")

    assert len(response_json["rows"]) == 5


@pytest.mark.skipif(
    _is_env_params_dont_exist(),
    reason=_build_skip_message(),
)
def test_grafana_individual_feature_analysis(db: Session, client: TestClient):
    endpoint_data = {
        "timestamp": "2021-02-28 21:02:58.642108",
        "project": TEST_PROJECT,
        "model": "test-model",
        "function": "v2-model-server",
        "tag": "latest",
        "model_class": "ClassifierModel",
        "endpoint_id": "test.test_id",
        "labels": "null",
        "latency_avg_1s": 42427.0,
        "predictions_per_second_count_1s": 141,
        "first_request": "2021-02-28 21:02:58.642108",
        "last_request": "2021-02-28 21:02:58.642108",
        "error_count": 0,
        "feature_names": '["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]',
        "feature_stats": '{"sepal length (cm)": {"count": 30, "mean": 5.946666666666668, "std": 0.8394305678023165, "min": 4.7, "max": 7.9, "hist": [[4, 4, 4, 4, 4, 3, 4, 0, 3, 4, 1, 1, 2, 1, 0, 1, 0, 0, 1, 1], [4.7, 4.86, 5.0200000000000005, 5.18, 5.34, 5.5, 5.66, 5.82, 5.98, 6.140000000000001, 6.300000000000001, 6.46, 6.62, 6.78, 6.94, 7.1, 7.26, 7.42, 7.58, 7.74, 7.9]]}, "sepal width (cm)": {"count": 30, "mean": 3.119999999999999, "std": 0.4088672324766359, "min": 2.2, "max": 3.8, "hist": [[1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 3, 3, 2, 2, 0, 3, 1, 1, 0, 4], [2.2, 2.2800000000000002, 2.3600000000000003, 2.44, 2.52, 2.6, 2.68, 2.7600000000000002, 2.84, 2.92, 3, 3.08, 3.16, 3.24, 3.3200000000000003, 3.4, 3.48, 3.56, 3.6399999999999997, 3.7199999999999998, 3.8]]}, "petal length (cm)": {"count": 30, "mean": 3.863333333333333, "std": 1.8212317418360753, "min": 1.3, "max": 6.7, "hist": [[6, 6, 6, 6, 6, 6, 0, 0, 1, 2, 0, 3, 3, 2, 2, 3, 1, 1, 1, 1], [1.3, 1.57, 1.84, 2.1100000000000003, 2.38, 2.6500000000000004, 2.92, 3.1900000000000004, 3.46, 3.7300000000000004, 4, 4.2700000000000005, 4.54, 4.8100000000000005, 5.08, 5.3500000000000005, 5.62, 5.89, 6.16, 6.430000000000001, 6.7]]}, "petal width (cm)": {"count": 30, "mean": 1.2733333333333334, "std": 0.8291804567674381, "min": 0.1, "max": 2.5, "hist": [[5, 5, 5, 5, 5, 5, 0, 0, 1, 2, 3, 2, 1, 0, 2, 3, 1, 1, 0, 4], [0.1, 0.22, 0.33999999999999997, 0.45999999999999996, 0.58, 0.7, 0.82, 0.94, 1.06, 1.1800000000000002, 1.3, 1.42, 1.54, 1.6600000000000001, 1.78, 1.9, 2.02, 2.14, 2.2600000000000002, 2.38, 2.5]]}}',  # noqa
        "current_stats": '{"petal length (cm)": {"count": 100.0, "mean": 2.861, "std": 1.4495485190537463, "min": 1.0, "max": 5.1, "hist": [[4, 20, 20, 4, 2, 0, 0, 0, 0, 1, 0, 2, 3, 2, 8, 7, 6, 10, 7, 4], [1.0, 1.205, 1.41, 1.615, 1.8199999999999998, 2.025, 2.23, 2.4349999999999996, 2.6399999999999997, 2.8449999999999998, 3.05, 3.255, 3.46, 3.665, 3.8699999999999997, 4.074999999999999, 4.279999999999999, 4.484999999999999, 4.6899999999999995, 4.895, 5.1]]}, "petal width (cm)": {"count": 100.0, "mean": 5.471000000000001, "std": 0.6416983463254116, "min": 4.3, "max": 7.0, "hist": [[4, 1, 6, 5, 5, 19, 4, 1, 13, 5, 7, 6, 4, 4, 5, 2, 1, 5, 1, 2], [4.3, 4.435, 4.57, 4.705, 4.84, 4.975, 5.109999999999999, 5.245, 5.38, 5.515, 5.65, 5.785, 5.92, 6.055, 6.1899999999999995, 6.325, 6.46, 6.595, 6.73, 6.865, 7.0]]}, "sepal length (cm)": {"count": 100.0, "mean": 0.7859999999999998, "std": 0.5651530587354012, "min": 0.1, "max": 1.8, "hist": [[5, 29, 7, 7, 1, 1, 0, 0, 0, 0, 7, 3, 5, 0, 13, 7, 10, 3, 1, 1], [0.1, 0.185, 0.27, 0.355, 0.43999999999999995, 0.5249999999999999, 0.61, 0.695, 0.7799999999999999, 0.8649999999999999, 0.9499999999999998, 1.035, 1.12, 1.205, 1.29, 1.375, 1.46, 1.545, 1.63, 1.7149999999999999, 1.8]]}, "sepal width (cm)": {"count": 100.0, "mean": 3.0989999999999998, "std": 0.4787388735948953, "min": 2.0, "max": 4.4, "hist": [[1, 2, 4, 3, 4, 8, 6, 8, 14, 7, 11, 10, 6, 3, 7, 2, 1, 1, 1, 1], [2.0, 2.12, 2.24, 2.3600000000000003, 2.48, 2.6, 2.72, 2.8400000000000003, 2.96, 3.08, 3.2, 3.3200000000000003, 3.4400000000000004, 3.5600000000000005, 3.6800000000000006, 3.8000000000000003, 3.9200000000000004, 4.040000000000001, 4.16, 4.28, 4.4]]}}',  # noqa
        "drift_measures": '{"petal width (cm)": {"tvd": 0.4, "hellinger": 0.38143130942893605, "kld": 1.3765624725652992}, "tvd_sum": 1.755886699507389, "tvd_mean": 0.43897167487684724, "hellinger_sum": 1.7802062191831514, "hellinger_mean": 0.44505155479578784, "kld_sum": 9.133613874253776, "kld_mean": 2.283403468563444, "sepal width (cm)": {"tvd": 0.3551724137931034, "hellinger": 0.4024622641158891, "kld": 1.7123635755188409}, "petal length (cm)": {"tvd": 0.445, "hellinger": 0.39975075965755447, "kld": 1.6449612084377268}, "sepal length (cm)": {"tvd": 0.5557142857142856, "hellinger": 0.5965618859807716, "kld": 4.399726617731908}}',  # noqa
    }

    v3io = get_v3io_client(endpoint=config.v3io_api, access_key=_get_access_key())

    v3io.kv.put(
        container="projects",
        table_path=f"{TEST_PROJECT}/model-endpoints/endpoints",
        key="test.test_id",
        attributes=endpoint_data,
    )

    response = client.post(
        url="grafana-proxy/model-endpoints/query",
        headers={"X-V3io-Session-Key": _get_access_key()},
        json={
            "targets": [
                {
                    "target": f"project={TEST_PROJECT};endpoint_id=test.test_id;target_endpoint=individual_feature_analysis"  # noqa
                }
            ]
        },
    )

    assert response.status_code == 200

    response_json = response.json()

    assert len(response_json) == 1
    assert "columns" in response_json[0]
    assert "rows" in response_json[0]
    assert len(response_json[0]["rows"]) == 4


@pytest.mark.skipif(
    _is_env_params_dont_exist(),
    reason=_build_skip_message(),
)
def test_grafana_individual_feature_analysis_missing_field_doesnt_fail(
    db: Session, client: TestClient
):
    endpoint_data = {
        "timestamp": "2021-02-28 21:02:58.642108",
        "project": TEST_PROJECT,
        "model": "test-model",
        "function": "v2-model-server",
        "tag": "latest",
        "model_class": "ClassifierModel",
        "endpoint_id": "test.test_id",
        "labels": "null",
        "latency_avg_1s": 42427.0,
        "predictions_per_second_count_1s": 141,
        "first_request": "2021-02-28 21:02:58.642108",
        "last_request": "2021-02-28 21:02:58.642108",
        "error_count": 0,
        "feature_names": '["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]',
        "feature_stats": '{"sepal length (cm)": {"count": 30, "mean": 5.946666666666668, "std": 0.8394305678023165, "min": 4.7, "max": 7.9, "hist": [[4, 4, 4, 4, 4, 3, 4, 0, 3, 4, 1, 1, 2, 1, 0, 1, 0, 0, 1, 1], [4.7, 4.86, 5.0200000000000005, 5.18, 5.34, 5.5, 5.66, 5.82, 5.98, 6.140000000000001, 6.300000000000001, 6.46, 6.62, 6.78, 6.94, 7.1, 7.26, 7.42, 7.58, 7.74, 7.9]]}, "sepal width (cm)": {"count": 30, "mean": 3.119999999999999, "std": 0.4088672324766359, "min": 2.2, "max": 3.8, "hist": [[1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 3, 3, 2, 2, 0, 3, 1, 1, 0, 4], [2.2, 2.2800000000000002, 2.3600000000000003, 2.44, 2.52, 2.6, 2.68, 2.7600000000000002, 2.84, 2.92, 3, 3.08, 3.16, 3.24, 3.3200000000000003, 3.4, 3.48, 3.56, 3.6399999999999997, 3.7199999999999998, 3.8]]}, "petal length (cm)": {"count": 30, "mean": 3.863333333333333, "std": 1.8212317418360753, "min": 1.3, "max": 6.7, "hist": [[6, 6, 6, 6, 6, 6, 0, 0, 1, 2, 0, 3, 3, 2, 2, 3, 1, 1, 1, 1], [1.3, 1.57, 1.84, 2.1100000000000003, 2.38, 2.6500000000000004, 2.92, 3.1900000000000004, 3.46, 3.7300000000000004, 4, 4.2700000000000005, 4.54, 4.8100000000000005, 5.08, 5.3500000000000005, 5.62, 5.89, 6.16, 6.430000000000001, 6.7]]}, "petal width (cm)": {"count": 30, "mean": 1.2733333333333334, "std": 0.8291804567674381, "min": 0.1, "max": 2.5, "hist": [[5, 5, 5, 5, 5, 5, 0, 0, 1, 2, 3, 2, 1, 0, 2, 3, 1, 1, 0, 4], [0.1, 0.22, 0.33999999999999997, 0.45999999999999996, 0.58, 0.7, 0.82, 0.94, 1.06, 1.1800000000000002, 1.3, 1.42, 1.54, 1.6600000000000001, 1.78, 1.9, 2.02, 2.14, 2.2600000000000002, 2.38, 2.5]]}}',  # noqa
        "drift_measures": '{"petal width (cm)": {"tvd": 0.4, "hellinger": 0.38143130942893605, "kld": 1.3765624725652992}, "tvd_sum": 1.755886699507389, "tvd_mean": 0.43897167487684724, "hellinger_sum": 1.7802062191831514, "hellinger_mean": 0.44505155479578784, "kld_sum": 9.133613874253776, "kld_mean": 2.283403468563444, "sepal width (cm)": {"tvd": 0.3551724137931034, "hellinger": 0.4024622641158891, "kld": 1.7123635755188409}, "petal length (cm)": {"tvd": 0.445, "hellinger": 0.39975075965755447, "kld": 1.6449612084377268}, "sepal length (cm)": {"tvd": 0.5557142857142856, "hellinger": 0.5965618859807716, "kld": 4.399726617731908}}',  # noqa
    }

    v3io = get_v3io_client(endpoint=config.v3io_api, access_key=_get_access_key())

    v3io.kv.put(
        container="projects",
        table_path=f"{TEST_PROJECT}/model-endpoints/endpoints",
        key="test.test_id",
        attributes=endpoint_data,
    )

    response = client.post(
        url="grafana-proxy/model-endpoints/query",
        headers={"X-V3io-Session-Key": _get_access_key()},
        json={
            "targets": [
                {
                    "target": f"project={TEST_PROJECT};endpoint_id=test.test_id;target_endpoint=individual_feature_analysis"  # noqa
                }
            ]
        },
    )

    assert response.status_code == 200

    response_json = response.json()

    assert len(response_json) == 1
    assert "columns" in response_json[0]
    assert "rows" in response_json[0]
    assert len(response_json[0]["rows"]) == 4

    for row in response_json[0]["rows"]:
        assert row[0] is not None
        assert all(map(lambda e: e is None, row[1:4]))
        assert all(map(lambda e: e is not None, row[4:10]))


@pytest.mark.skipif(
    _is_env_params_dont_exist(),
    reason=_build_skip_message(),
)
def test_grafana_overall_feature_analysis(db: Session, client: TestClient):
    endpoint_data = {
        "timestamp": "2021-02-28 21:02:58.642108",
        "project": TEST_PROJECT,
        "model": "test-model",
        "function": "v2-model-server",
        "tag": "latest",
        "model_class": "ClassifierModel",
        "endpoint_id": "test.test_id",
        "labels": "null",
        "latency_avg_1s": 42427.0,
        "predictions_per_second_count_1s": 141,
        "first_request": "2021-02-28 21:02:58.642108",
        "last_request": "2021-02-28 21:02:58.642108",
        "error_count": 0,
        "feature_names": '["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]',
        "feature_stats": '{"sepal length (cm)": {"count": 30, "mean": 5.946666666666668, "std": 0.8394305678023165, "min": 4.7, "max": 7.9, "hist": [[4, 4, 4, 4, 4, 3, 4, 0, 3, 4, 1, 1, 2, 1, 0, 1, 0, 0, 1, 1], [4.7, 4.86, 5.0200000000000005, 5.18, 5.34, 5.5, 5.66, 5.82, 5.98, 6.140000000000001, 6.300000000000001, 6.46, 6.62, 6.78, 6.94, 7.1, 7.26, 7.42, 7.58, 7.74, 7.9]]}, "sepal width (cm)": {"count": 30, "mean": 3.119999999999999, "std": 0.4088672324766359, "min": 2.2, "max": 3.8, "hist": [[1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 3, 3, 2, 2, 0, 3, 1, 1, 0, 4], [2.2, 2.2800000000000002, 2.3600000000000003, 2.44, 2.52, 2.6, 2.68, 2.7600000000000002, 2.84, 2.92, 3, 3.08, 3.16, 3.24, 3.3200000000000003, 3.4, 3.48, 3.56, 3.6399999999999997, 3.7199999999999998, 3.8]]}, "petal length (cm)": {"count": 30, "mean": 3.863333333333333, "std": 1.8212317418360753, "min": 1.3, "max": 6.7, "hist": [[6, 6, 6, 6, 6, 6, 0, 0, 1, 2, 0, 3, 3, 2, 2, 3, 1, 1, 1, 1], [1.3, 1.57, 1.84, 2.1100000000000003, 2.38, 2.6500000000000004, 2.92, 3.1900000000000004, 3.46, 3.7300000000000004, 4, 4.2700000000000005, 4.54, 4.8100000000000005, 5.08, 5.3500000000000005, 5.62, 5.89, 6.16, 6.430000000000001, 6.7]]}, "petal width (cm)": {"count": 30, "mean": 1.2733333333333334, "std": 0.8291804567674381, "min": 0.1, "max": 2.5, "hist": [[5, 5, 5, 5, 5, 5, 0, 0, 1, 2, 3, 2, 1, 0, 2, 3, 1, 1, 0, 4], [0.1, 0.22, 0.33999999999999997, 0.45999999999999996, 0.58, 0.7, 0.82, 0.94, 1.06, 1.1800000000000002, 1.3, 1.42, 1.54, 1.6600000000000001, 1.78, 1.9, 2.02, 2.14, 2.2600000000000002, 2.38, 2.5]]}}',  # noqa
        "drift_measures": '{"petal width (cm)": {"tvd": 0.4, "hellinger": 0.38143130942893605, "kld": 1.3765624725652992}, "tvd_sum": 1.755886699507389, "tvd_mean": 0.43897167487684724, "hellinger_sum": 1.7802062191831514, "hellinger_mean": 0.44505155479578784, "kld_sum": 9.133613874253776, "kld_mean": 2.283403468563444, "sepal width (cm)": {"tvd": 0.3551724137931034, "hellinger": 0.4024622641158891, "kld": 1.7123635755188409}, "petal length (cm)": {"tvd": 0.445, "hellinger": 0.39975075965755447, "kld": 1.6449612084377268}, "sepal length (cm)": {"tvd": 0.5557142857142856, "hellinger": 0.5965618859807716, "kld": 4.399726617731908}}',  # noqa
    }

    v3io = get_v3io_client(endpoint=config.v3io_api, access_key=_get_access_key())

    v3io.kv.put(
        container="projects",
        table_path=f"{TEST_PROJECT}/model-endpoints/endpoints",
        key="test.test_id",
        attributes=endpoint_data,
    )

    response = client.post(
        url="grafana-proxy/model-endpoints/query",
        headers={"X-V3io-Session-Key": _get_access_key()},
        json={
            "targets": [
                {
                    "target": f"project={TEST_PROJECT};endpoint_id=test.test_id;target_endpoint=overall_feature_analysis"  # noqa
                }
            ]
        },
    )

    assert response.status_code == 200

    response_json = response.json()

    assert len(response_json) == 1
    assert "columns" in response_json[0]
    assert "rows" in response_json[0]
    assert len(response_json[0]["rows"][0]) == 6


def test_parse_query_parameters_failure():
    # No 'targets' in body
    with pytest.raises(MLRunBadRequestError):
        _parse_query_parameters({})

    # No 'target' list in 'targets' dictionary
    with pytest.raises(MLRunBadRequestError):
        _parse_query_parameters({"targets": []})

    # Target query not separated by equals ('=') char
    with pytest.raises(MLRunBadRequestError):
        _parse_query_parameters({"targets": [{"target": "test"}]})


def test_parse_query_parameters_success():
    # Target query separated by equals ('=') char
    params = _parse_query_parameters({"targets": [{"target": "test=some_test"}]})
    assert params["test"] == "some_test"

    # Target query separated by equals ('=') char (multiple queries)
    params = _parse_query_parameters(
        {"targets": [{"target": "test=some_test;another_test=some_other_test"}]}
    )
    assert params["test"] == "some_test"
    assert params["another_test"] == "some_other_test"

    params = _parse_query_parameters(
        {"targets": [{"target": "test=some_test;another_test=some_other_test;"}]}
    )
    assert params["test"] == "some_test"
    assert params["another_test"] == "some_other_test"


def test_validate_query_parameters_failure():
    # No 'target_endpoint' in query parameters
    with pytest.raises(MLRunBadRequestError):
        _validate_query_parameters({})

    # target_endpoint unsupported
    with pytest.raises(MLRunBadRequestError):
        _validate_query_parameters(
            {"target_endpoint": "unsupported_endpoint"}, {"supported_endpoint"}
        )


def test_validate_query_parameters_success():
    _validate_query_parameters(
        {"target_endpoint": "list_endpoints"}, {"list_endpoints"}
    )


def _get_access_key() -> Optional[str]:
    return os.environ.get("V3IO_ACCESS_KEY")


@pytest.fixture(autouse=True)
def cleanup_endpoints(db: Session, client: TestClient):
    if not _is_env_params_dont_exist():
        kv_path = config.model_endpoint_monitoring.store_prefixes.default.format(
            project=TEST_PROJECT,
            kind=mlrun.api.schemas.ModelMonitoringStoreKinds.ENDPOINTS,
        )
        _, kv_container, kv_path = parse_model_endpoint_store_prefix(kv_path)

        tsdb_path = config.model_endpoint_monitoring.store_prefixes.default.format(
            project=TEST_PROJECT,
            kind=mlrun.api.schemas.ModelMonitoringStoreKinds.EVENTS,
        )
        _, tsdb_container, tsdb_path = parse_model_endpoint_store_prefix(tsdb_path)

        v3io = get_v3io_client(endpoint=config.v3io_api, access_key=_get_access_key())

        frames = get_frames_client(
            token=_get_access_key(),
            container=tsdb_container,
            address=config.v3io_framesd,
        )

        try:
            all_records = v3io.kv.new_cursor(
                container=kv_container,
                table_path=kv_path,
                raise_for_status=RaiseForStatus.never,
            ).all()

            all_records = [r["__name"] for r in all_records]

            # Cleanup KV
            for record in all_records:
                v3io.kv.delete(
                    container=kv_container,
                    table_path=kv_path,
                    key=record,
                    raise_for_status=RaiseForStatus.never,
                )
        except RuntimeError:
            pass

        try:
            # Cleanup TSDB
            frames.delete(
                backend="tsdb",
                table=tsdb_path,
                if_missing=fpb2.IGNORE,
            )
        except CreateError:
            pass


@pytest.mark.skipif(
    _is_env_params_dont_exist(),
    reason=_build_skip_message(),
)
def test_grafana_incoming_features(db: Session, client: TestClient):
    path = config.model_endpoint_monitoring.store_prefixes.default.format(
        project=TEST_PROJECT, kind=mlrun.api.schemas.ModelMonitoringStoreKinds.EVENTS
    )
    _, container, path = parse_model_endpoint_store_prefix(path)

    frames = get_frames_client(
        token=_get_access_key(),
        container=container,
        address=config.v3io_framesd,
    )

    frames.create(backend="tsdb", table=path, rate="10/m", if_exists=1)

    start = datetime.utcnow()
    endpoints = [_mock_random_endpoint() for _ in range(5)]
    for e in endpoints:
        e.spec.feature_names = ["f0", "f1", "f2", "f3"]

    # Initialize endpoint store target object
    store_type_object = mlrun.api.crud.model_monitoring.ModelEndpointStoreType(
        value="kv"
    )
    endpoint_store = store_type_object.to_endpoint_store(
        project=TEST_PROJECT, access_key=_get_access_key()
    )

    for endpoint in endpoints:
        endpoint_store.write_model_endpoint(endpoint)

        total = 0

        dfs = []

        for i in range(10):
            count = randint(1, 10)
            total += count
            data = {
                "f0": i,
                "f1": i + 1,
                "f2": i + 2,
                "f3": i + 3,
                "endpoint_id": endpoint.metadata.uid,
                "timestamp": start - timedelta(minutes=10 - i),
            }
            df = pd.DataFrame(data=[data])
            dfs.append(df)

        frames.write(
            backend="tsdb",
            table=path,
            dfs=dfs,
            index_cols=["timestamp", "endpoint_id"],
        )

    for endpoint in endpoints:
        response = client.post(
            url="grafana-proxy/model-endpoints/query",
            headers={"X-V3io-Session-Key": _get_access_key()},
            json={
                "targets": [
                    {
                        "target": f"project={TEST_PROJECT};endpoint_id={endpoint.metadata.uid};target_endpoint=incoming_features"  # noqa
                    }
                ]
            },
        )
        response = response.json()
        targets = [t["target"] for t in response]
        assert targets == ["f0", "f1", "f2", "f3"]

        lens = [t["datapoints"] for t in response]
        assert all(map(lambda l: len(l) == 10, lens))
