# Copyright 2024 Iguazio
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
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone

import pytest
import taosws

import mlrun
from mlrun.common.schemas.model_monitoring import (
    ModelEndpointMonitoringMetric,
    ModelEndpointMonitoringMetricType,
)
from mlrun.datastore.datastore_profile import DatastoreProfileTDEngine
from mlrun.model_monitoring.db.tsdb.tdengine import TDEngineConnector
from mlrun.model_monitoring.db.tsdb.tdengine.tdengine_connection import TDEngineError

project = "test-tdengine-connector"
connection_string = os.getenv("MLRUN_MODEL_ENDPOINT_MONITORING__TSDB_CONNECTION")


def drop_database(connection: taosws.Connection, name: str) -> None:
    connection.execute(f"DROP DATABASE IF EXISTS {name}")


def is_tdengine_defined() -> bool:
    return connection_string is not None and connection_string.startswith("taosws://")


@pytest.fixture
def connector(monkeypatch: pytest.MonkeyPatch) -> Iterator[TDEngineConnector]:
    profile = DatastoreProfileTDEngine.from_dsn(
        profile_name="mm-profile", dsn=connection_string
    )

    monkeypatch.setattr(mlrun.mlconf, "system_id", uuid.uuid4().hex)

    conn = TDEngineConnector(project, profile=profile)
    connection = taosws.connect(connection_string)
    drop_database(connection, conn.database)
    try:
        yield conn
    finally:
        drop_database(connection, conn.database)


@pytest.mark.parametrize(("with_result_extra_data"), [False, True])
@pytest.mark.skipif(not is_tdengine_defined(), reason="TDEngine is not defined")
def test_write_application_event(
    connector: TDEngineConnector, with_result_extra_data: bool
) -> None:
    endpoint_id = "1"
    app_name = "my_app"
    result_name = "my_Result"
    result_kind = 0
    start_infer_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end_infer_time = datetime(2024, 1, 1, second=1, tzinfo=timezone.utc)
    result_status = 0
    result_value = 123
    data = {
        "endpoint_id": endpoint_id,
        "application_name": app_name,
        "result_name": result_name,
        "result_kind": result_kind,
        "start_infer_time": start_infer_time,
        "end_infer_time": end_infer_time,
        "result_status": result_status,
        # make sure we can write apostrophes (ML-7535)
        "result_extra_data": """{"question": "Who wrote 'To Kill a Mockingbird'?"}""",
        "result_value": result_value,
    }

    with pytest.raises(TDEngineError, match="Database not exist"):
        connector.write_application_event(data)
    connector.create_tables()  # DB is created here
    connector.write_application_event(data)
    start_read_time = datetime(2023, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
    end_read_time = datetime(2025, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
    read_data_kwargs = {
        "endpoint_id": endpoint_id,
        "start": start_read_time,
        "end": end_read_time,
        "metrics": [
            ModelEndpointMonitoringMetric(
                project=project,
                app=app_name,
                name=result_name,
                type=ModelEndpointMonitoringMetricType.RESULT,
            ),
        ],
        "type": "results",
        "with_result_extra_data": with_result_extra_data,
    }

    # Write another event with different endpoint_id and result_status
    data_v2 = data.copy()
    data_v2["endpoint_id"] = "2"
    data_v2["result_status"] = 2

    connector.write_application_event(data_v2)

    read_back_results = connector.read_metrics_data(**read_data_kwargs)
    assert len(read_back_results) == 1
    read_back_result = read_back_results[0]
    assert read_back_result.full_name == f"{project}.{app_name}.result.{result_name}"
    assert read_back_result.data
    assert read_back_result.result_kind.value == result_kind
    assert read_back_result.type == "result"
    assert len(read_back_result.values) == 1
    read_back_values = read_back_result.values[0]
    assert read_back_values.timestamp == end_infer_time
    assert read_back_values.value == result_value
    assert read_back_values.status == result_status
    if with_result_extra_data:
        assert read_back_values.extra_data == data["result_extra_data"]

    # Check count results by status
    count_results_by_status = connector.count_results_by_status(
        start=start_infer_time, end=end_infer_time
    )
    assert len(count_results_by_status) == 2
    assert count_results_by_status[(data["application_name"], 0)] == 1
    assert count_results_by_status[(data_v2["application_name"], 2)] == 1

    # Check count results by status for specific endpoint_id
    count_results_by_status = connector.count_results_by_status(
        start=start_infer_time, end=end_infer_time, endpoint_ids=endpoint_id
    )
    assert len(count_results_by_status) == 1
    assert count_results_by_status[(data["application_name"], 0)] == 1

    # check processed model endpoints
    processed_model_endpoints = connector.count_processed_model_endpoints(
        start=start_infer_time, end=end_infer_time
    )
    assert processed_model_endpoints == {"my_app": 2}

    # calculate latest metrics
    latest_metrics = connector.calculate_latest_metrics(
        start=start_infer_time, end=end_infer_time, application_names="my_app"
    )

    assert len(latest_metrics) == 2
    first_metric = latest_metrics[0]
    assert first_metric.status == 0
    assert first_metric.value == 123

    second_metric = latest_metrics[1]
    assert second_metric.status == 2
    assert second_metric.value == 123

    # now let's write another result with different app and result_status
    data_v3 = data.copy()
    data_v3["application_name"] = "another_app"
    data_v3["result_status"] = 2
    connector.write_application_event(data_v3)

    # Check count results by status for specific application_name
    count_results_by_status = connector.count_results_by_status(
        start=start_infer_time, end=end_infer_time, application_names=["another_app"]
    )
    assert len(count_results_by_status) == 1
    assert count_results_by_status[(data_v3["application_name"], 2)] == 1

    # Check count results by status for specific result_status
    count_results_by_status = connector.count_results_by_status(
        start=start_infer_time, end=end_infer_time, result_status_list=[2]
    )
    assert len(count_results_by_status) == 2
    assert count_results_by_status[(data_v2["application_name"], 2)] == 1
    assert count_results_by_status[(data_v3["application_name"], 2)] == 1

    # check processed model endpoints
    processed_model_endpoints = connector.count_processed_model_endpoints(
        start=start_infer_time, end=end_infer_time
    )
    assert processed_model_endpoints == {"another_app": 1, "my_app": 2}

    # check latest metrics for specific application_name
    latest_metrics = connector.calculate_latest_metrics(
        start=start_infer_time, end=end_infer_time, application_names="another_app"
    )
    assert len(latest_metrics) == 1

    # Delete resources and verify that database is deleted
    connector.delete_tsdb_records(endpoint_ids=[endpoint_id, "123"])
    read_back_results = connector.read_metrics_data(**read_data_kwargs)
    read_back_result = read_back_results[0]
    assert not read_back_result.data

    # Delete database
    connector.delete_tsdb_resources()

    with pytest.raises(TDEngineError):
        connector.read_metrics_data(**read_data_kwargs)


def test_tdengine_connector_requires_system_id() -> None:
    """
    Test that TDEngineConnector raises an error when system_id is not set.

    This test verifies that the constructor validates system_id before attempting
    to construct the database name.
    """
    # Save the original system_id to restore later
    original_system_id = mlrun.mlconf.system_id

    try:
        # Clear system_id
        mlrun.mlconf.system_id = ""

        # Use a dummy DSN - we're not actually connecting to TDEngine
        dummy_dsn = "taosws://testuser:testpass@localhost:6041"
        profile = DatastoreProfileTDEngine.from_dsn(
            profile_name="test-profile", dsn=dummy_dsn
        )

        # Attempt to create TDEngineConnector without system_id should raise error
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError, match="system_id.*not set"
        ):
            TDEngineConnector(project, profile=profile)

    finally:
        # Restore original system_id
        mlrun.mlconf.system_id = original_system_id
