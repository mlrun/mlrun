# Copyright 2025 Iguazio
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
import random
import string
import time
import uuid
from datetime import datetime, timezone

import pytest

import mlrun.common.schemas.model_monitoring as mm_schemas
from mlrun.datastore.datastore_profile import DatastoreProfilePostgreSQL
from mlrun.model_monitoring.db.tsdb.preaggregate import PreAggregateConfig
from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connection import (
    TimescaleDBConnection,
)
from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_operations import (
    TimescaleDBOperationsHandler,
)
from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_query_handler import (
    TimescaleDBQueryHandler,
)

# ===== GLOBAL CONFIGURATION AND SKIP MARKS =====

# Connection string detection - used by ALL TimescaleDB tests
CONNECTION_STRING = os.getenv("MLRUN_MODEL_ENDPOINT_MONITORING__TSDB_CONNECTION")

# Global skip mark that ALL TimescaleDB tests can use
timescaledb_available = pytest.mark.skipif(
    not CONNECTION_STRING or not CONNECTION_STRING.startswith("postgres"),
    reason="TimescaleDB connection string not available or not PostgreSQL",
)

# Apply skip mark globally to this directory
pytestmark = timescaledb_available


# ===== UTILITY FUNCTIONS =====


def reset_global_connection_pool():
    """Reset the global connection pool to ensure clean test state.

    This function is needed by multiple test files to avoid connection leaks
    and ensure test isolation.
    """
    import mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connection as conn_module

    with conn_module._connection_lock:
        if conn_module._connection_pool:
            conn_module._connection_pool.closeall()
            conn_module._connection_pool = None


def generate_unique_name(prefix: str = "test") -> str:
    """Generate a unique name for test resources.

    Used for creating unique database names, project names, etc.
    to avoid conflicts between parallel test runs.
    """
    timestamp = int(time.time())
    random_suffix = uuid.uuid4().hex[:8]
    return f"{prefix}_{timestamp}_{random_suffix}"


# ===== CORE FIXTURES =====


@pytest.fixture(scope="session")
def connection_string():
    """TimescaleDB connection string from environment.

    Used by ALL TimescaleDB tests that need database connectivity.
    Session-scoped since connection string doesn't change during test session.
    """
    return CONNECTION_STRING


@pytest.fixture
def profile(connection_string):
    # Parse postgres://testuser:testpass@192.168.226.26:5432/postgres
    from urllib.parse import urlparse

    parsed = urlparse(connection_string)
    return DatastoreProfilePostgreSQL(
        name="test_profile",
        user=parsed.username,
        password=parsed.password,
        host=parsed.hostname,
        port=parsed.port,
        database=parsed.path.lstrip("/"),
    )


@pytest.fixture
def connection(connection_string):
    return TimescaleDBConnection(connection_string, max_connections=1, autocommit=False)


@pytest.fixture
def operations(connection, project_name):
    return TimescaleDBOperationsHandler(project=project_name, connection=connection)


@pytest.fixture
def pre_aggregate_config():
    """Create a test pre-aggregate configuration."""
    return PreAggregateConfig(
        aggregate_intervals=["10m", "1h"],
        agg_functions=["sum", "avg", "max", "count"],
        retention_policy={
            "raw": "7d",
            "10m": "30d",
            "1h": "1y",
        },
    )


@pytest.fixture
def query_handler(connection, project_name):
    # Create a combined handler that has both query and operations capabilities
    # This avoids the circular import issue
    class TestableTimescaleDBQueryHandler(
        TimescaleDBOperationsHandler, TimescaleDBQueryHandler
    ):
        def __init__(
            self,
            project: str,
            connection: TimescaleDBConnection,
            pre_aggregate_config=None,
        ):
            TimescaleDBOperationsHandler.__init__(
                self,
                project=project,
                connection=connection,
                pre_aggregate_config=pre_aggregate_config,
            )
            # Initialize query handler attributes
            self._pre_aggregate_config = pre_aggregate_config
            if self._pre_aggregate_config:
                from mlrun.model_monitoring.db.tsdb.preaggregate import (
                    PreAggregateHandler,
                )

                self._pre_aggregate_handler = PreAggregateHandler(
                    self._pre_aggregate_config
                )
            else:
                # Create a mock pre-aggregate handler
                class MockPreAggregateHandler:
                    def validate_interval_and_function(self, *args, **kwargs):
                        pass

                    def align_time_range(self, start, end, interval=None):
                        return start, end

                    def can_use_pre_aggregates(self, *args, **kwargs):
                        return False

                    def get_start_end(self, start, end):
                        start = start or datetime.min
                        end = end or datetime.now()
                        return start, end

                self._pre_aggregate_handler = MockPreAggregateHandler()

    handler = TestableTimescaleDBQueryHandler(
        project=project_name,
        connection=connection,
    )

    # Create tables
    handler.create_tables()

    yield handler

    # Cleanup
    handler.delete_tsdb_resources()


@pytest.fixture
def query_handler_with_aggregates(connection, project_name, pre_aggregate_config):
    """Create a TimescaleDB query handler with pre-aggregate configuration."""

    # Reuse the same testable handler class
    class TestableTimescaleDBQueryHandler(
        TimescaleDBOperationsHandler, TimescaleDBQueryHandler
    ):
        def __init__(
            self,
            project: str,
            connection: TimescaleDBConnection,
            pre_aggregate_config=None,
        ):
            TimescaleDBOperationsHandler.__init__(
                self,
                project=project,
                connection=connection,
                pre_aggregate_config=pre_aggregate_config,
            )
            self._pre_aggregate_config = pre_aggregate_config
            if self._pre_aggregate_config:
                from mlrun.model_monitoring.db.tsdb.preaggregate import (
                    PreAggregateHandler,
                )

                self._pre_aggregate_handler = PreAggregateHandler(
                    self._pre_aggregate_config
                )

    handler = TestableTimescaleDBQueryHandler(
        project=project_name,
        connection=connection,
        pre_aggregate_config=pre_aggregate_config,
    )

    # Create tables
    handler.create_tables()

    yield handler

    # Cleanup
    handler.delete_tsdb_resources()


@pytest.fixture
def sample_results():
    """Sample results data for testing."""
    return [
        {
            mm_schemas.WriterEvent.END_INFER_TIME: datetime(2024, 1, 15, 12, 0, 0),
            mm_schemas.WriterEvent.START_INFER_TIME: datetime(2024, 1, 15, 12, 0, 0),
            mm_schemas.WriterEvent.ENDPOINT_ID: "test_endpoint_1",
            mm_schemas.WriterEvent.APPLICATION_NAME: "drift_app",
            mm_schemas.ResultData.RESULT_NAME: "drift_detection",
            mm_schemas.ResultData.RESULT_VALUE: 0.85,
            mm_schemas.ResultData.RESULT_STATUS: mm_schemas.ResultStatusApp.detected.value,
            mm_schemas.ResultData.RESULT_KIND: mm_schemas.ResultKindApp.concept_drift.value,
        }
    ]


@pytest.fixture
def project_name():
    """Generate a unique project name for each test.

    Used by ALL tests that need isolated project resources.
    Replaces duplicate project name generation across test files.
    """
    return generate_unique_name("test_project")


@pytest.fixture
def endpoint_id():
    """Generate a unique endpoint ID for each test."""
    return "test-endpoint-" + "".join(random.choices(string.ascii_lowercase, k=8))


@pytest.fixture
def model_class():
    """Standard model class for tests."""
    return "test-model"


@pytest.fixture
def function_name():
    """Standard function name for tests."""
    return "test-function"


@pytest.fixture
def sample_metrics_data():
    """Sample metrics data for testing."""
    return {
        "accuracy": 0.95,
        "precision": 0.87,
        "recall": 0.92,
        "f1_score": 0.89,
    }


@pytest.fixture
def sample_prediction_data():
    """Sample prediction data for testing."""
    return {
        "predictions": [0.8, 0.2, 0.95, 0.1],
        "probabilities": [[0.2, 0.8], [0.8, 0.2], [0.05, 0.95], [0.9, 0.1]],
        "latency": 45.5,
    }


@pytest.fixture
def sample_drift_data():
    """Sample drift detection data for testing."""
    return {
        "feature_1": {"tvd": 0.15, "hellinger": 0.12, "kld": 0.08},
        "feature_2": {"tvd": 0.22, "hellinger": 0.18, "kld": 0.14},
    }


@pytest.fixture
def sample_timestamps():
    """Generate sample timestamps for testing."""
    base_time = datetime.now(timezone.utc)
    return [
        base_time.replace(hour=h, minute=0, second=0, microsecond=0)
        for h in range(10, 15)
    ]


def create_test_metrics_data(
    query_handler: TimescaleDBQueryHandler,
    project: str,
    endpoint_id: str,
    metrics: dict,
    timestamps: list[datetime],
) -> None:
    """Helper function to create test metrics data."""
    for timestamp in timestamps:
        metric_values = []
        for metric_name, base_value in metrics.items():
            # Add some variation to the values
            variation = random.uniform(0.9, 1.1)
            value = base_value * variation

            metric_values.append(
                mm_schemas.ModelEndpointMonitoringMetricValues(
                    full_name=f"test.{metric_name}",
                    values=[
                        mm_schemas.ModelEndpointMonitoringMetricValue(
                            timestamp=timestamp,
                            value=value,
                        )
                    ],
                )
            )

        query_handler._operations.write_monitoring_values(
            project=project,
            endpoint_id=endpoint_id,
            values=metric_values,
        )


def create_test_predictions_data(
    query_handler: TimescaleDBQueryHandler,
    project: str,
    endpoint_id: str,
    prediction_data: dict,
    timestamps: list[datetime],
) -> None:
    """Helper function to create test predictions data."""
    for timestamp in timestamps:
        latency = prediction_data["latency"] + random.uniform(-5, 5)

        query_handler._operations.write_prediction(
            project=project,
            endpoint_id=endpoint_id,
            prediction=prediction_data["predictions"][0],
            timestamp=timestamp,
            latency=latency,
        )


def create_test_drift_data(
    query_handler: TimescaleDBQueryHandler,
    project: str,
    endpoint_id: str,
    drift_data: dict,
    timestamps: list[datetime],
) -> None:
    """Helper function to create test drift data."""
    for timestamp in timestamps:
        for feature_name, metrics in drift_data.items():
            for metric_name, value in metrics.items():
                drift_value = value + random.uniform(-0.05, 0.05)

                query_handler._operations.write_monitoring_drift_result(
                    project=project,
                    endpoint_id=endpoint_id,
                    feature_name=feature_name,
                    metric_name=metric_name,
                    drift_value=drift_value,
                    timestamp=timestamp,
                    result_status=mm_schemas.ResultStatusApp.DETECTED
                    if drift_value > 0.2
                    else mm_schemas.ResultStatusApp.NO_DETECTION,
                )


def create_test_results_data(
    query_handler: TimescaleDBQueryHandler,
    project: str,
    endpoint_id: str,
    timestamps: list[datetime],
    result_kinds: list[str],
) -> None:
    """Helper function to create test results data."""
    statuses = [
        mm_schemas.ResultStatusApp.DETECTED,
        mm_schemas.ResultStatusApp.NO_DETECTION,
        mm_schemas.ResultStatusApp.ERROR,
    ]

    for timestamp in timestamps:
        for result_kind in result_kinds:
            status = random.choice(statuses)

            query_handler._operations.write_monitoring_result(
                project=project,
                endpoint_id=endpoint_id,
                result_kind=result_kind,
                result_status=status,
                timestamp=timestamp,
                result_extra_data={"test": "data"},
            )


# ===== ADDITIONAL FIXTURES FOR OTHER TEST FILES =====


@pytest.fixture
def clean_connection_pool():
    """Fixture to ensure clean connection pool state for each test.

    Useful for connection and operations tests that need isolated connection state.
    """
    # Clean before test
    reset_global_connection_pool()
    yield
    # Clean after test
    reset_global_connection_pool()


@pytest.fixture(scope="session")
def unique_database_name():
    """Generate a unique database name for tests that need their own database.

    Used by schema and connection tests that create/drop databases.
    Session-scoped since database name should be consistent within a session.
    """
    return generate_unique_name("mlrun_test_db")


@pytest.fixture
def basic_connection(connection_string):
    """Simple TimescaleDB connection without project setup.

    Used by connection and schema tests that don't need full query handler setup.
    """
    conn = TimescaleDBConnection(connection_string, max_connections=1, autocommit=False)
    yield conn
    # Connection automatically cleaned up


@pytest.fixture
def admin_connection(connection_string):
    """TimescaleDB connection with autocommit for DDL operations.

    Used by tests that need to create/drop databases or run administrative commands.
    """
    conn = TimescaleDBConnection(connection_string, max_connections=1, autocommit=True)
    yield conn


@pytest.fixture
def test_schema_config():
    """Standard test schema configuration.

    Used by schema tests to have consistent test schema setup.
    """
    return {
        "schema": "mlrun_model_monitoring_",
        "project": "test_project",
        "retention_days": 30,
        "chunk_time_interval": "1 day",
    }


@pytest.fixture
def sample_writer_event():
    """Sample writer event data for testing.

    Used by operations tests that need to test event writing.
    """
    return {
        mm_schemas.WriterEvent.END_INFER_TIME: datetime(2024, 1, 15, 12, 0, 0),
        mm_schemas.WriterEvent.START_INFER_TIME: datetime(2024, 1, 15, 12, 0, 0),
        mm_schemas.WriterEvent.ENDPOINT_ID: "test_endpoint",
        mm_schemas.WriterEvent.APPLICATION_NAME: "test_app",
    }


@pytest.fixture
def sample_metric_event(sample_writer_event):
    """Sample metric event data for testing.

    Extends writer event with metric-specific fields.
    """
    return {
        **sample_writer_event,
        mm_schemas.MetricData.METRIC_NAME: "accuracy",
        mm_schemas.MetricData.METRIC_VALUE: 0.95,
    }


@pytest.fixture
def sample_error_event(sample_writer_event):
    """Sample error event data for testing.

    Used by operations tests that need to test error event writing.
    """
    return {
        **sample_writer_event,
        mm_schemas.EventFieldType.TIME: sample_writer_event[
            mm_schemas.WriterEvent.END_INFER_TIME
        ],
        mm_schemas.EventFieldType.MODEL_ERROR: "Test error message",
        mm_schemas.EventFieldType.ERROR_TYPE: "inference_error",
    }
