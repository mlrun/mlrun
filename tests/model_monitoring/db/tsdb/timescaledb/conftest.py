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
import time
import uuid

import pytest

# Connection string detection - used by ALL TimescaleDB tests
CONNECTION_STRING = os.getenv("MLRUN_MODEL_ENDPOINT_MONITORING__TSDB_CONNECTION")


# Check if TimescaleDB is available for testing
def is_timescaledb_available():
    """Check if TimescaleDB connection is available and valid."""
    if not CONNECTION_STRING:
        return False
    if not CONNECTION_STRING.startswith("postgres"):
        return False
    return True


# Import TimescaleDB modules only if available
if is_timescaledb_available():
    import mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_schema as timescaledb_schema
    from mlrun.model_monitoring.db.tsdb.preaggregate import (
        PreAggregateConfig,
        PreAggregateManager,
    )
    from mlrun.model_monitoring.db.tsdb.timescaledb.queries.timescaledb_metrics_queries import (
        TimescaleDBMetricsQueries,
    )
    from mlrun.model_monitoring.db.tsdb.timescaledb.queries.timescaledb_predictions_queries import (
        TimescaleDBPredictionsQueries,
    )
    from mlrun.model_monitoring.db.tsdb.timescaledb.queries.timescaledb_results_queries import (
        TimescaleDBResultsQueries,
    )
    from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connection import (
        Statement,
        TimescaleDBConnection,
    )
    from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_operations import (
        TimescaleDBOperationsManager,
    )
    from mlrun.model_monitoring.db.tsdb.timescaledb.utils.timescaledb_query_builder import (
        TimescaleDBQueryBuilder,
    )
else:
    # Create dummy variables to avoid NameError in fixtures
    timescaledb_schema = None
    PreAggregateConfig = None
    PreAggregateManager = None
    TimescaleDBMetricsQueries = None
    TimescaleDBPredictionsQueries = None
    TimescaleDBResultsQueries = None
    TimescaleDBConnection = None
    TimescaleDBOperationsManager = None
    Statement = None
    TimescaleDBQueryBuilder = None
    # Global skip mark for this entire test file
    pytestmark = pytest.mark.skip(
        reason="TimescaleDB connection string not available or not PostgreSQL"
    )


@pytest.fixture(scope="session")
def connection_string():
    """TimescaleDB connection string from environment.

    Used by ALL TimescaleDB tests that need database connectivity.
    Session-scoped since connection string doesn't change during test session.
    """
    if not is_timescaledb_available():
        pytest.skip("TimescaleDB connection string not available or not PostgreSQL")
    return CONNECTION_STRING


@pytest.fixture
def project_name():
    """Generate a unique project name for test isolation."""
    timestamp = int(time.time())
    random_suffix = uuid.uuid4().hex[:8]
    return f"test-project-{timestamp}-{random_suffix}"


@pytest.fixture
def connection(connection_string):
    return TimescaleDBConnection(connection_string, max_connections=1, autocommit=False)


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
def table_schemas(project_name):
    """Create table schemas for testing using consolidated function."""
    return timescaledb_schema.create_table_schemas(project_name)


@pytest.fixture
def operations_handler(connection, project_name):
    """Operations handler for table management in tests."""
    handler = TimescaleDBOperationsManager(project=project_name, connection=connection)

    # Create tables
    handler.create_tables()

    yield handler

    # Cleanup
    handler.delete_tsdb_resources()


@pytest.fixture
def real_pre_aggregate_manager(pre_aggregate_config):
    """Real pre-aggregate manager for testing with aggregates."""
    return PreAggregateManager(pre_aggregate_config)


class QueryTestHelper:
    """Test helper class that packages commonly needed components for query testing."""

    def __init__(
        self,
        connection,
        project_name,
        table_schemas,
        pre_aggregate_manager,
        operations_handler,
    ):
        self.connection = connection
        self.project_name = project_name
        self.table_schemas = table_schemas
        self.pre_aggregate_manager = pre_aggregate_manager
        self.operations_handler = operations_handler

    def create_metrics_handler(self):
        """Create a TimescaleDBMetricsQueries instance."""
        return TimescaleDBMetricsQueries(
            project=self.project_name,
            connection=self.connection,
            pre_aggregate_manager=self.pre_aggregate_manager,
            tables=self.table_schemas,
        )

    def create_predictions_handler(self):
        """Create a TimescaleDBPredictionsQueries instance."""
        return TimescaleDBPredictionsQueries(
            project=self.project_name,
            connection=self.connection,
            pre_aggregate_manager=self.pre_aggregate_manager,
            tables=self.table_schemas,
        )

    def create_results_handler(self):
        """Create a TimescaleDBResultsQueries instance."""
        return TimescaleDBResultsQueries(
            connection=self.connection,
            project=self.project_name,
            pre_aggregate_manager=self.pre_aggregate_manager,
            tables=self.table_schemas,
        )

    def write_application_event(self, *args, **kwargs):
        """Convenience method to write application events."""
        return self.operations_handler.write_application_event(*args, **kwargs)


@pytest.fixture
def query_test_helper(
    connection,
    project_name,
    table_schemas,
    operations_handler,
) -> QueryTestHelper:
    """Test helper that packages all commonly needed query testing components."""
    # Create PreAggregateManager with no config for basic testing
    pre_aggregate_manager = PreAggregateManager()

    return QueryTestHelper(
        connection=connection,
        project_name=project_name,
        table_schemas=table_schemas,
        pre_aggregate_manager=pre_aggregate_manager,
        operations_handler=operations_handler,
    )


@pytest.fixture
def query_test_helper_with_aggregates(
    connection, project_name, table_schemas, real_pre_aggregate_manager
):
    """
    Test helper with real pre-aggregate functionality for testing aggregation features.
    Uses the same clean QueryTestHelper pattern but with pre-aggregates enabled.
    """
    # Create operations handler with pre-aggregate config for continuous aggregates
    pre_aggregate_config = PreAggregateConfig(
        aggregate_intervals=["10m", "1h"],
        agg_functions=["sum", "avg", "max", "count"],
        retention_policy={
            "raw": "7d",
            "10m": "30d",
            "1h": "1y",
        },
    )

    operations_handler = TimescaleDBOperationsManager(
        project=project_name,
        connection=connection,
        pre_aggregate_config=pre_aggregate_config,
    )

    # Create tables WITH pre-aggregates
    operations_handler.create_tables()

    # Return the same clean helper pattern but with real pre-aggregate manager
    yield QueryTestHelper(
        connection=connection,
        project_name=project_name,
        table_schemas=table_schemas,
        pre_aggregate_manager=real_pre_aggregate_manager,
        operations_handler=operations_handler,
    )

    # Cleanup - delete all resources including pre-aggregates
    operations_handler.delete_tsdb_resources()


@pytest.fixture
def admin_connection(connection_string):
    """TimescaleDB connection with autocommit for DDL operations.

    Used by tests that need to create/drop databases or run administrative commands.
    """
    yield TimescaleDBConnection(connection_string, max_connections=1, autocommit=True)


@pytest.fixture
def query_builder(connection_string):
    """TimescaleDBQueryBuilder instance for validation tests.

    This fixture ensures proper skipping when TimescaleDB is not available.
    """
    return TimescaleDBQueryBuilder


@pytest.fixture
def statement(connection_string):
    """Statement class for operations tests.

    This fixture ensures proper skipping when TimescaleDB is not available.
    """
    return Statement


@pytest.fixture
def connector(connection_string, project_name):
    """Create TimescaleDBConnector instance for cross-query testing."""
    from mlrun.datastore.datastore_profile import DatastoreProfilePostgreSQL
    from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connector import (
        TimescaleDBConnector,
    )

    # Create profile from DSN
    profile = DatastoreProfilePostgreSQL.from_dsn(
        dsn=connection_string, profile_name="test_profile"
    )

    connector_instance = TimescaleDBConnector(
        project=project_name,
        profile=profile,
    )

    # Create tables for this connector
    connector_instance.create_tables()

    yield connector_instance

    # Cleanup tables after test
    connector_instance.delete_tsdb_resources()
