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

import datetime

import pytest

import mlrun.common.schemas.model_monitoring as mm_schemas
from mlrun.model_monitoring.db.tsdb.timescaledb.schemas import (
    _MODEL_MONITORING_SCHEMA,
    AppResultTable,
    Errors,
    Metrics,
    PreAggregateConfig,
    Predictions,
    TimescaleDBSchema,
    _TimescaleDBColumn,
    _TimescaleDBColumnType,
)


class TestPreAggregateConfig:
    """Test cases for PreAggregateConfig class."""

    def test_default_initialization(self):
        """Test that PreAggregateConfig initializes with default values."""
        config = PreAggregateConfig()

        assert config.aggregate_intervals == ["10m", "1h", "6h", "1d", "1w", "1M"]
        assert config.agg_functions == ["sum", "avg", "min", "max", "count", "last"]
        assert config.retention_policy == {
            "raw": "7d",
            "10m": "30d",
            "1h": "1y",
            "6h": "1y",
            "1d": "5y",
            "1w": "5y",
            "1M": "5y",
        }

    def test_custom_initialization(self):
        """Test PreAggregateConfig with custom values."""
        custom_intervals = ["5m", "30m", "2h"]
        custom_functions = ["avg", "max"]
        custom_retention = {"raw": "3d", "5m": "14d"}

        config = PreAggregateConfig(
            aggregate_intervals=custom_intervals,
            agg_functions=custom_functions,
            retention_policy=custom_retention,
        )

        assert config.aggregate_intervals == custom_intervals
        assert config.agg_functions == custom_functions
        assert config.retention_policy == custom_retention

    def test_partial_custom_initialization(self):
        """Test PreAggregateConfig with some custom values."""
        config = PreAggregateConfig(aggregate_intervals=["1h", "1d"])

        assert config.aggregate_intervals == ["1h", "1d"]
        assert config.agg_functions == [
            "sum",
            "avg",
            "min",
            "max",
            "count",
            "last",
        ]  # default
        assert "raw" in config.retention_policy  # default retention policy


class TestTimescaleDBColumnType:
    """Test cases for _TimescaleDBColumnType class."""

    def test_basic_type(self):
        """Test basic column type without length."""
        col_type = _TimescaleDBColumnType("INTEGER")

        assert col_type.data_type == "INTEGER"
        assert col_type.length is None
        assert col_type.nullable is True
        assert str(col_type) == "INTEGER"

    def test_type_with_length(self):
        """Test column type with length parameter."""
        col_type = _TimescaleDBColumnType("VARCHAR", 255)

        assert col_type.data_type == "VARCHAR"
        assert col_type.length == 255
        assert str(col_type) == "VARCHAR(255)"

    def test_non_nullable_type(self):
        """Test non-nullable column type."""
        col_type = _TimescaleDBColumnType("INTEGER", nullable=False)

        assert col_type.nullable is False


class TestTimescaleDBColumn:
    """Test cases for _TimescaleDBColumn enum."""

    def test_column_definitions(self):
        """Test that all column types are properly defined."""
        assert _TimescaleDBColumn.TIMESTAMPTZ.data_type == "TIMESTAMPTZ"
        assert _TimescaleDBColumn.DOUBLE_PRECISION.data_type == "DOUBLE PRECISION"
        assert _TimescaleDBColumn.INTEGER.data_type == "INTEGER"
        assert _TimescaleDBColumn.VARCHAR_64.data_type == "VARCHAR"
        assert _TimescaleDBColumn.VARCHAR_64.length == 64
        assert _TimescaleDBColumn.VARCHAR_1000.length == 1000
        assert _TimescaleDBColumn.TEXT.data_type == "TEXT"


class TestTimescaleDBSchema:
    """Test cases for TimescaleDBSchema base class."""

    @pytest.fixture
    def sample_schema(self):
        """Create a sample schema for testing."""
        columns = {
            "timestamp_col": _TimescaleDBColumnType("TIMESTAMPTZ"),
            "value_col": _TimescaleDBColumnType("DOUBLE PRECISION"),
            "text_col": _TimescaleDBColumnType("VARCHAR", 64),
        }
        return TimescaleDBSchema(
            table_name="test_table",
            columns=columns,
            time_column="timestamp_col",
            project="test-project",
            indexes=["value_col", "text_col, timestamp_col"],
        )

    def test_initialization(self, sample_schema):
        """Test schema initialization."""
        assert sample_schema.table_name == "test_table_test_project"
        assert sample_schema.time_column == "timestamp_col"
        assert sample_schema.schema == _MODEL_MONITORING_SCHEMA
        assert sample_schema.chunk_time_interval == "1 day"
        assert len(sample_schema.columns) == 3
        assert len(sample_schema.indexes) == 2

    def test_project_name_sanitization(self):
        """Test that project names with hyphens are properly sanitized."""
        schema = TimescaleDBSchema(
            table_name="test",
            columns={"time": _TimescaleDBColumnType("TIMESTAMPTZ")},
            time_column="time",
            project="my-test-project",
        )
        assert schema.table_name == "test_my_test_project"

    def test_create_table_query(self, sample_schema):
        """Test table creation query generation."""
        query = sample_schema._create_table_query()

        expected_query = (
            f"CREATE TABLE IF NOT EXISTS {_MODEL_MONITORING_SCHEMA}.test_table_test_project "
            f"(timestamp_col TIMESTAMPTZ, value_col DOUBLE PRECISION, text_col VARCHAR(64));"
        )
        assert query == expected_query

    def test_create_hypertable_query(self, sample_schema):
        """Test hypertable creation query generation."""
        query = sample_schema._create_hypertable_query()

        expected_query = (
            f"SELECT create_hypertable('{_MODEL_MONITORING_SCHEMA}.test_table_test_project', "
            f"'timestamp_col', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);"
        )
        assert query == expected_query

    def test_create_indexes_query(self, sample_schema):
        """Test index creation queries."""
        queries = sample_schema._create_indexes_query()

        expected_queries = [
            f"CREATE INDEX IF NOT EXISTS idx_test_table_test_project_value_col "
            f"ON {_MODEL_MONITORING_SCHEMA}.test_table_test_project (value_col);",
            f"CREATE INDEX IF NOT EXISTS idx_test_table_test_project_text_col__timestamp_col "
            f"ON {_MODEL_MONITORING_SCHEMA}.test_table_test_project (text_col, timestamp_col);",
        ]

        assert len(queries) == 2
        assert queries == expected_queries

    def test_drop_table_query(self, sample_schema):
        """Test table drop query generation."""
        query = sample_schema.drop_table_query()

        expected_query = f"DROP TABLE IF EXISTS {_MODEL_MONITORING_SCHEMA}.test_table_test_project CASCADE;"
        assert query == expected_query

    def test_get_chunk_interval_for_agg(self, sample_schema):
        """Test chunk interval determination for aggregates."""
        assert sample_schema._get_chunk_interval_for_agg("10m") == "1 hour"
        assert sample_schema._get_chunk_interval_for_agg("1h") == "1 day"
        assert sample_schema._get_chunk_interval_for_agg("1d") == "7 days"
        assert sample_schema._get_chunk_interval_for_agg("1w") == "1 month"
        assert sample_schema._get_chunk_interval_for_agg("1M") == "3 months"
        assert (
            sample_schema._get_chunk_interval_for_agg("unknown") == "1 day"
        )  # default

    def test_create_pre_aggregate_tables_query(self, sample_schema):
        """Test pre-aggregate table creation."""
        config = PreAggregateConfig(
            aggregate_intervals=["1h", "1d"], agg_functions=["avg", "sum"]
        )
        queries = sample_schema._create_pre_aggregate_tables_query(config)

        expected_queries = [
            # 1h aggregate table
            f"CREATE TABLE IF NOT EXISTS {_MODEL_MONITORING_SCHEMA}.test_table_test_project_agg_1h "
            f"(time_bucket TIMESTAMPTZ NOT NULL, avg_value_col DOUBLE PRECISION,"
            f" sum_value_col DOUBLE PRECISION, text_col VARCHAR(64));",
            f"SELECT create_hypertable('{_MODEL_MONITORING_SCHEMA}.test_table_test_project_agg_1h', "
            f"'time_bucket', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);",
            # 1d aggregate table
            f"CREATE TABLE IF NOT EXISTS {_MODEL_MONITORING_SCHEMA}.test_table_test_project_agg_1d "
            f"(time_bucket TIMESTAMPTZ NOT NULL, avg_value_col DOUBLE PRECISION, sum_value_col DOUBLE PRECISION,"
            f" text_col VARCHAR(64));",
            f"SELECT create_hypertable('{_MODEL_MONITORING_SCHEMA}.test_table_test_project_agg_1d', "
            f"'time_bucket', chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);",
        ]

        assert len(queries) == 4
        assert queries == expected_queries

    def test_create_continuous_aggregates_query(self, sample_schema):
        """Test continuous aggregates creation."""
        config = PreAggregateConfig(
            aggregate_intervals=["1h"], agg_functions=["avg", "count"]
        )
        queries = sample_schema._create_continuous_aggregates_query(config)

        expected_query = (
            f"CREATE MATERIALIZED VIEW IF NOT EXISTS {_MODEL_MONITORING_SCHEMA}.test_table_test_project_cagg_1h "
            f"WITH (timescaledb.continuous) AS SELECT time_bucket(INTERVAL '1h', timestamp_col) AS time_bucket, "
            f"AVG(value_col) AS avg_value_col, COUNT(value_col) AS count_value_col "
            f"FROM {_MODEL_MONITORING_SCHEMA}.test_table_test_project "
            f"GROUP BY time_bucket WITH NO DATA;"
        )

        assert len(queries) == 1
        assert queries[0] == expected_query

    def test_create_retention_policies_query(self, sample_schema):
        """Test retention policies creation."""
        config = PreAggregateConfig(
            aggregate_intervals=["1h", "1d"],
            retention_policy={"raw": "7d", "1h": "30d", "1d": "1y"},
        )
        queries = sample_schema._create_retention_policies_query(config)

        expected_queries = [
            f"SELECT add_retention_policy('{_MODEL_MONITORING_SCHEMA}.test_table_test_project', INTERVAL '7d',"
            f" if_not_exists => TRUE);",
            f"SELECT add_retention_policy('{_MODEL_MONITORING_SCHEMA}.test_table_test_project_cagg_1h',"
            f" INTERVAL '30d', if_not_exists => TRUE);",
            f"SELECT add_retention_policy('{_MODEL_MONITORING_SCHEMA}.test_table_test_project_cagg_1d',"
            f" INTERVAL '1y', if_not_exists => TRUE);",
        ]

        assert len(queries) == 3
        assert queries == expected_queries

    def test_get_records_query_basic(self, sample_schema):
        """Test basic records query generation."""
        start = datetime.datetime(2024, 1, 1, 12, 0, 0)
        end = datetime.datetime(2024, 1, 2, 12, 0, 0)

        query = sample_schema._get_records_query(
            start=start, end=end, columns_to_filter=["timestamp_col", "value_col"]
        )

        expected_query = (
            f"SELECT timestamp_col, value_col FROM {_MODEL_MONITORING_SCHEMA}.test_table_test_project "
            f"WHERE timestamp_col >= '2024-01-01 12:00:00' AND timestamp_col <= '2024-01-02 12:00:00';"
        )
        assert query == expected_query

    def test_get_records_query_with_filter(self, sample_schema):
        """Test records query with additional filters."""
        start = datetime.datetime(2024, 1, 1)
        end = datetime.datetime(2024, 1, 2)

        query = sample_schema._get_records_query(
            start=start,
            end=end,
            filter_query="text_col = 'test'",
            order_by="timestamp_col",
            desc=True,
            limit=100,
        )

        expected_query = (
            f"SELECT * FROM {_MODEL_MONITORING_SCHEMA}.test_table_test_project "
            f"WHERE text_col = 'test' AND timestamp_col >= '2024-01-01 00:00:00' AND "
            f"timestamp_col <= '2024-01-02 00:00:00' ORDER BY timestamp_col DESC LIMIT 100;"
        )
        assert query == expected_query

    def test_get_records_query_with_pre_aggregates(self, sample_schema):
        """Test records query using pre-aggregates."""
        start = datetime.datetime(2024, 1, 1)
        end = datetime.datetime(2024, 1, 2)

        query = sample_schema._get_records_query(
            start=start,
            end=end,
            columns_to_filter=["timestamp_col", "value_col"],
            interval="1h",
            agg_funcs=["avg"],
            use_pre_aggregates=True,
        )

        expected_query = (
            f"SELECT time_bucket, avg_value_col FROM {_MODEL_MONITORING_SCHEMA}.test_table_test_project_cagg_1h "
            f"WHERE time_bucket >= '2024-01-01 00:00:00' AND time_bucket <= '2024-01-02 00:00:00';"
        )
        assert query == expected_query


class TestAppResultTable:
    """Test cases for AppResultTable schema."""

    @pytest.fixture
    def app_result_table(self):
        return AppResultTable(project="test-project")

    def test_initialization(self, app_result_table):
        """Test AppResultTable initialization."""
        assert "app_results_test_project" in app_result_table.table_name
        assert app_result_table.time_column == mm_schemas.WriterEvent.END_INFER_TIME
        assert len(app_result_table.columns) == 9
        assert mm_schemas.WriterEvent.ENDPOINT_ID in app_result_table.columns
        assert mm_schemas.ResultData.RESULT_VALUE in app_result_table.columns

    def test_required_columns_present(self, app_result_table):
        """Test that all required columns are present."""
        required_columns = [
            mm_schemas.WriterEvent.END_INFER_TIME,
            mm_schemas.WriterEvent.START_INFER_TIME,
            mm_schemas.ResultData.RESULT_VALUE,
            mm_schemas.ResultData.RESULT_STATUS,
            mm_schemas.ResultData.RESULT_EXTRA_DATA,
            mm_schemas.WriterEvent.ENDPOINT_ID,
            mm_schemas.WriterEvent.APPLICATION_NAME,
            mm_schemas.ResultData.RESULT_NAME,
            mm_schemas.ResultData.RESULT_KIND,
        ]

        for col in required_columns:
            assert col in app_result_table.columns

    def test_indexes_created(self, app_result_table):
        """Test that appropriate indexes are defined."""
        assert len(app_result_table.indexes) == 3
        assert mm_schemas.WriterEvent.ENDPOINT_ID in app_result_table.indexes


class TestMetrics:
    """Test cases for Metrics schema."""

    @pytest.fixture
    def metrics_table(self):
        return Metrics(project="test-project")

    def test_initialization(self, metrics_table):
        """Test Metrics table initialization."""
        assert "metrics_test_project" in metrics_table.table_name
        assert metrics_table.time_column == mm_schemas.WriterEvent.END_INFER_TIME
        assert len(metrics_table.columns) == 6

    def test_required_columns_present(self, metrics_table):
        """Test that all required columns are present."""
        required_columns = [
            mm_schemas.WriterEvent.END_INFER_TIME,
            mm_schemas.WriterEvent.START_INFER_TIME,
            mm_schemas.MetricData.METRIC_VALUE,
            mm_schemas.WriterEvent.ENDPOINT_ID,
            mm_schemas.WriterEvent.APPLICATION_NAME,
            mm_schemas.MetricData.METRIC_NAME,
        ]

        for col in required_columns:
            assert col in metrics_table.columns


class TestPredictions:
    """Test cases for Predictions schema."""

    @pytest.fixture
    def predictions_table(self):
        return Predictions(project="test-project")

    def test_initialization(self, predictions_table):
        """Test Predictions table initialization."""
        assert "predictions_test_project" in predictions_table.table_name
        assert predictions_table.time_column == mm_schemas.EventFieldType.TIME
        assert len(predictions_table.columns) == 6

    def test_required_columns_present(self, predictions_table):
        """Test that all required columns are present."""
        required_columns = [
            mm_schemas.EventFieldType.TIME,
            mm_schemas.EventFieldType.LATENCY,
            mm_schemas.EventKeyMetrics.CUSTOM_METRICS,
            mm_schemas.EventFieldType.ESTIMATED_PREDICTION_COUNT,
            mm_schemas.EventFieldType.EFFECTIVE_SAMPLE_COUNT,
            mm_schemas.WriterEvent.ENDPOINT_ID,
        ]

        for col in required_columns:
            assert col in predictions_table.columns


class TestErrors:
    """Test cases for Errors schema."""

    @pytest.fixture
    def errors_table(self):
        return Errors(project="test-project")

    def test_initialization(self, errors_table):
        """Test Errors table initialization."""
        assert "errors_test_project" in errors_table.table_name
        assert errors_table.time_column == mm_schemas.EventFieldType.TIME
        assert len(errors_table.columns) == 4

    def test_required_columns_present(self, errors_table):
        """Test that all required columns are present."""
        required_columns = [
            mm_schemas.EventFieldType.TIME,
            mm_schemas.EventFieldType.MODEL_ERROR,
            mm_schemas.WriterEvent.ENDPOINT_ID,
            mm_schemas.EventFieldType.ERROR_TYPE,
        ]

        for col in required_columns:
            assert col in errors_table.columns


class TestIntegration:
    """Integration tests for the schemas."""

    def test_all_tables_have_consistent_time_columns(self):
        """Test that all tables have properly defined time columns."""
        project = "integration-test"

        app_results = AppResultTable(project)
        metrics = Metrics(project)
        predictions = Predictions(project)
        errors = Errors(project)

        # Verify time columns exist in their respective column definitions
        assert app_results.time_column in app_results.columns
        assert metrics.time_column in metrics.columns
        assert predictions.time_column in predictions.columns
        assert errors.time_column in errors.columns

    def test_all_tables_have_endpoint_id(self):
        """Test that all tables include endpoint_id for filtering."""
        project = "integration-test"

        tables = [
            AppResultTable(project),
            Metrics(project),
            Predictions(project),
            Errors(project),
        ]

        for table in tables:
            assert mm_schemas.WriterEvent.ENDPOINT_ID in table.columns

    def test_pre_aggregate_config_compatibility(self):
        """Test that PreAggregateConfig works with all table types."""
        config = PreAggregateConfig(
            aggregate_intervals=["1h", "1d"], agg_functions=["avg", "sum", "count"]
        )

        project = "test-project"
        tables = [
            AppResultTable(project),
            Metrics(project),
            Predictions(project),
            Errors(project),
        ]

        for table in tables:
            # Should not raise any exceptions
            pre_agg_queries = table._create_pre_aggregate_tables_query(config)
            cont_agg_queries = table._create_continuous_aggregates_query(config)
            retention_queries = table._create_retention_policies_query(config)

            assert len(pre_agg_queries) > 0
            assert len(cont_agg_queries) > 0
            assert len(retention_queries) > 0


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_project_name(self):
        """Test handling of empty project name."""
        # Empty project name should work but result in empty suffix
        table = AppResultTable(project="")
        assert "app_results_" in table.table_name

    def test_invalid_interval_for_chunk(self):
        """Test handling of invalid intervals."""
        schema = TimescaleDBSchema(
            table_name="test",
            columns={"time": _TimescaleDBColumnType("TIMESTAMPTZ")},
            time_column="time",
            project="test",
        )

        # Should return default for unknown interval
        chunk_interval = schema._get_chunk_interval_for_agg("invalid_interval")
        assert chunk_interval == "1 day"

    def test_query_with_no_columns(self):
        """Test query generation with no specific columns."""
        schema = TimescaleDBSchema(
            table_name="test",
            columns={"time": _TimescaleDBColumnType("TIMESTAMPTZ")},
            time_column="time",
            project="test",
        )

        start = datetime.datetime(2024, 1, 1)
        end = datetime.datetime(2024, 1, 2)

        query = schema._get_records_query(start=start, end=end)

        expected_query = (
            f"SELECT * FROM {_MODEL_MONITORING_SCHEMA}.test_test "
            f"WHERE time >= '2024-01-01 00:00:00' AND time <= '2024-01-02 00:00:00';"
        )
        assert query == expected_query

    def test_custom_schema_name(self):
        """Test using custom schema name."""
        custom_schema = "custom_monitoring"
        table = AppResultTable(project="test", schema=custom_schema)

        assert table.schema == custom_schema

        query = table._create_table_query()

        expected_query = (
            f"CREATE TABLE IF NOT EXISTS {custom_schema}.app_results_test "
            f"(end_infer_time TIMESTAMPTZ, start_infer_time TIMESTAMPTZ, result_value DOUBLE PRECISION, "
            f"result_status INTEGER, result_extra_data VARCHAR(1000), endpoint_id VARCHAR(64), "
            f"application_name VARCHAR(64), result_name VARCHAR(64), result_kind INTEGER);"
        )
        assert query == expected_query


if __name__ == "__main__":
    pytest.main([__file__])
