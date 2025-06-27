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

from datetime import datetime
from typing import Optional, Union

import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.errors
import mlrun.model_monitoring.db.tsdb.timescaledb.schemas as timescaledb_schemas
from mlrun.datastore.datastore_profile import DatastoreProfile
from mlrun.model_monitoring.db.tsdb.timescaledb.schemas import PreAggregateConfig
from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connection import (
    Statement,
    TimescaleDBConnection,
)
from mlrun.utils import logger


class TimescaleDBOperationsHandler:
    """
    Handles all CRUD operations for TimescaleDB TSDB connector.

    This class implements all create/update/delete operations for model monitoring data:
    - Table and schema creation with optional pre-aggregates and continuous aggregates
    - Event writing with parameterized queries for safety against SQL injection
    - Record deletion with support for both raw and aggregate data cleanup
    - Resource deletion with automatic discovery of project-related tables and views
    - Schema management with automatic cleanup of empty schemas

    The handler uses dependency injection to receive a shared TimescaleDBConnection
    instance that manages the global connection pool. Each handler instance is
    project-scoped and manages its own set of table schemas within a dedicated
    database schema.

    Key Features:
    - Parameterized queries for all write/delete operations
    - Automatic discovery of aggregate tables for comprehensive cleanup
    - Transaction-based operations for data consistency
    - Unicode and special character support
    - Configurable pre-aggregation with retention policies
    - Thread-safe operations through shared connection pooling

    :param project: Project name used for table naming and schema organization
    :param profile: Datastore profile (used for table initialization)
    :param connection: Shared TimescaleDBConnection instance
    :param pre_aggregate_config: Optional configuration for pre-aggregated tables
    """

    def __init__(
        self,
        project: str,
        profile: DatastoreProfile,
        connection: TimescaleDBConnection,
        pre_aggregate_config: Optional[PreAggregateConfig] = None,
    ):
        """
        Initialize operations handler with a shared connection.

        :param project: The project name
        :param profile: Datastore profile for connection (used for table initialization)
        :param connection: Shared TimescaleDBConnection instance
        :param pre_aggregate_config: Optional pre-aggregation configuration
        """
        self.project = project
        self.profile = profile
        self._pre_aggregate_config = pre_aggregate_config

        # Use the injected shared connection
        self._connection = connection

        # Initialize table schemas
        self._init_tables()

    def _init_tables(self) -> None:
        """Initialize TimescaleDB table schemas."""
        schema_name = (
            f"{timescaledb_schemas._MODEL_MONITORING_SCHEMA}_{mlrun.mlconf.system_id}"
        )

        self.tables = {
            mm_schemas.TDEngineSuperTables.APP_RESULTS: timescaledb_schemas.AppResultTable(
                project=self.project, schema=schema_name
            ),
            mm_schemas.TDEngineSuperTables.METRICS: timescaledb_schemas.Metrics(
                project=self.project, schema=schema_name
            ),
            mm_schemas.TDEngineSuperTables.PREDICTIONS: timescaledb_schemas.Predictions(
                project=self.project, schema=schema_name
            ),
            mm_schemas.TDEngineSuperTables.ERRORS: timescaledb_schemas.Errors(
                project=self.project, schema=schema_name
            ),
        }

    def create_tables(
        self, pre_aggregate_config: Optional[PreAggregateConfig] = None
    ) -> None:
        """
        Create TimescaleDB hypertables with optional pre-aggregation configuration.

        When pre_aggregate_config is provided, creates additional pre-aggregated tables
        for the configured intervals and sets up retention policies.

        :param pre_aggregate_config: Override the instance config for this operation
        """
        # Use provided config or instance config
        config = pre_aggregate_config or self._pre_aggregate_config

        logger.debug(
            "Creating TimescaleDB tables for model monitoring",
            project=self.project,
            with_pre_aggregates=config is not None,
        )

        # Create schema if it doesn't exist
        schema_name = self.tables[mm_schemas.TDEngineSuperTables.PREDICTIONS].schema
        self._connection.run(statements=[f"CREATE SCHEMA IF NOT EXISTS {schema_name}"])

        # Create main tables and convert to hypertables
        for table_type, table in self.tables.items():
            statements = []

            # Create base table
            statements.append(table._create_table_query())

            # Convert to hypertable
            statements.append(table._create_hypertable_query())

            # Create indexes
            statements.extend(table._create_indexes_query())

            # Create pre-aggregate tables if config provided
            if config:
                statements.extend(table._create_continuous_aggregates_query(config))
                statements.extend(table._create_retention_policies_query(config))

            # Execute all statements for this table
            self._connection.run(statements=statements)

        logger.debug(
            "Successfully created TimescaleDB tables",
            project=self.project,
            table_count=len(self.tables),
        )

    def write_application_event(
        self,
        event: dict,
        kind: mm_schemas.WriterEventKind = mm_schemas.WriterEventKind.RESULT,
    ) -> None:
        """
        Write a single result or metric to TimescaleDB using parameterized queries.

        Uses PostgreSQL's parameterized queries for safety and performance.

        :param event: Event data to write
        :param kind: Type of event (RESULT or METRIC)
        """
        if kind == mm_schemas.WriterEventKind.RESULT:
            table = self.tables[mm_schemas.TDEngineSuperTables.APP_RESULTS]
        else:
            table = self.tables[mm_schemas.TDEngineSuperTables.METRICS]

        # Convert datetime strings to datetime objects if needed
        for time_field in [
            mm_schemas.WriterEvent.END_INFER_TIME,
            mm_schemas.WriterEvent.START_INFER_TIME,
        ]:
            if time_field in event:
                event[time_field] = self._convert_to_datetime(event[time_field])

        # Prepare the INSERT statement with parameterized query
        columns = list(table.columns.keys())
        placeholders = ", ".join(["%s"] * len(columns))

        insert_sql = f"""
            INSERT INTO {table.schema}.{table.table_name} ({', '.join(columns)})
            VALUES ({placeholders})
        """

        # Prepare values in the correct order
        values = tuple(event.get(col) for col in columns)

        # Create parameterized statement
        stmt = Statement(insert_sql, values)

        try:
            # Execute parameterized query
            self._connection.run(statements=[stmt])
        except Exception as e:
            logger.error(
                "Failed to write application event to TimescaleDB",
                project=self.project,
                table=table.table_name,
                error=str(e),
            )
            raise mlrun.errors.MLRunRuntimeError(
                f"Failed to write event to TimescaleDB: {e}"
            ) from e

    def delete_tsdb_records(
        self,
        endpoint_ids: list[str],
        include_aggregates: bool = True,
    ) -> None:
        """
        Delete model endpoint records from TimescaleDB using parameterized queries.

        :param endpoint_ids: List of endpoint IDs to delete
        :param include_aggregates: Whether to delete from pre-aggregate tables as well
        """
        if not endpoint_ids:
            logger.debug("No endpoint IDs provided for deletion", project=self.project)
            return

        logger.debug(
            "Deleting model endpoint records from TimescaleDB",
            project=self.project,
            number_of_endpoints_to_delete=len(endpoint_ids),
            include_aggregates=include_aggregates,
        )

        try:
            # Execute all deletions in a single transaction to prevent race conditions
            # Raw data must be deleted first to prevent continuous aggregates from repopulating
            all_deletion_statements = []

            # 1. Delete raw data first (removes source for continuous aggregates)
            all_deletion_statements.extend(
                self._get_raw_delete_statements(endpoint_ids)
            )

            # 2. Delete aggregate data second (cleanup existing aggregated data)
            if include_aggregates:
                # Always try to discover and delete aggregates, regardless of config
                all_deletion_statements.extend(
                    self._get_aggregate_delete_statements(endpoint_ids)
                )

            # Execute all deletions atomically
            self._connection.run(statements=all_deletion_statements)

            logger.debug(
                "Successfully deleted model endpoint records from TimescaleDB",
                project=self.project,
                number_of_endpoints_deleted=len(endpoint_ids),
            )

        except Exception as e:
            logger.error(
                "Failed to delete model endpoint records from TimescaleDB",
                project=self.project,
                endpoint_count=len(endpoint_ids),
                error=mlrun.errors.err_to_str(e),
            )
            raise

    def _get_raw_delete_statements(self, endpoint_ids: list[str]) -> list[Statement]:
        """
        Get parameterized DELETE statements for raw data tables.

        :param endpoint_ids: List of endpoint IDs to delete
        :return: List of Statement objects for raw data deletion
        """
        statements = []

        for table_schema in self.tables.values():
            if len(endpoint_ids) == 1:
                delete_sql = (
                    f"DELETE FROM {table_schema.schema}.{table_schema.table_name} "
                    f"WHERE {mm_schemas.WriterEvent.ENDPOINT_ID} = %s"
                )
                stmt = Statement(delete_sql, (endpoint_ids[0],))
            else:
                delete_sql = (
                    f"DELETE FROM {table_schema.schema}.{table_schema.table_name} "
                    f"WHERE {mm_schemas.WriterEvent.ENDPOINT_ID} = ANY(%s)"
                )
                stmt = Statement(delete_sql, (endpoint_ids,))

            statements.append(stmt)

        return statements

    def _get_aggregate_delete_statements(
        self, endpoint_ids: list[str]
    ) -> list[Statement]:
        """
        Get parameterized DELETE statements for aggregate data tables by discovering existing tables.

        This approach discovers all existing aggregate tables rather than relying on configuration,
        ensuring we don't miss any aggregate data.

        :param endpoint_ids: List of endpoint IDs to delete
        :return: List of Statement objects for aggregate data deletion
        """
        statements = []

        try:
            schema_name = self.tables[mm_schemas.TDEngineSuperTables.PREDICTIONS].schema

            # Get base table patterns for tables that have endpoint_id
            base_patterns = []
            for table_type in [
                mm_schemas.TDEngineSuperTables.PREDICTIONS,
                mm_schemas.TDEngineSuperTables.METRICS,
                mm_schemas.TDEngineSuperTables.APP_RESULTS,
            ]:
                if table_type in self.tables:
                    base_patterns.append(self.tables[table_type].table_name)

            if not base_patterns:
                return statements

            # Build query to find all aggregate tables and views
            pattern_conditions = []
            parameters = [schema_name]

            for pattern in base_patterns:
                pattern_conditions.extend(
                    [
                        "table_name LIKE %s",  # _agg_ tables
                        "table_name LIKE %s",  # _cagg_ views
                    ]
                )
                parameters.extend([f"{pattern}_agg_%", f"{pattern}_cagg_%"])

            # Build separate pattern conditions for materialized views
            view_pattern_conditions = []
            view_parameters = [schema_name]

            for pattern in base_patterns:
                view_pattern_conditions.append("matviewname LIKE %s")
                view_parameters.append(f"{pattern}_cagg_%")

            # Query for both tables and materialized views
            discovery_stmt = Statement(
                f"""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = %s
                AND table_type = 'BASE TABLE'
                AND ({' OR '.join(pattern_conditions)})
                UNION
                SELECT matviewname as table_name
                FROM pg_matviews
                WHERE schemaname = %s
                AND ({' OR '.join(view_pattern_conditions)})
                ORDER BY table_name
                """,
                tuple([schema_name] + parameters[1:] + view_parameters[1:]),
            )

            result = self._connection.run(query=discovery_stmt)
            discovered_objects = (
                [row[0] for row in result.data] if result and result.data else []
            )

            if not discovered_objects:
                logger.debug(
                    "No aggregate objects found for deletion",
                    project=self.project,
                    schema=schema_name,
                )
                return statements

            logger.debug(
                "Discovered aggregate objects for endpoint deletion",
                project=self.project,
                aggregate_objects=len(discovered_objects),
                endpoint_count=len(endpoint_ids),
            )

            # Create delete statements for all discovered aggregate objects
            for object_name in discovered_objects:
                if len(endpoint_ids) == 1:
                    delete_sql = f"DELETE FROM {schema_name}.{object_name} WHERE "
                    f" {mm_schemas.WriterEvent.ENDPOINT_ID} = %s"
                    stmt = Statement(delete_sql, (endpoint_ids[0],))
                else:
                    delete_sql = f"DELETE FROM {schema_name}.{object_name} WHERE "
                    f" {mm_schemas.WriterEvent.ENDPOINT_ID} = ANY(%s)"
                    stmt = Statement(delete_sql, (endpoint_ids,))

                statements.append(stmt)

        except Exception as e:
            logger.warning(
                "Failed to discover aggregate objects for deletion",
                project=self.project,
                error=mlrun.errors.err_to_str(e),
            )
            # Continue with empty statements list rather than failing completely

        return statements

    def delete_tsdb_resources(self) -> None:
        """
        Delete all project resources in TimescaleDB by discovering existing tables that match our patterns.

        This approach ensures we don't miss any tables, even if configurations are out of sync.
        """
        logger.debug(
            "Deleting all project resources from TimescaleDB",
            project=self.project,
        )

        try:
            schema_name = self.tables[mm_schemas.TDEngineSuperTables.PREDICTIONS].schema

            # Get the base table patterns for this project
            base_patterns = []
            for table_schema in self.tables.values():
                base_patterns.append(table_schema.table_name)

            # Build discovery query for all project objects
            pattern_conditions = []
            parameters = [schema_name]

            for pattern in base_patterns:
                # Match exact table name OR table name with _agg_/_cagg_ suffix
                pattern_conditions.extend(
                    [
                        "table_name = %s",
                        "table_name LIKE %s",  # _agg_ tables
                        "table_name LIKE %s",  # _cagg_ views
                    ]
                )
                parameters.extend([pattern, f"{pattern}_agg_%", f"{pattern}_cagg_%"])

            # Discover tables
            tables_stmt = Statement(
                f"""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = %s
                AND table_type = 'BASE TABLE'
                AND ({' OR '.join(pattern_conditions)})
                ORDER BY table_name
                """,
                tuple([schema_name] + parameters[1:]),
            )

            # Build separate pattern conditions for materialized views (use matviewname column)
            view_pattern_conditions = []
            view_parameters = [schema_name]

            for pattern in base_patterns:
                # For materialized views, only look for _cagg_ pattern
                view_pattern_conditions.append("matviewname LIKE %s")
                view_parameters.append(f"{pattern}_cagg_%")

            # Discover materialized views (continuous aggregates)
            views_stmt = Statement(
                f"""
                SELECT matviewname as table_name
                FROM pg_matviews
                WHERE schemaname = %s
                AND ({' OR '.join(view_pattern_conditions)})
                ORDER BY matviewname
                """,
                tuple(view_parameters),
            )

            tables_result = self._connection.run(query=tables_stmt)
            views_result = self._connection.run(query=views_stmt)

            discovered_tables = (
                [row[0] for row in tables_result.data]
                if tables_result and tables_result.data
                else []
            )
            discovered_views = (
                [row[0] for row in views_result.data]
                if views_result and views_result.data
                else []
            )

            if not discovered_tables and not discovered_views:
                logger.debug(
                    "No project resources found to delete",
                    project=self.project,
                    schema=schema_name,
                )
                return

            logger.debug(
                "Discovered project resources for deletion",
                project=self.project,
                tables=len(discovered_tables),
                views=len(discovered_views),
                schema=schema_name,
            )

            drop_statements = []

            # Drop materialized views first (they depend on tables)
            for view_name in discovered_views:
                drop_statements.append(
                    f"DROP MATERIALIZED VIEW IF EXISTS {schema_name}.{view_name} CASCADE"
                )

            # Drop tables second
            for table_name in discovered_tables:
                drop_statements.append(
                    f"DROP TABLE IF EXISTS {schema_name}.{table_name} CASCADE"
                )

            # Execute all drops
            if drop_statements:
                self._connection.run(statements=drop_statements)

                logger.debug(
                    "Successfully dropped project resources",
                    project=self.project,
                    dropped_objects=len(drop_statements),
                )

            # Check if schema is empty and drop it if so
            self._drop_schema_if_empty()

        except Exception as e:
            logger.warning(
                "Failed to delete all project resources from TimescaleDB",
                project=self.project,
                error=mlrun.errors.err_to_str(e),
            )
            raise

        logger.debug(
            "Successfully deleted all project resources from TimescaleDB",
            project=self.project,
        )

    def _drop_schema_if_empty(self) -> None:
        """Drop the schema if it contains no more tables using parameterized query."""
        try:
            schema_name = self.tables[mm_schemas.TDEngineSuperTables.PREDICTIONS].schema

            # Check if schema has any tables using parameterized query
            check_stmt = Statement(
                """
                SELECT COUNT(*) as table_count
                FROM information_schema.tables
                WHERE table_schema = %s
                """,
                (schema_name,),
            )

            result = self._connection.run(query=check_stmt)

            if result and result.data and result.data[0][0] == 0:
                # Schema is empty, drop it
                drop_schema_query = f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"
                self._connection.run(statements=[drop_schema_query])

                logger.debug(
                    "Dropped empty schema",
                    project=self.project,
                    schema=schema_name,
                )
        except Exception as e:
            logger.warning(
                "Failed to check/drop empty schema",
                project=self.project,
                error=mlrun.errors.err_to_str(e),
            )

    @staticmethod
    def _convert_to_datetime(val: Union[str, datetime]) -> datetime:
        """Convert string timestamps to datetime objects."""
        if isinstance(val, str):
            # Handle various timestamp formats
            if val.endswith("Z"):
                val = val.replace("Z", "+00:00")
            return datetime.fromisoformat(val)
        return val
