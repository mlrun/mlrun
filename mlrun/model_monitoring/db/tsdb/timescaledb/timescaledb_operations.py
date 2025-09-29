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

from typing import Optional

import psycopg

import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.errors
import mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_schema as timescaledb_schema
from mlrun.model_monitoring.db.tsdb.preaggregate import PreAggregateConfig
from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connection import (
    Statement,
    TimescaleDBConnection,
)
from mlrun.model_monitoring.db.tsdb.timescaledb.utils.timescaledb_query_builder import (
    TimescaleDBNaming,
)
from mlrun.utils import datetime_from_iso, logger


class TimescaleDBOperationsManager:
    """
    Handles all CRUD operations for TimescaleDB TSDB connector.

    This class implements all create/update/delete operations for model monitoring data:
    - Table and schema creation with optional pre-aggregates and continuous aggregates
    - Event writing with parameterized queries
    - Record deletion with support for both raw and aggregate data cleanup
    - Resource deletion with automatic discovery of project-related tables and views
    - Schema management with automatic cleanup of empty schemas


    Key Features:
    - Parameterized queries for all write/delete operations
    - Automatic discovery of aggregate tables for comprehensive cleanup
    - Transaction-based operations for data consistency
    - Configurable pre-aggregation with retention policies
    - Thread-safe operations through shared connection pooling

    :param project: Project name used for table naming and schema organization
    :param connection: Shared TimescaleDBConnection instance
    :param pre_aggregate_config: Optional configuration for pre-aggregated tables
    """

    def __init__(
        self,
        project: str,
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
        self._pre_aggregate_config = pre_aggregate_config

        # Use the injected shared connection
        self._connection = connection

        # Initialize table schemas
        self._init_tables()

    def _init_tables(self) -> None:
        self.tables = timescaledb_schema.create_table_schemas(self.project)

    def create_tables(
        self, pre_aggregate_config: Optional[PreAggregateConfig] = None
    ) -> None:
        config = pre_aggregate_config or self._pre_aggregate_config

        logger.debug(
            "Creating TimescaleDB tables for model monitoring",
            project=self.project,
            with_pre_aggregates=config is not None,
        )
        # Try to create extension, ignore if already exists
        try:
            self._connection.run(
                statements=["CREATE EXTENSION IF NOT EXISTS timescaledb"]
            )
        except psycopg.errors.DuplicateObject:
            # Extension already loaded - this is fine
            pass

        # Create schema if it doesn't exist
        schema_name = self.tables[mm_schemas.TimescaleDBTables.PREDICTIONS].schema
        self._connection.run(statements=[f"CREATE SCHEMA IF NOT EXISTS {schema_name}"])

        # Create main tables and convert to hypertables
        for table_type, table in self.tables.items():
            statements = [table._create_table_query()]

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
            table = self.tables[mm_schemas.TimescaleDBTables.APP_RESULTS]
        else:
            table = self.tables[mm_schemas.TimescaleDBTables.METRICS]

        # Convert datetime strings to datetime objects if needed
        for time_field in [
            mm_schemas.WriterEvent.END_INFER_TIME,
            mm_schemas.WriterEvent.START_INFER_TIME,
        ]:
            if time_field in event:
                if isinstance(event[time_field], str):
                    event[time_field] = datetime_from_iso(event[time_field])
                # datetime objects can stay as-is

        # Prepare the INSERT statement with parameterized query
        columns = list(table.columns.keys())
        placeholders = ", ".join(["%s"] * len(columns))

        insert_sql = f"""
            INSERT INTO {table.full_name()} ({', '.join(columns)})
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
                error=mlrun.errors.err_to_str(e),
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

            # Execute all deletions in a single transaction
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
                    f"DELETE FROM {table_schema.full_name()} "
                    f"WHERE {mm_schemas.WriterEvent.ENDPOINT_ID} = %s"
                )
                stmt = Statement(delete_sql, (endpoint_ids[0],))
            else:
                delete_sql = (
                    f"DELETE FROM {table_schema.full_name()} "
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
            schema_name = self.tables[mm_schemas.TimescaleDBTables.PREDICTIONS].schema

            # Get base table patterns for tables that have endpoint_id
            base_patterns = []
            base_patterns.extend(
                self.tables[table_type].table_name
                for table_type in [
                    mm_schemas.TimescaleDBTables.PREDICTIONS,
                    mm_schemas.TimescaleDBTables.METRICS,
                    mm_schemas.TimescaleDBTables.APP_RESULTS,
                ]
                if table_type in self.tables
            )
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
                parameters.extend(TimescaleDBNaming.get_all_aggregate_patterns(pattern))

            # Build separate pattern conditions for materialized views
            view_pattern_conditions = []
            view_parameters = [schema_name]

            for pattern in base_patterns:
                view_pattern_conditions.append("matviewname LIKE %s")
                view_parameters.append(TimescaleDBNaming.get_cagg_pattern(pattern))

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
                delete_sql = f"DELETE FROM {schema_name}.{object_name} WHERE "
                if len(endpoint_ids) == 1:
                    f" {mm_schemas.WriterEvent.ENDPOINT_ID} = %s"
                    stmt = Statement(delete_sql, (endpoint_ids[0],))
                else:
                    f" {mm_schemas.WriterEvent.ENDPOINT_ID} = ANY(%s)"
                    stmt = Statement(delete_sql, (endpoint_ids,))

                statements.append(stmt)

        except Exception as e:
            logger.debug(
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
            schema_name = self.tables[mm_schemas.TimescaleDBTables.PREDICTIONS].schema

            # Get the base table patterns for this project
            base_patterns = []
            base_patterns.extend(
                table_schema.table_name for table_schema in self.tables.values()
            )
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
                parameters.extend(TimescaleDBNaming.get_deletion_patterns(pattern))

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

            # Build separate pattern conditions for TimescaleDB continuous aggregates
            view_pattern_conditions = []
            view_parameters = [schema_name]

            for pattern in base_patterns:
                # For continuous aggregates, look for _cagg_ pattern
                view_pattern_conditions.append("view_name LIKE %s")
                view_parameters.append(TimescaleDBNaming.get_cagg_pattern(pattern))

            # Discover TimescaleDB continuous aggregates (use TimescaleDB catalog, not pg_matviews)
            views_stmt = Statement(
                f"""
                SELECT view_name as table_name
                FROM timescaledb_information.continuous_aggregates
                WHERE view_schema = %s
                AND ({' OR '.join(view_pattern_conditions)})
                ORDER BY view_name
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
            if discovered_views:
                view_list = ", ".join(
                    f"{schema_name}.{view_name}" for view_name in discovered_views
                )
                drop_statements.append(
                    f"DROP MATERIALIZED VIEW IF EXISTS {view_list} CASCADE"
                )

            # Drop tables second (one by one due to TimescaleDB hypertable limitations)
            drop_statements.extend(
                f"DROP TABLE IF EXISTS {schema_name}.{table_name} CASCADE"
                for table_name in discovered_tables
            )
            # Execute all drops
            if drop_statements:
                self._connection.run(statements=drop_statements)

                logger.debug(
                    "Successfully dropped project resources from TimescaleDB",
                    project=self.project,
                )

            # Optional cleanup: drop schema if empty (errors are logged but don't fail the operation)
            self._drop_schema_if_empty()

        except Exception as e:
            logger.error(
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
        """
        Drop the schema if it contains no more tables using parameterized query.

        This is a best-effort cleanup operation that should not fail the main resource deletion.
        Schema dropping may fail due to permissions, remaining objects, or concurrent operations,
        but the primary table deletion operation has already succeeded.
        """
        try:
            schema_name = self.tables[mm_schemas.TimescaleDBTables.PREDICTIONS].schema

            # Check if schema has any tables using parameterized query
            check_stmt = Statement(
                """
                SELECT COUNT(*) AS table_count
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
            # Schema dropping is optional cleanup - don't fail the main operation
            # This may happen due to permissions, remaining objects, or concurrent operations
            logger.warning(
                "Failed to check/drop empty schema (non-critical cleanup operation)",
                project=self.project,
                error=mlrun.errors.err_to_str(e),
            )

    def delete_application_records(
        self, application_name: str, endpoint_ids: Optional[list[str]] = None
    ) -> None:
        """
        Delete application records from TimescaleDB for the given model endpoints or all if endpoint_ids is None.

        This method deletes records from both app_results and metrics tables that match the specified
        application name and optionally filter by endpoint IDs.

        :param application_name: Name of the application whose records should be deleted
        :param endpoint_ids: Optional list of endpoint IDs to filter deletion. If None, deletes all records
                            for the application across all endpoints.
        """
        logger.debug(
            "Deleting application records from TimescaleDB",
            project=self.project,
            application_name=application_name,
            endpoint_ids=endpoint_ids,
        )

        if not application_name:
            logger.warning(
                "No application name provided for deletion", project=self.project
            )
            return

        try:
            self._delete_application_records(application_name, endpoint_ids)
        except Exception as e:
            logger.error(
                "Failed to delete application records from TimescaleDB",
                project=self.project,
                application_name=application_name,
                endpoint_ids=endpoint_ids,
                error=mlrun.errors.err_to_str(e),
            )
            raise mlrun.errors.MLRunRuntimeError(
                f"Failed to delete application records for {application_name}: {e}"
            ) from e

    def _delete_application_records(self, application_name, endpoint_ids):
        base_parameters = [application_name]

        # Add endpoint filter if provided
        if endpoint_ids:
            if len(endpoint_ids) == 1:
                endpoint_filter = f" AND {mm_schemas.WriterEvent.ENDPOINT_ID} = %s"
                parameters = base_parameters + [endpoint_ids[0]]
            else:
                endpoint_filter = f" AND {mm_schemas.WriterEvent.ENDPOINT_ID} = ANY(%s)"
                parameters = base_parameters + [endpoint_ids]
        else:
            endpoint_filter = ""
            parameters = base_parameters

        # Delete from app_results table
        app_results_table = self.tables[mm_schemas.TimescaleDBTables.APP_RESULTS]
        app_filter = f"{mm_schemas.WriterEvent.APPLICATION_NAME} = %s"
        app_results_sql = (
            f"DELETE FROM {app_results_table.full_name()} "
            f"WHERE {app_filter}{endpoint_filter}"
        )
        deletion_statements = [Statement(app_results_sql, tuple(parameters))]
        # Delete from metrics table
        metrics_table = self.tables[mm_schemas.TimescaleDBTables.METRICS]
        metrics_sql = (
            f"DELETE FROM {metrics_table.full_name()} "
            f"WHERE {app_filter}{endpoint_filter}"
        )
        deletion_statements.append(Statement(metrics_sql, tuple(parameters)))

        # Also delete from aggregate tables if they exist
        aggregate_statements = self._get_aggregate_delete_statements_by_application(
            application_name, endpoint_ids
        )
        deletion_statements.extend(aggregate_statements)

        # Execute all deletions in a single transaction
        self._connection.run(statements=deletion_statements)

        logger.debug(
            "Successfully deleted application records from TimescaleDB",
            project=self.project,
            application_name=application_name,
            endpoint_count=len(endpoint_ids) if endpoint_ids else "all",
        )

    def _get_aggregate_delete_statements_by_application(
        self, application_name: str, endpoint_ids: Optional[list[str]] = None
    ) -> list[Statement]:
        """
        Get parameterized DELETE statements for aggregate tables filtered by application name.

        This discovers existing aggregate tables and creates deletion statements that filter
        by both application name and optionally endpoint IDs.

        :param application_name: Application name to filter by
        :param endpoint_ids: Optional endpoint IDs to filter by
        :return: List of Statement objects for aggregate data deletion
        """
        statements = []

        try:
            schema_name = self.tables[mm_schemas.TimescaleDBTables.PREDICTIONS].schema

            # Discover all continuous aggregates and materialized views for this project
            discovery_stmt = Statement(
                """
                SELECT table_name
                FROM (
                    SELECT matviewname as table_name
                    FROM pg_matviews
                    WHERE schemaname = %s
                    AND matviewname LIKE %s

                    UNION ALL

                    SELECT view_name as table_name
                    FROM timescaledb_information.continuous_aggregates
                    WHERE view_schema = %s
                    AND view_name LIKE %s
                ) AS combined_objects
                ORDER BY table_name
                """,
                (schema_name, f"%{self.project}%", schema_name, f"%{self.project}%"),
            )

            result = self._connection.run(query=discovery_stmt)
            discovered_objects = (
                [row[0] for row in result.data] if result and result.data else []
            )

            if not discovered_objects:
                logger.debug(
                    "No aggregate objects found for application deletion",
                    project=self.project,
                    application_name=application_name,
                    schema=schema_name,
                )
                return statements

            logger.debug(
                "Discovered aggregate objects for application deletion",
                project=self.project,
                application_name=application_name,
                aggregate_objects=len(discovered_objects),
            )

            # Build filter conditions
            app_filter = f"{mm_schemas.WriterEvent.APPLICATION_NAME} = %s"
            base_parameters = [application_name]

            if endpoint_ids:
                if len(endpoint_ids) == 1:
                    endpoint_filter = f" AND {mm_schemas.WriterEvent.ENDPOINT_ID} = %s"
                    parameters = base_parameters + [endpoint_ids[0]]
                else:
                    endpoint_filter = (
                        f" AND {mm_schemas.WriterEvent.ENDPOINT_ID} = ANY(%s)"
                    )
                    parameters = base_parameters + [endpoint_ids]
            else:
                endpoint_filter = ""
                parameters = base_parameters

            # Create delete statements for all discovered aggregate objects
            for object_name in discovered_objects:
                delete_sql = (
                    f"DELETE FROM {schema_name}.{object_name} "
                    f"WHERE {app_filter}{endpoint_filter}"
                )
                stmt = Statement(delete_sql, tuple(parameters))
                statements.append(stmt)

        except Exception as e:
            logger.warning(
                "Failed to discover aggregate objects for application deletion",
                project=self.project,
                application_name=application_name,
                error=mlrun.errors.err_to_str(e),
            )
            # Continue with empty statements list rather than failing completely

        return statements
