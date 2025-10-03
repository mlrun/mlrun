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
import mlrun
import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_schema as timescaledb_schema
from mlrun.datastore.datastore_profile import DatastoreProfilePostgreSQL
from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connection import (
    TimescaleDBConnection,
)


class TimescaleDBStreamProcessor:
    """
    Handles stream processing operations for TimescaleDB TSDB connector.

    This class implements stream graph setup methods:
    - Monitoring stream steps configuration
    - Error handling setup
    - Real-time data ingestion pipeline

    Each instance creates its own TimescaleDBConnection that shares the global connection pool.
    """

    def __init__(
        self,
        project: str,
        profile: DatastoreProfilePostgreSQL,
        connection: TimescaleDBConnection,
    ):
        """
        Initialize stream handler with a shared connection.

        :param project: The project name
        :param profile: Datastore profile for connection (used for table initialization)
        :param connection: Shared TimescaleDBConnection instance
        """
        self.project = project
        self.profile = profile

        # Use the injected shared connection
        self._connection = connection

        # Initialize table schemas for stream operations
        self._init_tables()

    def _init_tables(self) -> None:
        """Initialize TimescaleDB table schemas for stream operations."""
        schema_name = (
            f"{timescaledb_schema._MODEL_MONITORING_SCHEMA}_{mlrun.mlconf.system_id}"
        )

        self.tables = {
            mm_schemas.TimescaleDBTables.PREDICTIONS: timescaledb_schema.Predictions(
                project=self.project, schema=schema_name
            ),
            mm_schemas.TimescaleDBTables.ERRORS: timescaledb_schema.Errors(
                project=self.project, schema=schema_name
            ),
        }

    def apply_monitoring_stream_steps(self, graph, **kwargs) -> None:
        """
        Apply TimescaleDB steps on the monitoring graph for real-time data ingestion.

        Sets up the stream processing pipeline to write prediction latency and
        custom metrics to TimescaleDB hypertables using the TimescaleDBTarget.

        :param graph: The stream processing graph to modify
        :param kwargs: Additional configuration parameters
        """

        def apply_process_before_timescaledb():
            """Add preprocessing step for TimescaleDB data format."""
            graph.add_step(
                "mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_stream_graph_steps.ProcessBeforeTimescaleDB",
                name="ProcessBeforeTimescaleDB",
                after="FilterNOP",
            )

        def apply_timescaledb_target(name: str, after: str):
            """Add TimescaleDB target for writing predictions data."""
            predictions_table = self.tables[mm_schemas.TimescaleDBTables.PREDICTIONS]

            graph.add_step(
                "mlrun.datastore.storeytargets.TimescaleDBStoreyTarget",
                name=name,
                after=after,
                url=f"ds://{self.profile.name}",
                time_col=mm_schemas.WriterEvent.END_INFER_TIME,
                table=predictions_table.full_name(),
                columns=[
                    mm_schemas.EventFieldType.LATENCY,
                    mm_schemas.EventKeyMetrics.CUSTOM_METRICS,
                    mm_schemas.EventFieldType.ESTIMATED_PREDICTION_COUNT,
                    mm_schemas.EventFieldType.EFFECTIVE_SAMPLE_COUNT,
                    mm_schemas.WriterEvent.ENDPOINT_ID,
                ],
                max_events=kwargs.get("tsdb_batching_max_events", 1000),
                flush_after_seconds=kwargs.get("tsdb_batching_timeout_secs", 30),
            )

        # Apply the processing steps
        apply_process_before_timescaledb()
        apply_timescaledb_target(
            name="TimescaleDBTarget",
            after="ProcessBeforeTimescaleDB",
        )

    def handle_model_error(
        self,
        graph,
        tsdb_batching_max_events: int = 1000,
        tsdb_batching_timeout_secs: int = 30,
        **kwargs,
    ) -> None:
        """
        Add error handling branch to the stream processing graph.

        Processes model errors and writes them to the TimescaleDB errors table
        for monitoring and alerting purposes.

        :param graph: The stream processing graph to modify
        :param tsdb_batching_max_events: Maximum events per batch
        :param tsdb_batching_timeout_secs: Batch timeout in seconds
        :param kwargs: Additional configuration parameters
        """

        errors_table = self.tables[mm_schemas.TimescaleDBTables.ERRORS]

        # Add error extraction step
        graph.add_step(
            "mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_stream_graph_steps.TimescaleDBErrorExtractor",
            name="error_extractor",
            after="ForwardError",
        )

        # Add TimescaleDB target for error data
        graph.add_step(
            "mlrun.datastore.storeytargets.TimescaleDBStoreyTarget",
            name="timescaledb_error",
            after="error_extractor",
            url=f"ds://{self.profile.name}",
            time_col=mm_schemas.EventFieldType.TIME,
            table=errors_table.full_name(),
            columns=[
                mm_schemas.EventFieldType.MODEL_ERROR,
                mm_schemas.WriterEvent.ENDPOINT_ID,
                mm_schemas.EventFieldType.ERROR_TYPE,
            ],
            max_events=tsdb_batching_max_events,
            flush_after_seconds=tsdb_batching_timeout_secs,
        )
