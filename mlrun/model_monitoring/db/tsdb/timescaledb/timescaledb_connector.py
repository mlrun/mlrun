# Alternative approach using composition instead of inheritance

from typing import Optional

import mlrun.common.schemas.model_monitoring as mm_schemas
from mlrun.datastore.datastore_profile import DatastoreProfileTimescaleDB
from mlrun.model_monitoring.db import TSDBConnector
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
from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_stream import (
    TimescaleDBStreamHandler,
)


class TimescaleDBConnector(TSDBConnector):
    """
    Complete TimescaleDB TSDB connector using composition
    """

    type: str = mm_schemas.TSDBTarget.TimescaleDB

    def __init__(
        self,
        project: str,
        profile: DatastoreProfileTimescaleDB,
        pre_aggregate_config: Optional[PreAggregateConfig] = None,
        **kwargs,
    ):
        super().__init__(project=project)

        # Create shared connection
        connection = TimescaleDBConnection(
            dsn=profile.dsn(),
            min_connections=kwargs.get("min_connections", 1),
            max_connections=kwargs.get("max_connections", 10),
            max_retries=kwargs.get("max_retries", 3),
            retry_delay=kwargs.get("retry_delay", 1.0),
            autocommit=kwargs.get("autocommit", False),
        )

        self._operations = TimescaleDBOperationsHandler(
            project=project,
            connection=connection,
            pre_aggregate_config=pre_aggregate_config,
        )

        self._queries = TimescaleDBQueryHandler(
            project=project,
            connection=connection,
            pre_aggregate_config=pre_aggregate_config,
        )

        self._stream = TimescaleDBStreamHandler(
            project=project, profile=profile, connection=connection
        )

        self._pre_aggregate_config = pre_aggregate_config

    def get_preaggregate_config(self) -> Optional[PreAggregateConfig]:
        """Returns the pre-aggregate configuration."""
        return self._pre_aggregate_config

    # Delegate operations methods
    def create_tables(self, **kwargs) -> None:
        return self._operations.create_tables(**kwargs)

    def write_application_event(self, **kwargs) -> None:
        return self._operations.write_application_event(**kwargs)

    def delete_tsdb_records(self, **kwargs) -> None:
        return self._operations.delete_tsdb_records(**kwargs)

    def delete_tsdb_resources(self) -> None:
        return self._operations.delete_tsdb_resources()

    # Delegate query methods
    def read_metrics_data(self, **kwargs):
        return self._queries.read_metrics_data(**kwargs)

    def read_predictions(self, **kwargs):
        return self._queries.read_predictions(**kwargs)

    def get_last_request(self, **kwargs):
        return self._queries.get_last_request(**kwargs)

    def get_drift_status(self, **kwargs):
        return self._queries.get_drift_status(**kwargs)

    def get_metrics_metadata(self, **kwargs):
        return self._queries.get_metrics_metadata(**kwargs)

    def get_results_metadata(self, **kwargs):
        return self._queries.get_results_metadata(**kwargs)

    def get_error_count(self, **kwargs):
        return self._queries.get_error_count(**kwargs)

    def get_avg_latency(self, **kwargs):
        return self._queries.get_avg_latency(**kwargs)

    def count_results_by_status(self, **kwargs):
        return self._queries.count_results_by_status(**kwargs)

    def get_model_endpoint_real_time_metrics(self, **kwargs):
        return self._queries.get_model_endpoint_real_time_metrics(**kwargs)

    async def add_basic_metrics(self, **kwargs):
        return await self._queries.add_basic_metrics(**kwargs)

    # Delegate stream methods
    def apply_monitoring_stream_steps(self, graph, **kwargs) -> None:
        return self._stream.apply_monitoring_stream_steps(graph, **kwargs)

    def handle_model_error(self, graph, **kwargs) -> None:
        return self._stream.handle_model_error(graph, **kwargs)
