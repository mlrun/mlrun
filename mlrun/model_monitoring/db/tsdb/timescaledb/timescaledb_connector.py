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

import mlrun.common.schemas.model_monitoring as mm_schemas
from mlrun.datastore.datastore_profile import DatastoreProfilePostgreSQL
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
    Complete TimescaleDB TSDB connector using composition pattern.

    Delegates functionality to specialized handlers:
    - TimescaleDBQueryHandler: All query operations
    - TimescaleDBOperationsHandler: Table management and write operations
    - TimescaleDBStreamHandler: Stream processing operations
    """

    type: str = mm_schemas.TSDBTarget.TimescaleDB

    def __init__(
        self,
        project: str,
        profile: DatastoreProfilePostgreSQL,
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

        # Create specialized handlers with shared connection
        self._queries = TimescaleDBQueryHandler(
            project=project,
            connection=connection,
            pre_aggregate_config=pre_aggregate_config,
        )

        self._operations = TimescaleDBOperationsHandler(
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
    def create_tables(self, *args, **kwargs) -> None:
        return self._operations.create_tables(*args, **kwargs)

    def write_application_event(self, *args, **kwargs) -> None:
        return self._operations.write_application_event(*args, **kwargs)

    def delete_tsdb_records(self, *args, **kwargs) -> None:
        return self._operations.delete_tsdb_records(*args, **kwargs)

    def delete_tsdb_resources(self, *args, **kwargs) -> None:
        return self._operations.delete_tsdb_resources(*args, **kwargs)

    def delete_application_records(self, *args, **kwargs) -> None:
        return self._operations.delete_application_records(*args, **kwargs)

    # Delegate query methods
    def read_metrics_data(self, *args, **kwargs):
        return self._queries.read_metrics_data(*args, **kwargs)

    def read_predictions(self, *args, **kwargs):
        return self._queries.read_predictions(*args, **kwargs)

    def get_last_request(self, *args, **kwargs):
        return self._queries.get_last_request(*args, **kwargs)

    def get_drift_status(self, *args, **kwargs):
        return self._queries.get_drift_status(*args, **kwargs)

    def get_metrics_metadata(self, *args, **kwargs):
        return self._queries.get_metrics_metadata(*args, **kwargs)

    def get_results_metadata(self, *args, **kwargs):
        return self._queries.get_results_metadata(*args, **kwargs)

    def get_error_count(self, *args, **kwargs):
        return self._queries.get_error_count(*args, **kwargs)

    def get_avg_latency(self, *args, **kwargs):
        return self._queries.get_avg_latency(*args, **kwargs)

    def count_results_by_status(self, *args, **kwargs):
        return self._queries.count_results_by_status(*args, **kwargs)

    def get_model_endpoint_real_time_metrics(self, *args, **kwargs):
        return self._queries.get_model_endpoint_real_time_metrics(*args, **kwargs)

    async def add_basic_metrics(self, *args, **kwargs):
        return await self._queries.add_basic_metrics(*args, **kwargs)

    # Delegate stream methods
    def apply_monitoring_stream_steps(self, *args, **kwargs) -> None:
        return self._stream.apply_monitoring_stream_steps(*args, **kwargs)

    def handle_model_error(self, *args, **kwargs) -> None:
        return self._stream.handle_model_error(*args, **kwargs)

    def calculate_latest_metrics(self, *args, **kwargs):
        return self._queries.calculate_latest_metrics(*args, **kwargs)

    def count_processed_model_endpoints(self, *args, **kwargs):
        return self._queries.count_processed_model_endpoints(*args, **kwargs)

    def get_drift_data(self, *args, **kwargs):
        return self._queries.get_drift_data(*args, **kwargs)
