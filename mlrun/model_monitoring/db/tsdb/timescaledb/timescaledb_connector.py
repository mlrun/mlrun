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
from typing import Callable, Optional

import pandas as pd

import mlrun
import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_schema as timescaledb_schema
from mlrun.datastore.datastore_profile import DatastoreProfilePostgreSQL
from mlrun.model_monitoring.db import TSDBConnector
from mlrun.model_monitoring.db.tsdb.preaggregate import (
    PreAggregateConfig,
    PreAggregateHandler,
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
    TimescaleDBConnection,
)
from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_operations import (
    TimescaleDBOperationsHandler,
)
from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_stream import (
    TimescaleDBStreamHandler,
)
from mlrun.utils import logger
from mlrun.utils.debug import _format_args, _repr, traced_call


class TimescaleDBConnectorIn(TSDBConnector):
    """
    Complete TimescaleDB TSDB connector using composition pattern.

    Uses composition for all specialized functionality:
    - TimescaleDBMetricsQueries, TimescaleDBPredictionsQueries, TimescaleDBResultsQueries: Direct query operations
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
        self._connection = TimescaleDBConnection(
            dsn=profile.dsn(),
            min_connections=kwargs.get("min_connections", 1),
            max_connections=kwargs.get("max_connections", 10),
            max_retries=kwargs.get("max_retries", 3),
            retry_delay=kwargs.get("retry_delay", 1.0),
            autocommit=kwargs.get("autocommit", False),
        )

        # Create shared components needed by query classes
        tables = timescaledb_schema.create_table_schemas(project)
        self._tables = tables  # Store for backward compatibility
        pre_aggregate_handler = PreAggregateHandler(pre_aggregate_config)

        # Create specialized query handlers with proper initialization
        self._metrics_queries = TimescaleDBMetricsQueries(
            project=project,
            connection=self._connection,
            pre_aggregate_handler=pre_aggregate_handler,
            tables=tables,
        )
        self._predictions_queries = TimescaleDBPredictionsQueries(
            project=project,
            connection=self._connection,
            pre_aggregate_handler=pre_aggregate_handler,
            tables=tables,
        )
        self._results_queries = TimescaleDBResultsQueries(
            project=project,
            connection=self._connection,
            pre_aggregate_handler=pre_aggregate_handler,
            tables=tables,
        )

        # Create operations and stream handlers
        self._operations = TimescaleDBOperationsHandler(
            project=project,
            connection=self._connection,
            pre_aggregate_config=pre_aggregate_config,
        )

        self._stream = TimescaleDBStreamHandler(
            project=project, profile=profile, connection=self._connection
        )

        self._pre_aggregate_config = pre_aggregate_config

    def get_preaggregate_config(self) -> Optional[PreAggregateConfig]:
        """Returns the pre-aggregate configuration."""
        return self._pre_aggregate_config

    @property
    def tables(self):
        """Returns the table schemas for backward compatibility."""
        return self._tables

    @property
    def _queries(self):
        """Returns a backward compatibility object with tables and connection properties."""

        class QueriesCompatibility:
            def __init__(self, tables, connection):
                self.tables = tables
                self._connection = connection

        return QueriesCompatibility(self._tables, self._connection)

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

    def read_metrics_data(
        self,
        *,
        endpoint_id: str,
        start: datetime.datetime,
        end: datetime.datetime,
        metrics: list[mm_schemas.ModelEndpointMonitoringMetric],
        type: str,
        with_result_extra_data: bool = False,
    ):
        """Read metrics or results data from TimescaleDB (cross-cutting coordination)."""

        if type == "metrics":
            df = self._metrics_queries.read_metrics_data_impl(
                endpoint_id=endpoint_id,
                start=start,
                end=end,
                metrics=metrics,
            )
            # Use inherited method to convert DataFrame to domain objects
            return self.df_to_metrics_values(
                df=df, metrics=metrics, project=self.project
            )

        else:  # results
            df = self._results_queries.read_results_data_impl(
                endpoint_id=endpoint_id,
                start=start,
                end=end,
                metrics=metrics,
                with_result_extra_data=with_result_extra_data,
            )
            # Use inherited method to convert DataFrame to domain objects
            return self.df_to_results_values(
                df=df, metrics=metrics, project=self.project
            )

    def get_model_endpoint_real_time_metrics(self, *args, **kwargs):
        return self._metrics_queries.get_model_endpoint_real_time_metrics(
            *args, **kwargs
        )

    def get_metrics_metadata(self, *args, **kwargs):
        return self._metrics_queries.get_metrics_metadata(*args, **kwargs)

    async def add_basic_metrics(
        self,
        model_endpoint_objects: list[mlrun.common.schemas.ModelEndpoint],
        project: str,
        run_in_threadpool: Callable,
        metric_list: Optional[list[str]] = None,
    ) -> list[mlrun.common.schemas.ModelEndpoint]:
        """
        Add basic metrics to the model endpoint object using TimescaleDB optimizations.

        :param model_endpoint_objects: A list of `ModelEndpoint` objects that will
                                        be filled with the relevant basic metrics.
        :param project:                The name of the project (unused - uses self.project from constructor).
        :param run_in_threadpool:      A function that runs another function in a thread pool
                                       (unused - TimescaleDB operations are synchronous).
        :param metric_list:            List of metrics to include from the time series DB. Defaults to all metrics.

        :return: A list of `ModelEndpointMonitoringMetric` objects.
        """
        # Note: project and run_in_threadpool parameters are part of the interface
        # but unused in TimescaleDB implementation (uses self.project, synchronous operations)
        del project, run_in_threadpool  # Suppress unused variable warnings

        uids = [mep.metadata.uid for mep in model_endpoint_objects]

        # Access methods directly from the respective query classes
        metric_name_to_function = {
            "error_count": self._results_queries.get_error_count,
            "last_request": self._predictions_queries.get_last_request,
            "avg_latency": self._predictions_queries.get_avg_latency,
            "result_status": self._results_queries.get_drift_status,
        }

        if metric_list is not None:
            for metric_name in list(metric_name_to_function):
                if metric_name not in metric_list:
                    del metric_name_to_function[metric_name]

        metric_name_to_df = {
            metric_name: function(endpoint_ids=uids)
            for metric_name, function in metric_name_to_function.items()
        }

        def add_metrics(
            mep: mlrun.common.schemas.ModelEndpoint,
            df_dictionary: dict[str, pd.DataFrame],
        ):
            for metric in df_dictionary:
                df = df_dictionary.get(metric, pd.DataFrame())
                if not df.empty:
                    line = df[df["endpoint_id"] == mep.metadata.uid]
                    if not line.empty and metric in line:
                        value = line[metric].item()
                        if isinstance(value, pd.Timestamp):
                            value = value.to_pydatetime()
                        setattr(mep.status, metric, value)

            return mep

        return list(
            map(
                lambda mep: add_metrics(
                    mep=mep,
                    df_dictionary=metric_name_to_df,
                ),
                model_endpoint_objects,
            )
        )

    def read_predictions(self, *args, **kwargs):
        return self._predictions_queries.read_predictions(*args, **kwargs)

    def get_last_request(self, *args, **kwargs):
        return self._predictions_queries.get_last_request(*args, **kwargs)

    def get_avg_latency(self, *args, **kwargs):
        return self._predictions_queries.get_avg_latency(*args, **kwargs)

    def count_processed_model_endpoints(self, *args, **kwargs):
        return self._predictions_queries.count_processed_model_endpoints(
            *args, **kwargs
        )

    def get_drift_status(self, *args, **kwargs):
        return self._results_queries.get_drift_status(*args, **kwargs)

    def get_results_metadata(self, *args, **kwargs):
        return self._results_queries.get_results_metadata(*args, **kwargs)

    def get_error_count(self, *args, **kwargs):
        return self._results_queries.get_error_count(*args, **kwargs)

    def count_results_by_status(self, *args, **kwargs):
        return self._results_queries.count_results_by_status(*args, **kwargs)

    def apply_monitoring_stream_steps(self, *args, **kwargs) -> None:
        return self._stream.apply_monitoring_stream_steps(*args, **kwargs)

    def handle_model_error(self, *args, **kwargs) -> None:
        return self._stream.handle_model_error(*args, **kwargs)

    def calculate_latest_metrics(self, *args, **kwargs):
        return self._metrics_queries.calculate_latest_metrics(*args, **kwargs)

    def get_drift_data(self, *args, **kwargs):
        return self._results_queries.get_drift_data(*args, **kwargs)


class TimescaleDBConnector(TimescaleDBConnectorIn):
    # Delegate operations methods
    def create_tables(self, *args, **kwargs):
        return traced_call(super().create_tables, *args, **kwargs)

    def write_application_event(self, *args, **kwargs):
        return traced_call(super().write_application_event, *args, **kwargs)

    def delete_tsdb_records(self, *args, **kwargs):
        return traced_call(super().delete_tsdb_records, *args, **kwargs)

    def delete_application_records(self, *args, **kwargs):
        return traced_call(super().delete_application_records, *args, **kwargs)

    def delete_tsdb_resources(self, *args, **kwargs):
        return traced_call(super().delete_tsdb_resources, *args, **kwargs)

    # Delegate query methods
    def read_metrics_data(self, *args, **kwargs):
        return traced_call(super().read_metrics_data, *args, **kwargs)

    def read_predictions(self, *args, **kwargs):
        return traced_call(super().read_predictions, *args, **kwargs)

    def get_last_request(self, *args, **kwargs):
        return traced_call(super().get_last_request, *args, **kwargs)

    def get_drift_status(self, *args, **kwargs):
        return traced_call(super().get_drift_status, *args, **kwargs)

    def get_metrics_metadata(self, *args, **kwargs):
        return traced_call(super().get_metrics_metadata, *args, **kwargs)

    def get_results_metadata(self, *args, **kwargs):
        return traced_call(super().get_results_metadata, *args, **kwargs)

    def get_error_count(self, *args, **kwargs):
        return traced_call(super().get_error_count, *args, **kwargs)

    def get_avg_latency(self, *args, **kwargs):
        return traced_call(super().get_avg_latency, *args, **kwargs)

    def count_results_by_status(self, *args, **kwargs):
        return traced_call(super().count_results_by_status, *args, **kwargs)

    def get_model_endpoint_real_time_metrics(self, *args, **kwargs):
        return traced_call(
            super().get_model_endpoint_real_time_metrics, *args, **kwargs
        )

    async def add_basic_metrics(self, *args, **kwargs):
        name = f"{super().add_basic_metrics.__module__}.{super().add_basic_metrics.__name__}"
        formatted_args = _format_args(super().add_basic_metrics, args, kwargs)

        logger.info(f"TDECALL: {name}({formatted_args})")

        try:
            result = await super().add_basic_metrics(*args, **kwargs)
            result_repr = "None" if result is None else _repr(result)
            logger.info(f"TDERETURN: {name} -> {result_repr}")
            return result
        except Exception as e:
            logger.info(f"TDEEXCEPTION: {name} -> {type(e).__name__}: {str(e)[:100]}")
            raise

    # Delegate stream methods
    def apply_monitoring_stream_steps(self, *args, **kwargs):
        return traced_call(super().apply_monitoring_stream_steps, *args, **kwargs)

    def handle_model_error(self, *args, **kwargs):
        return traced_call(super().handle_model_error, *args, **kwargs)

    def calculate_latest_metrics(self, *args, **kwargs):
        return traced_call(super().calculate_latest_metrics, *args, **kwargs)

    def count_processed_model_endpoints(self, *args, **kwargs):
        return traced_call(super().count_processed_model_endpoints, *args, **kwargs)

    def get_drift_data(self, *args, **kwargs):
        return traced_call(super().get_drift_data, *args, **kwargs)
