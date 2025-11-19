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
from typing import Optional

import pandas as pd

import mlrun
import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_schema as timescaledb_schema
from mlrun.config import config
from mlrun.datastore.datastore_profile import DatastoreProfilePostgreSQL
from mlrun.model_monitoring.db import TSDBConnector
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
    TimescaleDBConnection,
)
from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_operations import (
    TimescaleDBOperationsManager,
)
from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_stream import (
    TimescaleDBStreamProcessor,
)
from mlrun.utils import logger


class TimescaleDBConnector(TSDBConnector):
    """
    Complete TimescaleDB TSDB connector using composition pattern.

    Uses composition for all specialized functionality:
    - TimescaleDBMetricsQueries, TimescaleDBPredictionsQueries, TimescaleDBResultsQueries: Direct query operations
    - TimescaleDBOperationsManager: Table management and write operations
    - TimescaleDBStreamProcessor: Stream processing operations
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

        self.profile = profile

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
        pre_aggregate_manager = PreAggregateManager(pre_aggregate_config)

        # Create specialized query handlers with proper initialization
        self._metrics_queries = TimescaleDBMetricsQueries(
            project=project,
            connection=self._connection,
            pre_aggregate_manager=pre_aggregate_manager,
            tables=tables,
        )
        self._predictions_queries = TimescaleDBPredictionsQueries(
            project=project,
            connection=self._connection,
            pre_aggregate_manager=pre_aggregate_manager,
            tables=tables,
        )
        self._results_queries = TimescaleDBResultsQueries(
            connection=self._connection,
            project=project,
            pre_aggregate_manager=pre_aggregate_manager,
            tables=tables,
        )

        # Create operations and stream handlers
        self._operations = TimescaleDBOperationsManager(
            project=project,
            connection=self._connection,
            pre_aggregate_config=pre_aggregate_config,
        )

        self._stream = TimescaleDBStreamProcessor(
            project=project, profile=profile, connection=self._connection
        )

        self._pre_aggregate_config = pre_aggregate_config

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

    def add_basic_metrics(
        self,
        model_endpoint_objects: list[mlrun.common.schemas.ModelEndpoint],
        metric_list: Optional[list[str]] = None,
    ) -> list[mlrun.common.schemas.ModelEndpoint]:
        """
        Add basic metrics to the model endpoint object using TimescaleDB optimizations.

        :param model_endpoint_objects: A list of `ModelEndpoint` objects that will
                                        be filled with the relevant basic metrics.
        :param metric_list:            List of metrics to include from the time series DB. Defaults to all metrics.

        :return: A list of `ModelEndpointMonitoringMetric` objects.
        """
        uids = [mep.metadata.uid for mep in model_endpoint_objects]

        # Access methods directly from the respective query classes
        # Note: last_request is handled separately due to potential data synchronization issues
        metric_name_to_function = {
            mm_schemas.EventFieldType.ERROR_COUNT: self._results_queries.get_error_count,
            mm_schemas.ModelEndpointSchema.AVG_LATENCY: self._predictions_queries.get_avg_latency,
            mm_schemas.ResultData.RESULT_STATUS: self._results_queries.get_drift_status,
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
                    line = df[
                        df[mm_schemas.WriterEvent.ENDPOINT_ID] == mep.metadata.uid
                    ]
                    if not line.empty and metric in line:
                        value = line[metric].item()
                        if isinstance(value, pd.Timestamp):
                            value = value.to_pydatetime()
                        setattr(mep.status, metric, value)

            return mep

        enriched_endpoints = list(
            map(
                lambda mep: add_metrics(
                    mep=mep,
                    df_dictionary=metric_name_to_df,
                ),
                model_endpoint_objects,
            )
        )

        # Handle last_request separately with special enrichment
        if metric_list is None or "last_request" in metric_list:
            self._enrich_mep_with_last_request(
                model_endpoint_objects={
                    mep.metadata.uid: mep for mep in enriched_endpoints
                }
            )

        return enriched_endpoints

    def _enrich_mep_with_last_request(
        self,
        model_endpoint_objects: dict[str, mlrun.common.schemas.ModelEndpoint],
    ):
        """
        Enrich model endpoint objects with last_request data from predictions table.
        This method handles the special case of last_request which may have timing issues.
        """
        try:
            last_request_df = self._predictions_queries.get_last_request(
                endpoint_ids=list(model_endpoint_objects.keys())
            )

            if not last_request_df.empty:
                for _, row in last_request_df.iterrows():
                    endpoint_id = row.get(mm_schemas.WriterEvent.ENDPOINT_ID)
                    last_request = row.get("last_request")

                    if (
                        endpoint_id in model_endpoint_objects
                        and last_request is not None
                    ):
                        if isinstance(last_request, pd.Timestamp):
                            last_request = last_request.to_pydatetime()
                        model_endpoint_objects[
                            endpoint_id
                        ].status.last_request = last_request
        except Exception as e:
            # Log but don't fail - last_request is not critical for basic functionality
            logger.warning(
                "Failed to enrich model endpoints with last_request data",
                error=mlrun.errors.err_to_str(e),
                endpoint_count=len(model_endpoint_objects),
            )

    def read_predictions(self, *args, **kwargs):
        return self._predictions_queries.read_predictions(*args, **kwargs)

    def _get_records(
        self,
        table: str,
        start: datetime.datetime,
        end: datetime.datetime,
        endpoint_id: Optional[str] = None,
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Get raw records from TimescaleDB as pandas DataFrame.

        This method provides direct access to raw table data.

        :param table: Table name - "metrics", "results", or "predictions"
        :param start: Start time for the query
        :param end: End time for the query
        :param endpoint_id: Optional endpoint ID filter (None = all endpoints)
        :param columns: Optional list of specific columns to return (None = all columns)
        :return: Raw pandas DataFrame with all matching records
        """
        if table == "metrics":
            df = self._metrics_queries.read_metrics_data_impl(
                endpoint_id=endpoint_id,
                start=start,
                end=end,
                metrics=None,  # Get all metrics
            )
        elif table == "results":
            df = self._results_queries.read_results_data_impl(
                endpoint_id=endpoint_id,
                start=start,
                end=end,
                metrics=None,  # Get all results
                with_result_extra_data=True,
            )
        elif table == "predictions":
            df = self._predictions_queries.read_predictions_impl(
                endpoint_id=endpoint_id,
                start=start,
                end=end,
                columns=columns,
            )
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Invalid table '{table}'. Must be 'metrics', 'results', or 'predictions'"
            )

        if columns is not None and not df.empty:
            # Filter to requested columns if specified
            available_columns = [col for col in columns if col in df.columns]
            df = df[available_columns]

        return df

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

    def add_pre_writer_steps(self, graph, after):
        return graph.add_step(
            "mlrun.model_monitoring.db.tsdb.timescaledb.writer_graph_steps.ProcessBeforeTimescaleDBWriter",
            name="ProcessBeforeTimescaleDBWriter",
            after=after,
        )

    def apply_writer_steps(self, graph, after, **kwargs) -> None:
        tables = timescaledb_schema.create_table_schemas(self.project)

        graph.add_step(
            "mlrun.datastore.storeytargets.TimescaleDBStoreyTarget",
            name="tsdb_metrics",
            after=after,
            url=f"ds://{self.profile.name}",
            table=tables[mm_schemas.TimescaleDBTables.METRICS].full_name(),
            time_col=mm_schemas.WriterEvent.END_INFER_TIME,
            columns=[
                mm_schemas.WriterEvent.START_INFER_TIME,
                mm_schemas.MetricData.METRIC_VALUE,
                mm_schemas.WriterEvent.ENDPOINT_ID,
                mm_schemas.WriterEvent.APPLICATION_NAME,
                mm_schemas.MetricData.METRIC_NAME,
            ],
            max_events=config.model_endpoint_monitoring.writer_graph.max_events,
            flush_after_seconds=config.model_endpoint_monitoring.writer_graph.flush_after_seconds,
        )

        graph.add_step(
            "mlrun.datastore.storeytargets.TimescaleDBStoreyTarget",
            name="tsdb_app_results",
            after=after,
            url=f"ds://{self.profile.name}",
            table=tables[mm_schemas.TimescaleDBTables.APP_RESULTS].full_name(),
            time_col=mm_schemas.WriterEvent.END_INFER_TIME,
            columns=[
                mm_schemas.WriterEvent.START_INFER_TIME,
                mm_schemas.ResultData.RESULT_VALUE,
                mm_schemas.ResultData.RESULT_STATUS,
                mm_schemas.ResultData.RESULT_EXTRA_DATA,
                mm_schemas.WriterEvent.ENDPOINT_ID,
                mm_schemas.WriterEvent.APPLICATION_NAME,
                mm_schemas.ResultData.RESULT_NAME,
                mm_schemas.ResultData.RESULT_KIND,
            ],
            max_events=config.model_endpoint_monitoring.writer_graph.max_events,
            flush_after_seconds=config.model_endpoint_monitoring.writer_graph.flush_after_seconds,
        )
