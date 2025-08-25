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
from typing import Callable, Literal, Optional, Union

import pandas as pd

import mlrun
import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.errors
from mlrun.model_monitoring.db import TSDBConnector
from mlrun.model_monitoring.db.tsdb.timescaledb.utils.timescaledb_dataframe_processor import (
    TimescaleDBDataFrameProcessor,
)
from mlrun.model_monitoring.db.tsdb.timescaledb.utils.timescaledb_query_builder import (
    TimescaleDBQueryBuilder,
)


class TimescaleDBMetricsQueries:
    """
    Mixin class containing metrics-related query methods for TimescaleDB.

    This class expects the following attributes to be available (provided by TimescaleDBQueryHandler):
    - self._connection: TimescaleDBConnection
    - self.tables: Dict of table schemas
    - self._pre_aggregate_handler: PreAggregateHandler
    - self.project: str
    """

    def get_model_endpoint_real_time_metrics(
        self,
        endpoint_id: str,
        metrics: list[str],
        start: str,
        end: str,
        interval: Optional[str] = None,
        agg_function: Optional[str] = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """Get real-time metrics with optional pre-aggregate optimization."""

        # Validate parameters using the pre-aggregate handler
        self._pre_aggregate_handler.validate_interval_and_function(
            interval, agg_function
        )

        start_dt, end_dt = TimescaleDBQueryBuilder.parse_datetime_strings(start, end)

        # Align times if interval is provided
        start_dt, end_dt = self._pre_aggregate_handler.align_time_range(
            start_dt, end_dt, interval
        )

        # Check if we can use pre-aggregates
        use_pre_aggregates = self._pre_aggregate_handler.can_use_pre_aggregates(
            interval=interval, agg_funcs=[agg_function] if agg_function else None
        )

        table_schema = self.tables[mm_schemas.TimescaleDBTables.METRICS]

        # Build query using the schema's query builder
        columns = [table_schema.time_column, mm_schemas.MetricData.METRIC_VALUE]
        filter_query = TimescaleDBQueryBuilder.build_endpoint_filter(endpoint_id)

        query = table_schema._get_records_query(
            start=start_dt,
            end=end_dt,
            columns_to_filter=columns,
            filter_query=filter_query,
            interval=interval if use_pre_aggregates else None,
            agg_funcs=[agg_function] if use_pre_aggregates and agg_function else None,
            use_pre_aggregates=use_pre_aggregates,
        )

        result = self._connection.run(query=query)

        # Process results into expected format
        metrics_data = {}
        if result and result.data:
            metric_name = (
                "default_metric"  # Would need to be extracted from actual query
            )
            for row in result.data:
                timestamp, value = row[0], row[1]
                if metric_name not in metrics_data:
                    metrics_data[metric_name] = []
                metrics_data[metric_name].append((timestamp.isoformat(), float(value)))

        return metrics_data

    def read_metrics_data(
        self,
        *,
        endpoint_id: str,
        start: datetime,
        end: datetime,
        metrics: list[mm_schemas.ModelEndpointMonitoringMetric],
        type: Literal["metrics", "results"],
        with_result_extra_data: bool = False,
    ) -> Union[
        list[
            Union[
                mm_schemas.ModelEndpointMonitoringResultValues,
                mm_schemas.ModelEndpointMonitoringMetricNoData,
            ],
        ],
        list[
            Union[
                mm_schemas.ModelEndpointMonitoringMetricValues,
                mm_schemas.ModelEndpointMonitoringMetricNoData,
            ],
        ],
    ]:
        """Read metrics or results data from TimescaleDB."""

        if type == "metrics":
            table_schema = self.tables[mm_schemas.TimescaleDBTables.METRICS]
            name_column = mm_schemas.MetricData.METRIC_NAME
            value_column = mm_schemas.MetricData.METRIC_VALUE
            columns = [
                table_schema.time_column,
                mm_schemas.WriterEvent.APPLICATION_NAME,
                name_column,
                value_column,
            ]
            df_handler = TSDBConnector.df_to_metrics_values  # Use base class method
        else:  # results
            table_schema = self.tables[mm_schemas.TimescaleDBTables.APP_RESULTS]
            name_column = mm_schemas.ResultData.RESULT_NAME
            value_column = mm_schemas.ResultData.RESULT_VALUE
            columns = [
                table_schema.time_column,
                mm_schemas.WriterEvent.APPLICATION_NAME,
                name_column,
                value_column,
                mm_schemas.ResultData.RESULT_STATUS,
                mm_schemas.ResultData.RESULT_KIND,
            ]
            if with_result_extra_data:
                columns.append(mm_schemas.ResultData.RESULT_EXTRA_DATA)
            df_handler = TSDBConnector.df_to_results_values  # Use base class method

        # Build metrics condition using query builder utilities
        metrics_condition = TimescaleDBQueryBuilder.build_metrics_filter(metrics)
        endpoint_filter = TimescaleDBQueryBuilder.build_endpoint_filter(endpoint_id)

        # Combine filters using query builder utilities
        filters = [endpoint_filter, metrics_condition]
        filter_query = TimescaleDBQueryBuilder.combine_filters(filters, operator="AND")

        query = table_schema._get_records_query(
            start=start,
            end=end,
            columns_to_filter=columns,
            filter_query=filter_query,
        )

        result = self._connection.run(query=query)

        # Convert to DataFrame
        df = TimescaleDBDataFrameProcessor.from_query_result(result)

        if not df.empty:
            df[table_schema.time_column] = pd.to_datetime(df[table_schema.time_column])
            df.set_index(table_schema.time_column, inplace=True)

        if not with_result_extra_data and type == "results":
            df[mm_schemas.ResultData.RESULT_EXTRA_DATA] = ""

        return df_handler(df=df, metrics=metrics, project=self.project)

    def get_metrics_metadata(
        self,
        endpoint_id: Union[str, list[str]],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        interval: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get metrics metadata with optional pre-aggregate optimization."""

        start, end = self._pre_aggregate_handler.get_start_end(start, end)
        start, end = self._pre_aggregate_handler.align_time_range(start, end, interval)

        table_schema = self.tables[mm_schemas.TimescaleDBTables.METRICS]
        filter_query = TimescaleDBQueryBuilder.build_endpoint_filter(endpoint_id)

        columns = [
            mm_schemas.WriterEvent.APPLICATION_NAME,
            mm_schemas.MetricData.METRIC_NAME,
            mm_schemas.WriterEvent.ENDPOINT_ID,
        ]

        query = table_schema._get_records_query(
            start=start,
            end=end,
            columns_to_filter=columns,
            filter_query=filter_query,
        )

        result = self._connection.run(query=query)
        df = TimescaleDBDataFrameProcessor.from_query_result(result)

        # Get distinct values
        if not df.empty:
            df = df.drop_duplicates()

        return df

    def calculate_latest_metrics(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        application_names: Optional[list[str]] = None,
    ) -> list[
        Union[mm_schemas.ApplicationResultRecord, mm_schemas.ApplicationMetricRecord]
    ]:
        """
        Calculate the latest metrics and results across applications.

        Returns a list of ApplicationResultRecord and ApplicationMetricRecord objects.

        :param start:              The start time of the query. Last 24 hours is used by default.
        :param end:                The end time of the query. The current time is used by default.
        :param application_names:  A list of application names to filter the results by. If not provided, all
                                applications are included.
        :return:                   A list containing the latest metrics and results for each application.
        """
        if not application_names:
            return []

        start, end = self._pre_aggregate_handler.get_start_end(start, end)

        metric_objects = []

        for app_name in application_names:
            # Get latest results for this application
            results_df = self._get_latest_results_for_application(app_name, start, end)
            if not results_df.empty:
                for _, row in results_df.iterrows():
                    metric_objects.append(
                        mm_schemas.ApplicationResultRecord(
                            time=row.get("end_infer_time"),
                            value=row.get("result_value"),
                            kind=mm_schemas.ResultKindApp(row.get("result_kind")),
                            status=mm_schemas.ResultStatusApp(row.get("result_status")),
                            result_name=row.get("result_name"),
                        )
                    )

            # Get latest metrics for this application
            metrics_df = self._get_latest_metrics_for_application(app_name, start, end)
            if not metrics_df.empty:
                for _, row in metrics_df.iterrows():
                    metric_objects.append(
                        mm_schemas.ApplicationMetricRecord(
                            time=row.get("end_infer_time"),
                            value=row.get("metric_value"),
                            metric_name=row.get("metric_name"),
                        )
                    )

        return metric_objects

    def _get_latest_metrics_for_application(
        self, application_name: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Get the latest metrics for a specific application."""
        table_schema = self.tables[mm_schemas.TimescaleDBTables.METRICS]

        # Build filters using query builder utilities
        app_filter = TimescaleDBQueryBuilder.build_application_filter(application_name)
        time_filter = TimescaleDBQueryBuilder.build_time_range_filter(
            start, end, mm_schemas.WriterEvent.END_INFER_TIME
        )
        where_clause = TimescaleDBQueryBuilder.combine_filters(
            [app_filter, time_filter], "AND"
        )

        # DISTINCT ON is PostgreSQL-specific, keep as specialized query
        query = f"""
        SELECT DISTINCT ON (metric_name)
            {mm_schemas.WriterEvent.END_INFER_TIME},
            {mm_schemas.WriterEvent.APPLICATION_NAME},
            {mm_schemas.MetricData.METRIC_NAME},
            {mm_schemas.MetricData.METRIC_VALUE}
        FROM {table_schema.full_name()}
        WHERE {where_clause}
        ORDER BY metric_name, {mm_schemas.WriterEvent.END_INFER_TIME} DESC
        """

        from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connection import (
            Statement,
        )

        stmt = Statement(query)
        result = self._connection.run(query=stmt)
        df = TimescaleDBDataFrameProcessor.from_query_result(result)
        return df

    def _get_latest_results_for_application(
        self, application_name: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Get the latest results for a specific application."""
        table_schema = self.tables[mm_schemas.TimescaleDBTables.APP_RESULTS]

        # Build filters using query builder utilities
        app_filter = TimescaleDBQueryBuilder.build_application_filter(application_name)
        time_filter = TimescaleDBQueryBuilder.build_time_range_filter(
            start, end, mm_schemas.WriterEvent.END_INFER_TIME
        )
        where_clause = TimescaleDBQueryBuilder.combine_filters(
            [app_filter, time_filter], "AND"
        )

        # DISTINCT ON is PostgreSQL-specific, keep as specialized query
        query = f"""
        SELECT DISTINCT ON (result_name)
            {mm_schemas.WriterEvent.END_INFER_TIME},
            {mm_schemas.WriterEvent.APPLICATION_NAME},
            {mm_schemas.ResultData.RESULT_NAME},
            {mm_schemas.ResultData.RESULT_VALUE},
            {mm_schemas.ResultData.RESULT_STATUS},
            {mm_schemas.ResultData.RESULT_KIND}
        FROM {table_schema.full_name()}
        WHERE {where_clause}
        ORDER BY result_name, {mm_schemas.WriterEvent.END_INFER_TIME} DESC
        """

        from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connection import (
            Statement,
        )

        stmt = Statement(query)
        result = self._connection.run(query=stmt)
        df = TimescaleDBDataFrameProcessor.from_query_result(result)

        return df

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
        :param project:                The name of the project.
        :param run_in_threadpool:      A function that runs another function in a thread pool.
        :param metric_list:            List of metrics to include from the time series DB. Defaults to all metrics.

        :return: A list of `ModelEndpointMonitoringMetric` objects.
        """

        uids = [mep.metadata.uid for mep in model_endpoint_objects]

        # Import other query mixins for cross-domain operations
        # This will be resolved when we have the full inheritance hierarchy
        metric_name_to_function = {
            "error_count": getattr(
                self, "get_error_count", lambda endpoint_ids: pd.DataFrame()
            ),
            "last_request": getattr(
                self, "get_last_request", lambda endpoint_ids: pd.DataFrame()
            ),
            "avg_latency": getattr(
                self, "get_avg_latency", lambda endpoint_ids: pd.DataFrame()
            ),
            "result_status": getattr(
                self, "get_drift_status", lambda endpoint_ids: pd.DataFrame()
            ),
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
