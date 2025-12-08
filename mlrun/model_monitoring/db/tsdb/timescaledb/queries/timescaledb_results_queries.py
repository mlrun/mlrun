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

from datetime import datetime, timedelta
from typing import Optional, Union

import pandas as pd
import v3io_frames.client

import mlrun
import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_schema as timescaledb_schema
import mlrun.utils
from mlrun.model_monitoring.db.tsdb.timescaledb.utils.timescaledb_dataframe_processor import (
    TimescaleDBDataFrameProcessor,
)
from mlrun.model_monitoring.db.tsdb.timescaledb.utils.timescaledb_query_builder import (
    TimescaleDBQueryBuilder,
)


class TimescaleDBResultsQueries:
    """
    Query class containing results and drift-related query methods for TimescaleDB.

    Can be used as a mixin or standalone instance with proper initialization.
    """

    def __init__(
        self,
        connection,  # Required parameter
        project: Optional[str] = None,
        pre_aggregate_manager=None,
        tables: Optional[dict] = None,
    ):
        """
        Initialize TimescaleDB results query handler.

        :param connection: TimescaleDB connection instance (required)
        :param project: Project name
        :param pre_aggregate_manager: PreAggregateManager instance
        :param tables: Dictionary of table schemas
        """
        self.project = project
        self._connection = connection
        self._pre_aggregate_manager = pre_aggregate_manager
        self.tables = tables

    def get_drift_status(
        self,
        endpoint_ids: Union[str, list[str]],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        get_raw: bool = False,
    ) -> Union[pd.DataFrame, list[v3io_frames.client.RawFrame]]:
        """Get drift status for specified endpoints.

        :param endpoint_ids: Endpoint ID(s) to get drift status for
        :param start: Start datetime for filtering
        :param end: End datetime for filtering
        :param get_raw: If True, return raw frame data (not implemented)
        :return: DataFrame with drift status data
        """
        del get_raw  # Suppress unused variable warning (not implemented)

        if isinstance(endpoint_ids, str):
            endpoint_ids = [endpoint_ids]

        # Set default start time
        start = start or (mlrun.utils.datetime_now() - timedelta(hours=24))
        start, end = self._pre_aggregate_manager.get_start_end(start, end)

        table_schema = self.tables[mm_schemas.TimescaleDBTables.APP_RESULTS]
        endpoint_filter = TimescaleDBQueryBuilder.build_endpoint_filter(endpoint_ids)

        columns = [
            f"{mm_schemas.WriterEvent.ENDPOINT_ID} AS {mm_schemas.WriterEvent.ENDPOINT_ID}",
            f"MAX({mm_schemas.ResultData.RESULT_STATUS}) as {mm_schemas.ResultData.RESULT_STATUS}",
        ]
        group_by_columns = [mm_schemas.WriterEvent.ENDPOINT_ID]

        # Build filter using query builder utilities
        filters = [
            endpoint_filter,
            f"{mm_schemas.ResultData.RESULT_STATUS} IS NOT NULL",
        ]
        filter_query = TimescaleDBQueryBuilder.combine_filters(filters)

        query = table_schema._get_records_query(
            start=start,
            end=end,
            columns_to_filter=columns,
            filter_query=filter_query,
            group_by=group_by_columns,
            order_by=mm_schemas.WriterEvent.ENDPOINT_ID,
        )

        return self._connection.run_query_to_df(query)

    def get_error_count(
        self,
        endpoint_ids: Union[str, list[str]],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Get error count for specified endpoints."""

        if isinstance(endpoint_ids, str):
            endpoint_ids = [endpoint_ids]

        # Set default start time
        start = start or (mlrun.utils.datetime_now() - timedelta(hours=24))
        start, end = self._pre_aggregate_manager.get_start_end(start, end)

        table_schema = self.tables[mm_schemas.TimescaleDBTables.ERRORS]
        endpoint_filter = TimescaleDBQueryBuilder.build_endpoint_filter(endpoint_ids)

        # Build filter using query builder utilities
        filters = [
            endpoint_filter,
            f"{mm_schemas.EventFieldType.ERROR_TYPE} = '{mm_schemas.EventFieldType.INFER_ERROR}'",
        ]
        filter_query = TimescaleDBQueryBuilder.combine_filters(filters)

        columns = [
            f"{mm_schemas.WriterEvent.ENDPOINT_ID} AS {mm_schemas.WriterEvent.ENDPOINT_ID}",
            f"COUNT(*) AS {mm_schemas.EventFieldType.ERROR_COUNT}",
        ]
        group_by_columns = [mm_schemas.WriterEvent.ENDPOINT_ID]

        query = table_schema._get_records_query(
            start=start,
            end=end,
            columns_to_filter=columns,
            filter_query=filter_query,
            group_by=group_by_columns,
            order_by=mm_schemas.WriterEvent.ENDPOINT_ID,
        )

        return self._connection.run_query_to_df(query)

    def get_results_metadata(
        self,
        endpoint_id: Union[str, list[str]],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        interval: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get results metadata with optional pre-aggregate optimization."""

        start, end = self._pre_aggregate_manager.get_start_end(start, end)
        start, end = self._pre_aggregate_manager.align_time_range(start, end, interval)

        table_schema = self.tables[mm_schemas.TimescaleDBTables.APP_RESULTS]
        filter_query = TimescaleDBQueryBuilder.build_endpoint_filter(endpoint_id)

        columns = [
            mm_schemas.WriterEvent.APPLICATION_NAME,
            mm_schemas.ResultData.RESULT_NAME,
            mm_schemas.ResultData.RESULT_KIND,
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

    def count_results_by_status(
        self,
        start: Optional[Union[datetime, str]] = None,
        end: Optional[Union[datetime, str]] = None,
        endpoint_ids: Optional[Union[str, list[str]]] = None,
        application_names: Optional[Union[str, list[str]]] = None,
        result_status_list: Optional[list[int]] = None,
    ) -> dict[tuple[str, int], int]:
        """
        Read results status from the TSDB and return a dictionary of results statuses by application name.

        :param start:              The start time in which to read the results. By default, the last 24 hours are read.
        :param end:                The end time in which to read the results. Default is the current time (now).
        :param endpoint_ids:       Optional list of endpoint ids to filter the results by. By default, all
                                endpoint ids are included.
        :param application_names:  Optional list of application names to filter the results by. By default, all
                                application are included.
        :param result_status_list: Optional list of result statuses to filter the results by. By default, all
                                result statuses are included.

        :return: A dictionary where the key is a tuple of (application_name, result_status) and the value is the total
                number of results with that status for that application.
                For example:
                {
                    ('app1', 1): 10,
                    ('app1', 2): 5
                }
        """
        # Set defaults
        now = mlrun.utils.datetime_now()
        start = start or (now - timedelta(hours=24))
        end = end or now

        # Convert string dates to datetime if needed
        if isinstance(start, str):
            start = datetime.fromisoformat(start)
        if isinstance(end, str):
            end = datetime.fromisoformat(end)

        table_schema = self.tables[mm_schemas.TimescaleDBTables.APP_RESULTS]

        # Build filter conditions
        filter_conditions = []

        if endpoint_ids:
            endpoint_filter = TimescaleDBQueryBuilder.build_endpoint_filter(
                endpoint_ids
            )
            filter_conditions.append(endpoint_filter)

        if application_names:
            app_filter = TimescaleDBQueryBuilder.build_application_filter(
                application_names
            )
            filter_conditions.append(app_filter)

        if result_status_list:
            if len(result_status_list) == 1:
                status_filter = (
                    f"{mm_schemas.ResultData.RESULT_STATUS} = {result_status_list[0]}"
                )
            else:
                status_list = ", ".join(map(str, result_status_list))
                status_filter = (
                    f"{mm_schemas.ResultData.RESULT_STATUS} IN ({status_list})"
                )
            filter_conditions.append(status_filter)

        filter_query = TimescaleDBQueryBuilder.combine_filters(filter_conditions)

        # Build the aggregation query using the enhanced _get_records_query
        columns = [
            mm_schemas.WriterEvent.APPLICATION_NAME,
            mm_schemas.ResultData.RESULT_STATUS,
            "COUNT(*) AS count",
        ]

        group_by_columns = [
            mm_schemas.WriterEvent.APPLICATION_NAME,
            mm_schemas.ResultData.RESULT_STATUS,
        ]

        order_by_clause = f"{mm_schemas.WriterEvent.APPLICATION_NAME}, {mm_schemas.ResultData.RESULT_STATUS}"

        query = table_schema._get_records_query(
            start=start,
            end=end,
            columns_to_filter=columns,
            filter_query=filter_query,
            group_by=group_by_columns,
            order_by=order_by_clause,
        )

        result = self._connection.run(query=query)

        if not result or not result.data:
            return {}

        return {(row[0].lower(), row[1]): row[2] for row in result.data}

    def get_drift_data(
        self,
        start: datetime,
        end: datetime,
        interval: Optional[str] = None,
    ) -> mm_schemas.ModelEndpointDriftValues:
        """
        Get drift data aggregated by time intervals, showing the count of suspected and detected drift events.

        This method queries the app_results table for drift-related statuses (potential_detection=1, detected=2)
        and aggregates them by time intervals, counting the maximum drift status per endpoint per interval.
        Uses pre-aggregate optimization when available.

        :param start: Start time for the query
        :param end: End time for the query
        :param interval: Optional time interval for aggregation (e.g., "1 hour", "30 minutes").
                        If not provided, will be automatically determined based on query duration.
        :return: ModelEndpointDriftValues containing time-binned drift counts
        """
        # Prepare time range and interval using helper
        start, end, interval = TimescaleDBQueryBuilder.prepare_time_range_and_interval(
            self._pre_aggregate_manager, start, end, interval
        )

        # Build status filter for drift-related statuses only
        suspected_status = mm_schemas.ResultStatusApp.potential_detection.value  # 1
        detected_status = mm_schemas.ResultStatusApp.detected.value  # 2

        app_results_table = self.tables[mm_schemas.TimescaleDBTables.APP_RESULTS]

        def build_raw_query():
            # Use TimescaleDB's time_bucket function for interval aggregation
            return f"""
            WITH drift_intervals AS (
                SELECT
                    time_bucket('{interval}', {mm_schemas.WriterEvent.END_INFER_TIME}) AS bucket_start,
                    {mm_schemas.WriterEvent.ENDPOINT_ID},
                    MAX({mm_schemas.ResultData.RESULT_STATUS}) AS max_status
                FROM {app_results_table.full_name()}
                WHERE {mm_schemas.ResultData.RESULT_STATUS} IN (%s, %s)
                AND {mm_schemas.WriterEvent.END_INFER_TIME} >= %s
                AND {mm_schemas.WriterEvent.END_INFER_TIME} <= %s
                GROUP BY bucket_start, {mm_schemas.WriterEvent.ENDPOINT_ID}
            )
            SELECT
                bucket_start,
                max_status,
                COUNT(*) AS status_count
            FROM drift_intervals
            GROUP BY bucket_start, max_status
            ORDER BY bucket_start, max_status
            """

        raw_query = build_raw_query()
        from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connection import (
            Statement,
        )

        stmt = Statement(raw_query, (suspected_status, detected_status, start, end))
        result = self._connection.run(query=stmt)

        if not result or not result.data:
            return mm_schemas.ModelEndpointDriftValues(values=[])

        # Convert to DataFrame for easier processing
        df = TimescaleDBDataFrameProcessor.from_query_result(result)

        # Use the shared _df_to_drift_data method to convert to the expected format
        return self._df_to_drift_data(df)

    def _df_to_drift_data(
        self, df: pd.DataFrame
    ) -> mm_schemas.ModelEndpointDriftValues:
        """
        Convert DataFrame with drift data to ModelEndpointDriftValues format.

        Expected DataFrame columns:
        - bucket_start: timestamp of the interval bucket
        - max_status: the maximum drift status in that bucket (1=suspected, 2=detected)
        - status_count: count of endpoints with that status in the bucket

        :param df: DataFrame with aggregated drift data
        :return: ModelEndpointDriftValues with time-binned counts
        """
        if df.empty:
            return mm_schemas.ModelEndpointDriftValues(values=[])

        suspected_val = mm_schemas.ResultStatusApp.potential_detection.value  # 1
        detected_val = mm_schemas.ResultStatusApp.detected.value  # 2

        # Rename columns to match expected format
        df = df.rename(
            columns={
                "bucket_start": "_wstart",
                "max_status": f"max({mm_schemas.ResultData.RESULT_STATUS})",
                "status_count": "count",
            }
        )

        # Pivot the data to have separate columns for suspected and detected counts
        aggregated_df = (
            df.groupby(["_wstart", f"max({mm_schemas.ResultData.RESULT_STATUS})"])[
                "count"
            ]
            .sum()  # Sum counts for each interval x result-status combination
            .unstack()  # Create separate columns for each result-status
            .reindex(
                columns=[suspected_val, detected_val], fill_value=0
            )  # Ensure both columns exist
            .fillna(0)
            .astype(int)
            .rename(
                columns={
                    suspected_val: "count_suspected",
                    detected_val: "count_detected",
                }
            )
        )

        # Convert to list of tuples: (timestamp, count_suspected, count_detected)
        values = list(
            zip(
                aggregated_df.index,
                aggregated_df["count_suspected"],
                aggregated_df["count_detected"],
            )
        )

        return mm_schemas.ModelEndpointDriftValues(values=values)

    def read_results_data_impl(
        self,
        *,
        endpoint_id: Optional[str] = None,
        start: datetime,
        end: datetime,
        metrics: Optional[list[mm_schemas.ModelEndpointMonitoringMetric]] = None,
        with_result_extra_data: bool = False,
        timestamp_column: Optional[str] = None,
        agg_period: Optional[str] = None,
        agg_functions: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Read results data from TimescaleDB (app_results table only) - returns DataFrame.

        :param endpoint_id: Endpoint ID to filter by, or None to get all endpoints
        :param start: Start time
        :param end: End time
        :param metrics: List of metrics to filter by, or None to get all results
        :param with_result_extra_data: Whether to include extra data column
        :param timestamp_column: Optional timestamp column to use for time filtering
        :param agg_period: Optional aggregation period (e.g., '1h', '6h'). If provided,
                          reads from pre-aggregated continuous aggregates.
                          Note: extra_data is NOT included in aggregated results, but
                          status IS aggregated (e.g., max indicates detection in period).
        :param agg_functions: Optional list of aggregation functions to return
                             (e.g., ['avg', 'min', 'max']). Required when agg_period is provided.
        :return: DataFrame with results data
        """

        table_schema = self.tables[mm_schemas.TimescaleDBTables.APP_RESULTS]
        name_column = mm_schemas.ResultData.RESULT_NAME
        value_column = mm_schemas.ResultData.RESULT_VALUE
        columns = [
            table_schema.time_column,
            mm_schemas.WriterEvent.START_INFER_TIME,
            mm_schemas.WriterEvent.ENDPOINT_ID,
            mm_schemas.WriterEvent.APPLICATION_NAME,
            name_column,
            value_column,
            mm_schemas.ResultData.RESULT_STATUS,
            mm_schemas.ResultData.RESULT_KIND,
        ]
        if with_result_extra_data:
            columns.append(mm_schemas.ResultData.RESULT_EXTRA_DATA)

        metrics_condition = TimescaleDBQueryBuilder.build_results_filter(metrics)
        endpoint_filter = TimescaleDBQueryBuilder.build_endpoint_filter(endpoint_id)

        # Combine filters using query builder utilities
        filters = [endpoint_filter, metrics_condition]
        filter_query = TimescaleDBQueryBuilder.combine_filters(filters)

        # Two query paths based on agg_period:
        # 1. agg_period with functions (not "raw") → aggregation query from CAGG view (v2 API)
        # 2. agg_period=None or "raw" → raw query returning individual data points (v1 and v2 raw)
        # Note: Aggregated results include status aggregations but NOT extra_data
        use_aggregation = agg_period and agg_period != "raw" and agg_functions

        if use_aggregation:
            df = TimescaleDBQueryBuilder.build_read_data_with_aggregation(
                connection=self._connection,
                table_schema=table_schema,
                start=start,
                end=end,
                columns=columns,
                filter_query=filter_query,
                name_column=name_column,
                value_column=value_column,
                agg_period=agg_period,
                agg_functions=agg_functions,
                timestamp_column=timestamp_column,
                # Include status in aggregation (max indicates detection in period)
                additional_agg_columns=[mm_schemas.ResultData.RESULT_STATUS],
            )
        else:
            # Raw query for both v1 (agg_period=None) and v2 raw (agg_period="raw")
            df = TimescaleDBQueryBuilder.build_read_raw_data(
                connection=self._connection,
                table_schema=table_schema,
                start=start,
                end=end,
                columns=columns,
                filter_query=filter_query,
                timestamp_column=timestamp_column,
            )

        if not df.empty:
            # Use time_bucket for aggregated data, time_column for raw
            time_col = (
                timescaledb_schema.TIME_BUCKET_COLUMN
                if use_aggregation
                else table_schema.time_column
            )
            if time_col in df.columns:
                df[time_col] = pd.to_datetime(df[time_col])
                df.set_index(time_col, inplace=True)

        # For raw data without extra_data, add empty column
        if not use_aggregation and not with_result_extra_data:
            df[mm_schemas.ResultData.RESULT_EXTRA_DATA] = ""

        return df
