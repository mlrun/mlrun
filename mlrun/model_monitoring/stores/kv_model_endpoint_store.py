# Copyright 2023 Iguazio
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
#

import json
import os
import typing

import v3io.dataplane
import v3io_frames

import mlrun.common.model_monitoring.helpers
import mlrun.common.schemas.model_monitoring
import mlrun.utils.v3io_clients
from mlrun.utils import logger

from .model_endpoint_store import ModelEndpointStore

# Fields to encode before storing in the KV table or to decode after retrieving
fields_to_encode_decode = [
    mlrun.common.schemas.model_monitoring.EventFieldType.FEATURE_STATS,
    mlrun.common.schemas.model_monitoring.EventFieldType.CURRENT_STATS,
]


class KVModelEndpointStore(ModelEndpointStore):
    """
    Handles the DB operations when the DB target is from type KV. For the KV operations, we use an instance of V3IO
    client and usually the KV table can be found under v3io:///users/pipelines/project-name/model-endpoints/endpoints/.
    """

    def __init__(self, project: str, access_key: str):
        super().__init__(project=project)
        # Initialize a V3IO client instance
        self.access_key = access_key or os.environ.get("V3IO_ACCESS_KEY")
        self.client = mlrun.utils.v3io_clients.get_v3io_client(
            endpoint=mlrun.mlconf.v3io_api, access_key=self.access_key
        )
        # Get the KV table path and container
        self.path, self.container = self._get_path_and_container()

    def write_model_endpoint(self, endpoint: dict[str, typing.Any]):
        """
        Create a new endpoint record in the KV table.

        :param endpoint: model endpoint dictionary that will be written into the DB.
        """

        for field in fields_to_encode_decode:
            if field in endpoint:
                # Encode to binary data
                endpoint[field] = self._encode_field(endpoint[field])

        self.client.kv.put(
            container=self.container,
            table_path=self.path,
            key=endpoint[mlrun.common.schemas.model_monitoring.EventFieldType.UID],
            attributes=endpoint,
        )

        self._infer_kv_schema()

    def update_model_endpoint(
        self, endpoint_id: str, attributes: dict[str, typing.Any]
    ):
        """
        Update a model endpoint record with a given attributes.

        :param endpoint_id: The unique id of the model endpoint.
        :param attributes: Dictionary of attributes that will be used for update the model endpoint. Note that the keys
                           of the attributes dictionary should exist in the KV table.

        """

        for field in fields_to_encode_decode:
            if field in attributes:
                # Encode to binary data
                attributes[field] = self._encode_field(attributes[field])

        self.client.kv.update(
            container=self.container,
            table_path=self.path,
            key=endpoint_id,
            attributes=attributes,
        )

    def delete_model_endpoint(
        self,
        endpoint_id: str,
    ):
        """
        Deletes the KV record of a given model endpoint id.

        :param endpoint_id: The unique id of the model endpoint.
        """

        self.client.kv.delete(
            container=self.container,
            table_path=self.path,
            key=endpoint_id,
        )

    def get_model_endpoint(
        self,
        endpoint_id: str,
    ) -> dict[str, typing.Any]:
        """
        Get a single model endpoint record.

        :param endpoint_id: The unique id of the model endpoint.

        :return: A model endpoint record as a dictionary.

        :raise MLRunNotFoundError: If the endpoint was not found.
        """

        # Getting the raw data from the KV table
        endpoint = self.client.kv.get(
            container=self.container,
            table_path=self.path,
            key=endpoint_id,
            raise_for_status=v3io.dataplane.RaiseForStatus.never,
            access_key=self.access_key,
        )
        endpoint = endpoint.output.item

        for field in fields_to_encode_decode:
            if field in endpoint:
                # Decode binary data
                endpoint[field] = self._decode_field(endpoint[field])

        if not endpoint:
            raise mlrun.errors.MLRunNotFoundError(f"Endpoint {endpoint_id} not found")

        # For backwards compatability: replace null values for `error_count` and `metrics`
        self.validate_old_schema_fields(endpoint=endpoint)

        return endpoint

    def _get_path_and_container(self):
        """Getting path and container based on the model monitoring configurations"""
        path = mlrun.mlconf.model_endpoint_monitoring.store_prefixes.default.format(
            project=self.project,
            kind=mlrun.common.schemas.ModelMonitoringStoreKinds.ENDPOINTS,
        )
        (
            _,
            container,
            path,
        ) = mlrun.common.model_monitoring.helpers.parse_model_endpoint_store_prefix(
            path
        )
        return path, container

    def list_model_endpoints(
        self,
        model: str = None,
        function: str = None,
        labels: list[str] = None,
        top_level: bool = None,
        uids: list = None,
    ) -> list[dict[str, typing.Any]]:
        """
        Returns a list of model endpoint dictionaries, supports filtering by model, function, labels or top level.
        By default, when no filters are applied, all available model endpoints for the given project will
        be listed.

        :param model:           The name of the model to filter by.
        :param function:        The name of the function to filter by.
        :param labels:          A list of labels to filter by. Label filters work by either filtering a specific value
                                of a label (i.e. list("key=value")) or by looking for the existence of a given
                                key (i.e. "key").
        :param top_level:       If True will return only routers and endpoint that are NOT children of any router.
        :param uids:            List of model endpoint unique ids to include in the result.


        :return: A list of model endpoint dictionaries.
        """

        # # Initialize an empty model endpoints list
        endpoint_list = []

        # Retrieve the raw data from the KV table and get the endpoint ids
        try:
            cursor = self.client.kv.new_cursor(
                container=self.container,
                table_path=self.path,
                filter_expression=self._build_kv_cursor_filter_expression(
                    self.project,
                    function,
                    model,
                    labels,
                    top_level,
                ),
                raise_for_status=v3io.dataplane.RaiseForStatus.never,
            )
            items = cursor.all()

        except Exception as exc:
            logger.warning(
                "Failed retrieving raw data from kv table",
                exc=mlrun.errors.err_to_str(exc),
            )
            return endpoint_list

        # Create a list of model endpoints unique ids
        if uids is None:
            uids = []
            for item in items:
                if mlrun.common.schemas.model_monitoring.EventFieldType.UID not in item:
                    # This is kept for backwards compatibility - in old versions the key column named endpoint_id
                    uids.append(
                        item[
                            mlrun.common.schemas.model_monitoring.EventFieldType.ENDPOINT_ID
                        ]
                    )
                else:
                    uids.append(
                        item[mlrun.common.schemas.model_monitoring.EventFieldType.UID]
                    )

        # Add each relevant model endpoint to the model endpoints list
        for endpoint_id in uids:
            endpoint = self.get_model_endpoint(
                endpoint_id=endpoint_id,
            )
            endpoint_list.append(endpoint)

        return endpoint_list

    def delete_model_endpoints_resources(self, endpoints: list[dict[str, typing.Any]]):
        """
        Delete all model endpoints resources in both KV and the time series DB.

        :param endpoints: A list of model endpoints flattened dictionaries.
        """

        # Delete model endpoint record from KV table
        for endpoint_dict in endpoints:
            if (
                mlrun.common.schemas.model_monitoring.EventFieldType.UID
                not in endpoint_dict
            ):
                # This is kept for backwards compatibility - in old versions the key column named endpoint_id
                endpoint_id = endpoint_dict[
                    mlrun.common.schemas.model_monitoring.EventFieldType.ENDPOINT_ID
                ]
            else:
                endpoint_id = endpoint_dict[
                    mlrun.common.schemas.model_monitoring.EventFieldType.UID
                ]
            self.delete_model_endpoint(
                endpoint_id,
            )

        # Delete remain records in the KV
        all_records = self.client.kv.new_cursor(
            container=self.container,
            table_path=self.path,
            raise_for_status=v3io.dataplane.RaiseForStatus.never,
        ).all()

        all_records = [r["__name"] for r in all_records]

        # Cleanup KV
        for record in all_records:
            self.client.kv.delete(
                container=self.container,
                table_path=self.path,
                key=record,
                raise_for_status=v3io.dataplane.RaiseForStatus.never,
            )

        # Cleanup TSDB
        frames = self._get_frames_client()

        # Generate the required tsdb paths
        tsdb_path, filtered_path = self._generate_tsdb_paths()

        # Delete time series DB resources
        try:
            frames.delete(
                backend=mlrun.common.schemas.model_monitoring.TimeSeriesTarget.TSDB,
                table=filtered_path,
            )
        except v3io_frames.errors.DeleteError as e:
            if "No TSDB schema file found" not in str(e):
                logger.warning(
                    f"Failed to delete TSDB table '{filtered_path}'",
                    err=mlrun.errors.err_to_str(e),
                )
        # Final cleanup of tsdb path
        tsdb_path.replace("://u", ":///u")
        store, _, _ = mlrun.store_manager.get_or_create_store(tsdb_path)
        store.rm(tsdb_path, recursive=True)

    def get_endpoint_real_time_metrics(
        self,
        endpoint_id: str,
        metrics: list[str],
        start: str = "now-1h",
        end: str = "now",
        access_key: str = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Getting metrics from the time series DB. There are pre-defined metrics for model endpoints such as
        `predictions_per_second` and `latency_avg_5m` but also custom metrics defined by the user.

        :param endpoint_id:      The unique id of the model endpoint.
        :param metrics:          A list of real-time metrics to return for the model endpoint.
        :param start:            The start time of the metrics. Can be represented by a string containing an RFC 3339
                                 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days), or 0 for the
                                 earliest time.
        :param end:              The end time of the metrics. Can be represented by a string containing an RFC 3339
                                 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days), or 0 for the
                                 earliest time.
        :param access_key:       V3IO access key that will be used for generating Frames client object. If not
                                 provided, the access key will be retrieved from the environment variables.

        :return: A dictionary of metrics in which the key is a metric name and the value is a list of tuples that
                 includes timestamps and the values.
        """

        # Initialize access key
        access_key = access_key or mlrun.mlconf.get_v3io_access_key()

        if not metrics:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Metric names must be provided"
            )

        # Initialize metrics mapping dictionary
        metrics_mapping = {}

        # Getting the path for the time series DB
        events_path = (
            mlrun.mlconf.model_endpoint_monitoring.store_prefixes.default.format(
                project=self.project,
                kind=mlrun.common.schemas.ModelMonitoringStoreKinds.EVENTS,
            )
        )
        (
            _,
            container,
            events_path,
        ) = mlrun.common.model_monitoring.helpers.parse_model_endpoint_store_prefix(
            events_path
        )

        # Retrieve the raw data from the time series DB based on the provided metrics and time ranges
        frames_client = mlrun.utils.v3io_clients.get_frames_client(
            token=access_key,
            address=mlrun.mlconf.v3io_framesd,
            container=container,
        )

        try:
            data = frames_client.read(
                backend=mlrun.common.schemas.model_monitoring.TimeSeriesTarget.TSDB,
                table=events_path,
                columns=["endpoint_id", *metrics],
                filter=f"endpoint_id=='{endpoint_id}'",
                start=start,
                end=end,
            )

            # Fill the metrics mapping dictionary with the metric name and values
            data_dict = data.to_dict()
            for metric in metrics:
                metric_data = data_dict.get(metric)
                if metric_data is None:
                    continue

                values = [
                    (str(timestamp), value) for timestamp, value in metric_data.items()
                ]
                metrics_mapping[metric] = values

        except v3io_frames.errors.ReadError:
            logger.warn("Failed to read tsdb", endpoint=endpoint_id)

        return metrics_mapping

    def _generate_tsdb_paths(self) -> tuple[str, str]:
        """Generate a short path to the TSDB resources and a filtered path for the frames object
        :return: A tuple of:
             [0] = Short path to the TSDB resources
             [1] = Filtered path to TSDB events without schema and container
        """
        # Full path for the time series DB events
        full_path = (
            mlrun.mlconf.model_endpoint_monitoring.store_prefixes.default.format(
                project=self.project,
                kind=mlrun.common.schemas.ModelMonitoringStoreKinds.EVENTS,
            )
        )

        # Generate the main directory with the TSDB resources
        tsdb_path = (
            mlrun.common.model_monitoring.helpers.parse_model_endpoint_project_prefix(
                full_path, self.project
            )
        )

        # Generate filtered path without schema and container as required by the frames object
        (
            _,
            _,
            filtered_path,
        ) = mlrun.common.model_monitoring.helpers.parse_model_endpoint_store_prefix(
            full_path
        )
        return tsdb_path, filtered_path

    def _infer_kv_schema(self):
        """
        Create KV schema file if not exist. This schema is being used by the Grafana dashboards.
        """

        schema_file = self.client.kv.new_cursor(
            container=self.container,
            table_path=self.path,
            filter_expression='__name==".#schema"',
        )

        if not schema_file.all():
            logger.info("Generate a new V3IO KV schema file", kv_table_path=self.path)
            frames_client = self._get_frames_client()
            frames_client.execute(backend="kv", table=self.path, command="infer_schema")

    def _get_frames_client(self):
        return mlrun.utils.v3io_clients.get_frames_client(
            token=self.access_key,
            address=mlrun.mlconf.v3io_framesd,
            container=self.container,
        )

    @staticmethod
    def _build_kv_cursor_filter_expression(
        project: str,
        function: str = None,
        model: str = None,
        labels: list[str] = None,
        top_level: bool = False,
    ) -> str:
        """
        Convert the provided filters into a valid filter expression. The expected filter expression includes different
        conditions, divided by ' AND '.

        :param project:    The name of the project.
        :param model:      The name of the model to filter by.
        :param function:   The name of the function to filter by.
        :param labels:     A list of labels to filter by. Label filters work by either filtering a specific value of
                           a label (i.e. list("key=value")) or by looking for the existence of a given
                           key (i.e. "key").
        :param top_level:  If True will return only routers and endpoint that are NOT children of any router.

        :return: A valid filter expression as a string.

        :raise MLRunInvalidArgumentError: If project value is None.
        """

        if not project:
            raise mlrun.errors.MLRunInvalidArgumentError("project can't be empty")

        # Add project filter
        filter_expression = [f"project=='{project}'"]

        # Add function and model filters
        if function:
            filter_expression.append(f"function=='{function}'")
        if model:
            filter_expression.append(f"model=='{model}'")

        # Add labels filters
        if labels:
            for label in labels:
                if not label.startswith("_"):
                    label = f"_{label}"

                if "=" in label:
                    lbl, value = list(map(lambda x: x.strip(), label.split("=")))
                    filter_expression.append(f"{lbl}=='{value}'")
                else:
                    filter_expression.append(f"exists({label})")

        # Apply top_level filter (remove endpoints that considered a child of a router)
        if top_level:
            filter_expression.append(
                f"(endpoint_type=='{str(mlrun.common.schemas.model_monitoring.EndpointType.NODE_EP.value)}' "
                f"OR  endpoint_type=='{str(mlrun.common.schemas.model_monitoring.EndpointType.ROUTER.value)}')"
            )

        return " AND ".join(filter_expression)

    @staticmethod
    def validate_old_schema_fields(endpoint: dict):
        """
        Replace default null values for `error_count` and `metrics` for users that logged a model endpoint before 1.3.0.
        In addition, this function also validates that the key name of the endpoint unique id is `uid` and not
        `endpoint_id` that has been used before 1.3.0.

        Leaving here for backwards compatibility which related to the model endpoint schema.

        :param endpoint: An endpoint flattened dictionary.
        """

        # Validate default value for `error_count`
        # For backwards compatibility reasons, we validate that the model endpoint includes the `error_count` key
        if (
            mlrun.common.schemas.model_monitoring.EventFieldType.ERROR_COUNT in endpoint
            and endpoint[
                mlrun.common.schemas.model_monitoring.EventFieldType.ERROR_COUNT
            ]
            == "null"
        ):
            endpoint[
                mlrun.common.schemas.model_monitoring.EventFieldType.ERROR_COUNT
            ] = "0"

        # Validate default value for `metrics`
        # For backwards compatibility reasons, we validate that the model endpoint includes the `metrics` key
        if (
            mlrun.common.schemas.model_monitoring.EventFieldType.METRICS in endpoint
            and endpoint[mlrun.common.schemas.model_monitoring.EventFieldType.METRICS]
            == "null"
        ):
            endpoint[mlrun.common.schemas.model_monitoring.EventFieldType.METRICS] = (
                json.dumps(
                    {
                        mlrun.common.schemas.model_monitoring.EventKeyMetrics.GENERIC: {
                            mlrun.common.schemas.model_monitoring.EventLiveStats.LATENCY_AVG_1H: 0,
                            mlrun.common.schemas.model_monitoring.EventLiveStats.PREDICTIONS_PER_SECOND: 0,
                        }
                    }
                )
            )
        # Validate key `uid` instead of `endpoint_id`
        # For backwards compatibility reasons, we replace the `endpoint_id` with `uid` which is the updated key name
        if mlrun.common.schemas.model_monitoring.EventFieldType.ENDPOINT_ID in endpoint:
            endpoint[mlrun.common.schemas.model_monitoring.EventFieldType.UID] = (
                endpoint[
                    mlrun.common.schemas.model_monitoring.EventFieldType.ENDPOINT_ID
                ]
            )

    @staticmethod
    def _encode_field(field: typing.Union[str, bytes]) -> bytes:
        """Encode a provided field. Mainly used when storing data in the KV table."""

        if isinstance(field, str):
            return field.encode("ascii")
        return field

    @staticmethod
    def _decode_field(field: typing.Union[str, bytes]) -> str:
        """Decode a provided field. Mainly used when retrieving data from the KV table."""

        if isinstance(field, bytes):
            return field.decode()
        return field
