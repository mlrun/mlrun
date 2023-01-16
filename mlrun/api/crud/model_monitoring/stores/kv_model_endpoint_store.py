# Copyright 2018 Iguazio
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

import typing

import v3io.dataplane
import v3io_frames

import mlrun
import mlrun.api.schemas
import mlrun.model_monitoring.constants as model_monitoring_constants
import mlrun.utils.model_monitoring
import mlrun.utils.v3io_clients
from mlrun.utils import logger

from .model_endpoint_store import ModelEndpointStore


class KVModelEndpointStore(ModelEndpointStore):
    """
    Handles the DB operations when the DB target is from type KV. For the KV operations, we use an instance of V3IO
    client and usually the KV table can be found under v3io:///users/pipelines/project-name/model-endpoints/endpoints/.
    """

    def __init__(self, project: str, access_key: str):
        super().__init__(project=project)
        # Initialize a V3IO client instance
        self.access_key = access_key
        self.client = mlrun.utils.v3io_clients.get_v3io_client(
            endpoint=mlrun.mlconf.v3io_api, access_key=self.access_key
        )
        # Get the KV table path and container
        self.path, self.container = self._get_path_and_container()

    def write_model_endpoint(self, endpoint: mlrun.api.schemas.ModelEndpoint):
        """
        Create a new endpoint record in the KV table.

        :param endpoint: `ModelEndpoint` object that will be written into the DB.
        """

        # Retrieving the relevant attributes from the model endpoint object
        attributes = self.get_params(endpoint)
        # Create or update the model endpoint record
        self.client.kv.put(
            container=self.container,
            table_path=self.path,
            key=endpoint.metadata.uid,
            attributes=attributes,
        )

    def update_model_endpoint(self, endpoint_id: str, attributes: dict):
        """
        Update a model endpoint record with a given attributes.

        :param endpoint_id: The unique id of the model endpoint.
        :param attributes: Dictionary of attributes that will be used for update the model endpoint. Note that the keys
                           of the attributes dictionary should exist in the KV table.

        """

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
        start: str = "now-1h",
        end: str = "now",
        metrics: typing.List[str] = None,
        feature_analysis: bool = False,
        convert_to_endpoint_object: bool = True,
    ) -> typing.Union[mlrun.api.schemas.ModelEndpoint, dict]:
        """
        Get a single model endpoint object. You can apply different time series metrics that will be added to the
        result.

        :param endpoint_id:                The unique id of the model endpoint.
        :param start:                      The start time of the metrics. Can be represented by a string containing
                                           an RFC 3339 time, a Unix timestamp in milliseconds, a relative time (`'now'`
                                           or `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days),
                                           or 0 for the earliest time.
        :param end:                        The end time of the metrics. Can be represented by a string containing an
                                           RFC 3339 time, a Unix timestamp in milliseconds, a relative time (`'now'`
                                           or `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days),
                                           or 0 for the earliest time.
        :param metrics:                    A list of real-time metrics to return for the model endpoint. There are
                                           pre-defined real-time metrics for model endpoints such as
                                           predictions_per_second and latency_avg_5m but also custom metrics defined by
                                           the user. Please note that these metrics are stored in the time series DB
                                           and the results will appear under model_endpoint.spec.metrics.
        :param feature_analysis:           When True, the base feature statistics and current feature statistics will
                                           be added to the output of the resulting object.
        :param convert_to_endpoint_object: A boolean that indicates whether to convert the model endpoint dictionary
                                           into a `ModelEndpoint` or not. True by default.

        :return: A `ModelEndpoint` object or a model endpoint dictionary if `convert_to_endpoint_object` is False.

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

        if not endpoint:
            raise mlrun.errors.MLRunNotFoundError(f"Endpoint {endpoint_id} not found")

        # Generate a model endpoint object from the model endpoint KV record
        if convert_to_endpoint_object:
            endpoint = self._convert_into_model_endpoint_object(
                endpoint=endpoint, feature_analysis=feature_analysis
            )

            # If time metrics were provided, retrieve the results from the time series DB
            if metrics:
                if endpoint.status.metrics is None:
                    endpoint.status.metrics = {}
                endpoint_metrics = self.get_endpoint_real_time_metrics(
                    endpoint_id=endpoint_id,
                    start=start,
                    end=end,
                    metrics=metrics,
                    access_key=self.access_key,
                )
                if endpoint_metrics:
                    endpoint.status.metrics[
                        model_monitoring_constants.EventKeyMetrics.REAL_TIME
                    ] = endpoint_metrics

        return endpoint

    def _get_path_and_container(self):
        """Getting path and container based on the model monitoring configurations"""
        path = mlrun.mlconf.model_endpoint_monitoring.store_prefixes.default.format(
            project=self.project,
            kind=mlrun.api.schemas.ModelMonitoringStoreKinds.ENDPOINTS,
        )
        (
            _,
            container,
            path,
        ) = mlrun.utils.model_monitoring.parse_model_endpoint_store_prefix(path)
        return path, container

    def list_model_endpoints(
        self,
        model: str = None,
        function: str = None,
        labels: typing.Union[typing.List[str], typing.Dict] = None,
        top_level: bool = None,
        metrics: typing.List[str] = None,
        start: str = "now-1h",
        end: str = "now",
        uids: typing.List = None,
    ) -> mlrun.api.schemas.ModelEndpointList:
        """
        Returns a list of `ModelEndpoint` objects, supports filtering by model, function, labels or top level.
        By default, when no filters are applied, all available `ModelEndpoint` objects for the given project will
        be listed.

        :param model:           The name of the model to filter by.
        :param function:        The name of the function to filter by.
        :param labels:          A list of labels to filter by. Label filters work by either filtering a specific value
                                of a label (i.e. list("key==value")) or by looking for the existence of a given
                                key (i.e. "key").
        :param top_level:       If True will return only routers and endpoint that are NOT children of any router.
        :param metrics:         A list of real-time metrics to return for the model endpoint. There are pre-defined
                                real-time metrics for model endpoints such as predictions_per_second and latency_avg_5m
                                but also custom metrics defined by the user. Please note that these metrics are stored
                                in the time series DB and the results will be appeared under
                                model_endpoint.spec.metrics.
        :param start:           The start time of the metrics. Can be represented by a string containing an
                                RFC 3339 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days), or
                                0 for the earliest time.
        :param end:             The end time of the metrics. Can be represented by a string containing an
                                RFC 3339 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days),
                                or 0 for the earliest time.
        :param uids:            List of model endpoint unique ids to include in the result.


        :return: An object of `ModelEndpointList` which is literally a list of model endpoints along with some
                          metadata. To get a standard list of model endpoints use `ModelEndpointList.endpoints`.
        """

        # Initialize an empty model endpoints list
        endpoint_list = mlrun.api.schemas.model_endpoints.ModelEndpointList(
            endpoints=[]
        )

        # Labels from type list won't be supported from 1.4.0
        # TODO: Remove in 1.4.0
        if labels and isinstance(labels, typing.List):
            logger.warning(
                "Labels should be from type dictionary, not list,"
                "This is deprecated in 1.3.0, and will be removed in 1.4.0",
                FutureWarning,
                labels=labels,
            )

        if labels and isinstance(labels, dict):
            labels = [f"{key}={value}" for key, value in labels.items()]

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
                attribute_names=[model_monitoring_constants.EventFieldType.ENDPOINT_ID],
                raise_for_status=v3io.dataplane.RaiseForStatus.never,
            )
            items = cursor.all()
        except Exception as exc:
            logger.warning("Failed retrieving raw data from kv table", exc=exc)
            return endpoint_list

        # Create a list of model endpoints unique ids
        if uids is None:
            uids = [
                item[model_monitoring_constants.EventFieldType.ENDPOINT_ID]
                for item in items
            ]

        # Add each relevant model endpoint to the model endpoints list
        for endpoint_id in uids:
            endpoint = self.get_model_endpoint(
                metrics=metrics,
                endpoint_id=endpoint_id,
                start=start,
                end=end,
            )
            endpoint_list.endpoints.append(endpoint)

        return endpoint_list

    def delete_model_endpoints_resources(
        self, endpoints: mlrun.api.schemas.model_endpoints.ModelEndpointList
    ):
        """
        Delete all model endpoints resources in both KV and the time series DB.

        :param endpoints: An object of `ModelEndpointList` which is literally a list of model endpoints along with some
                          metadata. To get a standard list of model endpoints use `ModelEndpointList.endpoints`.
        """

        # Delete model endpoint record from KV table
        for endpoint in endpoints.endpoints:
            self.delete_model_endpoint(
                endpoint.metadata.uid,
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
        frames = mlrun.utils.v3io_clients.get_frames_client(
            token=self.access_key,
            address=mlrun.mlconf.v3io_framesd,
            container=self.container,
        )

        # Generate the required tsdb paths
        tsdb_path, filtered_path = self._generate_tsdb_paths()

        # Delete time series DB resources
        try:
            frames.delete(
                backend=model_monitoring_constants.TimeSeriesTarget.TSDB,
                table=filtered_path,
            )
        except (v3io_frames.errors.DeleteError, v3io_frames.errors.CreateError) as e:
            # Frames might raise an exception if schema file does not exist.
            logger.warning("Failed to delete TSDB schema file:", err=e)
            pass

        # Final cleanup of tsdb path
        tsdb_path.replace("://u", ":///u")
        store, _ = mlrun.store_manager.get_or_create_store(tsdb_path)
        store.rm(tsdb_path, recursive=True)

    def get_endpoint_real_time_metrics(
        self,
        endpoint_id: str,
        metrics: typing.List[str],
        start: str = "now-1h",
        end: str = "now",
        access_key: str = None,
    ) -> typing.Dict[str, typing.List]:
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
                kind=mlrun.api.schemas.ModelMonitoringStoreKinds.EVENTS,
            )
        )
        (
            _,
            container,
            events_path,
        ) = mlrun.utils.model_monitoring.parse_model_endpoint_store_prefix(events_path)

        # Retrieve the raw data from the time series DB based on the provided metrics and time ranges
        frames_client = mlrun.utils.v3io_clients.get_frames_client(
            token=access_key,
            address=mlrun.mlconf.v3io_framesd,
            container=container,
        )

        try:
            data = frames_client.read(
                backend=model_monitoring_constants.TimeSeriesTarget.TSDB,
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

    def _generate_tsdb_paths(self) -> typing.Tuple[str, str]:
        """Generate a short path to the TSDB resources and a filtered path for the frames object
        :return: A tuple of:
             [0] = Short path to the TSDB resources
             [1] = Filtered path to TSDB events without schema and container
        """
        # Full path for the time series DB events
        full_path = (
            mlrun.mlconf.model_endpoint_monitoring.store_prefixes.default.format(
                project=self.project,
                kind=mlrun.api.schemas.ModelMonitoringStoreKinds.EVENTS,
            )
        )

        # Generate the main directory with the TSDB resources
        tsdb_path = mlrun.utils.model_monitoring.parse_model_endpoint_project_prefix(
            full_path, self.project
        )

        # Generate filtered path without schema and container as required by the frames object
        (
            _,
            _,
            filtered_path,
        ) = mlrun.utils.model_monitoring.parse_model_endpoint_store_prefix(full_path)
        return tsdb_path, filtered_path

    @staticmethod
    def _build_kv_cursor_filter_expression(
        project: str,
        function: str = None,
        model: str = None,
        labels: typing.List[str] = None,
        top_level: bool = False,
    ) -> str:
        """
        Convert the provided filters into a valid filter expression. The expected filter expression includes different
        conditions, divided by ' AND '.

        :param project:    The name of the project.
        :param model:      The name of the model to filter by.
        :param function:   The name of the function to filter by.
        :param labels:     A list of labels to filter by. Label filters work by either filtering a specific value of
                           a label (i.e. list("key==value")) or by looking for the existence of a given
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
                f"(endpoint_type=='{str(mlrun.model_monitoring.EndpointType.NODE_EP.value)}' "
                f"OR  endpoint_type=='{str(mlrun.model_monitoring.EndpointType.ROUTER.value)}')"
            )

        return " AND ".join(filter_expression)
