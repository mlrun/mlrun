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

from .model_endpoint_store import _ModelEndpointStore


class _ModelEndpointKVStore(_ModelEndpointStore):
    """
    Handles the DB operations when the DB target is from type KV. For the KV operations, we use an instance of V3IO
    client and usually the KV table can be found under v3io:///users/pipelines/project-name/model-endpoints/endpoints/.
    """

    def __init__(self, access_key: str, project: str):
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

        :param endpoint: ModelEndpoint object that will be written into the DB.
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

        logger.info("Model endpoint table updated", endpoint_id=endpoint_id)

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

        logger.info("Model endpoint table cleared", endpoint_id=endpoint_id)

    def get_model_endpoint(
        self,
        endpoint_id: str = None,
        start: str = "now-1h",
        end: str = "now",
        metrics: typing.List[str] = None,
        feature_analysis: bool = False,
        convert_to_endpoint_object: bool = True,
    ):
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
        :param metrics:                    A list of metrics to return for the model endpoint. There are pre-defined
                                           metrics for model endpoints such as predictions_per_second and
                                           latency_avg_5m but also custom metrics defined by the user. Please note that
                                           these metrics are stored in the time series DB and the results will appear
                                           under model_endpoint.spec.metrics.
        :param feature_analysis:           When True, the base feature statistics and current feature statistics will
                                           be added to the output of the resulting object.
        :param convert_to_endpoint_object: A boolean that indicates whether to convert the model endpoint dictionary
                                           into a ModelEndpoint or not. True by default.

        :return: A ModelEndpoint object.
        """
        logger.info(
            "Getting model endpoint record from kv",
            endpoint_id=endpoint_id,
        )

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
                endpoint_metrics = self.get_endpoint_metrics(
                    endpoint_id=endpoint_id,
                    start=start,
                    end=end,
                    metrics=metrics,
                    access_key=self.access_key,
                )
                if endpoint_metrics:
                    endpoint.status.metrics = endpoint_metrics

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
        labels: typing.List = None,
        top_level: bool = None,
        metrics: typing.List[str] = None,
        start: str = "now-1h",
        end: str = "now",
        uids: typing.List = None,
    ) -> mlrun.api.schemas.ModelEndpointList:
        """
        Returns a list of ModelEndpoint objects, supports filtering by model, function, labels or top level.
        By default, when no filters are applied, all available ModelEndpoint objects for the given project will
        be listed.

        :param model:           The name of the model to filter by.
        :param function:        The name of the function to filter by.
        :param labels:          A list of labels to filter by. Label filters work by either filtering a specific value
                                of a label (i.e. list("key==value")) or by looking for the existence of a given
                                key (i.e. "key").
        :param top_level:       If True will return only routers and endpoint that are NOT children of any router.
        :param metrics:          A list of metrics to return for the model endpoint. There are pre-defined
                                 metrics for model endpoints such as predictions_per_second and
                                 latency_avg_5m but also custom metrics defined by the user. Please note that
                                 these metrics are stored in the time series DB and the results will be
                                 appeared under model_endpoint.spec.metrics.
        :param start:            The start time of the metrics. Can be represented by a string containing an
                                 RFC 3339 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days), or
                                 0 for the earliest time.
        :param end:              The end time of the metrics. Can be represented by a string containing an
                                 RFC 3339 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days),
                                 or 0 for the earliest time.
        :param uids:             List of model endpoint unique ids to include in the result.


        :return: An object of ModelEndpointList which is literally a list of model endpoints along with some
                          metadata. To get a standard list of model endpoints use ModelEndpointList.endpoints.
        """

        # Initialize an empty model endpoints list
        endpoint_list = mlrun.api.schemas.model_endpoints.ModelEndpointList(
            endpoints=[]
        )

        # Retrieve the raw data from the KV table and get the endpoint ids
        try:
            cursor = self.client.kv.new_cursor(
                container=self.container,
                table_path=self.path,
                filter_expression=self.build_kv_cursor_filter_expression(
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

        :param endpoints: An object of ModelEndpointList which is literally a list of model endpoints along with some
                          metadata. To get a standard list of model endpoints use ModelEndpointList.endpoints.
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

        # Getting the path for the time series DB
        events_path = (
            mlrun.mlconf.model_endpoint_monitoring.store_prefixes.default.format(
                project=self.project,
                kind=mlrun.api.schemas.ModelMonitoringStoreKinds.EVENTS,
            )
        )
        (
            _,
            _,
            events_path,
        ) = mlrun.utils.model_monitoring.parse_model_endpoint_store_prefix(events_path)

        # Delete time series DB resources
        try:
            frames.delete(
                backend=model_monitoring_constants.TimeSeriesTarget.TSDB,
                table=events_path,
                if_missing=v3io_frames.frames_pb2.IGNORE,
            )
        except v3io_frames.errors.CreateError:
            # Frames might raise an exception if schema file does not exist.
            pass

        # Final cleanup of tsdb path
        events_path.replace("://u", ":///u")
        store, _ = mlrun.store_manager.get_or_create_store(events_path)
        store.rm(events_path, recursive=True)

    @staticmethod
    def build_kv_cursor_filter_expression(
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
                f"(endpoint_type=='{str(mlrun.utils.model_monitoring.EndpointType.NODE_EP.value)}' "
                f"OR  endpoint_type=='{str(mlrun.utils.model_monitoring.EndpointType.ROUTER.value)}')"
            )

        return " AND ".join(filter_expression)
