# Copyright 2024 Iguazio
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

import json
import typing
from abc import ABC, abstractmethod

import mlrun.common.schemas.model_monitoring as mm_schemas


class StoreBase(ABC):
    type: typing.ClassVar[str]
    """
    An abstract class to handle the store object in the DB target.
    """

    def __init__(self, project: str):
        """
        Initialize a new store target.

        :param project: The name of the project.
        """
        self.project = project

    @abstractmethod
    def write_model_endpoint(self, endpoint: dict[str, typing.Any]):
        """
        Create a new endpoint record in the DB table.

        :param endpoint: model endpoint dictionary that will be written into the DB.
        """
        pass

    @abstractmethod
    def update_model_endpoint(
        self, endpoint_id: str, attributes: dict[str, typing.Any]
    ):
        """
        Update a model endpoint record with a given attributes.

        :param endpoint_id: The unique id of the model endpoint.
        :param attributes: Dictionary of attributes that will be used for update the model endpoint. Note that the keys
                           of the attributes dictionary should exist in the DB table.

        """
        pass

    @abstractmethod
    def delete_model_endpoint(self, endpoint_id: str):
        """
        Deletes the record of a given model endpoint id.

        :param endpoint_id: The unique id of the model endpoint.
        """
        pass

    @abstractmethod
    def delete_model_endpoints_resources(self):
        """
        Delete all model endpoints resources.

        """
        pass

    @abstractmethod
    def get_model_endpoint(
        self,
        endpoint_id: str,
    ) -> dict[str, typing.Any]:
        """
        Get a single model endpoint record.

        :param endpoint_id: The unique id of the model endpoint.

        :return: A model endpoint record as a dictionary.
        """
        pass

    @abstractmethod
    def list_model_endpoints(
        self,
        model: str = None,
        function: str = None,
        labels: list[str] = None,
        top_level: bool = None,
        uids: list = None,
        include_stats: bool = None,
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
        :param uids:             List of model endpoint unique ids to include in the result.
        :param include_stats:   If True, will include model endpoint statistics in the result.

        :return: A list of model endpoint dictionaries.
        """
        pass

    @abstractmethod
    def write_application_event(
        self,
        event: dict[str, typing.Any],
        kind: mm_schemas.WriterEventKind = mm_schemas.WriterEventKind.RESULT,
    ) -> None:
        """
        Write a new event in the target table.

        :param event: An event dictionary that represents the application result, should be corresponded to the
                      schema defined in the :py:class:`~mlrun.common.schemas.model_monitoring.constants.WriterEvent`
                      object.
        :param kind: The type of the event, can be either "result" or "metric".
        """

    @abstractmethod
    def get_last_analyzed(self, endpoint_id: str, application_name: str) -> int:
        """
        Get the last analyzed time for the provided model endpoint and application.

        :param endpoint_id:      The unique id of the model endpoint.
        :param application_name: Registered application name.

        :return: Timestamp as a Unix time.
        :raise:  MLRunNotFoundError if last analyzed value is not found.
        """
        pass

    @abstractmethod
    def update_last_analyzed(
        self,
        endpoint_id: str,
        application_name: str,
        last_analyzed: int,
    ):
        """
        Update the last analyzed time for the provided model endpoint and application.

        :param endpoint_id:      The unique id of the model endpoint.
        :param application_name: Registered application name.
        :param last_analyzed:    Timestamp as a Unix time that represents the last analyzed time of a certain
                                 application and model endpoint.

        """
        pass

    @abstractmethod
    def get_model_endpoint_metrics(
        self, endpoint_id: str, type: mm_schemas.ModelEndpointMonitoringMetricType
    ) -> list[mm_schemas.ModelEndpointMonitoringMetric]:
        """
        Get the model monitoring results and metrics of the requested model endpoint.

        :param: endpoint_id: The model endpoint identifier.
        :param: type:        The type of the requested metrics ("result" or "metric").

        :return:             A list of the available metrics.
        """

    @staticmethod
    def _validate_labels(
        endpoint_dict: dict,
        labels: list,
    ) -> bool:
        """Validate that the model endpoint dictionary has the provided labels. There are 2 possible cases:
        1 - Labels were provided as a list of key-values pairs (e.g. ['label_1=value_1', 'label_2=value_2']): Validate
            that each pair exist in the endpoint dictionary.
        2 - Labels were provided as a list of key labels (e.g. ['label_1', 'label_2']): Validate that each key exist in
            the endpoint labels dictionary.

        :param endpoint_dict: Dictionary of the model endpoint records.
        :param labels:        List of dictionary of required labels.

        :return: True if the labels exist in the endpoint labels dictionary, otherwise False.
        """

        # Convert endpoint labels into dictionary
        endpoint_labels = json.loads(
            endpoint_dict.get(mm_schemas.EventFieldType.LABELS)
        )

        for label in labels:
            # Case 1 - label is a key=value pair
            if "=" in label:
                lbl, value = list(map(lambda x: x.strip(), label.split("=")))
                if lbl not in endpoint_labels or str(endpoint_labels[lbl]) != value:
                    return False
            # Case 2 - label is just a key
            else:
                if label not in endpoint_labels:
                    return False

        return True

    def create_tables(self):
        pass
