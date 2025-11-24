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
import random
from copy import copy
from datetime import timedelta
from typing import Any, Optional, Union

import numpy as np
import storey

import mlrun
import mlrun.artifacts
import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.feature_store
import mlrun.serving
from mlrun.common.model_monitoring.helpers import (
    get_model_endpoints_creation_task_status,
)
from mlrun.common.schemas import MonitoringData
from mlrun.utils import get_data_from_path, logger


class MatchingEndpointsState(mlrun.common.types.StrEnum):
    all_matched = "all_matched"
    not_all_matched = "not_all_matched"
    no_check_needed = "no_check_needed"
    not_yet_checked = "not_yet_matched"

    @staticmethod
    def success_states() -> list[str]:
        return [
            MatchingEndpointsState.all_matched,
            MatchingEndpointsState.no_check_needed,
        ]


class MonitoringPreProcessor(storey.MapClass):
    """preprocess step, reconstructs the serving output event body to StreamProcessingEvent schema"""

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.server: mlrun.serving.GraphServer = (
            getattr(self.context, "server", None) if self.context else None
        )

    def reconstruct_request_resp_fields(
        self, event, model: str, model_monitoring_data: dict
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        result_path = model_monitoring_data.get(MonitoringData.RESULT_PATH)
        input_path = model_monitoring_data.get(MonitoringData.INPUT_PATH)

        output_schema = model_monitoring_data.get(MonitoringData.OUTPUTS)
        input_schema = model_monitoring_data.get(MonitoringData.INPUTS)
        logger.debug(
            "output and input schema retrieved",
            output_schema=output_schema,
            input_schema=input_schema,
        )

        outputs, new_output_schema = self.get_listed_data(
            event.body.get(model, event.body), result_path, output_schema
        )
        inputs, new_input_schema = self.get_listed_data(
            event._metadata.get("inputs", {}), input_path, input_schema
        )

        if outputs and isinstance(outputs[0], list):
            if output_schema and len(output_schema) != len(outputs[0]):
                logger.info(
                    "The number of outputs returned by the model does not match the number of outputs "
                    "specified in the model endpoint.",
                    model_endpoint=model,
                    output_len=len(outputs[0]),
                    schema_len=len(output_schema),
                )
        elif outputs:
            if output_schema and len(output_schema) != 1:
                logger.info(
                    "The number of outputs returned by the model does not match the number of outputs "
                    "specified in the model endpoint.",
                    model_endpoint=model,
                    output_len=len(outputs),
                    schema_len=len(output_schema),
                )
            if len(inputs) != len(outputs):
                logger.warn(
                    "outputs and inputs are not in the same length check 'input_path' and "
                    "'output_path' was specified if needed"
                )
        request = {
            "inputs": inputs,
            "id": getattr(event, "id", None),
            "input_schema": new_input_schema,
        }
        resp = {"outputs": outputs, "output_schema": new_output_schema}

        return request, resp

    def get_listed_data(
        self,
        raw_data: dict,
        data_path: Optional[Union[list[str], str]] = None,
        schema: Optional[list[str]] = None,
    ):
        """Get data from a path and transpose it by keys if dict is provided."""
        new_schema = None
        data_from_path = get_data_from_path(data_path, raw_data)
        if isinstance(data_from_path, dict):
            # transpose by key the inputs:
            listed_data, new_schema = self.transpose_by_key(data_from_path, schema)
            new_schema = new_schema or schema
            if not schema:
                logger.warn(
                    f"No schema provided through add_model(); the order of {data_from_path} "
                    "may not be preserved."
                )
        elif not isinstance(data_from_path, list):
            listed_data = [data_from_path]
        else:
            listed_data = data_from_path
        return listed_data, new_schema

    @staticmethod
    def transpose_by_key(
        data: dict, schema: Optional[Union[str, list[str]]] = None
    ) -> tuple[Union[list[Any], list[list[Any]]], list[str]]:
        """
        Transpose values from a dictionary by keys.

        Given a dictionary and an optional schema (a key or list of keys), this function:
        - Extracts the values for the specified keys (or all keys if no schema is provided).
        - Ensures the data is represented as a list of rows, then transposes it (i.e., switches rows to columns).
        - Handles edge cases:
            * If a single scalar or single-element list is provided, returns a flat list.
            * If a single key is provided (as a string or a list with one element), handles it properly.
            * If only one row with len of one remains after transposition, unwraps it to avoid nested list-of-one.

        Example::

            transpose_by_key({"a": 1})
            # returns: [1]

            transpose_by_key({"a": [1, 2]})
            # returns: [1 ,2]

            transpose_by_key({"a": [1, 2], "b": [3, 4]})
            # returns: [[1, 3], [2, 4]]

        :param data:     Dictionary with values that are either scalars or lists.
        :param schema:   Optional key or list of keys to extract. If not provided, all keys are used.
                         Can be a string (single key) or a list of strings.

        :return:         Transposed values:
                         * If result is a single column or row, returns a flat list.
                         * If result is a matrix, returns a list of lists.

        :raises ValueError: If the values include a mix of scalars and lists, or if the list lengths do not match.
                mlrun.MLRunInvalidArgumentError if the schema keys are not contained in the data keys.
        """
        new_schema = None
        # Normalize keys in data:
        normalize_data = {
            mlrun.feature_store.api.norm_column_name(k): copy(v)
            for k, v in data.items()
        }
        # Normalize schema to list
        if not schema:
            keys = list(normalize_data.keys())
            new_schema = keys
        elif isinstance(schema, str):
            keys = [mlrun.feature_store.api.norm_column_name(schema)]
        else:
            keys = [mlrun.feature_store.api.norm_column_name(key) for key in schema]

        values = [normalize_data[key] for key in keys if key in normalize_data]
        if len(values) != len(keys):
            raise mlrun.MLRunInvalidArgumentError(
                f"Schema keys {keys} are not contained in the data keys {list(data.keys())}."
            )

        # Detect if all are scalars ie: int,float,str
        all_scalars = all(not isinstance(v, (list, tuple, np.ndarray)) for v in values)
        all_lists = all(isinstance(v, (list, tuple, np.ndarray)) for v in values)

        if not (all_scalars or all_lists):
            raise ValueError(
                "All values must be either scalars or lists of equal length."
            )

        if all_scalars:
            transposed = np.array([values], dtype=object)
        elif all_lists and len(keys) > 1:
            arrays = [np.array(v, dtype=object) for v in values]
            mat = np.stack(arrays, axis=0)
            transposed = mat.T
        else:
            return values[0], new_schema

        if transposed.shape[1] == 1 and transposed.shape[0] == 1:
            # Transform [[0]] -> [0]:
            return transposed[:, 0].tolist(), new_schema
        return transposed.tolist(), new_schema

    def do(self, event):
        monitoring_event_list = []
        model_runner_name = event._metadata.get("model_runner_name", "")
        step = self.server.graph.steps[model_runner_name] if self.server else None
        if not step or not hasattr(step, "monitoring_data"):
            raise mlrun.errors.MLRunRuntimeError(
                f"ModelRunnerStep name {model_runner_name} is not found in the graph or does not have monitoring data"
            )
        monitoring_data = step.monitoring_data
        logger.debug(
            "monitoring preprocessor started",
            event=event,
            monitoring_data=monitoring_data,
            metadata=event._metadata,
        )
        if len(monitoring_data) > 1:
            for model in event.body.keys():
                if model in monitoring_data:
                    request, resp = self.reconstruct_request_resp_fields(
                        event, model, monitoring_data[model]
                    )
                    if hasattr(event, "_original_timestamp"):
                        when = event._original_timestamp
                    else:
                        when = event._metadata.get(model, {}).get(
                            mm_schemas.StreamProcessingEvent.WHEN
                        )
                    monitoring_event_list.append(
                        {
                            mm_schemas.StreamProcessingEvent.MODEL: model,
                            mm_schemas.StreamProcessingEvent.MODEL_CLASS: monitoring_data[
                                model
                            ].get(mm_schemas.StreamProcessingEvent.MODEL_CLASS),
                            mm_schemas.StreamProcessingEvent.MICROSEC: event._metadata.get(
                                model, {}
                            ).get(mm_schemas.StreamProcessingEvent.MICROSEC),
                            mm_schemas.StreamProcessingEvent.WHEN: when,
                            mm_schemas.StreamProcessingEvent.ENDPOINT_ID: monitoring_data[
                                model
                            ].get(
                                mlrun.common.schemas.MonitoringData.MODEL_ENDPOINT_UID
                            ),
                            mm_schemas.StreamProcessingEvent.LABELS: event.body[
                                model
                            ].get("labels")
                            or {},
                            mm_schemas.StreamProcessingEvent.FUNCTION_URI: self.server.function_uri
                            if self.server
                            else None,
                            mm_schemas.StreamProcessingEvent.REQUEST: request,
                            mm_schemas.StreamProcessingEvent.RESPONSE: resp,
                            mm_schemas.StreamProcessingEvent.ERROR: event.body[model][
                                mm_schemas.StreamProcessingEvent.ERROR
                            ]
                            if mm_schemas.StreamProcessingEvent.ERROR
                            in event.body[model]
                            else None,
                            mm_schemas.StreamProcessingEvent.METRICS: event.body[model][
                                mm_schemas.StreamProcessingEvent.METRICS
                            ]
                            if mm_schemas.StreamProcessingEvent.METRICS
                            in event.body[model]
                            else None,
                        }
                    )
        elif monitoring_data:
            model = list(monitoring_data.keys())[0]
            request, resp = self.reconstruct_request_resp_fields(
                event, model, monitoring_data[model]
            )
            if hasattr(event, "_original_timestamp"):
                when = event._original_timestamp
            else:
                when = event._metadata.get(mm_schemas.StreamProcessingEvent.WHEN)
            monitoring_event_list.append(
                {
                    mm_schemas.StreamProcessingEvent.MODEL: model,
                    mm_schemas.StreamProcessingEvent.MODEL_CLASS: monitoring_data[
                        model
                    ].get(mm_schemas.StreamProcessingEvent.MODEL_CLASS),
                    mm_schemas.StreamProcessingEvent.MICROSEC: event._metadata.get(
                        mm_schemas.StreamProcessingEvent.MICROSEC
                    ),
                    mm_schemas.StreamProcessingEvent.WHEN: when,
                    mm_schemas.StreamProcessingEvent.ENDPOINT_ID: monitoring_data[
                        model
                    ].get(mlrun.common.schemas.MonitoringData.MODEL_ENDPOINT_UID),
                    mm_schemas.StreamProcessingEvent.LABELS: event.body.get("labels")
                    or {},
                    mm_schemas.StreamProcessingEvent.FUNCTION_URI: self.server.function_uri
                    if self.server
                    else None,
                    mm_schemas.StreamProcessingEvent.REQUEST: request,
                    mm_schemas.StreamProcessingEvent.RESPONSE: resp,
                    mm_schemas.StreamProcessingEvent.ERROR: event.body.get(
                        mm_schemas.StreamProcessingEvent.ERROR
                    ),
                    mm_schemas.StreamProcessingEvent.METRICS: event.body[
                        mm_schemas.StreamProcessingEvent.METRICS
                    ]
                    if mm_schemas.StreamProcessingEvent.METRICS in event.body
                    else None,
                }
            )
        event.body = monitoring_event_list
        return event


class BackgroundTaskStatus(storey.MapClass):
    """
    background task status checker, prevent events from pushing to the model monitoring stream target if model endpoints
    creation failed or in progress
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.matching_endpoints = MatchingEndpointsState.not_yet_checked
        self.graph_model_endpoint_uids: set = set()
        self.listed_model_endpoint_uids: set = set()
        self.server: mlrun.serving.GraphServer = (
            getattr(self.context, "server", None) if self.context else None
        )
        self._background_task_check_timestamp = None
        self._background_task_state = mlrun.common.schemas.BackgroundTaskState.running

    def do(self, event):
        if self.server is None:
            return None
        if (
            self._background_task_state
            == mlrun.common.schemas.BackgroundTaskState.running
            and (
                self._background_task_check_timestamp is None
                or mlrun.utils.now_date() - self._background_task_check_timestamp
                >= timedelta(
                    seconds=mlrun.mlconf.model_endpoint_monitoring.model_endpoint_creation_check_period
                )
            )
        ):
            (
                self._background_task_state,
                self._background_task_check_timestamp,
                self.listed_model_endpoint_uids,
            ) = get_model_endpoints_creation_task_status(self.server)
        if (
            self.listed_model_endpoint_uids
            and self.matching_endpoints == MatchingEndpointsState.not_yet_checked
        ):
            if not self.graph_model_endpoint_uids:
                self.graph_model_endpoint_uids = collect_model_endpoint_uids(
                    self.server
                )

            if self.graph_model_endpoint_uids.issubset(self.listed_model_endpoint_uids):
                self.matching_endpoints = MatchingEndpointsState.all_matched
        elif self.listed_model_endpoint_uids is None:
            self.matching_endpoints = MatchingEndpointsState.no_check_needed

        if (
            self._background_task_state
            == mlrun.common.schemas.BackgroundTaskState.succeeded
            and self.matching_endpoints in MatchingEndpointsState.success_states()
        ):
            return event
        else:
            return None


def collect_model_endpoint_uids(server: mlrun.serving.GraphServer) -> set[str]:
    """Collects all model endpoint UIDs from the server's graph steps."""
    model_endpoint_uids = set()
    for step in server.graph.steps.values():
        if hasattr(step, "monitoring_data"):
            for model in step.monitoring_data.keys():
                uid = step.monitoring_data[model].get(
                    mlrun.common.schemas.MonitoringData.MODEL_ENDPOINT_UID
                )
                if uid:
                    model_endpoint_uids.add(uid)
    return model_endpoint_uids


class SamplingStep(storey.MapClass):
    """sampling step, samples the serving outputs for the model monitoring as sampling_percentage defines"""

    def __init__(
        self,
        sampling_percentage: Optional[float] = 100.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sampling_percentage = (
            sampling_percentage if 0 < sampling_percentage <= 100 else 100
        )

    def do(self, event):
        logger.debug(
            "sampling step runs",
            event=event,
            sampling_percentage=self.sampling_percentage,
        )
        if self.sampling_percentage != 100 and not event.get(
            mm_schemas.StreamProcessingEvent.ERROR
        ):
            request = event[mm_schemas.StreamProcessingEvent.REQUEST]
            num_of_inputs = len(request["inputs"])
            sampled_requests_indices = self._pick_random_requests(
                num_of_inputs, self.sampling_percentage
            )
            if not sampled_requests_indices:
                return None

            event[mm_schemas.StreamProcessingEvent.REQUEST]["inputs"] = [
                request["inputs"][i] for i in sampled_requests_indices
            ]

            if isinstance(
                event[mm_schemas.StreamProcessingEvent.RESPONSE]["outputs"], list
            ):
                event[mm_schemas.StreamProcessingEvent.RESPONSE]["outputs"] = [
                    event[mm_schemas.StreamProcessingEvent.RESPONSE]["outputs"][i]
                    for i in sampled_requests_indices
                ]
        event[mm_schemas.EventFieldType.SAMPLING_PERCENTAGE] = self.sampling_percentage
        event[mm_schemas.EventFieldType.EFFECTIVE_SAMPLE_COUNT] = len(
            event.get(mm_schemas.StreamProcessingEvent.REQUEST, {}).get("inputs", [])
        )
        return event

    @staticmethod
    def _pick_random_requests(num_of_reqs: int, percentage: float) -> list[int]:
        """
        Randomly selects indices of requests to sample based on the given percentage

        :param num_of_reqs: Number of requests to select from
        :param percentage: Sample percentage for each request
        :return: A list containing the indices of the selected requests
        """

        return [
            req for req in range(num_of_reqs) if random.random() < (percentage / 100)
        ]


class MockStreamPusher(storey.MapClass):
    def __init__(self, output_stream=None, **kwargs):
        super().__init__(**kwargs)
        stream = self.context.stream if self.context else None
        self.output_stream = output_stream or stream.output_stream

    def do(self, event):
        self.output_stream.push(
            [event], partition_key=mm_schemas.StreamProcessingEvent.ENDPOINT_ID
        )
