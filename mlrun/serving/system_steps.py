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
from copy import deepcopy
from datetime import timedelta
from typing import Any, Optional, Union

import storey

import mlrun
import mlrun.artifacts
import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.serving
from mlrun.common.schemas import MonitoringData
from mlrun.utils import logger


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

        result = self._get_data_from_path(
            result_path, event.body.get(model, event.body)
        )
        output_schema = model_monitoring_data.get(MonitoringData.OUTPUTS)
        input_schema = model_monitoring_data.get(MonitoringData.INPUTS)
        logger.debug("output schema retrieved", output_schema=output_schema)
        if isinstance(result, dict):
            if len(result) > 1:
                # transpose by key the outputs:
                outputs = self.transpose_by_key(result, output_schema)
            elif len(result) == 1:
                outputs = (
                    result[output_schema[0]]
                    if output_schema
                    else list(result.values())[0]
                )
            else:
                outputs = []
            if not output_schema:
                logger.warn(
                    "Output schema was not provided using Project:log_model or by ModelRunnerStep:add_model order "
                    "may not preserved"
                )
        else:
            outputs = result

        event_inputs = event._metadata.get("inputs", {})
        event_inputs = self._get_data_from_path(input_path, event_inputs)
        if isinstance(event_inputs, dict):
            if len(event_inputs) > 1:
                # transpose by key the inputs:
                inputs = self.transpose_by_key(event_inputs, input_schema)
            else:
                inputs = (
                    event_inputs[input_schema[0]]
                    if input_schema
                    else list(result.values())[0]
                )
        else:
            inputs = event_inputs

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
        request = {"inputs": inputs, "id": getattr(event, "id", None)}
        resp = {"outputs": outputs}

        return request, resp

    @staticmethod
    def transpose_by_key(
        data_to_transpose, schema: Optional[list[str]] = None
    ) -> list[list[float]]:
        values = (
            list(data_to_transpose.values())
            if not schema
            else [data_to_transpose[key] for key in schema]
        )
        if values and not isinstance(values[0], list):
            values = [values]
        transposed = (
            list(map(list, zip(*values)))
            if all(isinstance(v, list) for v in values) and len(values) > 1
            else values
        )
        return transposed

    @staticmethod
    def _get_data_from_path(
        path: Union[str, list[str], None], data: dict
    ) -> dict[str, Any]:
        if isinstance(path, str):
            output_data = data.get(path)
        elif isinstance(path, list):
            output_data = deepcopy(data)
            for key in path:
                output_data = output_data.get(key, {})
        elif path is None:
            output_data = data
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Expected path be of type str or list of str or None"
            )
        if isinstance(output_data, (int, float)):
            output_data = [output_data]
        return output_data

    def do(self, event):
        monitoring_event_list = []
        model_runner_name = event._metadata.get("model_runner_name", "")
        step = self.server.graph.steps[model_runner_name] if self.server else {}
        monitoring_data = step.monitoring_data
        logger.debug(
            "monitoring preprocessor started",
            event=event,
            model_endpoints=monitoring_data,
            metadata=event._metadata,
        )
        if len(monitoring_data) > 1:
            for model in event.body.keys():
                if model in monitoring_data:
                    request, resp = self.reconstruct_request_resp_fields(
                        event, model, monitoring_data[model]
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
                            mm_schemas.StreamProcessingEvent.WHEN: event._metadata.get(
                                model, {}
                            ).get(mm_schemas.StreamProcessingEvent.WHEN),
                            mm_schemas.StreamProcessingEvent.ENDPOINT_ID: monitoring_data[
                                model
                            ].get(
                                mlrun.common.schemas.MonitoringData.MODEL_ENDPOINT_UID
                            ),
                            mm_schemas.StreamProcessingEvent.LABELS: monitoring_data[
                                model
                            ].get(mlrun.common.schemas.MonitoringData.OUTPUTS),
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
            monitoring_event_list.append(
                {
                    mm_schemas.StreamProcessingEvent.MODEL: model,
                    mm_schemas.StreamProcessingEvent.MODEL_CLASS: monitoring_data[
                        model
                    ].get(mm_schemas.StreamProcessingEvent.MODEL_CLASS),
                    mm_schemas.StreamProcessingEvent.MICROSEC: event._metadata.get(
                        mm_schemas.StreamProcessingEvent.MICROSEC
                    ),
                    mm_schemas.StreamProcessingEvent.WHEN: event._metadata.get(
                        mm_schemas.StreamProcessingEvent.WHEN
                    ),
                    mm_schemas.StreamProcessingEvent.ENDPOINT_ID: monitoring_data[
                        model
                    ].get(mlrun.common.schemas.MonitoringData.MODEL_ENDPOINT_UID),
                    mm_schemas.StreamProcessingEvent.LABELS: monitoring_data[model].get(
                        mlrun.common.schemas.MonitoringData.OUTPUTS
                    ),
                    mm_schemas.StreamProcessingEvent.FUNCTION_URI: self.server.function_uri
                    if self.server
                    else None,
                    mm_schemas.StreamProcessingEvent.REQUEST: request,
                    mm_schemas.StreamProcessingEvent.RESPONSE: resp,
                    mm_schemas.StreamProcessingEvent.ERROR: event.body[
                        mm_schemas.StreamProcessingEvent.ERROR
                    ]
                    if mm_schemas.StreamProcessingEvent.ERROR in event.body
                    else None,
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
            background_task = mlrun.get_run_db().get_project_background_task(
                self.server.project, self.server.model_endpoint_creation_task_name
            )
            self._background_task_check_timestamp = mlrun.utils.now_date()
            self._log_background_task_state(background_task.status.state)
            self._background_task_state = background_task.status.state

        if (
            self._background_task_state
            == mlrun.common.schemas.BackgroundTaskState.succeeded
        ):
            return event
        else:
            return None

    def _log_background_task_state(
        self, background_task_state: mlrun.common.schemas.BackgroundTaskState
    ):
        logger.info(
            "Checking model endpoint creation task status",
            task_name=self.server.model_endpoint_creation_task_name,
        )
        if (
            background_task_state
            in mlrun.common.schemas.BackgroundTaskState.terminal_states()
        ):
            logger.info(
                f"Model endpoint creation task completed with state {background_task_state}"
            )
        else:  # in progress
            logger.info(
                f"Model endpoint creation task is still in progress with the current state: "
                f"{background_task_state}. Events will not be monitored for the next "
                f"{mlrun.mlconf.model_endpoint_monitoring.model_endpoint_creation_check_period} seconds",
                name=self.name,
                background_task_check_timestamp=self._background_task_check_timestamp.isoformat(),
            )


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
        if self.sampling_percentage != 100:
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
