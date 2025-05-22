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
from typing import Any, Union

import storey

import mlrun.artifacts
import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.serving
from mlrun.serving import ModelRunnerStep
from mlrun.serving.remote import RemoteStep
from mlrun.utils import logger


class MonitoringPreProcessor(storey.MapClass):  # TODO Roy is this necessary
    def __init__(
        self,
        context,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.models_uri: dict[str:str] = {}
        self.context = context
        self.model_endpoints: dict[str, dict[str, dict[str, str]]] = {}
        self.labels: dict[str, dict[str, str]] = {}
        self.input_path: dict[str, str] = {}
        self.result_path: dict[str, str] = {}
        self.output_schema: dict[str, Union[list[str], str]] = {}

        server: mlrun.serving.GraphServer = getattr(
            self.context, "_server", None
        ) or getattr(self.context, "server", None)

        for step in server.graph:
            if isinstance(step, ModelRunnerStep):
                self.model_endpoints[step.name] = {}
                monitoring_data = step.class_args.get(
                    mlrun.common.schemas.ModelRunnerStepData.MONITORING_DATA
                )
                for model in step.class_args.get(
                    mlrun.common.schemas.ModelRunnerStepData.MODELS, {}
                ).keys():
                    self.model_endpoints[step.name][
                        mm_schemas.StreamProcessingEvent.MODEL
                    ] = model
                    self.model_endpoints[step.name][
                        mm_schemas.StreamProcessingEvent.ENDPOINT_ID
                    ] = monitoring_data[model][
                        mm_schemas.StreamProcessingEvent.ENDPOINT_ID
                    ]
                    self.labels[model] = monitoring_data.get(model, {}).get(
                        mlrun.common.schemas.MonitoringData.OUTPUTS
                    )
                    self.input_path[model] = monitoring_data.get(model, {}).get(
                        mlrun.common.schemas.MonitoringData.INPUT_PATH
                    )
                    self.result_path[model] = monitoring_data.get(model, {}).get(
                        mlrun.common.schemas.MonitoringData.RESULT_PATH
                    )
                    self.output_schema[model] = monitoring_data.get(model, {}).get(
                        mlrun.common.schemas.MonitoringData.OUTPUTS
                    )
                    self.models_uri[model] = monitoring_data.get(model, {}).get(
                        mlrun.common.schemas.MonitoringData.MODEL_PATH
                    )

    def get_model_output_schema(self, model: str) -> list[str]:
        if self.output_schema.get(model) is None and self.models_uri.get(model):
            _, model_spec, extra_datitems = mlrun.artifacts.get_model(
                self.output_schema.get(model), ""
            )
            self.output_schema[model] = [feature.name for feature in model_spec.outputs]
        return self.output_schema.get(model)

    def reconstruct_request_field(self, event, model: str) -> dict[str, Any]:
        output_schema = self.get_model_output_schema(model)
        result_path = self.result_path.get(model) or ""
        result = event.body.get(model) or event.body.get(result_path) or event.body
        result = (
            result.get(result_path) if (isinstance(result, dict)) else result
        ) or result

        if isinstance(result, dict):
            outputs = []
            list_apply = False
            for loc, key in enumerate(output_schema):
                if key in result:
                    if isinstance(result[key], list):
                        if not list_apply:
                            list_apply = True
                            # cols - len(output_schema), rows - len(result[key])
                            outputs = [
                                [None] * len(output_schema)
                                for _ in range(len(result[key]))
                            ]
                        for event_index in range(len(result[key])):
                            outputs[event_index][loc] = result[key][event_index]
                    else:
                        outputs.append(result[key])
        else:
            outputs = result

        event_inputs = (event.metadat.get("inputs", {}).get(self.input_path[model]),)
        if isinstance(event_inputs, dict):
            inputs = []
            list_apply = False
            for loc, key in enumerate(event_inputs):
                if isinstance(event_inputs[key], list):
                    if not list_apply:
                        list_apply = True
                        # cols - len(event_inputs), rows - len(event_inputs[key])
                        inputs = [
                            [None] * len(event_inputs)
                            for _ in range(len(event_inputs[key]))
                        ]
                    for event_index in range(len(event_inputs[key])):
                        inputs[event_index][loc] = event_inputs[key][event_index]
                else:
                    inputs.append(result[key])
        else:
            inputs = event_inputs

        if outputs and isinstance(outputs[0], list):
            if output_schema and len(output_schema) != len(outputs[0]):
                logger.info(
                    "The number of outputs returned by the model does not match the number of outputs "
                    "specified in the model endpoint.",
                    model_endpoint=model,
                    output_len=len([outputs][0]),
                    schema_len=len(output_schema),
                )

        return {
            "inputs": inputs,
            "outputs": outputs,
        }

    def _do(self, event):
        if self.context is not None and self.context.is_mock:
            return event
        monitoring_event_list = []
        server: mlrun.serving.GraphServer = getattr(
            self.context, "_server", None
        ) or getattr(self.context, "server", None)
        model_runner_endpoints = self.model_endpoints.get(
            event.metadata.get("model_runner_name", "")
        )
        if len(model_runner_endpoints) > 1:
            for model in event.body.keys():
                if model in model_runner_endpoints:
                    monitoring_event_list.append(
                        {
                            mm_schemas.StreamProcessingEvent.MODEL: model,
                            mm_schemas.StreamProcessingEvent.MODEL_CLASS: model_runner_endpoints[
                                model
                            ].get(mm_schemas.StreamProcessingEvent.MODEL_CLASS),
                            mm_schemas.StreamProcessingEvent.MICROSEC: event.metadata.get(
                                model, {}
                            ).get(mm_schemas.StreamProcessingEvent.MICROSEC),
                            mm_schemas.StreamProcessingEvent.WHEN: event.metadata.get(
                                model, {}
                            ).get(mm_schemas.StreamProcessingEvent.WHEN),
                            mm_schemas.StreamProcessingEvent.ENDPOINT_ID: model_runner_endpoints[
                                model
                            ].get(mm_schemas.StreamProcessingEvent.ENDPOINT_ID),
                            mm_schemas.StreamProcessingEvent.LABELS: self.labels[model],
                            mm_schemas.StreamProcessingEvent.FUNCTION_URI: server.function_uri,
                            mm_schemas.StreamProcessingEvent.REQUEST: self.reconstruct_request_field(
                                event, model
                            ),
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
        elif model_runner_endpoints:
            model = list(model_runner_endpoints.keys())[0]
            monitoring_event_list.append(
                {
                    mm_schemas.StreamProcessingEvent.MODEL: model,
                    mm_schemas.StreamProcessingEvent.MODEL_CLASS: self.model_endpoints[
                        model
                    ].get(mm_schemas.StreamProcessingEvent.MODEL_CLASS),
                    mm_schemas.StreamProcessingEvent.MICROSEC: event.metadata.get(
                        mm_schemas.StreamProcessingEvent.MICROSEC
                    ),
                    mm_schemas.StreamProcessingEvent.WHEN: event.metadata.get(
                        mm_schemas.StreamProcessingEvent.WHEN
                    ),
                    mm_schemas.StreamProcessingEvent.ENDPOINT_ID: self.model_endpoints[
                        model
                    ].get(mm_schemas.StreamProcessingEvent.ENDPOINT_ID),
                    mm_schemas.StreamProcessingEvent.LABELS: self.labels[model],
                    mm_schemas.StreamProcessingEvent.FUNCTION_URI: server.function_uri,
                    mm_schemas.StreamProcessingEvent.REQUEST: self.reconstruct_request_field(
                        event, model
                    ),
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


class BackgroundTaskStatus(RemoteStep):
    def __init__(self, context, **kwargs):
        self.context = context
        self.server: mlrun.serving.GraphServer = getattr(
            self.context, "_server", None
        ) or getattr(self.context, "server", None)
        self._background_task_check_timestamp = None
        self._background_task_status = mlrun.common.schemas.BackgroundTaskState.running

        path = f"projects/{self.server.project}/background-tasks/{self.server.model_endpoint_creation_task_name}"
        if not self.context.is_mock:
            super().__init__(
                url=self.context.get_run_db().get_base_api_url(path), method="GET", **kwargs
            )
        else:
            super().__init__(
                url=path, method="GET", **kwargs
            )

    async def _process_event(self, event):
        if self.context is not None and (not self.context.is_mock or self.context.monitoring_mock):
            if (
                self._background_task_status
                == mlrun.common.schemas.BackgroundTaskState.running
            ):
                response = await super()._process_event(event)
                background_task = mlrun.common.schemas.BackgroundTask(**response.json())
                self._background_task_check_timestamp = mlrun.utils.now_date()
                if self._background_task_succeeded(background_task):
                    return event
                else:
                    return None
            elif (
                self._background_task_status
                == mlrun.common.schemas.BackgroundTaskState.failed
            ):
                return None
        return event

    def do_event(self, event):
        if self.context is not None and (not self.context.is_mock or self.context.monitoring_mock):
            if (
                self._background_task_status
                == mlrun.common.schemas.BackgroundTaskState.running
            ):
                response = super().do_event(event)
                background_task = mlrun.common.schemas.BackgroundTask(**response.json())
                self._background_task_check_timestamp = mlrun.utils.now_date()
                if self._background_task_succeeded(background_task):
                    return event
                else:
                    return None
            elif (
                self._background_task_status
                == mlrun.common.schemas.BackgroundTaskState.failed
            ):
                return None
        return event



    def _background_task_succeeded(
        self, background_task: mlrun.common.schemas.BackgroundTask
    ):
        logger.debug(
            "Checking model endpoint creation task status",
            task_name=self.server.model_endpoint_creation_task_name,
        )
        self._background_task_status = background_task.status.state
        if (
            background_task.status.state
            in mlrun.common.schemas.BackgroundTaskState.terminal_states()
        ):
            logger.debug(
                f"Model endpoint creation task completed with state {background_task.status.state}"
            )
        else:  # in progress
            logger.debug(
                f"Model endpoint creation task is still in progress with the current state: "
                f"{background_task.status.state}. Events will not be monitored for the next 15 seconds",
                name=self.name,
                background_task_check_timestamp=self._background_task_check_timestamp.isoformat(),
            )
        return (
            background_task.status.state
            == mlrun.common.schemas.BackgroundTaskState.succeeded
        )


class SamplingStep(storey.MapClass):
    def __init__(
        self,
        context,
        sampling_percentage: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        server: mlrun.serving.GraphServer = getattr(
            self.context, "_server", None
        ) or getattr(context, "server", None)
        self.sampling_percentage = sampling_percentage
        self.input_path: dict[str, str] = {}
        self.result_path : dict[str, str] = {}
        monitoring_data = {}
        for step in server.graph:
            if isinstance(step, ModelRunnerStep):
                monitoring_data.update(
                    step.class_args.get(
                        mlrun.common.schemas.ModelRunnerStepData.MONITORING_DATA, {}
                    )
                )
        for model in server.graph.model_endpoints_names:
            self.input_path[model] = monitoring_data.get(model, {}).get(
                mlrun.common.schemas.MonitoringData.INPUT_PATH
            )
            self.result_path[model] = monitoring_data.get(model, {}).get(
                mlrun.common.schemas.MonitoringData.RESULT_PATH
            )

    def do(self, event):
        if self.context is not None and self.context.is_mock:
            return event
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

            if isinstance(request["outputs"], list):
                event[mm_schemas.StreamProcessingEvent.REQUEST]["outputs"] = [
                    request["outputs"][i] for i in sampled_requests_indices
                ]
        event[mm_schemas.EventFieldType.SAMPLING_PERCENTAGE] = self.sampling_percentage
        event[mm_schemas.EventFieldType.EFFECTIVE_SAMPLE_COUNT] = len(
            event.get("inputs", [])
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
