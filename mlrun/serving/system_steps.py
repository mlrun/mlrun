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
from datetime import timedelta
from typing import Any, Union

import storey

import mlrun
import mlrun.artifacts
import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.serving
from mlrun.serving import ModelRunnerStep
from mlrun.utils import logger


class MonitoringPreProcessor(storey.MapClass):
    def __init__(
        self,
        context,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.models_uri: dict[str:str] = {}
        self.context = context
        self.model_endpoints: dict[str, [dict[str, str]]] = {}
        self.labels: dict[str, dict[str, str]] = {}
        self.input_path: dict[str, str] = {}
        self.result_path: dict[str, str] = {}
        self.output_schema: dict[str, Union[list[str], str]] = {}

        server: mlrun.serving.GraphServer = getattr(
            self.context, "_server", None
        ) or getattr(self.context, "server", None)

        for step in server.graph.steps.values():
            if isinstance(step, ModelRunnerStep):
                self.model_endpoints[step.name] = {}
                monitoring_data = step.class_args.get(
                    mlrun.common.schemas.ModelRunnerStepData.MONITORING_DATA
                )
                for model, (model_class, _) in step.class_args.get(
                    mlrun.common.schemas.ModelRunnerStepData.MODELS, {}
                ).items():
                    self.model_endpoints[step.name][model] = {
                        mlrun.common.schemas.MonitoringData.MODEL_ENDPOINT_UID: monitoring_data[
                            model
                        ][mlrun.common.schemas.MonitoringData.MODEL_ENDPOINT_UID],
                        mm_schemas.StreamProcessingEvent.MODEL_CLASS: model_class,
                    }

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
        if self.output_schema.get(model) is None:
            if self.models_uri.get(model) is not None:
                _, model_spec, extra_datitems = mlrun.artifacts.get_model(
                    self.models_uri.get(model), ""
                )
                self.output_schema[model] = [
                    feature.name for feature in model_spec.outputs
                ]
            else:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "model uri or output schema must be provided in ModelRunnerStep.add_model"
                )
        return self.output_schema.get(model)

    def reconstruct_request_resp_field(self, event, model: str) -> tuple[dict[str, Any], dict[str, Any]]:
        output_schema = self.get_model_output_schema(model)
        logger.info("output schema retrieved", output_schema=output_schema)
        result_path = self.result_path.get(model) or ""
        input_path = self.input_path.get(model) or ""
        result = event.body.get(model) or event.body.get(result_path) or event.body
        result = (
            result.get(result_path, result) if (isinstance(result, dict)) else result
        )

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

        event_inputs = event.headers.get("inputs", {})
        event_inputs = (
            event_inputs.get(input_path, event_inputs)
            if isinstance(event_inputs, dict)
            else event_inputs
        )
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
        request = {"inputs": inputs, "id": getattr(event, "id", None)}
        resp = {"outputs": outputs}

        return request, resp

    def do(self, event):
        monitoring_event_list = []
        server: mlrun.serving.GraphServer = getattr(
            self.context, "_server", None
        ) or getattr(self.context, "server", None)
        model_runner_name = event.headers.get("model_runner_name", "")
        model_runner_endpoints = self.model_endpoints.get(model_runner_name)
        logger.info(
            "monitoring pre processor runs",
            event=event,
            model_endpoints=self.model_endpoints,
            metadata=event._metadata,
            headers=event.headers,
            model_runner_endpoints=model_runner_endpoints,
        )
        if len(model_runner_endpoints) > 1:
            for model in event.body.keys():
                if model in model_runner_endpoints:
                    request, resp = self.reconstruct_request_resp_field(event, model)
                    monitoring_event_list.append(
                        {
                            mm_schemas.StreamProcessingEvent.MODEL: model,
                            mm_schemas.StreamProcessingEvent.MODEL_CLASS: model_runner_endpoints[
                                model
                            ].get(mm_schemas.StreamProcessingEvent.MODEL_CLASS),
                            mm_schemas.StreamProcessingEvent.MICROSEC: event._metadata.get(
                                model, {}
                            ).get(mm_schemas.StreamProcessingEvent.MICROSEC),
                            mm_schemas.StreamProcessingEvent.WHEN: event._metadata.get(
                                model, {}
                            ).get(mm_schemas.StreamProcessingEvent.WHEN),
                            mm_schemas.StreamProcessingEvent.ENDPOINT_ID: model_runner_endpoints[
                                model
                            ].get(
                                mlrun.common.schemas.MonitoringData.MODEL_ENDPOINT_UID
                            ),
                            mm_schemas.StreamProcessingEvent.LABELS: self.labels[model],
                            mm_schemas.StreamProcessingEvent.FUNCTION_URI: server.function_uri,
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
        elif model_runner_endpoints:
            model = list(model_runner_endpoints.keys())[0]
            request, resp = self.reconstruct_request_resp_field(event, model)
            monitoring_event_list.append(
                {
                    mm_schemas.StreamProcessingEvent.MODEL: model,
                    mm_schemas.StreamProcessingEvent.MODEL_CLASS: model_runner_endpoints[
                        model
                    ].get(mm_schemas.StreamProcessingEvent.MODEL_CLASS),
                    mm_schemas.StreamProcessingEvent.MICROSEC: event._metadata[0].get(
                        mm_schemas.StreamProcessingEvent.MICROSEC
                    ),
                    mm_schemas.StreamProcessingEvent.WHEN: event._metadata[0].get(
                        mm_schemas.StreamProcessingEvent.WHEN
                    ),
                    mm_schemas.StreamProcessingEvent.ENDPOINT_ID: model_runner_endpoints[
                        model
                    ].get(mlrun.common.schemas.MonitoringData.MODEL_ENDPOINT_UID),
                    mm_schemas.StreamProcessingEvent.LABELS: self.labels[model],
                    mm_schemas.StreamProcessingEvent.FUNCTION_URI: server.function_uri,
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
        logger.info("monitoring pre processor ended", event=event)
        return event


class BackgroundTaskStatus(storey.MapClass):
    def __init__(self, context, **kwargs):
        self.context = context
        self.server: mlrun.serving.GraphServer = getattr(
            self.context, "_server", None
        ) or getattr(self.context, "server", None)
        self._background_task_check_timestamp = None
        self._background_task_status = mlrun.common.schemas.BackgroundTaskState.running
        super().__init__(**kwargs)

    def do(self, event):
        if (
            self._background_task_status
            == mlrun.common.schemas.BackgroundTaskState.running
            and (
                self._background_task_check_timestamp is None
                or mlrun.utils.now_date() - self._background_task_check_timestamp
                >= timedelta(seconds=15)
            )
        ):
            background_task = mlrun.get_run_db().get_project_background_task(
                self.server.project, self.server.model_endpoint_creation_task_name
            )
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
        logger.info(
            "Checking model endpoint creation task status",
            task_name=self.server.model_endpoint_creation_task_name,
        )
        self._background_task_status = background_task.status.state
        if (
            background_task.status.state
            in mlrun.common.schemas.BackgroundTaskState.terminal_states()
        ):
            logger.info(
                f"Model endpoint creation task completed with state {background_task.status.state}"
            )
        else:  # in progress
            logger.info(
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
        self.context = context
        server: mlrun.serving.GraphServer = getattr(
            context, "_server", None
        ) or getattr(context, "server", None)
        self.sampling_percentage = sampling_percentage
        self.input_path: dict[str, str] = {}
        self.result_path: dict[str, str] = {}
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
        logger.info(
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

            if isinstance(request["outputs"], list):
                event[mm_schemas.StreamProcessingEvent.RESPONSE]["outputs"] = [
                    request["outputs"][i] for i in sampled_requests_indices
                ]
        event[mm_schemas.EventFieldType.SAMPLING_PERCENTAGE] = self.sampling_percentage
        event[mm_schemas.EventFieldType.EFFECTIVE_SAMPLE_COUNT] = len(
            event.get(mm_schemas.StreamProcessingEvent.REQUEST,{}).get("inputs", [])
        )
        logger.info("sampling step ended", event=event)
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
