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

import storey

import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.serving
from mlrun.serving.remote import RemoteStep
from mlrun.utils import logger


class MonitoringPreProcessor(storey.MapClass):  # TODO Roy is this necessary
    def __init__(
        self,
        model_endpoints: dict[str, dict[str, dict[str, str]]],
        labels: dict[str, dict[str, str]],
        context,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_endpoints = model_endpoints
        self.labels = labels
        self.context = context

    def _do(self, event):
        monitoring_event_list = []
        server: mlrun.serving.GraphServer = getattr(
            self.context, "_server", None
        ) or getattr(self.context, "server", None)
        model_runner_endpoints = self.model_endpoints.get(
            event.metadata.get("model_runner_name")
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
                            mm_schemas.StreamProcessingEvent.REQUEST: event.body[model],
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
                    mm_schemas.StreamProcessingEvent.REQUEST: event.body,
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
        super().__init__(url=self.context.get_run_db().get_base_api_url(path), method="GET" ,**kwargs)

    async def _process_event(self, event):
        if self._background_task_status == mlrun.common.schemas.BackgroundTaskState.running:
            response = await super()._process_event(event)
            background_task = mlrun.common.schemas.BackgroundTask(**response.json())
            self._background_task_check_timestamp = mlrun.utils.now_date()
            if self._background_task_succeeded(background_task):
                return event
            else:
                return None
        elif self._background_task_status == mlrun.common.schemas.BackgroundTaskState.failed:
            return None
        return event

    def _background_task_succeeded(self, background_task: mlrun.common.schemas.BackgroundTask):
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
        return background_task.status.state == mlrun.common.schemas.BackgroundTaskState.succeeded


class SamplingStep(storey.MapClass):
    def __init__(
        self, sampling_rate: float, inputs_path: str, result_path: str, **kwargs
    ):
        super().__init__(**kwargs)
        self.sampling_rate = sampling_rate
        self.inputs_path = inputs_path
        self.result_path = result_path

    def do(self, event):
        sampled_requests_indices = self._pick_random_requests(
            len(event[mm_schemas.StreamProcessingEvent.REQUEST].get(self.inputs_path)),
            self.sampling_rate,
        )

        event[mm_schemas.StreamProcessingEvent.REQUEST][self.inputs_path] = [
            event[mm_schemas.StreamProcessingEvent.REQUEST][self.inputs_path][i]
            for i in sampled_requests_indices
        ]

        if (
            event
            and self.result_path in event[mm_schemas.StreamProcessingEvent.REQUEST]
            and isinstance(
                event[mm_schemas.StreamProcessingEvent.REQUEST][self.result_path], list
            )
        ):
            event[mm_schemas.StreamProcessingEvent.REQUEST][self.result_path] = [
                event[mm_schemas.StreamProcessingEvent.REQUEST][self.result_path][i]
                for i in sampled_requests_indices
            ]
        event[mm_schemas.EventFieldType.SAMPLING_PERCENTAGE] = self.sampling_rate
        event[mm_schemas.EventFieldType.EFFECTIVE_SAMPLE_COUNT] = len(
            event.get(self.inputs_path)
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
