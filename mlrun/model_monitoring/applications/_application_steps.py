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

import collections
import traceback
from collections import OrderedDict
from datetime import datetime
from typing import Any, Union

import mlrun.common.schemas
import mlrun.common.schemas.alert as alert_objects
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.errors
import mlrun.model_monitoring.helpers
import mlrun.platforms.iguazio
from mlrun.serving import GraphContext
from mlrun.serving.utils import StepToDict
from mlrun.utils import logger

from .base import _serialize_context_and_result
from .context import MonitoringApplicationContext
from .results import (
    ModelMonitoringApplicationMetric,
    ModelMonitoringApplicationResult,
    _ModelMonitoringApplicationStats,
)


class _PushToMonitoringWriter(StepToDict):
    kind = "monitoring_application_stream_pusher"

    def __init__(self, project: str) -> None:
        """
        Class for pushing application results to the monitoring writer stream.

        :param project: Project name.
        """
        self.project = project
        self._output_stream = None

    def do(
        self,
        event: tuple[
            list[
                ModelMonitoringApplicationResult
                | ModelMonitoringApplicationMetric
                | _ModelMonitoringApplicationStats,
            ],
            MonitoringApplicationContext,
        ],
    ) -> None:
        """
        Push application results to the monitoring writer stream.

        :param event: Monitoring result(s) to push and the original event from the controller.
        """
        application_results, application_context = event

        writer_events = [
            _serialize_context_and_result(context=application_context, result=result)
            for result in application_results
        ]

        logger.debug("Pushing data to output stream", writer_events=str(writer_events))
        self.output_stream.push(
            writer_events, partition_key=application_context.endpoint_id
        )
        logger.debug("Pushed data to output stream successfully")

    @property
    def output_stream(
        self,
    ) -> Union[
        mlrun.platforms.iguazio.OutputStream, mlrun.platforms.iguazio.KafkaOutputStream
    ]:
        if self._output_stream is None:
            self._output_stream = mlrun.model_monitoring.helpers.get_output_stream(
                project=self.project,
                function_name=mm_constants.MonitoringFunctionNames.WRITER,
            )
        return self._output_stream


class _PrepareOTelEvent(StepToDict):
    """Adapter step between an MM application's output and
    :class:`mlrun.serving.OTelMetricsExporter`.

    The application emits a ``(results, MonitoringApplicationContext)`` tuple
    where ``results`` mixes :class:`ModelMonitoringApplicationResult` (drift
    detection / scoring) and :class:`ModelMonitoringApplicationMetric`
    (free-form numeric metrics). The OTel exporter expects events shaped as::

        {
            "metrics": [
                {
                    "metric_name": "...",
                    "value": 0.42,
                    "type": "gauge",
                    "attributes": {...},
                },
                ...,
            ]
        }

    Naming convention — a single fixed instrument name per family, with the
    specific result/metric name carried as an attribute rather than encoded
    in the instrument name:
        * ``mlrun.model_monitoring.result`` for results, with the result name
          in the ``result.name`` attribute
        * ``mlrun.model_monitoring.metric`` for metrics, with the metric name
          in the ``metric.name`` attribute

    Both are emitted as gauges — raw value flows through, and
    ``result.status`` is the normalized signal for alerting / dashboarding.

    Attributes (shared from MonitoringApplicationContext):
        * ``project``
        * ``app.name``
        * ``function.name``
        * ``endpoint.uid``
        * ``endpoint.name``

    Result-only attributes:
        * ``result.name``   (e.g. ``"general_drift"``)
        * ``result.kind``   (e.g. ``"data_drift"``)
        * ``result.status`` (e.g. ``"detected"``)

    Metric-only attributes:
        * ``metric.name``   (e.g. ``"hellinger"``)

    Attribute key names live in
    :class:`mlrun.common.schemas.model_monitoring.constants.OTelMonitoringAttribute`
    so downstream consumers (alerts, dashboards) have one canonical
    reference rather than scattered string literals.

    ``_ModelMonitoringApplicationStats`` entries (histogram drift app
    internal stats) are skipped — they don't map cleanly onto an OTel
    instrument.
    """

    kind = "monitoring_otel_event_preparer"

    def do(
        self,
        event: tuple[
            list[
                Union[
                    ModelMonitoringApplicationResult,
                    ModelMonitoringApplicationMetric,
                    _ModelMonitoringApplicationStats,
                ]
            ],
            MonitoringApplicationContext,
        ],
    ) -> dict[str, Any]:
        results, ctx = event
        attr = mm_constants.OTelMonitoringAttribute
        metric_name_enum = mm_constants.OTelMonitoringMetricName
        base_attributes = {
            attr.PROJECT.value: ctx.project_name,
            attr.APP_NAME.value: ctx.application_name,
            attr.FUNCTION_NAME.value: ctx.model_endpoint.spec.function_name,
            attr.ENDPOINT_UID.value: ctx.endpoint_id,
            attr.ENDPOINT_NAME.value: ctx.endpoint_name,
        }
        # Strip None-valued attributes — the OTel SDK rejects them with a
        # warning on every record() call.
        base_attributes = {k: v for k, v in base_attributes.items() if v is not None}

        metrics: list[dict[str, Any]] = []
        for entry in results:
            if isinstance(entry, _ModelMonitoringApplicationStats):
                # Histogram stats are a side payload, not a metric value.
                continue
            attributes = dict(base_attributes)
            if isinstance(entry, ModelMonitoringApplicationResult):
                metric_name = metric_name_enum.RESULT.value
                attributes[attr.RESULT_NAME.value] = entry.name
                attributes[attr.RESULT_KIND.value] = entry.kind.name
                attributes[attr.RESULT_STATUS.value] = entry.status.name
            elif isinstance(entry, ModelMonitoringApplicationMetric):
                metric_name = metric_name_enum.METRIC.value
                attributes[attr.METRIC_NAME.value] = entry.name
            else:
                logger.warning(
                    "Skipping unexpected entry type in OTel event preparer",
                    entry_type=type(entry).__name__,
                )
                continue
            metrics.append(
                {
                    "metric_name": metric_name,
                    "value": float(entry.value),
                    "type": "gauge",
                    "attributes": attributes,
                }
            )
        return {"metrics": metrics, "endpoint_id": ctx.endpoint_id}


class _PrepareMonitoringEvent(StepToDict):
    MAX_MODEL_ENDPOINTS: int = 1500

    def __init__(self, context: GraphContext, application_name: str) -> None:
        """
        Class for preparing the application event for the application step.

        :param application_name: Application name.
        """
        self.graph_context = context
        _ = self.graph_context.project_obj  # Ensure project exists
        self.application_name = application_name
        self.model_endpoints: OrderedDict[str, mlrun.common.schemas.ModelEndpoint] = (
            collections.OrderedDict()
        )
        self.feature_sets: dict[str, mlrun.common.schemas.FeatureSet] = {}

    def do(self, event: dict[str, Any]) -> MonitoringApplicationContext:
        """
        Prepare the application event for the application step.

        :param event: Application event.
        :return: Application context.
        """
        endpoint_id = event.get(mm_constants.ApplicationEvent.ENDPOINT_ID)
        endpoint_updated = datetime.fromisoformat(
            event.get(mm_constants.ApplicationEvent.ENDPOINT_UPDATED)
        )
        if (
            endpoint_id in self.model_endpoints
            and endpoint_updated != self.model_endpoints[endpoint_id].metadata.updated
        ):
            logger.debug(
                "Updated endpoint removing endpoint from cash",
                new_updated=endpoint_updated.isoformat(),
                old_updated=self.model_endpoints[
                    endpoint_id
                ].metadata.updated.isoformat(),
            )
            self.model_endpoints.pop(endpoint_id)

        application_context = MonitoringApplicationContext._from_graph_ctx(
            application_name=self.application_name,
            event=event,
            model_endpoint_dict=self.model_endpoints,
            graph_context=self.graph_context,
            feature_sets_dict=self.feature_sets,
        )

        self.model_endpoints.setdefault(
            application_context.endpoint_id, application_context.model_endpoint
        )
        self.feature_sets.setdefault(
            application_context.endpoint_id, application_context.feature_set
        )
        # every used endpoint goes to first location allowing to pop last used:
        self.model_endpoints.move_to_end(application_context.endpoint_id, last=False)
        if len(self.model_endpoints) > self.MAX_MODEL_ENDPOINTS:
            removed_endpoint_id, _ = self.model_endpoints.popitem(
                last=True
            )  # Removing the LRU endpoint
            self.feature_sets.pop(removed_endpoint_id, None)
            logger.debug(
                "Exceeded maximum number of model endpoints removing the LRU from cash",
                endpoint_id=removed_endpoint_id,
            )

        return application_context


class _ApplicationErrorHandler(StepToDict):
    def __init__(
        self,
        project: str,
        name: str | None = None,
        application_name: str | None = None,
        user_step_name: str | None = None,
    ):
        """Single error-handler shared across the MM serving graph.

        Storey sets ``event.origin_state`` to the failing step's name.
        Failures from the user step (``user_step_name``) produce an alert
        with entity id ``<project>_<app>``; failures from any other step
        produce ``<project>_<app>_<origin_state>`` so alert configs can
        route by failing step.

        :param project: Project name; goes on the alert event entity.
        :param name: Step name in the graph (default
            ``"ApplicationErrorHandler"``).
        :param application_name: Application name; included on every alert.
        :param user_step_name: The user app step name; failures whose
            ``origin_state`` matches this are tagged as the main-app branch
            (no suffix). Anything else is treated as an auxiliary step.
        """
        self.project = project
        self.name = name or "ApplicationErrorHandler"
        self.application_name = application_name
        self.user_step_name = user_step_name

    def do(self, event):
        """
        Handle a failure from any step in the MM serving graph and emit
        an ``MM_APP_FAILED`` alert tagged with the failing step's name.

        :param event: Application event (storey adds ``origin_state``
            and ``error`` before routing to this step).
        """
        origin_state = getattr(event, "origin_state", None)
        if isinstance(event.body, tuple) and len(event.body) == 2:
            endpoint_id = event.body[1].endpoint_id
        elif isinstance(event.body, dict):
            endpoint_id = event.body["endpoint_id"]
        elif isinstance(event.body, MonitoringApplicationContext):
            endpoint_id = event.body.endpoint_id
        else:
            endpoint_id = None
        error_data = {
            "Endpoint ID": endpoint_id,
            "Application Class": self.application_name,
            "Error": "".join(
                traceback.format_exception(
                    None, value=event.error, tb=event.error.__traceback__
                )
            ),
            "Timestamp": event.timestamp,
            "Step Name": origin_state,
        }
        logger.error("Error in application", **error_data)
        error_data["Error"] = event.error

        entity_id = f"{self.project}_{self.application_name}"
        if origin_state != self.user_step_name:
            entity_id += f"_{origin_state}"

        event_data = alert_objects.Event(
            kind=alert_objects.EventKind.MM_APP_FAILED,
            entity=alert_objects.EventEntities(
                kind=alert_objects.EventEntityKind.MODEL_MONITORING_APPLICATION,
                project=self.project,
                ids=[entity_id],
            ),
            value_dict=error_data,
        )

        mlrun.get_run_db().generate_event(
            name=alert_objects.EventKind.MM_APP_FAILED, event_data=event_data
        )
        logger.info("Event generated successfully")
        return event
