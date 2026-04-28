# Copyright 2026 Iguazio
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
import time
from datetime import UTC, datetime, timedelta

import deepdiff
import pytest

import mlrun
import mlrun.common.schemas
import mlrun.common.schemas.alert as alert_objects
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.model_monitoring.api
import tests.system.common.helpers.notifications as notification_helpers
from mlrun import mlconf
from mlrun.datastore import get_stream_pusher
from mlrun.datastore.datastore_profile import (
    register_temporary_client_datastore_profile,
)
from mlrun.model_monitoring.helpers import get_stream_path
from tests.system.base import TestMLRunSystem
from tests.system.model_monitoring import TestMLRunSystemModelMonitoring


@TestMLRunSystem.skip_test_if_env_not_configured
class TestWriterLag(TestMLRunSystemModelMonitoring):
    """System tests for model monitoring writer lag detection and alerting."""

    # Set image to "<repo>/mlrun:<tag>" for local testing
    image: str | None = None

    @pytest.mark.model_monitoring
    def test_writer_lag_alert(self):
        """
        Validate that a lag event is generated and triggers an alert when the
        writer processes events with old inference timestamps.

        Flow:
        1. Enable model monitoring with a low lag threshold.
        2. Configure a lag alert using set_model_monitoring_lag_alert().
        3. Push events with old END_INFER_TIME through the writer stream.
        4. Verify that the notification is sent and alert activation is created.
        """
        self.set_mm_credentials()
        register_temporary_client_datastore_profile(self.mm_stream_profile)

        lag_threshold = 5
        lag_event_cooldown = 1
        self.project.enable_model_monitoring(
            image=self.image or mlrun.mlconf.function_defaults.image_by_kind.job,
            lag_threshold=lag_threshold,
            lag_event_cooldown=lag_event_cooldown,
        )

        nuclio_function_url = notification_helpers.deploy_notification_nuclio(
            self.project, self.image
        )

        notification_data = "writer lag detected"
        notification = mlrun.common.schemas.Notification(
            kind="webhook",
            name="lag-notif",
            params={
                "url": nuclio_function_url,
                "override_body": {
                    "operation": "add",
                    "data": notification_data,
                },
            },
        )
        self.project.set_model_monitoring_lag_alert(
            notifications=notification,
        )

        model_endpoint = mlrun.model_monitoring.api.get_or_create_model_endpoint(
            project=self.project.metadata.name,
            model_endpoint_name="test-lag-endpoint",
            context=mlrun.get_or_create_ctx("demo"),
        )

        writer = self.project.get_function(
            key=mm_constants.MonitoringFunctionNames.WRITER
        )
        writer._wait_for_function_deployment(db=writer._get_db())

        stream_uri = get_stream_path(
            project=self.project.metadata.name,
            function_name=mm_constants.MonitoringFunctionNames.WRITER,
            profile=self.mm_stream_profile,
        )
        output_stream = get_stream_pusher(stream_uri)

        old_time = (datetime.now(tz=UTC) - timedelta(hours=1)).isoformat()
        result_name = (
            mm_constants.HistogramDataDriftApplicationConstants.GENERAL_RESULT_NAME
        )
        output_stream.push(
            {
                mm_constants.WriterEvent.ENDPOINT_ID: model_endpoint.metadata.uid,
                mm_constants.WriterEvent.ENDPOINT_NAME: model_endpoint.metadata.name,
                mm_constants.WriterEvent.APPLICATION_NAME: mm_constants.HistogramDataDriftApplicationConstants.NAME,
                mm_constants.WriterEvent.START_INFER_TIME: old_time,
                mm_constants.WriterEvent.END_INFER_TIME: old_time,
                mm_constants.WriterEvent.EVENT_KIND: "result",
                mm_constants.WriterEvent.DATA: json.dumps(
                    {
                        mm_constants.ResultData.RESULT_NAME: result_name,
                        mm_constants.ResultData.RESULT_KIND: mm_constants.ResultKindApp.data_drift.value,
                        mm_constants.ResultData.RESULT_VALUE: 0.1,
                        mm_constants.ResultData.RESULT_STATUS: mm_constants.ResultStatusApp.no_detection.value,
                        mm_constants.ResultData.RESULT_EXTRA_DATA: json.dumps({}),
                    }
                ),
            }
        )

        time.sleep(
            5 + mlconf.model_endpoint_monitoring.writer_graph.flush_after_seconds
        )

        def _check_notification():
            sent = list(
                notification_helpers.get_notifications_from_nuclio_and_reset_notification_cache(
                    nuclio_function_url
                )
            )
            assert deepdiff.DeepDiff(sent, [notification_data], ignore_order=True) == {}

        self.wait_for_condition(
            _check_notification,
            timeout=30,
            retry_interval=3,
            condition_description="lag notification received",
        )

        activations = mlrun.get_run_db().list_alert_activations(
            project=self.project_name,
            entity_kind=alert_objects.EventEntityKind.MODEL_MONITORING_INFRA,
            event_kind=alert_objects.EventKind.MODEL_MONITORING_LAG_DETECTED,
        )
        assert len(activations) >= 1
        assert activations[0].entity_id.startswith(f"{self.project_name}.writer.")

        self.project.delete_model_monitoring_lag_alert()
