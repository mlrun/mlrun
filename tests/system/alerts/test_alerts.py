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
#
import json
import time
import typing

import pytest

import mlrun
import mlrun.alerts
import mlrun.common.schemas.alert as alert_constants
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.model_monitoring.api
import tests.system.common.helpers.notifications as notification_helpers
from mlrun.datastore import get_stream_pusher
from mlrun.model_monitoring.helpers import get_stream_path
from tests.system.base import TestMLRunSystem


@TestMLRunSystem.skip_test_if_env_not_configured
class TestAlerts(TestMLRunSystem):
    project_name = "alerts-test-project"

    # Set image to "<repo>/mlrun:<tag>" for local testing
    image: typing.Optional[str] = None

    def test_job_failure_alert(self):
        """
        validate that an alert is sent in case a job fails
        """
        self.project.set_function(
            name="test-func",
            func="assets/function.py",
            handler="handler",
            image="mlrun/mlrun" if self.image is None else self.image,
            kind="job",
        )

        # nuclio function for storing notifications, to validate that alert notifications were sent on the failed job
        nuclio_function_url = notification_helpers.deploy_notification_nuclio(
            self.project, self.image
        )

        # create an alert with webhook notification
        alert_name = "failure_webhook"
        alert_summary = "Job failed"
        notifications = self._generate_failure_notifications(nuclio_function_url)
        self._create_alert_config(
            self.project_name,
            alert_name,
            alert_constants.EventEntityKind.JOB,
            alert_summary,
            alert_constants.EventKind.FAILED,
            notifications,
        )

        with pytest.raises(Exception):
            self.project.run_function("test-func")

        # in order to trigger the periodic monitor runs function, to detect the failed run and send an event on it
        time.sleep(35)

        # Validate that the notifications was sent on the failed job
        expected_notifications = ["notification failure"]
        self._validate_notifications_on_nuclio(
            nuclio_function_url, expected_notifications
        )

    def test_drift_detection_alert(self):
        """
        validate that an alert is sent in case of a model drift detection
        """
        # enable model monitoring - deploy writer function
        self.project.enable_model_monitoring(image=self.image or "mlrun/mlrun")
        # deploy nuclio func for storing notifications, to validate an alert notifications were sent on drift detection
        nuclio_function_url = notification_helpers.deploy_notification_nuclio(
            self.project, self.image
        )

        # create an alert with two webhook notifications
        alert_name = "drift_webhook"
        alert_summary = "Model is drifting"
        notifications = self._generate_drift_notifications(nuclio_function_url)
        self._create_alert_config(
            self.project_name,
            alert_name,
            alert_constants.EventEntityKind.MODEL,
            alert_summary,
            alert_constants.EventKind.DRIFT_DETECTED,
            notifications,
        )

        # waits for the writer function to be deployed
        writer = self.project.get_function(
            key=mm_constants.MonitoringFunctionNames.WRITER
        )
        writer._wait_for_function_deployment(db=writer._get_db())

        endpoint_id = "demo-endpoint"
        mlrun.model_monitoring.api.get_or_create_model_endpoint(
            project=self.project.metadata.name,
            endpoint_id=endpoint_id,
            context=mlrun.get_or_create_ctx("demo"),
        )
        stream_uri = get_stream_path(
            project=self.project.metadata.name,
            function_name=mm_constants.MonitoringFunctionNames.WRITER,
        )
        output_stream = get_stream_pusher(
            stream_uri,
        )

        data = {
            mm_constants.WriterEvent.ENDPOINT_ID: endpoint_id,
            mm_constants.WriterEvent.APPLICATION_NAME: mm_constants.HistogramDataDriftApplicationConstants.NAME,
            mm_constants.WriterEvent.START_INFER_TIME: "2023-09-11T12:00:00",
            mm_constants.WriterEvent.END_INFER_TIME: "2023-09-11T12:01:00",
            mm_constants.WriterEvent.EVENT_KIND: "result",
            mm_constants.WriterEvent.DATA: json.dumps(
                {
                    mm_constants.ResultData.RESULT_NAME: "data_drift_test",
                    mm_constants.ResultData.RESULT_KIND: mm_constants.ResultKindApp.data_drift.value,
                    mm_constants.ResultData.RESULT_VALUE: 0.5,
                    mm_constants.ResultData.RESULT_STATUS: mm_constants.ResultStatusApp.detected.value,
                    mm_constants.ResultData.RESULT_EXTRA_DATA: {"threshold": 0.3},
                    mm_constants.ResultData.CURRENT_STATS: "",
                }
            ),
        }
        output_stream.push([data])

        # wait for the nuclio function to check for the stream inputs
        time.sleep(10)

        # Validate that the notifications were sent on the drift
        expected_notifications = ["first drift", "second drift"]
        self._validate_notifications_on_nuclio(
            nuclio_function_url, expected_notifications
        )

    @staticmethod
    def _generate_failure_notifications(nuclio_function_url):
        notification = mlrun.common.schemas.Notification(
            kind="webhook",
            name="failure",
            message="job failed !",
            severity="warning",
            when=["now"],
            condition="failed",
            params={
                "url": nuclio_function_url,
                "override_body": {
                    "operation": "add",
                    "data": "notification failure",
                },
            },
            secret_params={
                "webhook": "some-webhook",
            },
        )
        return [alert_constants.AlertNotification(notification=notification)]

    @staticmethod
    def _generate_drift_notifications(nuclio_function_url):
        first_notification = mlrun.common.schemas.Notification(
            kind="webhook",
            name="drift",
            message="A drift was detected",
            severity="warning",
            when=["now"],
            condition="failed",
            params={
                "url": nuclio_function_url,
                "override_body": {
                    "operation": "add",
                    "data": "first drift",
                },
            },
            secret_params={
                "webhook": "some-webhook",
            },
        )
        second_notification = mlrun.common.schemas.Notification(
            kind="webhook",
            name="drift2",
            message="A drift was detected",
            severity="warning",
            when=["now"],
            condition="failed",
            params={
                "url": nuclio_function_url,
                "override_body": {
                    "operation": "add",
                    "data": "second drift",
                },
            },
            secret_params={
                "webhook": "some-webhook",
            },
        )
        return [
            alert_constants.AlertNotification(notification=first_notification),
            alert_constants.AlertNotification(notification=second_notification),
        ]

    @staticmethod
    def _create_alert_config(
        project,
        name,
        entity_kind,
        summary,
        event_name,
        notifications,
        criteria=None,
    ):
        alert_data = mlrun.alerts.alert.AlertConfig(
            project=project,
            name=name,
            summary=summary,
            severity=alert_constants.AlertSeverity.LOW,
            entities=alert_constants.EventEntities(
                kind=entity_kind, project=project, ids=["*"]
            ),
            trigger=alert_constants.AlertTrigger(events=[event_name]),
            criteria=criteria,
            notifications=notifications,
        )

        mlrun.get_run_db().store_alert_config(name, alert_data)

    @staticmethod
    def _validate_notifications_on_nuclio(nuclio_function_url, expected_notifications):
        for notification in notification_helpers.get_notifications_from_nuclio_and_reset_notification_cache(
            nuclio_function_url
        ):
            assert notification in expected_notifications
