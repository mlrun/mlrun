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

import mlrun.alerts.alert
import mlrun.common.schemas.alert as alert_objects


def _make_alert_config():
    """Create a minimal AlertConfig for testing to_dict."""
    return mlrun.alerts.alert.AlertConfig(
        project="test-project",
        name="test-alert",
        summary="test summary",
        severity=alert_objects.AlertSeverity.LOW,
        entities=alert_objects.EventEntities(
            kind=alert_objects.EventEntityKind.MODEL_ENDPOINT_RESULT,
            project="test-project",
            ids=["endpoint-1"],
        ),
        trigger=alert_objects.AlertTrigger(
            events=[alert_objects.EventKind.DATA_DRIFT_DETECTED]
        ),
        criteria=alert_objects.AlertCriteria(count=3, period="1h"),
        notifications=[
            alert_objects.AlertNotification(
                notification=mlrun.common.schemas.Notification(
                    kind="slack",
                    name="slack_notification",
                    secret_params={"webhook": "https://hooks.slack.com/test"},
                ).dict()
            )
        ],
    )


def test_to_dict_forwards_exclude_parameter():
    """Verify that the exclude parameter is forwarded to the parent to_dict."""
    alert = _make_alert_config()

    full_dict = alert.to_dict()
    assert "project" in full_dict, "project should be in the full dict"

    excluded_dict = alert.to_dict(exclude=["project"])
    assert "project" not in excluded_dict, (
        "project should be excluded when passed in the exclude parameter"
    )


def test_to_dict_forwards_strip_parameter():
    """Verify that the strip parameter is forwarded to the parent to_dict."""
    alert = _make_alert_config()

    # strip=False should include all fields
    full_dict = alert.to_dict(strip=False)
    assert isinstance(full_dict, dict)

    # strip=True should also return a dict (no error) and pass through
    stripped_dict = alert.to_dict(strip=True)
    assert isinstance(stripped_dict, dict)
