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

import pytest

import mlrun.common.schemas
import mlrun.common.schemas.alert as alert_objects
from mlrun.common.schemas import (
    AlertNotification,
    Event,
    Notification,
    NotificationState,
    NotificationSummary,
)

import framework.utils.notifications.notification_pusher


@pytest.mark.parametrize(
    "notifications, expected_states",
    [
        # Case 1: All notifications of one kind fail
        (
            [
                AlertNotification(
                    notification=Notification(
                        name="test", kind="slack", reason="Error 1"
                    )
                ),
                AlertNotification(
                    notification=Notification(
                        name="test2", kind="slack", reason="Error 2"
                    )
                ),
            ],
            [
                NotificationState(
                    kind="slack",
                    err="All slack notifications failed. Errors: Error 1, Error 2",
                    summary=NotificationSummary(failed=2, succeeded=0),
                )
            ],
        ),
        # Case 2: Some notifications of one kind fail, some succeed
        (
            [
                AlertNotification(
                    notification=Notification(
                        name="test", kind="slack", reason="Error 1"
                    )
                ),
                AlertNotification(
                    notification=Notification(name="test2", kind="slack", reason=None)
                ),
            ],
            [
                NotificationState(
                    kind="slack",
                    err="Some slack notifications failed. Errors: Error 1",
                    summary=NotificationSummary(failed=1, succeeded=1),
                )
            ],
        ),
        # Case 3: All notifications of one kind succeed
        (
            [
                AlertNotification(
                    notification=Notification(name="test", kind="slack", reason=None),
                ),
                AlertNotification(
                    notification=Notification(name="test2", kind="slack", reason=None)
                ),
            ],
            [
                NotificationState(
                    kind="slack",
                    err="",
                    summary=NotificationSummary(failed=0, succeeded=2),
                )
            ],
        ),
        # Case 4: Mixed kinds, with some failures and successes
        (
            [
                AlertNotification(
                    notification=Notification(
                        name="test", kind="slack", reason="Error 1"
                    )
                ),
                AlertNotification(
                    notification=Notification(name="test2", kind="slack", reason=None)
                ),
                AlertNotification(
                    notification=Notification(
                        name="test3", kind="git", reason="Error 2"
                    )
                ),
                AlertNotification(
                    notification=Notification(
                        name="test4", kind="git", reason="Error 3"
                    )
                ),
            ],
            [
                NotificationState(
                    kind="slack",
                    err="Some slack notifications failed. Errors: Error 1",
                    summary=NotificationSummary(failed=1, succeeded=1),
                ),
                NotificationState(
                    kind="git",
                    err="All git notifications failed. Errors: Error 2, Error 3",
                    summary=NotificationSummary(failed=2, succeeded=0),
                ),
            ],
        ),
    ],
)
def test_prepare_notifications_states(notifications, expected_states):
    result = framework.utils.notifications.notification_pusher.AlertNotificationPusher._prepare_notification_states(
        notifications
    )

    # normalize the error strings for comparison
    result_dicts = [_normalize_error_messages(state.__dict__) for state in result]

    assert result_dicts == expected_states


def _normalize_error_messages(state):
    # normalize the 'err' field of a NotificationState dictionary by sorting error messages.
    if "Errors:" in state["err"]:
        prefix, errors = state["err"].split("Errors: ")
        state["err"] = f"{prefix}Errors: {', '.join(sorted(errors.split(', ')))}"
    return state


class TestPrepareNotificationArgs:
    """Tests for AlertNotificationPusher._prepare_notification_args (ML-12248)."""

    def test_message_resolves_all_placeholders(self):
        """When no custom notification message is set, the fallback message
        must resolve all template placeholders ({{project}}, {{name}}, {{entity}})."""
        alert, event_data = self._create_alert_and_event(
            project="lag-detection-tutorial",
            name="my-lag-alert",
            summary="Alert {{name}} in project {{project}}, entity {{entity}}.",
            entity_id="lag-detection-tutorial.writer.0",
        )
        notification_object = Notification(name="n", kind="slack")

        message, _ = (
            framework.utils.notifications.notification_pusher.AlertNotificationPusher._prepare_notification_args(
                alert, notification_object, event_data
            )
        )

        assert message == (
            "Alert my-lag-alert in project lag-detection-tutorial, "
            "entity lag-detection-tutorial.writer.0."
        )

    def test_custom_notification_message_used_as_is(self):
        """When the notification has its own message, use it directly."""
        alert, event_data = self._create_alert_and_event()
        notification_object = Notification(name="n", kind="slack", message="custom msg")

        message, _ = (
            framework.utils.notifications.notification_pusher.AlertNotificationPusher._prepare_notification_args(
                alert, notification_object, event_data
            )
        )

        assert message == ": custom msg"

    @staticmethod
    def _create_alert_and_event(
        project="my-project",
        name="monitoring-lag-detected",
        summary="Lag in project {{project}}.",
        entity_id="my-project.writer.0",
    ):
        """Build a minimal AlertConfig and Event for _prepare_notification_args tests."""
        entity_kind = alert_objects.EventEntityKind.MODEL_MONITORING_INFRA
        alert = mlrun.common.schemas.AlertConfig(
            project=project,
            name=name,
            summary=summary,
            severity=alert_objects.AlertSeverity.MEDIUM,
            entities=alert_objects.EventEntities(
                kind=entity_kind, project=project, ids=[entity_id]
            ),
            trigger=alert_objects.AlertTrigger(
                events=[alert_objects.EventKind.MODEL_MONITORING_LAG_DETECTED]
            ),
            notifications=[
                AlertNotification(notification=Notification(name="n", kind="slack"))
            ],
        )
        event_data = Event(
            kind=alert_objects.EventKind.MODEL_MONITORING_LAG_DETECTED,
            entity=alert_objects.EventEntities(
                kind=entity_kind, project=project, ids=[entity_id]
            ),
        )
        return alert, event_data
