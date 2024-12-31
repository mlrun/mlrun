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

import copy
import hashlib
import json
import unittest.mock

import mlrun.common.runtimes.constants as runtimes_constants
import mlrun.common.schemas.notification

import framework.constants
import framework.utils.notifications
import services.api.crud


def test_notification_params_masking_on_run(monkeypatch):
    def _store_project_secrets(*args, **kwargs):
        pass

    monkeypatch.setattr(
        services.api.crud.Secrets, "store_project_secrets", _store_project_secrets
    )
    params = {"sensitive": "sensitive-value"}
    params_hash = hashlib.sha224(
        json.dumps(params, sort_keys=True).encode("utf-8")
    ).hexdigest()
    run_uid = "test-run-uid"
    run = {
        "metadata": {"uid": run_uid, "project": "test-project"},
        "spec": {"notifications": [{"when": "completed", "secret_params": params}]},
    }
    framework.utils.notifications.mask_notification_params_on_task(
        run, framework.constants.MaskOperations.CONCEAL
    )
    assert "sensitive" not in run["spec"]["notifications"][0]["secret_params"]
    assert "secret" in run["spec"]["notifications"][0]["secret_params"]
    assert (
        run["spec"]["notifications"][0]["secret_params"]["secret"]
        == f"mlrun.notifications.{params_hash}"
    )


def test_notification_params_unmasking_on_run(monkeypatch):
    secret_value = {"sensitive": "sensitive-value"}
    run = {
        "metadata": {"uid": "test-run-uid", "project": "test-project"},
        "spec": {
            "notifications": [
                {
                    "name": "test-notification",
                    "when": ["completed"],
                    "secret_params": {"secret": "secret-name"},
                },
            ],
        },
    }

    def _get_valid_project_secret(*args, **kwargs):
        return json.dumps(secret_value)

    def _get_invalid_project_secret(*args, **kwargs):
        return json.dumps(secret_value)[:5]

    db_mock = unittest.mock.Mock()
    db_session_mock = unittest.mock.Mock()

    monkeypatch.setattr(
        services.api.crud.Secrets, "get_project_secret", _get_valid_project_secret
    )

    unmasked_run = (
        framework.utils.notifications.unmask_notification_params_secret_on_task(
            db_mock, db_session_mock, copy.deepcopy(run)
        )
    )
    assert "sensitive" in unmasked_run.spec.notifications[0].secret_params
    assert "secret" not in unmasked_run.spec.notifications[0].secret_params
    assert unmasked_run.spec.notifications[0].secret_params == secret_value

    monkeypatch.setattr(
        services.api.crud.Secrets, "get_project_secret", _get_invalid_project_secret
    )
    unmasked_run = (
        framework.utils.notifications.unmask_notification_params_secret_on_task(
            db_mock, db_session_mock, copy.deepcopy(run)
        )
    )
    assert len(unmasked_run.spec.notifications) == 0
    db_mock.store_run_notifications.assert_called_once()
    args, _ = db_mock.store_run_notifications.call_args
    assert args[1][0].status == mlrun.common.schemas.NotificationStatus.ERROR


class TestKFPNotificationPusher:
    def test_push(self):
        project = "test-project"
        run_id = "test-run-id"
        notifications = [
            mlrun.common.schemas.Notification(
                name="webhook-notification",
                kind=mlrun.common.schemas.notification.NotificationKind.webhook,
                message="test-message",
                severity=mlrun.common.schemas.notification.NotificationSeverity.INFO,
                when=[runtimes_constants.RunStates.completed],
            ),
            mlrun.common.schemas.Notification(
                name="mail-notification",
                kind=mlrun.common.schemas.notification.NotificationKind.mail,
                message="test-message",
                severity=mlrun.common.schemas.notification.NotificationSeverity.INFO,
                when=[runtimes_constants.RunStates.completed],
            ),
            mlrun.common.schemas.Notification(
                name="console-notification",
                kind=mlrun.common.schemas.notification.NotificationKind.console,
                message="test-message",
                severity=mlrun.common.schemas.notification.NotificationSeverity.INFO,
                when=[runtimes_constants.RunStates.completed],
            ),
        ]

        kfp_notification_pusher = (
            framework.utils.notifications.notification_pusher.KFPNotificationPusher(
                project, run_id, notifications, {}
            )
        )
        assert len(kfp_notification_pusher._sync_notifications) == 1
        assert len(kfp_notification_pusher._async_notifications) == 2
        with (
            unittest.mock.patch(
                "mlrun.utils.Workflow.get_workflow_steps"
            ) as get_workflow_steps_mock,
            unittest.mock.patch("mlrun.config.is_running_as_api", return_value=False),
        ):
            kfp_notification_pusher.push()
            assert get_workflow_steps_mock.call_count == 3
