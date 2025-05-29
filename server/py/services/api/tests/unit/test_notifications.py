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

import pytest

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


@pytest.mark.parametrize("running_from_api", [True, False])
def test_push_kfp_notification(running_from_api):
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

    with unittest.mock.patch("framework.api.utils.get_run_db_instance"):
        kfp_notification_pusher = (
            framework.utils.notifications.notification_pusher.KFPNotificationPusher(
                unittest.mock.Mock(), project, run_id, notifications, {}
            )
        )
    kfp_notification_pusher._push_workflow_notification_async = (
        unittest.mock.AsyncMock()
    )
    kfp_notification_pusher._push_workflow_notification_sync = unittest.mock.Mock()
    assert len(kfp_notification_pusher._sync_notifications) == 1
    assert len(kfp_notification_pusher._async_notifications) == 2
    with (
        unittest.mock.patch(
            "mlrun.utils.Workflow.get_workflow_steps"
        ) as get_workflow_steps_mock,
        unittest.mock.patch(
            "mlrun.config.is_running_as_api", return_value=running_from_api
        ),
    ):
        kfp_notification_pusher.push()

        # get the workflow steps once, send notification many times
        assert get_workflow_steps_mock.call_count == 1
        assert kfp_notification_pusher._push_workflow_notification_async.call_count == 2
        assert (
            kfp_notification_pusher._push_workflow_notification_sync.call_count == 0
            if running_from_api
            else 1
        )


def test_get_workflow_steps_called():
    """Test that mlrun.utils.Workflow.get_workflow_steps is called and returns expected steps."""

    db_mock = unittest.mock.Mock()
    workflow_id = "workflow-123"
    project = "test-project"

    # Patch _get_workflow_manifest to return a mock with get_steps
    class MockStep:
        def __init__(self, display_name, step_type, skipped=False):
            self.display_name = display_name
            self.step_type = step_type
            self.skipped = skipped
            self.node_name = display_name
            self.phase = "Succeeded"

        def to_dict(self):
            return {"display_name": self.display_name, "step_type": self.step_type}

    class MockManifest:
        def get_steps(self):
            return [
                MockStep("step1", "run"),
            ]

    db_mock.list_runs.side_effect = [
        # no runs as label is incorrect
        [],
        # trying again yields it
        [
            {
                "metadata": {"name": "step1", "project": project},
                "status": {"state": "completed"},
            }
        ],
    ]

    with unittest.mock.patch(
        "mlrun.utils.Workflow._get_workflow_manifest", return_value=MockManifest()
    ):
        steps = mlrun.utils.Workflow.get_workflow_steps(db_mock, workflow_id, project)

    # TODO: change to 1 when deprecating kfp 1.8 server all together
    assert (
        db_mock.list_runs.call_count == 2
    )  # called twice, first with no runs, then with the run
    assert len(steps) == 1  # first step is the actual mlrun run
    assert steps[0]["metadata"]["name"] == "step1"
