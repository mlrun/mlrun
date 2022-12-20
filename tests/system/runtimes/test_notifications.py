# Copyright 2018 Iguazio
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
import datetime

import mlrun
import tests.system.base


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
class TestNotifications(tests.system.base.TestMLRunSystem):

    project_name = "notifications-test"

    def test_run_notifications(self):
        error_notification_name = "slack-shoul-fail"
        success_notification_name = "console-should-succeed"

        def _assert_notifications():
            runs = self._run_db.list_runs(
                project=self.project_name,
                join_notifications=True,
            )
            assert len(runs) == 1
            assert len(runs[0]["spec"]["notifications"]) == 2
            for notification in runs[0]["spec"]["notifications"]:
                if notification["name"] == error_notification.name:
                    assert notification["status"] == "error"
                elif notification["name"] == success_notification.name:
                    assert notification["status"] == "sent"

        error_notification = self._create_notification(
            name=error_notification_name,
            message="should-fail",
            params={
                "webhook": "https://invalid.slack.url.com",
            },
        )
        success_notification = self._create_notification(
            kind="console",
            name=success_notification_name,
            message="should-succeed",
        )

        function = mlrun.new_function(
            "function-from-module",
            kind="job",
            project=self.project_name,
            image="mlrun/mlrun",
        )
        run = function.run(
            handler="json.dumps",
            params={"obj": {"x": 99}},
            notifications=[error_notification, success_notification],
        )
        assert run.output("return") == '{"x": 99}'

        # the notifications are sent asynchronously, so we need to wait for them
        mlrun.utils.retry_until_successful(
            1,
            20,
            self._logger,
            True,
            _assert_notifications,
        )

    @staticmethod
    def _create_notification(
        kind=None,
        name=None,
        message=None,
        severity=None,
        when=None,
        condition=None,
        params=None,
    ):
        return mlrun.model.Notification(
            kind=kind or "slack",
            when=when or ["completed"],
            name=name or "test-notification",
            message=message or "test-notification-message",
            condition=condition or "",
            severity=severity or "info",
            params=params or {},
        )
