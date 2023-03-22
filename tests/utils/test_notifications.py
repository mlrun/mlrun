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

import builtins
import unittest.mock

import pytest
import requests
import tabulate

import mlrun.utils.notifications


@pytest.mark.parametrize(
    "when,condition,run_state,expected",
    [
        (["success"], "", "success", True),
        (["success"], "", "error", False),
        (["success"], "True", "success", True),
        (["success"], "False", "success", False),
        (["failure"], "", "success", False),
        (["failure"], "", "error", True),
        (["success", "failure"], "", "success", True),
        (["success", "failure"], "", "error", True),
        (["success", "failure"], "True", "success", True),
        (["success", "failure"], "True", "error", True),
        (["success", "failure"], "False", "success", False),
        (["success", "failure"], "False", "error", True),
    ],
)
def test_notification_should_notify(when, condition, run_state, expected):
    run = {"status": {"state": run_state}}
    notification_config = {"when": when, "condition": condition}

    assert (
        mlrun.utils.notifications.notification_pusher.NotificationPusher._should_notify(
            run, notification_config
        )
        == expected
    )


@pytest.mark.parametrize(
    "runs,expected,is_table",
    [
        ([], "[info] test-message", False),
        (
            [
                {
                    "metadata": {"name": "test-run", "uid": "test-run-uid"},
                    "status": {"state": "success"},
                }
            ],
            [["success", "test-run", "..un-uid", ""]],
            True,
        ),
        (
            [
                {
                    "metadata": {"name": "test-run", "uid": "test-run-uid"},
                    "status": {"state": "error"},
                }
            ],
            [["error", "test-run", "..un-uid", ""]],
            True,
        ),
    ],
)
def test_console_notification(monkeypatch, runs, expected, is_table):
    console_notification = mlrun.utils.notifications.ConsoleNotification()
    print_result = ""

    def set_result(result):
        nonlocal print_result
        print_result = result

    monkeypatch.setattr(builtins, "print", set_result)
    console_notification.send("test-message", "info", runs)

    if is_table:
        expected = tabulate.tabulate(
            expected, headers=["status", "name", "uid", "results"]
        )
    assert print_result == expected


@pytest.mark.parametrize(
    "runs,expected",
    [
        (
            [],
            {
                "blocks": [
                    {
                        "text": {"text": "[info] test-message", "type": "mrkdwn"},
                        "type": "section",
                    }
                ]
            },
        ),
        (
            [
                {
                    "metadata": {"name": "test-run", "uid": "test-run-uid"},
                    "status": {"state": "success"},
                }
            ],
            {
                "blocks": [
                    {
                        "text": {"text": "[info] test-message", "type": "mrkdwn"},
                        "type": "section",
                    },
                    {
                        "fields": [
                            {"text": "*Runs*", "type": "mrkdwn"},
                            {"text": "*Results*", "type": "mrkdwn"},
                            {"text": ":question:  test-run", "type": "mrkdwn"},
                            {"text": "None", "type": "mrkdwn"},
                        ],
                        "type": "section",
                    },
                ]
            },
        ),
        (
            [
                {
                    "metadata": {"name": "test-run", "uid": "test-run-uid"},
                    "status": {"state": "error"},
                }
            ],
            {
                "blocks": [
                    {
                        "text": {"text": "[info] test-message", "type": "mrkdwn"},
                        "type": "section",
                    },
                    {
                        "fields": [
                            {"text": "*Runs*", "type": "mrkdwn"},
                            {"text": "*Results*", "type": "mrkdwn"},
                            {"text": ":x:  test-run", "type": "mrkdwn"},
                            {"text": "**", "type": "mrkdwn"},
                        ],
                        "type": "section",
                    },
                ]
            },
        ),
    ],
)
def test_slack_notification(runs, expected):
    slack_notification = mlrun.utils.notifications.SlackNotification()
    slack_data = slack_notification._generate_slack_data("test-message", "info", runs)

    assert slack_data == expected


@pytest.mark.parametrize(
    "params,expected_url,expected_headers",
    [
        (
            {
                "repo": "test-repo",
                "issue": "test-issue",
                "token": "test-token",
            },
            "https://api.github.com/repos/test-repo/issues/test-issue/comments",
            {
                "Accept": "application/vnd.github.v3+json",
                "Authorization": "token test-token",
            },
        ),
        (
            {
                "repo": "test-repo",
                "issue": "test-issue",
                "token": "test-token",
                "gitlab": True,
            },
            "https://gitlab.com/api/v4/projects/test-repo/merge_requests/test-issue/notes",
            {
                "PRIVATE-TOKEN": "test-token",
            },
        ),
        (
            {
                "repo": "test-repo",
                "issue": "test-issue",
                "token": "test-token",
                "server": "custom-gitlab",
            },
            "https://custom-gitlab/api/v4/projects/test-repo/merge_requests/test-issue/notes",
            {
                "PRIVATE-TOKEN": "test-token",
            },
        ),
    ],
)
def test_git_notification(monkeypatch, params, expected_url, expected_headers):
    git_notification = mlrun.utils.notifications.GitNotification(params)
    expected_body = "[info] test-message"

    requests_mock = unittest.mock.MagicMock()
    monkeypatch.setattr(requests, "post", requests_mock)
    git_notification.send("test-message", "info", [])

    requests_mock.assert_called_once_with(
        url=expected_url, json={"body": expected_body}, headers=expected_headers
    )


@pytest.mark.parametrize(
    "ipython_active,expected_console_call_amount,expected_ipython_call_amount",
    [
        (True, 0, 1),
        (False, 1, 0),
    ],
)
def test_inverse_dependencies(
    monkeypatch,
    ipython_active,
    expected_console_call_amount,
    expected_ipython_call_amount,
):
    custom_notification_pusher = mlrun.utils.notifications.CustomNotificationPusher(
        [
            mlrun.utils.notifications.NotificationTypes.console,
            mlrun.utils.notifications.NotificationTypes.ipython,
        ]
    )

    mock_console_send = unittest.mock.MagicMock()
    mock_ipython_send = unittest.mock.MagicMock()
    monkeypatch.setattr(
        mlrun.utils.notifications.ConsoleNotification, "send", mock_console_send
    )
    monkeypatch.setattr(
        mlrun.utils.notifications.IPythonNotification, "send", mock_ipython_send
    )
    monkeypatch.setattr(
        mlrun.utils.notifications.IPythonNotification, "active", ipython_active
    )

    custom_notification_pusher.push("test-message", "info", [])
    assert mock_console_send.call_count == expected_console_call_amount
    assert mock_ipython_send.call_count == expected_ipython_call_amount
