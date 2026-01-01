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

import asyncio
import builtins
import unittest.mock
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from typing import Any, Optional

import aiohttp
import pytest
import tabulate
from aiohttp.typedefs import StrOrURL

import mlrun.common.runtimes.constants as runtimes_constants
import mlrun.common.schemas
import mlrun.common.schemas.notification
import mlrun.utils
import mlrun.utils.helpers
import mlrun.utils.notifications
import mlrun.utils.notifications.notification.mail as mail
import mlrun.utils.notifications.notification.webhook


@pytest.fixture
def inline_run_in_threadpool(monkeypatch):
    async def _inline(func, *args, **kwargs):
        if kwargs:
            func = partial(func, **kwargs)
        return func(*args)

    monkeypatch.setattr(mlrun.utils.helpers, "run_in_threadpool", _inline)


@pytest.mark.parametrize(
    "notification_kind, params, default_params, expected_params",
    [
        (
            mlrun.common.schemas.notification.NotificationKind.webhook,
            {"webhook": "some-webhook"},
            {"webhook": "some-default"},
            {"webhook": "some-webhook"},
        ),
        (
            mlrun.common.schemas.notification.NotificationKind.webhook,
            {"webhook": "some-webhook"},
            {"hello": "world"},
            {"webhook": "some-webhook", "hello": "world"},
        ),
    ],
)
def test_process_notification(
    notification_kind, params, default_params, expected_params
):
    run_uid = "test-run-uid"
    notification_name = "test-notification-name"
    when_state = runtimes_constants.RunStates.completed
    notification = mlrun.model.Notification.from_dict(
        {
            "kind": notification_kind,
            "when": [when_state],
            "status": "pending",
            "name": notification_name,
            "params": params,
        }
    )
    run = mlrun.model.RunObject.from_dict(
        {
            "metadata": {"uid": run_uid},
            "spec": {"notifications": [notification]},
            "status": {"state": when_state},
        }
    )

    default_params = {
        notification_kind: default_params,
    }
    notification_pusher = (
        mlrun.utils.notifications.notification_pusher.NotificationPusher(
            [run], default_params
        )
    )
    loaded_notifications = (
        notification_pusher._sync_notifications
        + notification_pusher._async_notifications
    )
    assert len(loaded_notifications) == 1
    assert loaded_notifications[0][0].params == expected_params
    assert loaded_notifications[0][0].name == notification_name


@pytest.mark.parametrize(
    "when,condition,run_state,notification_previously_sent,expected",
    [
        (
            [runtimes_constants.RunStates.completed],
            "",
            runtimes_constants.RunStates.completed,
            False,
            True,
        ),
        (
            [runtimes_constants.RunStates.completed],
            "",
            runtimes_constants.RunStates.completed,
            True,
            False,
        ),
        (
            [runtimes_constants.RunStates.completed],
            "",
            runtimes_constants.RunStates.error,
            False,
            False,
        ),
        (
            [runtimes_constants.RunStates.completed],
            "",
            runtimes_constants.RunStates.error,
            True,
            False,
        ),
        (
            [runtimes_constants.RunStates.completed],
            "> 4",
            runtimes_constants.RunStates.completed,
            False,
            True,
        ),
        (
            [runtimes_constants.RunStates.completed],
            "> 4",
            runtimes_constants.RunStates.completed,
            True,
            False,
        ),
        (
            [runtimes_constants.RunStates.completed],
            "< 4",
            runtimes_constants.RunStates.completed,
            False,
            False,
        ),
        (
            [runtimes_constants.RunStates.completed],
            "< 4",
            runtimes_constants.RunStates.completed,
            True,
            False,
        ),
        (
            [runtimes_constants.RunStates.error],
            "",
            runtimes_constants.RunStates.completed,
            False,
            False,
        ),
        (
            [runtimes_constants.RunStates.error],
            "",
            runtimes_constants.RunStates.completed,
            True,
            False,
        ),
        (
            [runtimes_constants.RunStates.error],
            "",
            runtimes_constants.RunStates.error,
            False,
            True,
        ),
        (
            [runtimes_constants.RunStates.error],
            "",
            runtimes_constants.RunStates.error,
            True,
            False,
        ),
        (
            [
                runtimes_constants.RunStates.completed,
                runtimes_constants.RunStates.error,
            ],
            "",
            runtimes_constants.RunStates.completed,
            False,
            True,
        ),
        (
            [
                runtimes_constants.RunStates.completed,
                runtimes_constants.RunStates.error,
            ],
            "",
            runtimes_constants.RunStates.completed,
            True,
            False,
        ),
        (
            [
                runtimes_constants.RunStates.completed,
                runtimes_constants.RunStates.error,
            ],
            "",
            runtimes_constants.RunStates.error,
            False,
            True,
        ),
        (
            [
                runtimes_constants.RunStates.completed,
                runtimes_constants.RunStates.error,
            ],
            "",
            runtimes_constants.RunStates.error,
            True,
            False,
        ),
        (
            [
                runtimes_constants.RunStates.completed,
                runtimes_constants.RunStates.error,
            ],
            "> 4",
            runtimes_constants.RunStates.completed,
            False,
            True,
        ),
        (
            [
                runtimes_constants.RunStates.completed,
                runtimes_constants.RunStates.error,
            ],
            "> 4",
            runtimes_constants.RunStates.completed,
            True,
            False,
        ),
        (
            [
                runtimes_constants.RunStates.completed,
                runtimes_constants.RunStates.error,
            ],
            "> 4",
            runtimes_constants.RunStates.error,
            False,
            True,
        ),
        (
            [
                runtimes_constants.RunStates.completed,
                runtimes_constants.RunStates.error,
            ],
            "> 4",
            runtimes_constants.RunStates.error,
            True,
            False,
        ),
        (
            [
                runtimes_constants.RunStates.completed,
                runtimes_constants.RunStates.error,
            ],
            "< 4",
            runtimes_constants.RunStates.completed,
            False,
            False,
        ),
        (
            [
                runtimes_constants.RunStates.completed,
                runtimes_constants.RunStates.error,
            ],
            "< 4",
            runtimes_constants.RunStates.completed,
            True,
            False,
        ),
        (
            [
                runtimes_constants.RunStates.completed,
                runtimes_constants.RunStates.error,
            ],
            "< 4",
            runtimes_constants.RunStates.error,
            False,
            True,
        ),
        (
            [
                runtimes_constants.RunStates.completed,
                runtimes_constants.RunStates.error,
            ],
            "< 4",
            runtimes_constants.RunStates.error,
            True,
            False,
        ),
    ],
)
def test_notification_should_notify(
    when, condition, run_state, notification_previously_sent, expected
):
    if condition:
        condition = f'{{{{ run["status"]["results"]["val"] {condition} }}}}'

    run = mlrun.model.RunObject.from_dict(
        {"status": {"state": run_state, "results": {"val": 5}}}
    )
    notification = mlrun.model.Notification.from_dict(
        {
            "when": when,
            "condition": condition,
            "status": "pending" if not notification_previously_sent else "sent",
        }
    )

    notification_pusher = (
        mlrun.utils.notifications.notification_pusher.NotificationPusher([run])
    )
    assert notification_pusher._should_notify(run, notification) == expected


@pytest.mark.parametrize(
    "notification_kind",
    [
        mlrun.common.schemas.notification.NotificationKind.console,
        mlrun.common.schemas.notification.NotificationKind.slack,
        mlrun.common.schemas.notification.NotificationKind.git,
        mlrun.common.schemas.notification.NotificationKind.webhook,
        mlrun.common.schemas.notification.NotificationKind.ipython,
        mlrun.common.schemas.notification.NotificationKind.mail,
    ],
)
def test_notification_reason(notification_kind):
    error_exc = Exception("Blew up")
    run = mlrun.model.RunObject.from_dict(
        {"status": {"state": runtimes_constants.RunStates.completed}}
    )
    run.spec.notifications = [
        mlrun.model.Notification.from_dict(
            {
                "kind": notification_kind,
                "status": "pending",
                "message": "test-abc",
            }
        ),
    ]

    notification_pusher = (
        mlrun.utils.notifications.notification_pusher.NotificationPusher([run])
    )

    # dont really update, just mock it for later assertions
    notification_pusher._update_notification_status = unittest.mock.MagicMock()

    # mock the push method to raise an exception
    notification_kind_type = getattr(
        mlrun.utils.notifications.NotificationTypes, notification_kind
    ).get_notification()
    if asyncio.iscoroutinefunction(notification_kind_type.push):
        concrete_notification = notification_pusher._async_notifications[0][0]
    else:
        concrete_notification = notification_pusher._sync_notifications[0][0]

    concrete_notification.push = unittest.mock.MagicMock(side_effect=error_exc)

    # send notifications
    notification_pusher.push()

    # asserts
    notification_pusher._update_notification_status.assert_called_once()
    concrete_notification.push.assert_called_once()

    assert (
        str(error_exc)
        in notification_pusher._update_notification_status.call_args.kwargs["reason"]
    )


@pytest.mark.usefixtures("inline_run_in_threadpool")
@pytest.mark.parametrize(
    "when, run_state, store_count",
    [
        (
            [runtimes_constants.RunStates.running],
            runtimes_constants.RunStates.running,
            1,
        ),
        (
            [
                runtimes_constants.RunStates.running,
                runtimes_constants.RunStates.completed,
            ],
            runtimes_constants.RunStates.running,
            0,
        ),
        (
            [
                runtimes_constants.RunStates.running,
                runtimes_constants.RunStates.completed,
            ],
            runtimes_constants.RunStates.completed,
            1,
        ),
    ],
)
def test_notification_update_notification_status(when, run_state, store_count):
    notification_kind = mlrun.common.schemas.notification.NotificationKind.mail
    run = mlrun.model.RunObject.from_dict({"status": {"state": run_state}})
    run.spec.notifications = [
        mlrun.model.Notification.from_dict(
            {
                "kind": notification_kind,
                "status": "pending",
                "message": "test-abc",
                "when": when,
            }
        ),
    ]

    db = mlrun.get_run_db()
    db.store_run_notifications = unittest.mock.MagicMock()

    notification_pusher = (
        mlrun.utils.notifications.notification_pusher.NotificationPusher([run])
    )

    # mock the push method to raise an exception
    notification_kind_type = getattr(
        mlrun.utils.notifications.NotificationTypes, notification_kind
    ).get_notification()
    if asyncio.iscoroutinefunction(notification_kind_type.push):
        concrete_notification = notification_pusher._async_notifications[0][0]
        concrete_notification.push = unittest.mock.AsyncMock()
    else:
        concrete_notification = notification_pusher._sync_notifications[0][0]
        concrete_notification.push = unittest.mock.MagicMock()

    # send notifications
    notification_pusher.push()

    # asserts
    concrete_notification.push.assert_called_once()

    assert db.store_run_notifications.call_count == store_count


@pytest.mark.usefixtures("inline_run_in_threadpool")
@pytest.mark.parametrize(
    "notification_kind",
    [
        mlrun.common.schemas.notification.NotificationKind.console,
        mlrun.common.schemas.notification.NotificationKind.slack,
        mlrun.common.schemas.notification.NotificationKind.git,
        mlrun.common.schemas.notification.NotificationKind.webhook,
        mlrun.common.schemas.notification.NotificationKind.ipython,
        mlrun.common.schemas.notification.NotificationKind.mail,
    ],
)
@pytest.mark.parametrize(
    "run_status",
    [runtimes_constants.RunStates.running, runtimes_constants.RunStates.completed],
)
def test_update_notification_status(notification_kind, run_status):
    run = mlrun.model.RunObject.from_dict({"status": {"state": run_status}})
    run.spec.notifications = [
        mlrun.model.Notification.from_dict(
            {
                "kind": notification_kind,
                "status": "pending",
                "message": "test-abc",
                "when": [
                    runtimes_constants.RunStates.running,
                    runtimes_constants.RunStates.completed,
                ],
            }
        ),
    ]

    notification_pusher = (
        mlrun.utils.notifications.notification_pusher.NotificationPusher([run])
    )

    # mock the push method to raise an exception
    notification_kind_type = getattr(
        mlrun.utils.notifications.NotificationTypes, notification_kind
    ).get_notification()
    if asyncio.iscoroutinefunction(notification_kind_type.push):
        concrete_notification = notification_pusher._async_notifications[0][0]
        concrete_notification.push = unittest.mock.AsyncMock()
    else:
        concrete_notification = notification_pusher._sync_notifications[0][0]
        concrete_notification.push = unittest.mock.MagicMock()

    db = mlrun.get_run_db()
    db.store_run_notifications = unittest.mock.MagicMock()

    # send notifications
    notification_pusher.push()

    # we don't want to call the store_run_notifications method on running
    expected_store_run_notifications_call_count = (
        0 if run_status == runtimes_constants.RunStates.running else 1
    )
    assert (
        expected_store_run_notifications_call_count
        == db.store_run_notifications.call_count
    )


def test_condition_evaluation_timeout():
    condition = """
        {% for i in range(100000) %}
            {% for i in range(100000) %}
                {% for i in range(100000) %}
                    {{ i }}
                {% endfor %}
            {% endfor %}
        {% endfor %}
    """

    run = mlrun.model.RunObject.from_dict(
        {
            "status": {
                "state": runtimes_constants.RunStates.completed,
                "results": {"val": 5},
            }
        }
    )
    notification = mlrun.model.Notification.from_dict(
        {
            "when": [runtimes_constants.RunStates.completed],
            "condition": condition,
            "status": "pending",
        }
    )

    notification_pusher = (
        mlrun.utils.notifications.notification_pusher.NotificationPusher([run])
    )
    assert notification_pusher._should_notify(run, notification)


@pytest.mark.parametrize(
    "override_body",
    [
        ({"message": "runs: {{runs}}"}),
        ({"message": "runs: {{ runs }}"}),
        ({"message": "runs: {{ runs}}"}),
        ({"message": "runs: {{runs }}"}),
    ],
)
async def test_webhook_override_body_job_succeed(monkeypatch, override_body):
    requests_mock = _mock_async_response(monkeypatch, "post", {"id": "response-id"})
    run = _generate_run_result(
        state=runtimes_constants.RunStates.completed, results={"return": 1}
    )
    await mlrun.utils.notifications.notification.webhook.WebhookNotification(
        params={"override_body": override_body, "url": "http://test.com"}
    ).push("test-message", "info", [run])
    expected_body = {
        "message": "runs: [{'project': 'test-remote-workflow', 'name': 'func-func', "
        "'status': {'state': 'completed', 'results': {'return': 1}}, 'host': 'func-func-8lvl8'}]"
    }
    requests_mock.assert_called_once_with(
        "http://test.com", headers={}, json=expected_body, ssl=None
    )


@pytest.mark.parametrize(
    "run,input_override_body,expected_override_body",
    [
        (
            {
                "metadata": {"name": "x", "project": "y"},
                "status": {"state": runtimes_constants.RunStates.completed},
            },
            {"message": "{{runs}}"},
            {
                "message": "[{'project': 'y', 'name': 'x', 'status': {'state': 'completed'}}]"
            },
        ),
        (
            {
                "metadata": {"name": "x", "project": "y"},
                "status": {"state": runtimes_constants.RunStates.completed},
            },
            {"message": "{{runs}}", "ignore_non_str_values": ["{{runs}}"]},
            {
                "message": "[{'project': 'y', 'name': 'x', 'status': {'state': 'completed'}}]",
                "ignore_non_str_values": ["{{runs}}"],
            },
        ),
    ],
)
async def test_serialize_runs_in_request_body(
    run, input_override_body, expected_override_body
):
    # just to make line shorter
    webhook_cls = mlrun.utils.notifications.notification.webhook.WebhookNotification
    override_body = webhook_cls._serialize_runs_in_request_body(
        override_body=input_override_body,
        runs=[run],
    )
    assert override_body == expected_override_body


@pytest.mark.parametrize(
    "override_body",
    [({"message": "runs: {{runs}}"}), ({"message": "runs: {{ runs }}"})],
)
async def test_webhook_override_body_job_failed(monkeypatch, override_body):
    requests_mock = _mock_async_response(monkeypatch, "post", {"id": "response-id"})
    run = _generate_run_result(
        state=runtimes_constants.RunStates.error, error="some_error"
    )
    await mlrun.utils.notifications.notification.webhook.WebhookNotification(
        params={"override_body": override_body, "url": "http://test.com"}
    ).push("test-message", "info", [run])
    expected_body = {
        "message": "runs: [{'project': 'test-remote-workflow', 'name': 'func-func', "
        "'status': {'state': 'error', 'error': 'some_error'}, 'host': 'func-func-8lvl8'}]"
    }
    requests_mock.assert_called_once_with(
        "http://test.com", headers={}, json=expected_body, ssl=None
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
                    "status": {"state": runtimes_constants.RunStates.error},
                }
            ],
            [["error", "test-run", "..un-uid", ""]],
            True,
        ),
    ],
)
def test_console_notification(monkeypatch, runs, expected, is_table):
    console_notification = mlrun.utils.notifications.console.ConsoleNotification()
    print_result = ""

    def set_result(result):
        nonlocal print_result
        print_result = result

    monkeypatch.setattr(builtins, "print", set_result)
    console_notification.push("test-message", "info", runs)

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
                        "text": {"text": "[info] test-message", "type": "plain_text"},
                        "type": "header",
                    }
                ]
            },
        ),
        (
            [
                {
                    "metadata": {"name": "test-run", "uid": "test-run-uid"},
                    "status": {"state": runtimes_constants.RunStates.completed},
                }
            ],
            {
                "blocks": [
                    {
                        "text": {"text": "[info] test-message", "type": "plain_text"},
                        "type": "header",
                    },
                    {
                        "fields": [
                            {"text": "*Runs*", "type": "mrkdwn"},
                            {"text": "*Results*", "type": "mrkdwn"},
                            {"text": ":smiley:  test-run", "type": "mrkdwn"},
                            {"text": "completed", "type": "mrkdwn"},
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
                    "status": {"state": runtimes_constants.RunStates.error},
                }
            ],
            {
                "blocks": [
                    {
                        "text": {"text": "[info] test-message", "type": "plain_text"},
                        "type": "header",
                    },
                    {
                        "fields": [
                            {"text": "*Runs*", "type": "mrkdwn"},
                            {"text": "*Results*", "type": "mrkdwn"},
                            {"text": ":x:  test-run", "type": "mrkdwn"},
                            {"text": "*error*", "type": "mrkdwn"},
                        ],
                        "type": "section",
                    },
                ]
            },
        ),
    ],
)
def test_slack_notification(runs, expected):
    slack_notification = mlrun.utils.notifications.slack.SlackNotification()
    slack_data = slack_notification._generate_slack_data("test-message", "info", runs)

    assert slack_data == expected


@pytest.mark.asyncio
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
                mlrun.common.schemas.HeaderNames.authorization: "token test-token",
            },
        ),
        (
            {
                "repo": "test-repo",
                "issue": "test-issue",
                "token": "test-token",
                "gitlab": True,
            },
            "https://gitlab.com/api/v4/projects/test-repo/issues/test-issue/notes",
            {
                "PRIVATE-TOKEN": "test-token",
            },
        ),
        (
            {
                "repo": "test-repo",
                "merge_request": "test-merge-request",
                "token": "test-token",
                "gitlab": True,
            },
            "https://gitlab.com/api/v4/projects/test-repo/merge_requests/test-merge-request/notes",
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
            "https://custom-gitlab/api/v4/projects/test-repo/issues/test-issue/notes",
            {
                "PRIVATE-TOKEN": "test-token",
            },
        ),
    ],
)
async def test_git_notification(monkeypatch, params, expected_url, expected_headers):
    git_notification = mlrun.utils.notifications.git.GitNotification("git", params)
    expected_body = "[info] git: test-message"

    requests_mock = _mock_async_response(monkeypatch, "post", {"id": "response-id"})

    await git_notification.push("test-message", "info", [])

    requests_mock.assert_called_once_with(
        expected_url,
        headers=expected_headers,
        json={"body": expected_body},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("test_method", ["GET", "POST", "PUT", "PATCH", "DELETE"])
async def test_webhook_notification(monkeypatch, test_method):
    requests_mock = _mock_async_response(monkeypatch, test_method.lower(), None)

    test_url = "https://test-url"
    test_headers = {"test-header": "test-value"}
    test_override_body = {
        "test-key": "test-value",
    }
    test_message = "test-message"
    test_severity = "info"
    test_runs_info = ["some-run"]
    webhook_notification = mlrun.utils.notifications.webhook.WebhookNotification(
        "webhook",
        {
            "url": test_url,
            "method": test_method,
            "headers": test_headers,
        },
    )
    await webhook_notification.push(test_message, test_severity, test_runs_info)

    requests_mock.assert_called_once_with(
        test_url,
        headers=test_headers,
        json={
            "message": test_message,
            "severity": test_severity,
            "runs": test_runs_info,
        },
        ssl=None,
    )

    webhook_notification.params["override_body"] = test_override_body

    await webhook_notification.push("test-message", "info", ["some-run"])

    requests_mock.assert_called_with(
        test_url,
        headers=test_headers,
        json=test_override_body,
        ssl=None,
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

    mock_console_push = unittest.mock.MagicMock(return_value=Exception())
    mock_ipython_push = unittest.mock.MagicMock(return_value=Exception())
    monkeypatch.setattr(
        mlrun.utils.notifications.console.ConsoleNotification, "push", mock_console_push
    )
    monkeypatch.setattr(
        mlrun.utils.notifications.ipython.IPythonNotification, "push", mock_ipython_push
    )
    monkeypatch.setattr(
        mlrun.utils.notifications.ipython.IPythonNotification, "active", ipython_active
    )

    custom_notification_pusher.push("test-message", "info", [])

    assert mock_console_push.call_count == expected_console_call_amount
    assert mock_ipython_push.call_count == expected_ipython_call_amount


NOTIFICATION_VALIDATION_PARMETRIZE = [
    (
        {
            "kind": "invalid-kind",
        },
        pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
    ),
    (
        {
            "kind": mlrun.common.schemas.notification.NotificationKind.slack,
        },
        does_not_raise(),
    ),
    (
        {
            "severity": "invalid-severity",
        },
        pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
    ),
    (
        {
            "severity": mlrun.common.schemas.notification.NotificationSeverity.INFO,
        },
        does_not_raise(),
    ),
    (
        {
            "status": "invalid-status",
        },
        pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
    ),
    (
        {
            "status": mlrun.common.schemas.notification.NotificationStatus.PENDING,
        },
        does_not_raise(),
    ),
    (
        {
            "when": "invalid-when",
        },
        pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
    ),
    (
        {
            "when": [
                runtimes_constants.RunStates.completed,
                runtimes_constants.RunStates.error,
            ],
        },
        does_not_raise(),
    ),
    (
        {
            "message": {"my-message": "invalid"},
        },
        pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
    ),
    (
        {
            "message": runtimes_constants.RunStates.completed,
        },
        does_not_raise(),
    ),
    (
        {
            "condition": ["invalid-condition"],
        },
        pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
    ),
    (
        {
            "condition": "valid-condition",
        },
        does_not_raise(),
    ),
]


@pytest.mark.parametrize(
    "notification_kwargs,expectation",
    NOTIFICATION_VALIDATION_PARMETRIZE,
)
def test_notification_validation_on_object(
    monkeypatch, notification_kwargs, expectation
):
    with expectation:
        mlrun.model.Notification(**notification_kwargs)


def test_notification_validation_defaults(monkeypatch):
    notification = mlrun.model.Notification()
    notification_fields = {
        "kind": mlrun.common.schemas.notification.NotificationKind.slack,
        "message": "",
        "severity": mlrun.common.schemas.notification.NotificationSeverity.INFO,
        "when": [runtimes_constants.RunStates.completed],
        "condition": "",
        "name": "",
    }

    for notification_field, expected_value in notification_fields.items():
        value = getattr(notification, notification_field)
        assert (
            value == expected_value
        ), f"{notification_field} field value is {value}, expected {expected_value}"


@pytest.mark.parametrize(
    "notification_kwargs,expectation",
    NOTIFICATION_VALIDATION_PARMETRIZE,
)
def test_notification_validation_on_run(monkeypatch, notification_kwargs, expectation):
    notification = mlrun.model.Notification(
        name="test-notification", when=[runtimes_constants.RunStates.completed]
    )
    for key, value in notification_kwargs.items():
        setattr(notification, key, value)
    function = mlrun.new_function(
        "function-from-module",
        kind="job",
        project="test-project",
        image="mlrun/mlrun",
    )
    with expectation:
        function.run(
            handler="json.dumps",
            params={"obj": {"x": 99}},
            notifications=[notification],
            local=True,
        )


def test_notification_sent_on_handler_run(monkeypatch):
    run_many_mock = unittest.mock.Mock(return_value=[])
    push_mock = unittest.mock.Mock()

    monkeypatch.setattr(mlrun.runtimes.HandlerRuntime, "_run_many", run_many_mock)
    monkeypatch.setattr(mlrun.utils.notifications.NotificationPusher, "push", push_mock)

    def hyper_func(context, p1, p2):
        print(f"p1={p1}, p2={p2}, result={p1 * p2}")
        context.log_result("multiplier", p1 * p2)

    notification = mlrun.model.Notification(
        name="test-notification", when=[runtimes_constants.RunStates.completed]
    )

    grid_params = {"p1": [2, 4, 1], "p2": [10, 20]}
    task = mlrun.new_task("grid-demo").with_hyper_params(
        grid_params, selector="max.multiplier"
    )
    mlrun.new_function().run(task, handler=hyper_func, notifications=[notification])
    run_many_mock.assert_called_once()
    push_mock.assert_called_once()


def test_notification_sent_on_dask_run(monkeypatch):
    run_mock = unittest.mock.Mock(return_value=None)
    push_mock = unittest.mock.Mock()

    monkeypatch.setattr(mlrun.runtimes.LocalRuntime, "_run", run_mock)
    monkeypatch.setattr(mlrun.utils.notifications.NotificationPusher, "push", push_mock)

    notification = mlrun.model.Notification(
        name="test-notification", when=[runtimes_constants.RunStates.completed]
    )

    function = mlrun.new_function(
        "function-from-module",
        kind="dask",
        project="test-project",
        image="mlrun/mlrun",
    )

    function.run(
        handler="json.dumps",
        params={"obj": {"x": 99}},
        notifications=[notification],
        local=True,
    )

    run_mock.assert_called_once()
    push_mock.assert_called_once()


@pytest.mark.parametrize(
    "notification1_name,notification2_name,expectation",
    [
        ("n1", "n1", pytest.raises(mlrun.errors.MLRunInvalidArgumentError)),
        ("n1", "n2", does_not_raise()),
    ],
)
def test_notification_name_uniqueness_validation(
    notification1_name, notification2_name, expectation
):
    notification1 = mlrun.model.Notification(
        name=notification1_name, when=[runtimes_constants.RunStates.completed]
    )
    notification2 = mlrun.model.Notification(
        name=notification2_name, when=[runtimes_constants.RunStates.completed]
    )
    function = mlrun.new_function(
        "function-from-module",
        kind="job",
        project="test-project",
        image="mlrun/mlrun",
    )
    with expectation:
        function.run(
            handler="json.dumps",
            params={"obj": {"x": 99}},
            notifications=[notification1, notification2],
            local=True,
        )


def generate_notification_validation_params():
    validation_params = []
    valid_params_by_kind = {
        mlrun.common.schemas.notification.NotificationKind.slack: {
            "webhook": "some-webhook"
        },
        mlrun.common.schemas.notification.NotificationKind.git: {
            "repo": "some-repo",
            "issue": "some-issue",
            "token": "some-token",
        },
        mlrun.common.schemas.notification.NotificationKind.webhook: {"url": "some-url"},
    }

    for kind, valid_params in valid_params_by_kind.items():
        # Both are None
        validation_params.append(
            (
                {
                    "kind": kind,
                    "secret_params": None,
                    "params": None,
                },
                pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
            )
        )
        # Both are not None and equal
        validation_params.append(
            (
                {
                    "kind": kind,
                    "secret_params": valid_params,
                    "params": valid_params,
                },
                does_not_raise(),
            )
        )
        # Only secret_params is not None
        validation_params.append(
            (
                {
                    "kind": kind,
                    "secret_params": valid_params,
                    "params": None,
                },
                does_not_raise(),
            )
        )
        # Only params is not None
        validation_params.append(
            (
                {
                    "kind": kind,
                    "secret_params": None,
                    "params": valid_params,
                },
                does_not_raise(),
            )
        )

        # Specific invalid cases for each kind
        if kind == mlrun.common.schemas.notification.NotificationKind.slack:
            # invalid webhook
            validation_params.append(
                (
                    {
                        "kind": kind,
                        "secret_params": {"webhook": None},
                    },
                    pytest.raises(
                        ValueError,
                        match="Parameter 'webhook' is required for SlackNotification",
                    ),
                )
            )

        if kind == mlrun.common.schemas.notification.NotificationKind.git:
            # invalid repo
            validation_params.append(
                (
                    {
                        "kind": kind,
                        "secret_params": {
                            "repo": None,
                            "issue": "some-issue",
                            "token": "some-token",
                        },
                    },
                    pytest.raises(
                        ValueError,
                        match="Parameter 'repo' is required for GitNotification",
                    ),
                )
            )
            # invalid token
            validation_params.append(
                (
                    {
                        "kind": kind,
                        "params": {
                            "repo": "some-repo",
                            "issue": "some-issue",
                            "token": None,
                        },
                    },
                    pytest.raises(
                        ValueError,
                        match="Parameter 'token' is required for GitNotification",
                    ),
                )
            )
            # invalid issue
            validation_params.append(
                (
                    {
                        "kind": kind,
                        "params": {
                            "repo": "some-repo",
                            "issue": None,
                            "token": "some-token",
                        },
                    },
                    pytest.raises(
                        ValueError,
                        match="At least one of 'issue' or 'merge_request' is required for GitNotification",
                    ),
                )
            )

        if kind == mlrun.common.schemas.notification.NotificationKind.webhook:
            # invalid url
            validation_params.append(
                (
                    {
                        "kind": kind,
                        "params": {"url": None},
                    },
                    pytest.raises(
                        ValueError,
                        match="Parameter 'url' is required for WebhookNotification",
                    ),
                )
            )
            # valid url with secret params
            validation_params.append(
                (
                    {
                        "kind": kind,
                        "secret_params": {"webhook": "some-webhook"},
                        "params": valid_params,
                    },
                    does_not_raise(),
                )
            )

    return validation_params


@pytest.mark.parametrize(
    "notification_kwargs, expectation",
    generate_notification_validation_params(),
)
def test_validate_notification_params(monkeypatch, notification_kwargs, expectation):
    notification = mlrun.model.Notification(**notification_kwargs)
    with expectation:
        notification.validate_notification_params()


@pytest.mark.parametrize(
    "secret_params, get_secret_or_env_return_value, expected_params, should_raise",
    [
        (
            {"web": "secret-web"},
            "check",
            {"web": "secret-web"},
            False,
        ),
        ({"secret": "Hello"}, "Hello", {}, True),
        ({"secret": "Hello"}, '{"webhook": "Hello"}', {"webhook": "Hello"}, False),
    ],
)
def test_enrich_unmasked_secret_params_from_project_secret(
    secret_params, get_secret_or_env_return_value, expected_params, should_raise
):
    with unittest.mock.patch(
        "mlrun.get_secret_or_env", return_value=get_secret_or_env_return_value
    ):
        notification = mlrun.model.Notification(
            kind=mlrun.common.schemas.notification.NotificationKind.slack,
            secret_params=secret_params,
        )
        if should_raise:
            with pytest.raises(mlrun.errors.MLRunValueError):
                notification.enrich_unmasked_secret_params_from_project_secret()
        else:
            notification.enrich_unmasked_secret_params_from_project_secret()
            assert notification.secret_params == expected_params


@pytest.mark.parametrize(
    "retry_count, max_retries, run_state, expected_retry, expected_final_failure",
    [
        # No retries attempted — no retry info should be included
        (None, None, runtimes_constants.RunStates.completed, False, False),
        (0, 0, runtimes_constants.RunStates.completed, False, False),
        # Job succeeded after retries — include retry log line only
        (2, 5, runtimes_constants.RunStates.completed, True, False),
        # Retries exhausted and job failed — both log lines should be included
        (3, 3, runtimes_constants.RunStates.error, True, True),
        # retry_count is None — no retry info expected
        (None, 3, runtimes_constants.RunStates.completed, False, False),
        (None, 3, runtimes_constants.RunStates.error, False, False),
        # Job was aborted after retries
        (2, 3, runtimes_constants.RunStates.aborted, True, False),
    ],
)
def test_prepare_notification_retry_messages(
    retry_count, max_retries, run_state, expected_retry, expected_final_failure
):
    run_dict = {
        "metadata": {"project": "test"},
        "status": {"state": run_state},
        "spec": {},
    }
    if retry_count is not None:
        run_dict["status"]["retry_count"] = retry_count
    if max_retries is not None:
        run_dict["spec"]["retry"] = {"count": max_retries}

    run = mlrun.model.RunObject.from_dict(run_dict)
    notification = mlrun.model.Notification(name="notify", when=[run_state])
    notification_pusher = (
        mlrun.utils.notifications.notification_pusher.NotificationPusher([run])
    )
    message, _, _ = notification_pusher._prepare_notification_args(run, notification)

    if expected_retry and retry_count is not None:
        assert f"Retries attempted: {retry_count}" in message
    else:
        assert "Retries attempted" not in message

    if expected_final_failure:
        assert "Retry limit reached" in message
    else:
        assert "Retry limit reached" not in message


def _mock_async_response(monkeypatch, method, result):
    response_json_future = asyncio.Future()
    response_json_future.set_result(result)
    response_mock = unittest.mock.MagicMock()
    response_mock.json = unittest.mock.MagicMock(return_value=response_json_future)

    request_future = asyncio.Future()
    request_future.set_result(response_mock)

    requests_mock = unittest.mock.MagicMock(return_value=request_future)
    monkeypatch.setattr(aiohttp.ClientSession, method, requests_mock)

    return requests_mock


def _generate_run_result(
    state: str, error: Optional[str] = None, results: Optional[dict] = None
):
    run_example = {
        "status": {
            "notifications": {
                "Test": {"status": "pending", "sent_time": None, "reason": None}
            },
            "last_update": "2024-06-18T13:46:37.686443+00:00",
            "start_time": "2024-06-18T13:46:37.392158+00:00",
        },
        "metadata": {
            "uid": "b176e54e4ed24b28883aa69dce981601",
            "project": "test-remote-workflow",
            "name": "func-func",
            "labels": {
                "v3io_user": "admin",
                "kind": "job",
                "owner": "admin",
                "mlrun/client_version": "1.7.0-rc21",
                "mlrun/client_python_version": "3.9.18",
                "host": "func-func-8lvl8",
            },
            "iteration": 0,
        },
        "spec": {
            "function": "test-remote-workflow/func@8e0ddc3926470d5b97733679bb96738fa6dfd01b",
            "parameters": {"x": 1},
            "state_thresholds": {
                "pending_scheduled": "1h",
                "pending_not_scheduled": "-1",
                "image_pull_backoff": "1h",
                "executing": "24h",
            },
            "output_path": "v3io:///projects/test-remote-workflow/artifacts",
            "notifications": [
                {
                    "when": [
                        runtimes_constants.RunStates.error,
                        runtimes_constants.RunStates.completed,
                    ],
                    "name": "Test",
                    "params": {
                        "url": "https://webhook.site/5da7ac4d-39dc-4896-b18f-e13c5712a96a",
                        "method": "POST",
                    },
                    "message": "",
                    "status": "pending",
                    "condition": "",
                    "kind": "webhook",
                    "severity": "info",
                }
            ],
            "handler": "func",
        },
    }
    if state == runtimes_constants.RunStates.completed:
        run_example["status"]["results"] = results
        run_example["status"]["state"] = state
    elif state == runtimes_constants.RunStates.error:
        run_example["status"]["error"] = error
        run_example["status"]["state"] = state
    return run_example


class TestMailNotification:
    DEFAULT_PARAMS = {
        "server_host": "smtp.gmail.com",
        "server_port": 587,
        "sender_address": "sender@example.com",
        "username": "user",
        "password": "pass",
        "default_email_addresses": "a@example.com",
        "use_tls": True,
        "validate_certs": True,
        "start_tls": False,
    }
    MOCKED_HTML = "mocked_html"

    @pytest.mark.parametrize(
        "params, expectation",
        [
            (
                {
                    "server_host": "smtp.gmail.com",
                    "server_port": 587,
                    "sender_address": "sender@example.com",
                    "username": "user",
                    "password": "pass",
                    "email_addresses": "a@example.com",
                    "use_tls": True,
                    "validate_certs": True,
                    "start_tls": False,
                },
                does_not_raise(),
            ),
            (
                {
                    "server_host": "smtp.gmail.com",
                    "server_port": 587,
                    "sender_address": "sender@example.com",
                    "username": "user",
                    "password": "pass",
                    "email_addresses": ["a@example.com", "b@example.com"],
                    "use_tls": True,
                    "validate_certs": True,
                    "start_tls": False,
                },
                does_not_raise(),
            ),
            (
                {
                    "server_host": "smtp.gmail.com",
                    "server_port": 587,
                    "sender_address": "sender@example.com",
                    "username": "user",
                    "password": "pass",
                    "email_addresses": "a,b",
                    "use_tls": True,
                    "validate_certs": True,
                    "start_tls": False,
                },
                pytest.raises(ValueError, match="Invalid email address 'a'"),
            ),
            (
                {
                    "server_port": 587,
                    "sender_address": "sender@example.com",
                    "username": "user",
                    "password": "pass",
                    "email_addresses": "a@example.com",
                    "use_tls": True,
                    "validate_certs": True,
                    "start_tls": False,
                },
                pytest.raises(
                    ValueError,
                    match="Parameter 'server_host' is required for MailNotification",
                ),
            ),
            (
                {
                    "server_host": "smtp.gmail.com",
                    "server_port": 587,
                    "sender_address": "sender@example.com",
                    "username": "user",
                    "password": "pass",
                    "email_addresses": ["a@example.com", 1],
                    "use_tls": True,
                    "validate_certs": True,
                    "start_tls": False,
                },
                pytest.raises(
                    ValueError,
                    match="Email address '1' must be a string",
                ),
            ),
            (
                {
                    "server_host": "smtp.gmail.com",
                    "server_port": 587,
                    "sender_address": "sender@example.com",
                    "username": "user",
                    "password": "pass",
                    "email_addresses": ["a@example.com", "aaa"],
                    "use_tls": True,
                    "validate_certs": True,
                    "start_tls": False,
                },
                pytest.raises(ValueError, match="Invalid email address 'aaa'"),
            ),
            (
                {
                    "server_host": "smtp.gmail.com",
                    "server_port": 587,
                    "sender_address": "sender@example.com",
                    "username": "user",
                    "password": "pass",
                    "email_addresses": ["a@example.com", "aaa"],
                    "use_tls": "True",
                    "validate_certs": True,
                    "start_tls": False,
                },
                pytest.raises(
                    ValueError,
                    match="Parameter 'use_tls' must be a boolean for MailNotification",
                ),
            ),
            (  # missing username and password should pass validation - no auth case
                {
                    "server_host": "smtp.gmail.com",
                    "server_port": 587,
                    "sender_address": "sender@example.com",
                    "username": "",
                    "password": "",
                    "email_addresses": "a@example.com",
                    "use_tls": True,
                    "validate_certs": True,
                    "start_tls": False,
                },
                does_not_raise(),
            ),
            (  # missing password should pass validation - some servers allow username only auth
                {
                    "server_host": "smtp.gmail.com",
                    "server_port": 587,
                    "sender_address": "sender@example.com",
                    "username": "user",
                    "password": "",
                    "email_addresses": "a@example.com",
                    "use_tls": True,
                    "validate_certs": True,
                    "start_tls": False,
                },
                does_not_raise(),
            ),
            (  # missing username and password provided should fail validation
                {
                    "server_host": "smtp.gmail.com",
                    "server_port": 587,
                    "sender_address": "sender@example.com",
                    "username": "",
                    "password": "pass",
                    "email_addresses": "a@example.com",
                    "use_tls": True,
                    "validate_certs": True,
                    "start_tls": False,
                },
                pytest.raises(
                    ValueError,
                    match="Parameter 'username' is required when 'password' is provided for MailNotification",
                ),
            ),
        ],
    )
    def test_validate_mail_params(self, params, expectation):
        with expectation:
            mail.MailNotification.validate_params(params)

    @pytest.mark.parametrize(
        ["name", "params", "expected_params"],
        [
            (
                "missing_all_params",
                {},
                {},
            ),
            (
                "overriding_some_params",
                {
                    "server_host": "another@smtp.com",
                    "server_port": 589,
                },
                {
                    "server_host": "another@smtp.com",
                    "server_port": 589,
                },
            ),
            (
                "email_addresses_as_list",
                {
                    "email_addresses": ["a@b.com", "b@b.com", "c@c.com"],
                },
                {"email_addresses": "a@b.com,b@b.com,c@c.com,a@example.com"},
            ),
        ],
    )
    def test_enrich_default_params(self, name, params, expected_params):
        mlrun.utils.logger.debug(f"Testing {name}")
        enriched_params = mail.MailNotification.enrich_default_params(
            params, TestMailNotification.DEFAULT_PARAMS
        )
        default_params_copy = TestMailNotification.DEFAULT_PARAMS.copy()
        default_params_copy["email_addresses"] = default_params_copy.pop(
            "default_email_addresses"
        )
        default_params_copy.update(expected_params)
        assert enriched_params == default_params_copy

    @pytest.mark.parametrize(
        ["name", "params", "message", "severity", "expected"],
        [
            (
                "no_username_or_password",
                {},
                "test-message",
                "info",
                {
                    "subject": "[info] test-message",
                    "body": MOCKED_HTML,
                    # commented out to reflect the fact that username and password are not required
                    # "username": None,
                    # "password": None,
                },
            ),
            (
                "empty_params",
                {},
                "test-message",
                "info",
                {
                    "subject": "[info] test-message",
                    "body": MOCKED_HTML,
                    "username": None,
                    "password": None,
                },
            ),
            (
                "with_params_message",
                {"message_body_override": "runs: {{runs}}"},
                "test-message",
                "warning",
                {
                    "subject": "[warning] test-message",
                    "body": f"runs: {MOCKED_HTML}",
                    "username": None,
                    "password": None,
                },
            ),
            (
                "empty_auth_params",
                {"username": "", "password": ""},
                "test-message",
                "info",
                {
                    "subject": "[info] test-message",
                    "body": MOCKED_HTML,
                    "username": None,
                    "password": None,
                },
            ),
            (
                "with_auth_params",
                {"username": "user", "password": "pass"},
                "test-message",
                "info",
                {
                    "subject": "[info] test-message",
                    "body": MOCKED_HTML,
                    "username": "user",
                    "password": "pass",
                },
            ),
        ],
    )
    async def test_push(self, name, params, message, severity, expected):
        params.update(
            {
                "sender_address": "test@example.com",
                "server_host": "smtp.example.com",
            }
        )
        notification = mail.MailNotification(params=params)
        notification._get_html = unittest.mock.MagicMock(return_value=self.MOCKED_HTML)

        with unittest.mock.patch(
            "aiosmtplib.send", new_callable=unittest.mock.AsyncMock
        ):
            await notification.push(message, severity, [])
        assert notification.params["subject"] == expected["subject"]
        assert notification.params["body"] == expected["body"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ["username", "password", "expected_auth_kwargs"],
        [
            ("", "", {}),
            (None, None, {}),
            ("user", None, {"username": "user"}),
            ("user", "pass", {"username": "user", "password": "pass"}),
        ],
    )
    async def test_send_email_auth_handling(
        self, username, password, expected_auth_kwargs, monkeypatch: pytest.MonkeyPatch
    ):
        send_mock = unittest.mock.AsyncMock()
        monkeypatch.setattr(mail.aiosmtplib, "send", send_mock)

        await mail.MailNotification(
            params={
                "server_host": "smtp.example.com",
                "server_port": 25,
                "sender_address": "",
                "username": username,
                "password": password,
                "use_tls": False,
                "validate_certs": True,
                "start_tls": False,
            }
        ).push(
            message="Test Message",
            severity="info",
        )

        assert send_mock.await_count == 1
        assert send_mock.await_args.kwargs["hostname"] == "smtp.example.com"
        assert send_mock.await_args.kwargs["port"] == 25
        assert send_mock.await_args.kwargs["use_tls"] is False
        assert send_mock.await_args.kwargs["start_tls"] is False
        assert send_mock.await_args.kwargs["validate_certs"] is True

        for key, value in expected_auth_kwargs.items():
            assert send_mock.await_args.kwargs.get(key) == value

        for key in ("username", "password"):
            if key not in expected_auth_kwargs:
                assert key not in send_mock.await_args.kwargs


class DummyResponse:
    def __init__(self) -> None:
        self.status: int = 200

    def raise_for_status(self) -> None:
        pass


class DummySession:
    def __init__(self, json_serialize: Callable) -> None:
        self.request_args: Optional[dict[str, Any]] = None
        self._json_serialize = json_serialize

    async def post(
        self,
        url: str,
        headers: Optional[dict[str, str]] = None,
        json: Any = None,
        ssl: Optional[bool] = None,
    ) -> DummyResponse:
        await self._request(
            "post",
            url,
            headers=headers,
            json=json,
            ssl=ssl,
        )
        return DummyResponse()

    async def put(
        self,
        url: str,
        headers: Optional[dict[str, str]] = None,
        json: Any = None,
        ssl: Optional[bool] = None,
    ) -> DummyResponse:
        await self._request(
            "put",
            url,
            headers=headers,
            json=json,
            ssl=ssl,
        )
        return DummyResponse()

    async def _request(
        self,
        method: str,
        str_or_url: StrOrURL,
        **kwargs: Any,
    ) -> DummyResponse:
        if kwargs.get("data") is not None and kwargs.get("json") is not None:
            raise ValueError(
                "data and json parameters can not be used at the same time"
            )
        elif kwargs.get("json") is not None:
            data = self._json_serialize(kwargs["json"])
        self.request_args = {
            "method": method,
            "url": str_or_url,
            "data": data,
            **kwargs,
        }


class DummySessionContext:
    dummy_session_holder: dict[str, DummySession] = {}

    def __init__(self, json_serialize: Callable) -> None:
        self._session = DummySession(json_serialize)

    async def __aenter__(self) -> DummySession:
        self.dummy_session_holder["session"] = self._session
        return self._session

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        pass


@pytest.fixture
def client_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Patch the session factory used by webhook notifications so tests
    get a DummySessionContext instead of a real aiohttp session.
    """

    def make_dummy_session(self, **kwargs):
        return DummySessionContext(**kwargs)

    monkeypatch.setattr(
        mlrun.utils.notifications.notification.webhook.TimedHTTPClient,
        "session",
        make_dummy_session,
    )


@dataclass
class DummyRun:
    project: str = "proj"
    name: str = "run1"
    host: Optional[str] = None
    state: str = "s"
    error: Optional[str] = None
    results: Optional[list[Any]] = None
    metadata: dict[str, Any] = field(init=False)
    status: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        self.metadata = {"project": self.project, "name": self.name, "labels": {}}
        if self.host:
            self.metadata["labels"]["host"] = self.host
        self.status = {"state": self.state}
        if self.error:
            self.status["error"] = self.error
        elif self.results is not None:
            self.status["results"] = self.results

    def to_dict(self) -> dict[str, Any]:
        return {"metadata": self.metadata, "status": self.status}


@dataclass
class DummyAlert:
    name: str
    project: str
    severity: str
    summary: Optional[str] = None


@dataclass
class DummyEntity:
    ids: list[Any]


@dataclass
class DummyEvent:
    value_dict: dict[str, Any]
    ids: list[Any]
    entity: DummyEntity = field(init=False)

    def __post_init__(self) -> None:
        self.entity = DummyEntity(self.ids)


@pytest.mark.asyncio
async def test_push_full_payload(client_session: Any) -> None:
    runs: list[DummyRun] = [
        DummyRun(
            project="p", name="n", host="h", state="running", error="err"
        ).to_dict()
    ]
    alert = DummyAlert("alertName", "alertProj", "alertSeverity", summary="summaryText")
    event = DummyEvent({"key": "val"}, ["id1", "id2"])
    custom_html = "<b>html</b>"

    notif = mlrun.utils.notifications.notification.webhook.WebhookNotification(
        params={
            "url": "https://example.com/hook",
            "method": "PUT",
            "headers": {"H": "v"},
        }
    )
    await notif.push(
        message="hello",
        severity="origSeverity",
        runs=runs,
        custom_html=custom_html,
        alert=alert,
        event_data=event,
    )

    session = DummySessionContext.dummy_session_holder["session"]
    args = session.request_args or {}
    assert args["method"] == "put"
    assert args["url"] == "https://example.com/hook"
    assert args["headers"] == {"H": "v"}

    payload = args["json"]

    assert payload["message"] == "hello"
    assert payload["severity"] == alert.severity
    assert payload["runs"] == runs
    assert payload["name"] == alert.name
    assert payload["project"] == alert.project
    assert payload["summary"] == "summaryText"
    assert payload["value"] == {"key": "val"}
    assert payload["id"] == "id1"
    assert payload["custom_html"] == custom_html
    assert args["ssl"] is None
    raw_data = args["data"]
    assert raw_data == notif._encoder(payload)


@pytest.mark.asyncio
async def test_override_list_passthrough(client_session: Any) -> None:
    override_body = ["a", "b"]
    notif = mlrun.utils.notifications.notification.webhook.WebhookNotification(
        params={
            "url": "http://example.com",
            "override_body": override_body,
            "verify_ssl": True,
        },
    )
    await notif.push("ignored")
    session = DummySessionContext.dummy_session_holder["session"]
    assert session.request_args and session.request_args["json"] == override_body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "override_body, runs, key, expected",
    [
        (
            {"dict": {"x": datetime(2025, 1, 1, 0, 0, 0)}},
            None,
            "dict",
            {"x": datetime(2025, 1, 1, 0, 0, 0)},
        ),
        ({"float": 1.23}, None, "float", 1.23),
        ({"bool": True}, None, "bool", True),
        ({"none": None}, None, "none", None),
        ({"list": [1, 2, "a"]}, None, "list", [1, 2, "a"]),
        ({"mixed": "val{{ runs }}end"}, [], "mixed", "val[]end"),
    ],
)
async def test_override_values(
    client_session: Any,
    override_body: dict[str, Any],
    runs: Optional[list[Any]],
    key: str,
    expected: Any,
) -> None:
    notif = mlrun.utils.notifications.notification.webhook.WebhookNotification(
        params={
            "url": "http://example.com",
            "override_body": override_body.copy(),
        }
    )
    await notif.push("ignored", runs=runs)
    sent = DummySessionContext.dummy_session_holder["session"].request_args["json"]
    assert sent[key] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url, verify_ssl, expected_ssl",
    [
        ("https://example.com", None, None),
        ("https://example.com", False, False),
        ("http://example.com", True, None),
    ],
)
async def test_ssl_logic(
    client_session: Any,
    url: str,
    verify_ssl: Optional[bool],
    expected_ssl: Optional[bool],
) -> None:
    params: dict[str, Any] = {"url": url}
    if verify_ssl is not None:
        params["verify_ssl"] = verify_ssl
    notification = mlrun.utils.notifications.notification.webhook.WebhookNotification(
        params=params
    )
    await notification.push("ignored")
    ssl_arg = DummySessionContext.dummy_session_holder["session"].request_args["ssl"]
    assert ssl_arg is expected_ssl
