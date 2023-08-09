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
import copy
import json
import ssl
import unittest.mock
from contextlib import nullcontext as does_not_raise

import aiohttp
import pytest
import tabulate

import mlrun.api.api.utils
import mlrun.api.crud
import mlrun.common.schemas.notification
import mlrun.utils.notifications


@pytest.mark.parametrize(
    "notification_kind", mlrun.common.schemas.notification.NotificationKind
)
def test_load_notification(notification_kind):
    run_uid = "test-run-uid"
    notification_name = "test-notification-name"
    when_state = "completed"
    notification = mlrun.model.Notification.from_dict(
        {
            "kind": notification_kind,
            "when": when_state,
            "status": "pending",
            "name": notification_name,
        }
    )
    run = mlrun.model.RunObject.from_dict(
        {
            "metadata": {"uid": run_uid},
            "spec": {"notifications": [notification]},
            "status": {"state": when_state},
        }
    )

    notification_pusher = (
        mlrun.utils.notifications.notification_pusher.NotificationPusher([run])
    )
    notification_pusher._load_notification(run, notification)
    loaded_notifications = (
        notification_pusher._sync_notifications
        + notification_pusher._async_notifications
    )
    assert len(loaded_notifications) == 1
    assert loaded_notifications[0][0].name == notification_name


@pytest.mark.parametrize(
    "when,condition,run_state,notification_previously_sent,expected",
    [
        (["completed"], "", "completed", False, True),
        (["completed"], "", "completed", True, False),
        (["completed"], "", "error", False, False),
        (["completed"], "", "error", True, False),
        (["completed"], "> 4", "completed", False, True),
        (["completed"], "> 4", "completed", True, False),
        (["completed"], "< 4", "completed", False, False),
        (["completed"], "< 4", "completed", True, False),
        (["error"], "", "completed", False, False),
        (["error"], "", "completed", True, False),
        (["error"], "", "error", False, True),
        (["error"], "", "error", True, False),
        (["completed", "error"], "", "completed", False, True),
        (["completed", "error"], "", "completed", True, False),
        (["completed", "error"], "", "error", False, True),
        (["completed", "error"], "", "error", True, False),
        (["completed", "error"], "> 4", "completed", False, True),
        (["completed", "error"], "> 4", "completed", True, False),
        (["completed", "error"], "> 4", "error", False, True),
        (["completed", "error"], "> 4", "error", True, False),
        (["completed", "error"], "< 4", "completed", False, False),
        (["completed", "error"], "< 4", "completed", True, False),
        (["completed", "error"], "< 4", "error", False, True),
        (["completed", "error"], "< 4", "error", True, False),
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
        {"status": {"state": "completed", "results": {"val": 5}}}
    )
    notification = mlrun.model.Notification.from_dict(
        {"when": ["completed"], "condition": condition, "status": "pending"}
    )

    notification_pusher = (
        mlrun.utils.notifications.notification_pusher.NotificationPusher([run])
    )
    assert notification_pusher._should_notify(run, notification)


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
    git_notification = mlrun.utils.notifications.GitNotification("git", params)
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
    webhook_notification = mlrun.utils.notifications.WebhookNotification(
        "webhook",
        {
            "url": test_url,
            "method": test_method,
            "headers": test_headers,
        },
    )
    await webhook_notification.push(test_message, test_severity, test_runs_info)

    call_args = requests_mock.call_args.args
    call_kwargs = requests_mock.call_args.kwargs
    assert call_args[0] == test_url
    assert call_kwargs.get("headers") == test_headers
    assert call_kwargs.get("json") == {
        "message": test_message,
        "severity": test_severity,
        "runs": test_runs_info,
    }
    assert call_kwargs.get("ssl").verify_mode == ssl.CERT_REQUIRED

    webhook_notification.params["override_body"] = test_override_body

    await webhook_notification.push("test-message", "info", ["some-run"])

    call_args = requests_mock.call_args
    assert call_args.args[0] == test_url
    assert call_args.kwargs.get("headers") == test_headers
    assert call_args.kwargs.get("json") == test_override_body


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
        mlrun.utils.notifications.ConsoleNotification, "push", mock_console_push
    )
    monkeypatch.setattr(
        mlrun.utils.notifications.IPythonNotification, "push", mock_ipython_push
    )
    monkeypatch.setattr(
        mlrun.utils.notifications.IPythonNotification, "active", ipython_active
    )

    custom_notification_pusher.push("test-message", "info", [])

    assert mock_console_push.call_count == expected_console_call_amount
    assert mock_ipython_push.call_count == expected_ipython_call_amount


def test_notification_params_masking_on_run(monkeypatch):
    def _store_project_secrets(*args, **kwargs):
        pass

    monkeypatch.setattr(
        mlrun.api.crud.Secrets, "store_project_secrets", _store_project_secrets
    )
    run_uid = "test-run-uid"
    run = {
        "metadata": {"uid": run_uid, "project": "test-project"},
        "spec": {
            "notifications": [
                {"when": "completed", "params": {"sensitive": "sensitive-value"}}
            ]
        },
    }
    mlrun.api.api.utils.mask_notification_params_on_task(run)
    assert "sensitive" not in run["spec"]["notifications"][0]["params"]
    assert "secret" in run["spec"]["notifications"][0]["params"]
    assert (
        run["spec"]["notifications"][0]["params"]["secret"]
        == f"mlrun.notifications.{run_uid}"
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
                    "params": {"secret": "secret-name"},
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
        mlrun.api.crud.Secrets, "get_project_secret", _get_valid_project_secret
    )

    unmasked_run = mlrun.api.api.utils.unmask_notification_params_secret_on_task(
        db_mock, db_session_mock, copy.deepcopy(run)
    )
    assert "sensitive" in unmasked_run.spec.notifications[0].params
    assert "secret" not in unmasked_run.spec.notifications[0].params
    assert unmasked_run.spec.notifications[0].params == secret_value

    monkeypatch.setattr(
        mlrun.api.crud.Secrets, "get_project_secret", _get_invalid_project_secret
    )
    unmasked_run = mlrun.api.api.utils.unmask_notification_params_secret_on_task(
        db_mock, db_session_mock, copy.deepcopy(run)
    )
    assert len(unmasked_run.spec.notifications) == 0
    db_mock.store_run_notifications.assert_called_once()
    args, _ = db_mock.store_run_notifications.call_args
    assert args[1][0].status == mlrun.common.schemas.NotificationStatus.ERROR


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
            "when": ["completed", "error"],
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
            "message": "completed",
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
        "when": ["completed"],
        "condition": "",
        "name": "",
    }

    for field, expected_value in notification_fields.items():
        value = getattr(notification, field)
        assert (
            value == expected_value
        ), f"{field} field value is {value}, expected {expected_value}"


@pytest.mark.parametrize(
    "notification_kwargs,expectation",
    NOTIFICATION_VALIDATION_PARMETRIZE,
)
def test_notification_validation_on_run(monkeypatch, notification_kwargs, expectation):
    notification = mlrun.model.Notification(
        name="test-notification", when=["completed"]
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
        name="test-notification", when=["completed"]
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
        name="test-notification", when=["completed"]
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
        name=notification1_name, when=["completed"]
    )
    notification2 = mlrun.model.Notification(
        name=notification2_name, when=["completed"]
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
