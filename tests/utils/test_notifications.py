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
import hashlib
import json
import unittest.mock
from contextlib import nullcontext as does_not_raise

import aiohttp
import pytest
import tabulate

import mlrun.common.schemas.notification
import mlrun.utils.notifications
import server.api.api.utils
import server.api.constants
import server.api.crud
from mlrun.utils.notifications.notification.webhook import WebhookNotification


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


@pytest.mark.parametrize(
    "notification_kind",
    [
        mlrun.common.schemas.notification.NotificationKind.console,
        mlrun.common.schemas.notification.NotificationKind.slack,
        mlrun.common.schemas.notification.NotificationKind.git,
        mlrun.common.schemas.notification.NotificationKind.webhook,
        mlrun.common.schemas.notification.NotificationKind.ipython,
    ],
)
def test_notification_reason(notification_kind):
    error_exc = Exception("Blew up")
    run = mlrun.model.RunObject.from_dict({"status": {"state": "completed"}})
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
    "override_body",
    [({"message": "runs: {{runs}}"}), ({"message": "runs: {{ runs }}"})],
)
async def test_webhook_override_body_job_succeed(monkeypatch, override_body):
    requests_mock = _mock_async_response(monkeypatch, "post", {"id": "response-id"})
    runs = _generate_run_result(state="completed", results={"return": 1})
    await WebhookNotification(
        params={"override_body": override_body, "url": "http://test.com"}
    ).push("test-message", "info", [runs])
    expected_body = {
        "message": "runs: [{'project': 'test-remote-workflow', 'name': 'func-func', 'host': 'func-func-8lvl8', "
        "'status': {'state': 'completed', 'results': {'return': 1}}}]"
    }
    requests_mock.assert_called_once_with(
        "http://test.com", headers={}, json=expected_body, ssl=None
    )


@pytest.mark.parametrize(
    "override_body",
    [({"message": "runs: {{runs}}"}), ({"message": "runs: {{ runs }}"})],
)
async def test_webhook_override_body_job_failed(monkeypatch, override_body):
    requests_mock = _mock_async_response(monkeypatch, "post", {"id": "response-id"})
    runs = _generate_run_result(
        state="error", error='can only concatenate str (not "int") to str'
    )
    await WebhookNotification(
        params={"override_body": override_body, "url": "http://test.com"}
    ).push("test-message", "info", [runs])
    expected_body = {
        "message": "runs: [{'project': 'test-remote-workflow', 'name': 'func-func', 'host': 'func-func-8lvl8', "
        "'status': {'state': 'error', 'error': 'can only concatenate str (not \"int\") to str'}}]"
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
                    "status": {"state": "completed"},
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
                    "status": {"state": "error"},
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
        server.api.crud.Secrets, "store_project_secrets", _store_project_secrets
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
    server.api.api.utils.mask_notification_params_on_task(
        run, server.api.constants.MaskOperations.CONCEAL
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
        server.api.crud.Secrets, "get_project_secret", _get_valid_project_secret
    )

    unmasked_run = server.api.api.utils.unmask_notification_params_secret_on_task(
        db_mock, db_session_mock, copy.deepcopy(run)
    )
    assert "sensitive" in unmasked_run.spec.notifications[0].secret_params
    assert "secret" not in unmasked_run.spec.notifications[0].secret_params
    assert unmasked_run.spec.notifications[0].secret_params == secret_value

    monkeypatch.setattr(
        server.api.crud.Secrets, "get_project_secret", _get_invalid_project_secret
    )
    unmasked_run = server.api.api.utils.unmask_notification_params_secret_on_task(
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


def _generate_run_result(state: str, error: str = None, results: dict = None):
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
                    "when": ["error", "completed"],
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
    if state == "completed":
        run_example["status"]["results"] = results
        run_example["status"]["state"] = state
    elif state == "error":
        run_example["status"]["error"] = error
        run_example["status"]["state"] = state
    return run_example
