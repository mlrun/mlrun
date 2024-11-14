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
import asyncio
import json
import typing
from hashlib import sha224

from sqlalchemy.orm import Session

import mlrun.common.schemas
import mlrun.errors
from mlrun.errors import err_to_str
from mlrun.utils import get_in, logger

import framework.constants
import framework.db.base
import framework.utils.notifications.notification_pusher as notification_pusher
import framework.utils.singletons.k8s
import services.api.crud


def delete_notification_params_secret(
    project: str, notification_object: mlrun.model.Notification
) -> None:
    secret_params = notification_object.secret_params or {}
    params_secret = secret_params.get("secret", "")
    if not params_secret:
        return

    k8s = framework.utils.singletons.k8s.get_k8s_helper()
    if not k8s:
        raise mlrun.errors.MLRunRuntimeError(
            "Not running in k8s environment, cannot delete notification params secret"
        )

    if services.api.crud.Secrets().is_internal_project_secret_key(params_secret):
        services.api.crud.Secrets().delete_project_secret(
            project,
            mlrun.common.schemas.SecretProviderName.kubernetes,
            secret_key=params_secret,
            allow_internal_secrets=True,
            allow_secrets_from_k8s=True,
        )


def mask_notification_params_on_task(
    task: dict,
    action: framework.constants.MaskOperations,
):
    """
    Mask notification config params from the task dictionary
    :param task:    The task object to mask
    :param action:  The masking operation to perform on the notification config params (conceal/redact)
    """
    run_uid = get_in(task, "metadata.uid")
    project = get_in(task, "metadata.project")
    notifications = task.get("spec", {}).get("notifications", [])
    if notifications:
        notifications_objects = _mask_notifications_params(
            run_uid,
            project,
            [
                mlrun.model.Notification.from_dict(notification)
                for notification in notifications
            ],
            action,
        )
        task.setdefault("spec", {})["notifications"] = [
            notification.to_dict() for notification in notifications_objects
        ]


def mask_notification_params_on_task_object(
    task: mlrun.model.RunObject,
    action: framework.constants.MaskOperations,
):
    """
    Mask notification config params from the task object
    :param task:    The task object to mask
    :param action:  The masking operation to perform on the notification config params (conceal/redact)
    """
    run_uid = task.metadata.uid
    project = task.metadata.project
    notifications = task.spec.notifications
    if notifications:
        task.spec.notifications = _mask_notifications_params(
            run_uid, project, notifications, action
        )


def unmask_notification_params_secret_on_task(
    db: framework.db.base.DBInterface,
    db_session: Session,
    run: typing.Union[dict, mlrun.model.RunObject],
):
    if isinstance(run, dict):
        run = mlrun.model.RunObject.from_dict(run)

    notifications = []
    for notification in run.spec.notifications:
        invalid_notifications = []
        try:
            notifications.append(
                unmask_notification_params_secret(run.metadata.project, notification)
            )
        except Exception as exc:
            logger.warning(
                "Failed to unmask notification params, notification will not be sent",
                project=run.metadata.project,
                run_uid=run.metadata.uid,
                notification=notification.name,
                exc=err_to_str(exc),
            )
            # set error status in order to later save the db
            notification.status = mlrun.common.schemas.NotificationStatus.ERROR
            invalid_notifications.append(notification)

        if invalid_notifications:
            db.store_run_notifications(
                db_session,
                invalid_notifications,
                run.metadata.uid,
                run.metadata.project,
            )

    run.spec.notifications = notifications

    return run


def unmask_notification_params_secret(
    project: str, notification_object: mlrun.model.Notification
) -> mlrun.model.Notification:
    secret_params = notification_object.secret_params or {}
    params_secret = secret_params.get("secret", "")
    if not params_secret:
        return notification_object

    k8s = framework.utils.singletons.k8s.get_k8s_helper()
    if not k8s:
        raise mlrun.errors.MLRunRuntimeError(
            "Not running in k8s environment, cannot load notification params secret"
        )

    secret = services.api.crud.Secrets().get_project_secret(
        project,
        mlrun.common.schemas.SecretProviderName.kubernetes,
        secret_key=params_secret,
        allow_internal_secrets=True,
        allow_secrets_from_k8s=True,
    )

    if secret is None:
        # we don't want to provide a message that is too detailed due to security considerations here
        raise mlrun.errors.MLRunPreconditionFailedError()

    notification_object.secret_params = json.loads(secret)

    return notification_object


def validate_and_mask_notification_list(
    notifications: list[
        typing.Union[mlrun.model.Notification, mlrun.common.schemas.Notification, dict]
    ],
    parent: str,
    project: str,
) -> list[mlrun.model.Notification]:
    """
    Validates notification schema, uniqueness and masks notification params with secret if needed.
    If at least one of the validation steps fails, the function will raise an exception and cause the API to return
    an error response.
    :param notifications: list of notification objects
    :param parent: parent identifier
    :param project: project name
    :return: list of validated and masked notification objects
    """
    notification_objects = []

    for notification in notifications:
        if isinstance(notification, dict):
            notification_object = mlrun.model.Notification.from_dict(notification)
        elif isinstance(notification, mlrun.common.schemas.Notification):
            notification_object = mlrun.model.Notification.from_dict(
                notification.dict()
            )
        elif isinstance(notification, mlrun.model.Notification):
            notification_object = notification
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "notification must be a dict or a Notification object"
            )

        # validate notification schema
        mlrun.common.schemas.Notification(**notification_object.to_dict())

        default_notification_params = notification_pusher.RunNotificationPusher.resolve_notifications_default_params()
        notification_object.validate_notification_params(default_notification_params)

        notification_objects.append(notification_object)

    mlrun.model.Notification.validate_notification_uniqueness(notification_objects)

    return [
        _conceal_notification_params_with_secret(project, parent, notification_object)
        for notification_object in notification_objects
    ]


def _mask_notifications_params(
    run_uid: str,
    project: str,
    notifications: list[mlrun.model.Notification],
    action: framework.constants.MaskOperations,
):
    """
    Mask notification config params from notifications list
    :param run_uid:         The run UID
    :param project:         The project name
    :param notifications:   The list of notification objects to mask
    :param action:          The masking operation to perform on the notification config params (conceal/redact)
    """
    mask_op = _notification_params_mask_op(action)
    return [mask_op(project, run_uid, notification) for notification in notifications]


def _notification_params_mask_op(
    action,
) -> typing.Callable[[str, str, mlrun.model.Notification], mlrun.model.Notification]:
    return {
        framework.constants.MaskOperations.CONCEAL: _conceal_notification_params_with_secret,
        framework.constants.MaskOperations.REDACT: _redact_notification_params,
    }[action]


def _conceal_notification_params_with_secret(
    project: str, parent: str, notification_object: mlrun.model.Notification
) -> mlrun.model.Notification:
    if (
        notification_object.secret_params
        and "secret" not in notification_object.secret_params
    ):
        # create secret key from a hash of the secret params. this will allow multiple notifications with the same
        # params to share the same secret (saving secret storage space).
        # TODO: add holders to the secret content, so we can monitor when all runs that use the secret are deleted.
        #       as we currently don't delete runs unless the project is deleted (in which case, the entire secret is
        #       deleted), we don't need the mechanism yet.
        secret_key = services.api.crud.Secrets().generate_client_project_secret_key(
            services.api.crud.SecretsClientType.notifications,
            _generate_notification_secret_key(notification_object),
        )
        services.api.crud.Secrets().store_project_secrets(
            project,
            mlrun.common.schemas.SecretsData(
                provider=mlrun.common.schemas.SecretProviderName.kubernetes,
                secrets={secret_key: json.dumps(notification_object.secret_params)},
            ),
            allow_internal_secrets=True,
        )
        notification_object.secret_params = {"secret": secret_key}

    return notification_object


def _redact_notification_params(
    project: str, parent: str, notification_object: mlrun.model.Notification
) -> mlrun.model.Notification:
    if not notification_object.secret_params:
        return notification_object

    # If the notification params contain a secret key, we consider them concealed and don't redact them
    if "secret" in notification_object.secret_params:
        return notification_object

    for param in notification_object.secret_params:
        notification_object.secret_params[param] = "REDACTED"

    return notification_object


def _generate_notification_secret_key(
    notification_object: mlrun.model.Notification,
) -> str:
    # hash notification params to generate a unique secret key
    return sha224(
        json.dumps(notification_object.secret_params, sort_keys=True).encode("utf-8")
    ).hexdigest()
