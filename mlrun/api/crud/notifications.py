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
#
import typing

import sqlalchemy.orm

import mlrun.api.api.utils
import mlrun.api.db.sqldb.db
import mlrun.api.utils.scheduler
import mlrun.api.utils.singletons.db
import mlrun.api.utils.singletons.scheduler
import mlrun.common.schemas
import mlrun.utils.singleton


class Notifications(
    metaclass=mlrun.utils.singleton.Singleton,
):
    def store_run_notifications(
        self,
        session: sqlalchemy.orm.Session,
        notification_objects: typing.List[mlrun.model.Notification],
        run_uid: str,
        project: str = None,
        mask_params: bool = True,
    ):
        project = project or mlrun.mlconf.default_project

        # we don't mask the notification params when it's a status update as they are already masked
        notification_objects_to_store = notification_objects
        if mask_params:
            notification_objects_to_store = (
                mlrun.api.api.utils.validate_and_mask_notification_list(
                    notification_objects, run_uid, project
                )
            )

        mlrun.api.utils.singletons.db.get_db().store_run_notifications(
            session, notification_objects_to_store, run_uid, project
        )

    def list_run_notifications(
        self,
        session: sqlalchemy.orm.Session,
        run_uid: str,
        project: str = "",
    ) -> typing.List[mlrun.model.Notification]:
        project = project or mlrun.mlconf.default_project
        return mlrun.api.utils.singletons.db.get_db().list_run_notifications(
            session, run_uid, project
        )

    def delete_run_notifications(
        self,
        session: sqlalchemy.orm.Session,
        name: str = None,
        run_uid: str = None,
        project: str = None,
    ):
        project = project or mlrun.mlconf.default_project

        # Delete notification param project secret
        notifications = [
            notification
            for notification in self.list_run_notifications(session, run_uid, project)
            if notification.name == name
        ]
        if notifications:
            # unique constraint on name, run_uid, project, so the list will contain one item at most
            notification = notifications[0]
            mlrun.api.api.utils.delete_notification_params_secret(project, notification)

        mlrun.api.utils.singletons.db.get_db().delete_run_notifications(
            session, name, run_uid, project
        )

    @staticmethod
    def set_object_notifications(
        db_session: sqlalchemy.orm.Session,
        auth_info: mlrun.common.schemas.AuthInfo,
        project: str,
        notifications: typing.List[mlrun.common.schemas.Notification],
        notification_parent: typing.Union[
            mlrun.common.schemas.RunIdentifier, mlrun.common.schemas.ScheduleIdentifier
        ],
    ):
        """
        Sets notifications on given object (run or schedule, might be extended in the future).
        This will replace any existing notifications.
        :param db_session: DB session
        :param auth_info: Authorization info
        :param project: Project name
        :param notifications: List of notifications to set
        :param notification_parent: Identifier of the object on which to set the notifications
        """
        set_notification_methods = {
            "run": {
                "factory": mlrun.api.utils.singletons.db.get_db,
                "method_name": mlrun.api.db.sqldb.db.SQLDB.set_run_notifications.__name__,
                "identifier_key": "uid",
            },
            "schedule": {
                "factory": mlrun.api.utils.singletons.scheduler.get_scheduler,
                "method_name": mlrun.api.utils.scheduler.Scheduler.set_schedule_notifications.__name__,
                "identifier_key": "name",
            },
        }

        set_notification_method = set_notification_methods.get(
            notification_parent.kind, {}
        )
        factory = set_notification_method.get("factory")
        if not factory:
            raise mlrun.errors.MLRunNotFoundError(
                f"couldn't find factory for object kind: {notification_parent.kind}"
            )
        set_func = set_notification_method.get("method_name")
        if not set_func:
            raise mlrun.errors.MLRunNotFoundError(
                f"couldn't find set notification function for object kind: {notification_parent.kind}"
            )
        identifier_key = set_notification_method.get("identifier_key")
        if not identifier_key:
            raise mlrun.errors.MLRunNotFoundError(
                f"couldn't find identifier key for object kind: {notification_parent.kind}"
            )

        notification_objects_to_set = (
            mlrun.api.api.utils.validate_and_mask_notification_list(
                notifications,
                getattr(notification_parent, identifier_key),
                project,
            )
        )

        getattr(factory(), set_func)(
            session=db_session,
            project=project,
            notifications=notification_objects_to_set,
            identifier=notification_parent,
            auth_info=auth_info,
        )
