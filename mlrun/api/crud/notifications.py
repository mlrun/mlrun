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
import typing

import sqlalchemy.orm

import mlrun.api.api.utils
import mlrun.api.db.sqldb.db
import mlrun.api.utils.singletons.db
import mlrun.common.schemas
import mlrun.utils.singleton

kind_to_function_names = {
    "run": mlrun.api.db.sqldb.db.SQLDB.set_run_notifications.__name__
}

kind_to_indentifier_key = {
    "run": "uid",
}


class Notifications(
    metaclass=mlrun.utils.singleton.Singleton,
):
    def store_run_notifications(
        self,
        session: sqlalchemy.orm.Session,
        notification_objects: typing.List[mlrun.model.Notification],
        run_uid: str,
        project: str = None,
    ):
        project = project or mlrun.mlconf.default_project
        notification_objects_to_store = []
        for notification_object in notification_objects:
            notification_objects_to_store.append(
                mlrun.api.api.utils.mask_notification_params_with_secret(
                    project, run_uid, notification_object
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
        project: str,
        notifications: typing.List[mlrun.common.schemas.Notification],
        notification_parent: mlrun.common.schemas.NotificationParent,
    ):
        set_func = kind_to_function_names.get(notification_parent.kind, {})
        if not set_func:
            raise mlrun.errors.MLRunNotFoundError(
                f"couldn't find set notification function for object kind: {notification_parent.kind}"
            )
        identifier_key = kind_to_indentifier_key.get(notification_parent.kind, {})
        if not identifier_key:
            raise mlrun.errors.MLRunNotFoundError(
                f"couldn't find identifier key for object kind: {notification_parent.kind}"
            )
        mlrun.model.Notification.validate_notification_uniqueness(notifications)

        notification_objects_to_set = []
        for notification_object in notifications:
            notification_objects_to_set.append(
                mlrun.api.api.utils.mask_notification_params_with_secret(
                    project,
                    getattr(notification_parent.identifier, identifier_key),
                    notification_object,
                )
            )

        getattr(mlrun.api.utils.singletons.db.get_db(), set_func)(
            session=db_session,
            project=project,
            notifications=notification_objects_to_set,
            identifier=notification_parent.identifier,
        )
