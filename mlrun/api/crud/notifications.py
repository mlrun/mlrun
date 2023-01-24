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
import mlrun.api.utils.singletons.db
import mlrun.utils.singleton


class Notifications(
    metaclass=mlrun.utils.singleton.Singleton,
):
    def store_notifications(
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

        mlrun.api.utils.singletons.db.get_db().store_notifications(
            session, notification_objects_to_store, run_uid, project
        )

    def list_notifications(
        self,
        session: sqlalchemy.orm.Session,
        run_uid: str,
        project: str = "",
    ) -> typing.List[mlrun.model.Notification]:
        project = project or mlrun.mlconf.default_project
        return mlrun.api.utils.singletons.db.get_db().list_notifications(
            session, run_uid, project
        )

    def delete_notifications(
        self,
        session: sqlalchemy.orm.Session,
        name: str = None,
        run_uid: str = None,
        project: str = None,
    ):
        project = project or mlrun.mlconf.default_project
        mlrun.api.utils.singletons.db.get_db().delete_notifications(
            session, name, run_uid, project
        )
