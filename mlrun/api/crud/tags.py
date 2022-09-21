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

import mlrun.api.db.sqldb.db
import mlrun.api.schemas
import mlrun.api.utils.projects.remotes.follower
import mlrun.api.utils.singletons.db
import mlrun.api.utils.singletons.project_member
import mlrun.config
import mlrun.errors
import mlrun.utils.singleton

kind_to_function = {
    "artifact": {
        "overwrite": mlrun.api.db.sqldb.db.SQLDB.overwrite_artifacts_with_tag.__name__,
        "append": mlrun.api.db.sqldb.db.SQLDB.append_tag_to_artifacts.__name__,
        "delete": mlrun.api.db.sqldb.db.SQLDB.delete_tag_from_artifacts.__name__,
    }
}


class Tags(
    metaclass=mlrun.utils.singleton.Singleton,
):
    @staticmethod
    def overwrite_object_tags_with_tag(
        db_session: sqlalchemy.orm.Session,
        project: str,
        tag: str,
        tag_objects: mlrun.api.schemas.TagObjects,
    ):
        overwrite_func = kind_to_function.get(tag_objects.kind, {}).get("overwrite")
        getattr(mlrun.api.utils.singletons.db.get_db(), overwrite_func)(
            session=db_session,
            project=project,
            tag=tag,
            identifiers=tag_objects.identifiers,
        )

    @staticmethod
    def append_tag_to_objects(
        db_session: sqlalchemy.orm.Session,
        project: str,
        tag: str,
        tag_objects: mlrun.api.schemas.TagObjects,
    ):
        append_func = kind_to_function.get(tag_objects.kind, {}).get("append")
        getattr(mlrun.api.utils.singletons.db.get_db(), append_func)(
            session=db_session,
            project=project,
            tag=tag,
            identifiers=tag_objects.identifiers,
        )

    @staticmethod
    def delete_tag_from_objects(
        db_session: sqlalchemy.orm.Session,
        project: str,
        tag: str,
        tag_objects: mlrun.api.schemas.TagObjects,
    ):
        delete_func = kind_to_function.get(tag_objects.kind, {}).get("delete")
        getattr(mlrun.api.utils.singletons.db.get_db(), delete_func)(
            session=db_session,
            project=project,
            tag=tag,
            identifiers=tag_objects.identifiers,
        )
