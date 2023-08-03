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
import mlrun.api.runtime_handlers
import mlrun.api.utils.projects.remotes.follower
import mlrun.api.utils.singletons.db
import mlrun.api.utils.singletons.project_member
import mlrun.common.schemas
import mlrun.config
import mlrun.errors
import mlrun.utils.singleton


class Functions(
    metaclass=mlrun.utils.singleton.Singleton,
):
    def store_function(
        self,
        db_session: sqlalchemy.orm.Session,
        function: dict,
        name: str,
        project: str = mlrun.mlconf.default_project,
        tag: str = "",
        versioned: bool = False,
        auth_info: mlrun.common.schemas.AuthInfo = None,
    ) -> str:
        project = project or mlrun.mlconf.default_project
        if auth_info:
            function_obj = mlrun.new_function(
                name=name, project=project, runtime=function, tag=tag
            )
            # not raising exception if no access key was provided as the store of the function can be part of
            # intermediate steps or temporary objects which might not be executed at any phase and therefore we don't
            # want to enrich if user didn't requested.
            # (The way user will request to generate is by passing $generate in the metadata.credentials.access_key)
            mlrun.api.api.utils.ensure_function_auth_and_sensitive_data_is_masked(
                function_obj, auth_info, allow_empty_access_key=True
            )
            function = function_obj.to_dict()

        return mlrun.api.utils.singletons.db.get_db().store_function(
            db_session,
            function,
            name,
            project,
            tag,
            versioned,
        )

    def get_function(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        project: str = mlrun.mlconf.default_project,
        tag: str = "",
        hash_key: str = "",
    ) -> dict:
        project = project or mlrun.mlconf.default_project
        return mlrun.api.utils.singletons.db.get_db().get_function(
            db_session, name, project, tag, hash_key
        )

    def delete_function(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        name: str,
    ):
        return mlrun.api.utils.singletons.db.get_db().delete_function(
            db_session, project, name
        )

    def list_functions(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str = mlrun.mlconf.default_project,
        name: str = "",
        tag: str = "",
        labels: typing.List[str] = None,
        hash_key: str = "",
    ) -> typing.List:
        project = project or mlrun.mlconf.default_project
        if labels is None:
            labels = []
        return mlrun.api.utils.singletons.db.get_db().list_functions(
            session=db_session,
            name=name,
            project=project,
            tag=tag,
            labels=labels,
            hash_key=hash_key,
        )

    def get_function_status(
        self,
        kind,
        selector,
    ):
        resource = mlrun.api.runtime_handlers.runtime_resources_map.get(kind)
        if "status" not in resource:
            raise mlrun.errors.MLRunBadRequestError(
                reason="Runtime error: 'status' not supported by this runtime"
            )

        return resource["status"](selector)

    def start_function(self, function, client_version=None, client_python_version=None):
        resource = mlrun.api.runtime_handlers.runtime_resources_map.get(function.kind)
        if "start" not in resource:
            raise mlrun.errors.MLRunBadRequestError(
                reason="Runtime error: 'start' not supported by this runtime"
            )

        resource["start"](
            function,
            client_version=client_version,
            client_python_version=client_python_version,
        )
        function.save(versioned=False)
