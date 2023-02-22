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
import mlrun.api.schemas
import mlrun.api.utils.projects.remotes.follower
import mlrun.api.utils.singletons.db
import mlrun.api.utils.singletons.project_member
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
        auth_info: mlrun.api.schemas.AuthInfo = None,
    ) -> str:
        project = project or mlrun.mlconf.default_project
        if auth_info:
            function_obj = mlrun.new_function(
                name=name, project=project, runtime=function, tag=tag
            )
            # only need to ensure auth and sensitive data is masked
            mlrun.api.api.utils.apply_enrichment_and_validation_on_function(
                function_obj,
                auth_info,
                ensure_auth=True,
                mask_sensitive_data=True,
                # below is not needed as part of storing the function but rather validated when running the function
                perform_auto_mount=False,
                validate_service_account=False,
                ensure_security_context=False,
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
    ) -> typing.List:
        project = project or mlrun.mlconf.default_project
        if labels is None:
            labels = []
        return mlrun.api.utils.singletons.db.get_db().list_functions(
            db_session,
            name,
            project,
            tag,
            labels,
        )
