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

import mlrun.common.schemas
import mlrun.common.schemas.artifact
import mlrun.config
import mlrun.errors
import mlrun.utils.singleton
import server.api.utils.singletons.db


class Artifacts(
    metaclass=mlrun.utils.singleton.Singleton,
):
    def store_artifact(
        self,
        db_session: sqlalchemy.orm.Session,
        key: str,
        artifact: dict,
        object_uid: str = None,
        tag: str = "latest",
        iter: int = 0,
        project: str = None,
        producer_id: str = None,
    ):
        project = project or mlrun.mlconf.default_project
        # In case project is an empty string the setdefault won't catch it
        if not artifact.setdefault("project", project):
            artifact["project"] = project

        if artifact["project"] != project:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Conflicting project name - storing artifact with project {artifact['project']}"
                f" into a different project: {project}."
            )
        return server.api.utils.singletons.db.get_db().store_artifact(
            db_session,
            key,
            artifact,
            object_uid,
            iter,
            tag,
            project,
            producer_id=producer_id,
        )

    def create_artifact(
        self,
        db_session: sqlalchemy.orm.Session,
        key: str,
        artifact: dict,
        tag: str = "latest",
        iter: int = 0,
        producer_id: str = None,
        project: str = None,
    ):
        project = project or mlrun.mlconf.default_project
        # In case project is an empty string the setdefault won't catch it
        if not artifact.setdefault("project", project):
            artifact["project"] = project

        best_iteration = artifact.get("metadata", {}).get("best_iteration", False)

        if artifact["project"] != project:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Conflicting project name - storing artifact with project {artifact['project']}"
                f" into a different project: {project}."
            )

        return server.api.utils.singletons.db.get_db().create_artifact(
            db_session,
            project,
            artifact,
            key,
            tag,
            iteration=iter,
            producer_id=producer_id,
            best_iteration=best_iteration,
        )

    def get_artifact(
        self,
        db_session: sqlalchemy.orm.Session,
        key: str,
        tag: str = "latest",
        iter: int = None,
        project: str = mlrun.mlconf.default_project,
        format_: mlrun.common.schemas.artifact.ArtifactsFormat = mlrun.common.schemas.artifact.ArtifactsFormat.full,
        producer_id: str = None,
        object_uid: str = None,
    ) -> dict:
        project = project or mlrun.mlconf.default_project
        artifact = server.api.utils.singletons.db.get_db().read_artifact(
            db_session,
            key,
            tag,
            iter,
            project,
            producer_id,
            object_uid,
        )
        return artifact

    def list_artifacts(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str = mlrun.mlconf.default_project,
        name: str = "",
        tag: str = "",
        labels: typing.List[str] = None,
        since=None,
        until=None,
        kind: typing.Optional[str] = None,
        category: typing.Optional[mlrun.common.schemas.ArtifactCategories] = None,
        iter: typing.Optional[int] = None,
        best_iteration: bool = False,
        format_: mlrun.common.schemas.artifact.ArtifactsFormat = mlrun.common.schemas.artifact.ArtifactsFormat.full,
        producer_id: str = None,
    ) -> typing.List:
        project = project or mlrun.mlconf.default_project
        if labels is None:
            labels = []
        artifacts = server.api.utils.singletons.db.get_db().list_artifacts(
            db_session,
            name,
            project,
            tag,
            labels,
            since,
            until,
            kind,
            category,
            iter,
            best_iteration,
            producer_id=producer_id,
        )
        return artifacts

    def list_artifact_tags(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str = mlrun.mlconf.default_project,
        category: mlrun.common.schemas.ArtifactCategories = None,
    ):
        project = project or mlrun.mlconf.default_project
        return server.api.utils.singletons.db.get_db().list_artifact_tags(
            db_session, project, category
        )

    def delete_artifact(
        self,
        db_session: sqlalchemy.orm.Session,
        key: str,
        tag: str = "latest",
        project: str = mlrun.mlconf.default_project,
        object_uid: str = None,
        producer_id: str = None,
    ):
        project = project or mlrun.mlconf.default_project
        return server.api.utils.singletons.db.get_db().del_artifact(
            db_session, key, tag, project, object_uid, producer_id=producer_id
        )

    def delete_artifacts(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str = mlrun.mlconf.default_project,
        name: str = "",
        tag: str = "latest",
        labels: typing.List[str] = None,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
        producer_id: str = None,
    ):
        project = project or mlrun.mlconf.default_project
        server.api.utils.singletons.db.get_db().del_artifacts(
            db_session, name, project, tag, labels, producer_id=producer_id
        )
