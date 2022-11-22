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

import mlrun.api.schemas
import mlrun.api.utils.projects.remotes.follower
import mlrun.api.utils.singletons.db
import mlrun.api.utils.singletons.project_member
import mlrun.config
import mlrun.errors
import mlrun.utils.singleton
from mlrun.api.schemas.artifact import ArtifactsFormat


class Artifacts(
    metaclass=mlrun.utils.singleton.Singleton,
):
    def store_artifact(
        self,
        db_session: sqlalchemy.orm.Session,
        key: str,
        data: dict,
        uid: str,
        tag: str = "latest",
        iter: int = 0,
        project: str = mlrun.mlconf.default_project,
    ):
        project = project or mlrun.mlconf.default_project
        # In case project is an empty string the setdefault won't catch it
        if not data.setdefault("project", project):
            data["project"] = project

        if data["project"] != project:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Artifact with conflicting project name - {data['project']} while request project : {project}."
                f"key={key}, uid={uid}, data={data}"
            )
        mlrun.api.utils.singletons.db.get_db().store_artifact(
            db_session,
            key,
            data,
            uid,
            iter,
            tag,
            project,
        )

    def get_artifact(
        self,
        db_session: sqlalchemy.orm.Session,
        key: str,
        tag: str = "latest",
        iter: int = 0,
        project: str = mlrun.mlconf.default_project,
        format_: ArtifactsFormat = ArtifactsFormat.full,
    ) -> dict:
        project = project or mlrun.mlconf.default_project
        artifact = mlrun.api.utils.singletons.db.get_db().read_artifact(
            db_session,
            key,
            tag,
            iter,
            project,
        )
        if format_ == ArtifactsFormat.legacy:
            return _transform_artifact_struct_to_legacy_format(artifact)
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
        category: typing.Optional[mlrun.api.schemas.ArtifactCategories] = None,
        iter: typing.Optional[int] = None,
        best_iteration: bool = False,
        format_: ArtifactsFormat = ArtifactsFormat.full,
    ) -> typing.List:
        project = project or mlrun.mlconf.default_project
        if labels is None:
            labels = []
        artifacts = mlrun.api.utils.singletons.db.get_db().list_artifacts(
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
        )
        if format_ != ArtifactsFormat.legacy:
            return artifacts
        return [
            _transform_artifact_struct_to_legacy_format(artifact)
            for artifact in artifacts
        ]

    def list_artifact_tags(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str = mlrun.mlconf.default_project,
        category: mlrun.api.schemas.ArtifactCategories = None,
    ):
        project = project or mlrun.mlconf.default_project
        return mlrun.api.utils.singletons.db.get_db().list_artifact_tags(
            db_session, project, category
        )

    def delete_artifact(
        self,
        db_session: sqlalchemy.orm.Session,
        key: str,
        tag: str = "latest",
        project: str = mlrun.mlconf.default_project,
    ):
        project = project or mlrun.mlconf.default_project
        return mlrun.api.utils.singletons.db.get_db().del_artifact(
            db_session, key, tag, project
        )

    def delete_artifacts(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str = mlrun.mlconf.default_project,
        name: str = "",
        tag: str = "latest",
        labels: typing.List[str] = None,
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
    ):
        project = project or mlrun.mlconf.default_project
        mlrun.api.utils.singletons.db.get_db().del_artifacts(
            db_session, name, project, tag, labels
        )


def _transform_artifact_struct_to_legacy_format(artifact):
    # Check if this is already in legacy format
    if "metadata" not in artifact:
        return artifact

    # Simply flatten the dictionary
    legacy_artifact = {"kind": artifact["kind"]}
    for section in ["metadata", "spec", "status"]:
        for key, value in artifact[section].items():
            legacy_artifact[key] = value
    return legacy_artifact
