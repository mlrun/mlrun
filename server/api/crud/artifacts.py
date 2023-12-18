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
from http import HTTPStatus

import sqlalchemy.orm
from fastapi import HTTPException

import mlrun.common.schemas
import mlrun.common.schemas.artifact
import mlrun.config
import mlrun.errors
import mlrun.utils.singleton
import server.api.utils.singletons.db
from mlrun.utils import logger


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
        auth_info: mlrun.common.schemas.AuthInfo = None,
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

        # calculate the size of the artifact
        self._resolve_artifact_size(data, auth_info)

        server.api.utils.singletons.db.get_db().store_artifact(
            db_session,
            key,
            data,
            uid,
            iter,
            tag,
            project,
        )

    @staticmethod
    def _resolve_artifact_size(data, auth_info):
        if "spec" in data and "size" not in data["spec"]:
            if "target_path" in data["spec"]:
                path = data["spec"].get("target_path")
                try:
                    file_stat = server.api.crud.Files().get_filestat(
                        auth_info, path=path
                    )
                    data["spec"]["size"] = file_stat["size"]
                except HTTPException as exc:
                    if (
                        exc.status_code == HTTPStatus.NOT_FOUND.value
                    ):  # if the path was not found the size will be N/A
                        logger.debug("Path was not found", path=path)
                        pass

    def get_artifact(
        self,
        db_session: sqlalchemy.orm.Session,
        key: str,
        tag: str = "latest",
        iter: int = 0,
        project: str = mlrun.mlconf.default_project,
        format_: mlrun.common.schemas.artifact.ArtifactsFormat = mlrun.common.schemas.artifact.ArtifactsFormat.full,
    ) -> dict:
        project = project or mlrun.mlconf.default_project
        artifact = server.api.utils.singletons.db.get_db().read_artifact(
            db_session,
            key,
            tag,
            iter,
            project,
        )
        if format_ == mlrun.common.schemas.artifact.ArtifactsFormat.legacy:
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
        category: typing.Optional[mlrun.common.schemas.ArtifactCategories] = None,
        iter: typing.Optional[int] = None,
        best_iteration: bool = False,
        format_: mlrun.common.schemas.artifact.ArtifactsFormat = mlrun.common.schemas.artifact.ArtifactsFormat.full,
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
        )
        if format_ != mlrun.common.schemas.artifact.ArtifactsFormat.legacy:
            return artifacts
        return [
            _transform_artifact_struct_to_legacy_format(artifact)
            for artifact in artifacts
        ]

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
    ):
        project = project or mlrun.mlconf.default_project
        return server.api.utils.singletons.db.get_db().del_artifact(
            db_session, key, tag, project
        )

    def delete_artifacts(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str = mlrun.mlconf.default_project,
        name: str = "",
        tag: str = "latest",
        labels: typing.List[str] = None,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ):
        project = project or mlrun.mlconf.default_project
        server.api.utils.singletons.db.get_db().del_artifacts(
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
