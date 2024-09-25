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
import datetime
import typing

import sqlalchemy.orm

import mlrun.artifacts.base
import mlrun.common.formatters
import mlrun.common.schemas
import mlrun.common.schemas.artifact
import mlrun.config
import mlrun.errors
import mlrun.utils.helpers
import mlrun.utils.singleton
import server.api.utils.singletons.db
from mlrun.errors import err_to_str
from mlrun.utils import logger


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
        iter: int = None,
        project: str = None,
        producer_id: str = None,
        auth_info: mlrun.common.schemas.AuthInfo = None,
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

        # calculate the size of the artifact
        self._resolve_artifact_size(artifact, auth_info)

        # TODO: remove this in 1.8.0
        if mlrun.utils.helpers.is_legacy_artifact(artifact):
            artifact = mlrun.artifacts.base.convert_legacy_artifact_to_new_format(
                artifact
            ).to_dict()

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
        iter: int = None,
        producer_id: str = None,
        project: str = None,
        auth_info: mlrun.common.schemas.AuthInfo = None,
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

        # calculate the size of the artifact
        self._resolve_artifact_size(artifact, auth_info)

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
        project: str = None,
        format_: mlrun.common.formatters.ArtifactFormat = mlrun.common.formatters.ArtifactFormat.full,
        producer_id: str = None,
        object_uid: str = None,
        raise_on_not_found: bool = True,
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
            raise_on_not_found,
            format_=format_,
        )
        return artifact

    def list_artifacts(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str = None,
        name: str = "",
        tag: str = "",
        labels: list[str] = None,
        since: datetime.datetime = None,
        until: datetime.datetime = None,
        kind: typing.Optional[str] = None,
        category: typing.Optional[mlrun.common.schemas.ArtifactCategories] = None,
        iter: typing.Optional[int] = None,
        best_iteration: bool = False,
        format_: mlrun.common.formatters.ArtifactFormat = mlrun.common.formatters.ArtifactFormat.full,
        producer_id: str = None,
        producer_uri: str = None,
        limit: int = None,
    ) -> list:
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
            producer_uri=producer_uri,
            format_=format_,
            limit=limit,
        )
        return artifacts

    def list_artifacts_for_producer_id(
        self,
        db_session: sqlalchemy.orm.Session,
        producer_id: str,
        project: str,
        key_tag_iteration_pairs: list[tuple] = "",
    ):
        return server.api.utils.singletons.db.get_db().list_artifacts_for_producer_id(
            db_session,
            producer_id=producer_id,
            project=project,
            key_tag_iteration_pairs=key_tag_iteration_pairs,
        )

    def list_artifact_tags(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str = None,
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
        project: str = None,
        object_uid: str = None,
        producer_id: str = None,
        iteration: int = None,
        deletion_strategy: mlrun.common.schemas.artifact.ArtifactsDeletionStrategies = (
            mlrun.common.schemas.artifact.ArtifactsDeletionStrategies.metadata_only
        ),
        secrets: dict = None,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ):
        project = project or mlrun.mlconf.default_project

        # delete artifacts data by deletion strategy
        if deletion_strategy in [
            mlrun.common.schemas.artifact.ArtifactsDeletionStrategies.data_optional,
            mlrun.common.schemas.artifact.ArtifactsDeletionStrategies.data_force,
        ]:
            self._delete_artifact_data(
                db_session=db_session,
                key=key,
                tag=tag,
                project=project,
                object_uid=object_uid,
                producer_id=producer_id,
                iteration=iteration,
                deletion_strategy=deletion_strategy,
                secrets=secrets,
                auth_info=auth_info,
            )

        return server.api.utils.singletons.db.get_db().del_artifact(
            session=db_session,
            key=key,
            tag=tag,
            project=project,
            uid=object_uid,
            producer_id=producer_id,
            iter=iteration,
        )

    def delete_artifacts(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str = None,
        name: str = "",
        tag: str = "latest",
        labels: list[str] = None,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
        producer_id: str = None,
    ):
        project = project or mlrun.mlconf.default_project
        server.api.utils.singletons.db.get_db().del_artifacts(
            db_session, name, project, tag, labels, producer_id=producer_id
        )

    @staticmethod
    def _resolve_artifact_size(artifact, auth_info):
        if "spec" in artifact and "size" not in artifact["spec"]:
            if "target_path" in artifact["spec"]:
                path = artifact["spec"].get("target_path")
                try:
                    file_stat = server.api.crud.Files().get_filestat(
                        auth_info, path=path
                    )
                    artifact["spec"]["size"] = file_stat["size"]
                except Exception as err:
                    logger.debug(
                        "Failed calculating artifact size",
                        path=path,
                        err=err_to_str(err),
                    )
        if "spec" in artifact and "inline" in artifact["spec"]:
            mlrun.utils.helpers.validate_inline_artifact_body_size(
                artifact["spec"]["inline"]
            )

    def _delete_artifact_data(
        self,
        db_session: sqlalchemy.orm.Session,
        key: str,
        tag: str = "latest",
        project: str = None,
        object_uid: str = None,
        producer_id: str = None,
        iteration: int = None,
        deletion_strategy: mlrun.common.schemas.artifact.ArtifactsDeletionStrategies = (
            mlrun.common.schemas.artifact.ArtifactsDeletionStrategies.metadata_only
        ),
        secrets: dict = None,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ):
        logger.debug("Deleting artifact data", project=project, key=key, tag=tag)

        try:
            artifact = self.get_artifact(
                db_session,
                key,
                tag,
                project=project,
                producer_id=producer_id,
                object_uid=object_uid,
                iter=iteration,
            )

            path = artifact["spec"]["target_path"]

            # Data artifacts that are ModelArtifact, DirArtifact must not be removed because we do not yet
            # support the deletion of artifacts that contain multiple files
            # We support deleting DatasetArtifact data that contains one file
            # TODO: must be removed once it is supported
            artifact_kind = artifact["kind"]
            if artifact_kind in ["model", "dir"]:
                raise mlrun.errors.MLRunNotImplementedServerError(
                    f"Deleting artifact data kind: {artifact_kind} is currently not supported"
                )
            if artifact_kind == "dataset" and not mlrun.utils.helpers.is_parquet_file(
                path
            ):
                raise mlrun.errors.MLRunNotImplementedServerError(
                    "Deleting artifact data of kind 'dataset' is currently supported for a single file only"
                )

            server.api.crud.Files().delete_artifact_data(
                auth_info, project, path, secrets=secrets
            )
        except Exception as exc:
            logger.debug(
                "Failed delete artifact data",
                key=key,
                project=project,
                deletion_strategy=deletion_strategy,
                err=err_to_str(exc),
            )

            if (
                deletion_strategy
                == mlrun.common.schemas.artifact.ArtifactsDeletionStrategies.data_force
            ):
                raise
