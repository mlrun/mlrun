import typing

import sqlalchemy.orm

import mlrun.api.schemas
import mlrun.api.utils.clients.opa
import mlrun.api.utils.projects.remotes.follower
import mlrun.api.utils.singletons.db
import mlrun.api.utils.singletons.project_member
import mlrun.config
import mlrun.errors
import mlrun.utils.singleton


class Artifacts(metaclass=mlrun.utils.singleton.Singleton,):
    def store_artifact(
        self,
        db_session: sqlalchemy.orm.Session,
        key: str,
        data: dict,
        uid: str,
        tag: str = "latest",
        iter: int = 0,
        project: str = mlrun.mlconf.default_project,
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
    ):
        project = project or mlrun.mlconf.default_project
        mlrun.api.utils.singletons.project_member.get_project_member().ensure_project(
            db_session, project, leader_session=auth_info.session
        )
        mlrun.api.utils.clients.opa.Client().query_resource_permissions(
            mlrun.api.schemas.AuthorizationResourceTypes.artifact,
            project,
            key,
            mlrun.api.schemas.AuthorizationAction.store,
            auth_info,
        )
        mlrun.api.utils.singletons.db.get_db().store_artifact(
            db_session, key, data, uid, iter, tag, project,
        )

    def get_artifact(
        self,
        db_session: sqlalchemy.orm.Session,
        key: str,
        tag: str = "latest",
        iter: int = 0,
        project: str = mlrun.mlconf.default_project,
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
    ) -> dict:
        project = project or mlrun.mlconf.default_project
        mlrun.api.utils.clients.opa.Client().query_resource_permissions(
            mlrun.api.schemas.AuthorizationResourceTypes.artifact,
            project,
            key,
            mlrun.api.schemas.AuthorizationAction.read,
            auth_info,
        )
        return mlrun.api.utils.singletons.db.get_db().read_artifact(
            db_session, key, tag, iter, project,
        )

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
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
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
        return mlrun.api.utils.clients.opa.Client().filter_resources_by_permissions(
            mlrun.api.schemas.AuthorizationResourceTypes.artifact,
            artifacts,
            lambda artifact: (
                artifact.get("project", mlrun.mlconf.default_project),
                artifact["db_key"],
            ),
            auth_info,
        )

    def delete_artifact(
        self,
        db_session: sqlalchemy.orm.Session,
        key: str,
        tag: str = "latest",
        project: str = mlrun.mlconf.default_project,
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
    ):
        project = project or mlrun.mlconf.default_project
        mlrun.api.utils.clients.opa.Client().query_resource_permissions(
            mlrun.api.schemas.AuthorizationResourceTypes.artifact,
            project,
            key,
            mlrun.api.schemas.AuthorizationAction.delete,
            auth_info,
        )
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
        artifacts = self.list_artifacts(
            db_session, project, name, tag, labels, auth_info=auth_info
        )
        mlrun.api.utils.clients.opa.Client().query_resources_permissions(
            mlrun.api.schemas.AuthorizationResourceTypes.artifact,
            artifacts,
            lambda artifact: (artifact["project"], artifact["db_key"]),
            mlrun.api.schemas.AuthorizationAction.delete,
            auth_info,
        )
        mlrun.api.utils.singletons.db.get_db().del_artifacts(
            db_session, name, project, tag, labels
        )
