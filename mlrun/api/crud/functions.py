import typing

import sqlalchemy.orm

import mlrun.api.api.utils
import mlrun.api.schemas
import mlrun.api.utils.clients.opa
import mlrun.api.utils.projects.remotes.follower
import mlrun.api.utils.singletons.db
import mlrun.api.utils.singletons.project_member
import mlrun.config
import mlrun.errors
import mlrun.runtimes
import mlrun.utils.singleton


class Functions(metaclass=mlrun.utils.singleton.Singleton,):
    def store_function(
        self,
        db_session: sqlalchemy.orm.Session,
        function: dict,
        name: str,
        project: str = mlrun.mlconf.default_project,
        tag: str = "",
        versioned: bool = False,
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
    ) -> str:
        mlrun.api.utils.singletons.project_member.get_project_member().ensure_project(
            db_session, project, leader_session=auth_info.session
        )
        mlrun.api.utils.clients.opa.Client().query_resource_permissions(
            mlrun.api.schemas.AuthorizationResourceTypes.function,
            project,
            name,
            mlrun.api.schemas.AuthorizationAction.store,
            auth_info,
        )
        return mlrun.api.utils.singletons.db.get_db().store_function(
            db_session, function, name, project, tag, versioned,
        )

    def get_function(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        project: str = mlrun.mlconf.default_project,
        tag: str = "",
        hash_key: str = "",
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
    ) -> dict:
        mlrun.api.utils.clients.opa.Client().query_resource_permissions(
            mlrun.api.schemas.AuthorizationResourceTypes.function,
            project,
            name,
            mlrun.api.schemas.AuthorizationAction.read,
            auth_info,
        )
        return mlrun.api.utils.singletons.db.get_db().get_function(
            db_session, name, project, tag, hash_key
        )

    def delete_function(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        name: str,
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
    ):
        mlrun.api.utils.clients.opa.Client().query_resource_permissions(
            mlrun.api.schemas.AuthorizationResourceTypes.function,
            project,
            name,
            mlrun.api.schemas.AuthorizationAction.delete,
            auth_info,
        )
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
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
    ) -> typing.List:
        if labels is None:
            labels = []
        functions = mlrun.api.utils.singletons.db.get_db().list_functions(
            db_session, name, project, tag, labels,
        )
        return mlrun.api.utils.clients.opa.Client().filter_resources_by_permissions(
            mlrun.api.schemas.AuthorizationResourceTypes.function,
            functions,
            lambda function: (
                function.get("metadata", {}).get(
                    "project", mlrun.mlconf.default_project
                ),
                function["metadata"]["name"],
            ),
            auth_info,
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
