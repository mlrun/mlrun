import typing

import sqlalchemy.orm

import mlrun.api.schemas
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
            db_session, key, data, uid, iter, tag, project,
        )

    def get_artifact(
        self,
        db_session: sqlalchemy.orm.Session,
        key: str,
        tag: str = "latest",
        iter: int = 0,
        project: str = mlrun.mlconf.default_project,
    ) -> dict:
        project = project or mlrun.mlconf.default_project
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
        return artifacts

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
