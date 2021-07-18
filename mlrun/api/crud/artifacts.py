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
        mlrun.api.utils.clients.opa.Client().query_artifact_permissions(
            project, key, mlrun.api.schemas.AuthorizationAction.store, auth_info
        )
        mlrun.api.utils.singletons.project_member.get_project_member().ensure_project(
            db_session, project, leader_session=auth_info.session
        )
        mlrun.api.utils.singletons.db.get_db().store_artifact(
            db_session, key, data, uid, iter, tag, project,
        )
