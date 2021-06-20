import typing

import sqlalchemy.orm

import mlrun.api.crud
import mlrun.api.schemas
import mlrun.api.utils.projects.remotes.follower
import mlrun.api.utils.singletons.db
import mlrun.errors
import mlrun.utils.singleton
from mlrun.utils import logger


class Projects(
    mlrun.api.utils.projects.remotes.follower.Member,
    metaclass=mlrun.utils.singleton.AbstractSingleton,
):
    def create_project(
        self, session: sqlalchemy.orm.Session, project: mlrun.api.schemas.Project
    ):
        logger.debug("Creating project", project=project)
        mlrun.api.utils.singletons.db.get_db().create_project(session, project)

    def store_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        project: mlrun.api.schemas.Project,
    ):
        logger.debug("Storing project", name=name, project=project)
        mlrun.api.utils.singletons.db.get_db().store_project(session, name, project)

    def patch_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        project: dict,
        patch_mode: mlrun.api.schemas.PatchMode = mlrun.api.schemas.PatchMode.replace,
    ):
        logger.debug(
            "Patching project", name=name, project=project, patch_mode=patch_mode
        )
        mlrun.api.utils.singletons.db.get_db().patch_project(
            session, name, project, patch_mode
        )

    def delete_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        deletion_strategy: mlrun.api.schemas.DeletionStrategy = mlrun.api.schemas.DeletionStrategy.default(),
        leader_session: typing.Optional[str] = None,
        # In follower the store of the projects objects themselves is just a dict in the follower member class
        # therefore two methods here (existence check + deletion) need to happen on the store itself (and not the db
        # like the rest of the actions) so enabling to overriding this store with this arg..
        # I felt like defining another layer and interface only for these two methods is an overkill, so although it's a
        # bit ugly I feel like it's fine
        projects_store_override=None,
    ):
        logger.debug("Deleting project", name=name, deletion_strategy=deletion_strategy)
        projects_store = (
            projects_store_override or mlrun.api.utils.singletons.db.get_db()
        )
        if deletion_strategy.is_restricted():
            if not projects_store.is_project_exists(session, name):
                return
            mlrun.api.utils.singletons.db.get_db().verify_project_has_no_related_resources(
                session, name
            )
        elif deletion_strategy.is_cascading():
            self.delete_project_resources(session, name, leader_session)
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Unknown deletion strategy: {deletion_strategy}"
            )
        projects_store.delete_project(session, name, deletion_strategy)

    def delete_project_resources(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        leader_session: typing.Optional[str] = None,
    ):
        # delete runtime resources
        mlrun.api.crud.Runtimes().delete_runtimes(
            session,
            label_selector=f"mlrun/project={name}",
            force=True,
            leader_session=leader_session,
        )

        # delete db resources
        mlrun.api.utils.singletons.db.get_db().delete_project_related_resources(
            session, name
        )

    def get_project(
        self, session: sqlalchemy.orm.Session, name: str
    ) -> mlrun.api.schemas.Project:
        return mlrun.api.utils.singletons.db.get_db().get_project(session, name)

    def list_projects(
        self,
        session: sqlalchemy.orm.Session,
        owner: str = None,
        format_: mlrun.api.schemas.Format = mlrun.api.schemas.Format.full,
        labels: typing.List[str] = None,
        state: mlrun.api.schemas.ProjectState = None,
    ) -> mlrun.api.schemas.ProjectsOutput:
        return mlrun.api.utils.singletons.db.get_db().list_projects(
            session, owner, format_, labels, state
        )
