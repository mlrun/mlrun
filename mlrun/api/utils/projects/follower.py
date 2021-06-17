import typing

import humanfriendly
import sqlalchemy.orm

import mlrun.api.db.session
import mlrun.api.schemas
import mlrun.api.utils.clients.iguazio
import mlrun.api.utils.clients.nuclio
import mlrun.api.utils.periodic
import mlrun.api.utils.projects.member
import mlrun.api.utils.projects.remotes.nop_leader
import mlrun.config
import mlrun.errors
import mlrun.utils
import mlrun.utils.helpers
import mlrun.utils.regex
import mlrun.utils.singleton
from mlrun.utils import logger


class Member(
    mlrun.api.utils.projects.member.Member,
    metaclass=mlrun.utils.singleton.AbstractSingleton,
):
    class ProjectsStore:
        """
        See mlrun.api.crud.projects.delete_project for explanation for this ugly thing
        """

        def __init__(self, project_member):
            self.project_member = project_member

        def is_project_exists(self, session, name: str):
            return name in self.project_member._projects

        def delete_project(
            self,
            session,
            name: str,
            deletion_strategy: mlrun.api.schemas.DeletionStrategy = mlrun.api.schemas.DeletionStrategy.default(),
        ):
            if name in self.project_member._projects:
                del self.project_member._projects[name]

    def initialize(self):
        logger.info("Initializing projects follower")
        self._projects: typing.Dict[str, mlrun.api.schemas.Project] = {}
        self._projects_store_for_deletion = self.ProjectsStore(self)
        self._leader_name = mlrun.config.config.httpdb.projects.leader
        self._sync_session = None
        if self._leader_name == "iguazio":
            self._leader_client = mlrun.api.utils.clients.iguazio.Client()
            if not mlrun.config.config.httpdb.projects.iguazio_access_key:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Iguazio access key must be configured when the leader is Iguazio"
                )
            self._sync_session = mlrun.config.config.httpdb.projects.iguazio_access_key
        elif self._leader_name == "nop":
            self._leader_client = mlrun.api.utils.projects.remotes.nop_leader.Member()
        else:
            raise NotImplementedError("Unsupported project leader")
        self._periodic_sync_interval_seconds = humanfriendly.parse_timespan(
            mlrun.config.config.httpdb.projects.periodic_sync_interval
        )
        self._synced_until_datetime = None
        # run one sync to start off on the right foot and fill out the cache but don't fail initialization on it
        try:
            self._sync_projects()
        except Exception as exc:
            logger.warning("Initial projects sync failed", exc=str(exc))
        self._start_periodic_sync()

    def shutdown(self):
        logger.info("Shutting down projects leader")
        self._stop_periodic_sync()

    def create_project(
        self,
        db_session: sqlalchemy.orm.Session,
        project: mlrun.api.schemas.Project,
        projects_role: typing.Optional[mlrun.api.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
        wait_for_completion: bool = True,
    ) -> typing.Tuple[mlrun.api.schemas.Project, bool]:
        if self._is_request_from_leader(projects_role):
            if project.metadata.name in self._projects:
                raise mlrun.errors.MLRunConflictError("Project already exists")
            self._projects[project.metadata.name] = project
            return project, False
        else:
            return self._leader_client.create_project(
                leader_session, project, wait_for_completion
            )

    def store_project(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        project: mlrun.api.schemas.Project,
        projects_role: typing.Optional[mlrun.api.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
        wait_for_completion: bool = True,
    ) -> typing.Tuple[mlrun.api.schemas.Project, bool]:
        if self._is_request_from_leader(projects_role):
            self._projects[project.metadata.name] = project
            return project, False
        else:
            return self._leader_client.store_project(
                leader_session, name, project, wait_for_completion
            )

    def patch_project(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        project: dict,
        patch_mode: mlrun.api.schemas.PatchMode = mlrun.api.schemas.PatchMode.replace,
        projects_role: typing.Optional[mlrun.api.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
        wait_for_completion: bool = True,
    ) -> typing.Tuple[mlrun.api.schemas.Project, bool]:
        if self._is_request_from_leader(projects_role):
            # No real scenario for this to be useful currently - in iguazio patch is transformed to store request
            raise NotImplementedError("Patch operation not supported from leader")
        else:
            return self._leader_client.patch_project(
                leader_session, name, project, patch_mode, wait_for_completion,
            )

    def delete_project(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        deletion_strategy: mlrun.api.schemas.DeletionStrategy = mlrun.api.schemas.DeletionStrategy.default(),
        projects_role: typing.Optional[mlrun.api.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
        wait_for_completion: bool = True,
    ) -> bool:
        if self._is_request_from_leader(projects_role):
            # importing here to avoid circular import (db using project member using mlrun follower using db)
            import mlrun.api.crud

            mlrun.api.crud.Projects().delete_project(
                db_session,
                name,
                deletion_strategy,
                leader_session,
                self._projects_store_for_deletion,
            )
        else:
            return self._leader_client.delete_project(
                leader_session, name, deletion_strategy, wait_for_completion,
            )
        return False

    def get_project(
        self, db_session: sqlalchemy.orm.Session, name: str
    ) -> mlrun.api.schemas.Project:
        if name not in self._projects:
            raise mlrun.errors.MLRunNotFoundError(f"Project not found {name}")
        return self._projects[name]

    def list_projects(
        self,
        db_session: sqlalchemy.orm.Session,
        owner: str = None,
        format_: mlrun.api.schemas.Format = mlrun.api.schemas.Format.full,
        labels: typing.List[str] = None,
        state: mlrun.api.schemas.ProjectState = None,
    ) -> mlrun.api.schemas.ProjectsOutput:
        projects = list(self._projects.values())
        # filter projects
        if owner:
            raise NotImplementedError(
                "Filtering projects by owner is currently not supported in follower mode"
            )
        if state:
            projects = list(
                filter(lambda project: project.status.state == state, projects)
            )
        if labels:
            projects = list(
                filter(
                    lambda project: self._is_project_matching_labels(labels, project),
                    projects,
                )
            )
        project_names = list(map(lambda project: project.metadata.name, projects))
        # format output
        if format_ == mlrun.api.schemas.Format.name_only:
            projects = project_names
        elif format_ == mlrun.api.schemas.Format.full:
            pass
        elif format_ == mlrun.api.schemas.Format.summary:
            # importing here to avoid circular import (db using project member using mlrun follower using db)
            from mlrun.api.utils.singletons.db import get_db

            projects = get_db().generate_projects_summaries(db_session, project_names)
        else:
            raise NotImplementedError(
                f"Provided format is not supported. format={format_}"
            )
        return mlrun.api.schemas.ProjectsOutput(projects=projects)

    def _start_periodic_sync(self):
        # the > 0 condition is to allow ourselves to disable the sync from configuration
        if self._periodic_sync_interval_seconds > 0:
            logger.info(
                "Starting periodic projects sync",
                interval=self._periodic_sync_interval_seconds,
            )
            mlrun.api.utils.periodic.run_function_periodically(
                self._periodic_sync_interval_seconds,
                self._sync_projects.__name__,
                False,
                self._sync_projects,
            )

    def _stop_periodic_sync(self):
        mlrun.api.utils.periodic.cancel_periodic_function(self._sync_projects.__name__)

    def _sync_projects(self):
        projects, latest_updated_at = self._leader_client.list_projects(
            self._sync_session, self._synced_until_datetime
        )
        # Don't add projects in non terminal state if they didn't exist before to prevent race conditions
        filtered_projects = []
        for project in projects:
            if (
                project.status.state
                not in mlrun.api.schemas.ProjectState.terminal_states()
                and project.metadata.name not in self._projects
            ):
                continue
            filtered_projects.append(project)
        for project in filtered_projects:
            self._projects[project.metadata.name] = project
        self._synced_until_datetime = latest_updated_at

    def _is_request_from_leader(
        self, projects_role: typing.Optional[mlrun.api.schemas.ProjectsRole]
    ):
        if projects_role and projects_role.value == self._leader_name:
            return True
        return False

    @staticmethod
    def _is_project_matching_labels(
        labels: typing.List[str], project: mlrun.api.schemas.Project
    ):
        if not project.metadata.labels:
            return False
        for label in labels:
            if "=" in label:
                name, value = [v.strip() for v in label.split("=", 1)]
                if name not in project.metadata.labels:
                    return False
                return value == project.metadata.labels[name]
            else:
                return label in project.metadata.labels
