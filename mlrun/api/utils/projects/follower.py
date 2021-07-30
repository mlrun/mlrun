import typing

import humanfriendly
import sqlalchemy.orm

import mlrun.api.db.session
import mlrun.api.schemas
import mlrun.api.utils.clients.iguazio
import mlrun.api.utils.clients.nuclio
import mlrun.api.utils.clients.opa
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
    class ProjectsStoreMode:
        none = "none"
        cache = "cache"

        @staticmethod
        def all():
            return [
                Member.ProjectsStoreMode.none,
                Member.ProjectsStoreMode.cache,
            ]

    class ProjectsStore:
        """
        See mlrun.api.crud.projects.delete_project for explanation for this ugly thing
        """

        def __init__(self, project_member):
            self.project_member = project_member

        def is_project_exists(
            self, session, name: str, leader_session: typing.Optional[str] = None
        ):
            if (
                self.project_member.projects_store_mode
                == self.project_member.ProjectsStoreMode.cache
            ):
                return name in self.project_member._projects
            elif (
                self.project_member.projects_store_mode
                == self.project_member.ProjectsStoreMode.none
            ):
                projects_output = self.project_member.list_projects(
                    session,
                    format_=mlrun.api.schemas.ProjectsFormat.name_only,
                    leader_session=leader_session,
                )
                return name in projects_output.projects

        def delete_project(
            self,
            session,
            name: str,
            deletion_strategy: mlrun.api.schemas.DeletionStrategy = mlrun.api.schemas.DeletionStrategy.default(),
        ):
            if (
                self.project_member.projects_store_mode
                == self.project_member.ProjectsStoreMode.cache
            ):
                if name in self.project_member._projects:
                    del self.project_member._projects[name]
            return

    def initialize(self):
        logger.info("Initializing projects follower")
        self.projects_store_mode = (
            mlrun.mlconf.httpdb.projects.follower_projects_store_mode
        )
        if self.projects_store_mode not in self.ProjectsStoreMode.all():
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Provided projects store mode is not supported. mode={self.projects_store_mode}"
            )
        self._projects: typing.Dict[str, mlrun.api.schemas.Project] = {}
        self._projects_store_for_deletion = self.ProjectsStore(self)
        self._leader_name = mlrun.mlconf.httpdb.projects.leader
        self._sync_session = None
        if self._leader_name == "iguazio":
            self._leader_client = mlrun.api.utils.clients.iguazio.Client()
            if not mlrun.mlconf.httpdb.projects.iguazio_access_key:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Iguazio access key must be configured when the leader is Iguazio"
                )
            self._sync_session = mlrun.mlconf.httpdb.projects.iguazio_access_key
        elif self._leader_name == "nop":
            self._leader_client = mlrun.api.utils.projects.remotes.nop_leader.Member()
        else:
            raise NotImplementedError("Unsupported project leader")
        self._periodic_sync_interval_seconds = humanfriendly.parse_timespan(
            mlrun.mlconf.httpdb.projects.periodic_sync_interval
        )
        self._synced_until_datetime = None
        # Only if we're storing the projects in cache, we need to maintain this cache i.e. run the periodic sync
        if self.projects_store_mode == self.ProjectsStoreMode.cache:
            # run one sync to start off on the right foot and fill out the cache but don't fail initialization on it
            try:
                self._sync_projects()
            except Exception as exc:
                logger.warning("Initial projects sync failed", exc=str(exc))
            self._start_periodic_sync()

    def shutdown(self):
        logger.info("Shutting down projects leader")
        self._stop_periodic_sync()

    def ensure_project(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        wait_for_completion: bool = True,
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
    ) -> bool:
        is_project_created = super().ensure_project(
            db_session, name, wait_for_completion, auth_info
        )
        if is_project_created:
            mlrun.api.utils.clients.opa.Client().add_allowed_project_for_owner(
                name, auth_info,
            )
        return is_project_created

    def create_project(
        self,
        db_session: sqlalchemy.orm.Session,
        project: mlrun.api.schemas.Project,
        projects_role: typing.Optional[mlrun.api.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
        wait_for_completion: bool = True,
    ) -> typing.Tuple[mlrun.api.schemas.Project, bool]:
        if self._is_request_from_leader(projects_role):
            if self.projects_store_mode == self.ProjectsStoreMode.cache:
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
            if self.projects_store_mode == self.ProjectsStoreMode.cache:
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
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
        wait_for_completion: bool = True,
    ) -> bool:
        if self._is_request_from_leader(projects_role):
            # importing here to avoid circular import (db using project member using mlrun follower using db)
            import mlrun.api.crud

            mlrun.api.crud.Projects().delete_project(
                db_session,
                name,
                deletion_strategy,
                auth_info,
                self._projects_store_for_deletion,
            )
        else:
            return self._leader_client.delete_project(
                auth_info.session, name, deletion_strategy, wait_for_completion,
            )
        return False

    def get_project(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        leader_session: typing.Optional[str] = None,
    ) -> mlrun.api.schemas.Project:
        if self.projects_store_mode == self.ProjectsStoreMode.cache:
            if name not in self._projects:
                raise mlrun.errors.MLRunNotFoundError(f"Project not found {name}")
            return self._projects[name]
        elif self.projects_store_mode == self.ProjectsStoreMode.none:
            return self._leader_client.get_project(leader_session, name)

    def list_projects(
        self,
        db_session: sqlalchemy.orm.Session,
        owner: str = None,
        format_: mlrun.api.schemas.ProjectsFormat = mlrun.api.schemas.ProjectsFormat.full,
        labels: typing.List[str] = None,
        state: mlrun.api.schemas.ProjectState = None,
        # needed only for external usage when requesting leader format
        projects_role: typing.Optional[mlrun.api.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
    ) -> mlrun.api.schemas.ProjectsOutput:
        projects = []
        if format_ == mlrun.api.schemas.ProjectsFormat.leader:
            if not self._is_request_from_leader(projects_role):
                raise mlrun.errors.MLRunAccessDeniedError(
                    "Leader format is allowed only to the leader"
                )
            # importing here to avoid circular import (db using project member using mlrun follower using db)
            from mlrun.api.utils.singletons.db import get_db

            # Basically in follower mode our projects source of truth is the leader and the data in the DB is not
            # relevant or maintained. The leader format purpose is a specific upgrade scenario where we're moving from
            # leader mode (in which the projects are maintained in the DB) to follower mode in which the leader needs
            # to be aware of the already existing projects so we're allowing only to the leader, to read from the DB,
            # and return it in the leader's format
            projects = get_db().list_projects(db_session, owner, format_, labels, state)
            leader_projects = [
                self._leader_client.format_as_leader_project(project)
                for project in projects.projects
            ]
            return mlrun.api.schemas.ProjectsOutput(projects=leader_projects)

        if self.projects_store_mode == self.ProjectsStoreMode.cache:
            projects = list(self._projects.values())
        elif self.projects_store_mode == self.ProjectsStoreMode.none:
            projects, _ = self._leader_client.list_projects(leader_session)

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
        if format_ == mlrun.api.schemas.ProjectsFormat.name_only:
            projects = project_names
        elif format_ == mlrun.api.schemas.ProjectsFormat.full:
            pass
        elif format_ == mlrun.api.schemas.ProjectsFormat.summary:
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
