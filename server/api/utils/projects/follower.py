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
import traceback
import typing

import humanfriendly
import mergedeep
import pytz
import sqlalchemy.orm

import mlrun.common.formatters
import mlrun.common.schemas
import mlrun.config
import mlrun.errors
import mlrun.utils
import mlrun.utils.helpers
import mlrun.utils.regex
import mlrun.utils.singleton
import server.api.crud
import server.api.db.session
import server.api.utils.auth.verifier
import server.api.utils.background_tasks
import server.api.utils.clients.iguazio
import server.api.utils.helpers
import server.api.utils.periodic
import server.api.utils.projects.member as project_member
import server.api.utils.projects.remotes.leader
import server.api.utils.projects.remotes.nop_leader
from mlrun.errors import err_to_str
from mlrun.utils import logger


class Member(
    project_member.Member,
    metaclass=mlrun.utils.singleton.AbstractSingleton,
):
    def initialize(self):
        logger.info("Initializing projects follower")
        self._should_sync = (
            mlrun.mlconf.httpdb.clusterization.role
            == mlrun.common.schemas.ClusterizationRole.chief
            and mlrun.mlconf.httpdb.clusterization.chief.feature_gates.project_sync
            == "enabled"
        )
        self._leader_name = mlrun.mlconf.httpdb.projects.leader
        self._sync_session = None
        self._leader_client: server.api.utils.projects.remotes.leader.Member
        if self._leader_name == "iguazio":
            self._leader_client = server.api.utils.clients.iguazio.Client()
            if not mlrun.mlconf.httpdb.projects.iguazio_access_key:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Iguazio access key must be configured when the leader is Iguazio"
                )
            self._sync_session = mlrun.mlconf.httpdb.projects.iguazio_access_key
        elif self._leader_name == "nop":
            self._leader_client = server.api.utils.projects.remotes.nop_leader.Member()
        else:
            raise NotImplementedError("Unsupported project leader")
        self._periodic_sync_interval_seconds = humanfriendly.parse_timespan(
            mlrun.mlconf.httpdb.projects.periodic_sync_interval
        )
        self._synced_until_datetime = None
        # run one sync to start off on the right foot and fill out the cache but don't fail initialization on it
        if self._should_sync:
            try:
                # full_sync=True was a temporary measure to handle the move of mlrun from single instance to
                # chief-worker model. Now it is possible to delete projects that are not in the leader therefore
                # we don't necessarily need to archive projects that are not in the leader.
                # A full sync occurs only during initialization (and not during the periodic task) to avoid issues where
                # a bug in Iguazio might return an empty list of projects. During a full sync, if projects exist in
                # MLRun but are not found in Iguazio, they could be archived, which might result in their deletion
                # if a delete request is subsequently received.
                # TODO: Discuss maybe removing full_sync=True in 1.8.0
                self._sync_projects(full_sync=True)
            except Exception as exc:
                logger.warning(
                    "Initial projects sync failed",
                    exc=err_to_str(exc),
                    traceback=traceback.format_exc(),
                )

    def start(self):
        if self._should_sync:
            self._start_periodic_sync()

    def shutdown(self):
        logger.info("Shutting down projects leader")
        if self._should_sync:
            self._stop_periodic_sync()

    def create_project(
        self,
        db_session: sqlalchemy.orm.Session,
        project: mlrun.common.schemas.Project,
        projects_role: typing.Optional[mlrun.common.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
        wait_for_completion: bool = True,
        commit_before_get: bool = False,
    ) -> tuple[typing.Optional[mlrun.common.schemas.Project], bool]:
        self._validate_project(project)
        if server.api.utils.helpers.is_request_from_leader(
            projects_role, leader_name=self._leader_name
        ):
            server.api.crud.Projects().create_project(db_session, project)
            return project, False
        else:
            is_running_in_background = self._leader_client.create_project(
                leader_session, project, wait_for_completion
            )
            created_project = None
            if not is_running_in_background:
                # not running in background means long-project creation operation might stale
                # its db session, so we need to create a new one
                # https://jira.iguazeng.com/browse/ML-5764
                created_project = (
                    server.api.db.session.run_function_with_new_db_session(
                        self.get_project, project.metadata.name, leader_session
                    )
                )
            return created_project, is_running_in_background

    def store_project(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        project: mlrun.common.schemas.Project,
        projects_role: typing.Optional[mlrun.common.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
        wait_for_completion: bool = True,
    ) -> tuple[typing.Optional[mlrun.common.schemas.Project], bool]:
        self._validate_project(project)
        if server.api.utils.helpers.is_request_from_leader(
            projects_role, leader_name=self._leader_name
        ):
            server.api.crud.Projects().store_project(db_session, name, project)
            return project, False
        else:
            try:
                self.get_project(db_session, name, leader_session)
            except mlrun.errors.MLRunNotFoundError:
                return self.create_project(
                    db_session,
                    project,
                    projects_role,
                    leader_session,
                    wait_for_completion,
                    commit_before_get=True,
                )
            else:
                self._leader_client.update_project(leader_session, name, project)
                return server.api.db.session.run_function_with_new_db_session(
                    self.get_project, name, leader_session
                ), False

    def patch_project(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        project: dict,
        patch_mode: mlrun.common.schemas.PatchMode = mlrun.common.schemas.PatchMode.replace,
        projects_role: typing.Optional[mlrun.common.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
        wait_for_completion: bool = True,
    ) -> tuple[typing.Optional[mlrun.common.schemas.Project], bool]:
        if server.api.utils.helpers.is_request_from_leader(
            projects_role, leader_name=self._leader_name
        ):
            # No real scenario for this to be useful currently - in iguazio patch is transformed to store request
            raise NotImplementedError("Patch operation not supported from leader")
        else:
            current_project = self.get_project(db_session, name, leader_session)
            strategy = patch_mode.to_mergedeep_strategy()
            current_project_dict = current_project.dict(exclude_unset=True)
            mergedeep.merge(current_project_dict, project, strategy=strategy)
            patched_project = mlrun.common.schemas.Project(**current_project_dict)
            return self.store_project(
                db_session,
                name,
                patched_project,
                projects_role,
                leader_session,
                wait_for_completion,
            )

    def delete_project(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        deletion_strategy: mlrun.common.schemas.DeletionStrategy = mlrun.common.schemas.DeletionStrategy.default(),
        projects_role: typing.Optional[mlrun.common.schemas.ProjectsRole] = None,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
        wait_for_completion: bool = True,
        background_task_name: str = None,
        model_monitoring_access_key: str = None,
    ) -> bool:
        if server.api.utils.helpers.is_request_from_leader(
            projects_role, leader_name=self._leader_name
        ):
            server.api.crud.Projects().delete_project(
                session=db_session,
                name=name,
                deletion_strategy=deletion_strategy,
                auth_info=auth_info,
                background_task_name=background_task_name,
                model_monitoring_access_key=model_monitoring_access_key,
            )
        else:
            return self._leader_client.delete_project(
                session=auth_info.session,
                name=name,
                deletion_strategy=deletion_strategy,
                wait_for_completion=wait_for_completion,
            )
        return False

    def get_project(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        leader_session: typing.Optional[str] = None,
        from_leader: bool = False,
        format_: mlrun.common.formatters.ProjectFormat = mlrun.common.formatters.ProjectFormat.full,
    ) -> mlrun.common.schemas.ProjectOutput:
        # by default, get project will use mlrun db to get/list the project.
        # from leader is relevant for cases where we want to get the project from the leader
        if from_leader:
            return self._leader_client.get_project(leader_session, name)

        # format_ is relevant for cases where we want to get the project from mlrun db
        projects = self.list_projects(
            db_session, format_=format_, leader_session=leader_session, names=[name]
        ).projects
        if not projects:
            raise mlrun.errors.MLRunNotFoundError(f"Project {name} not found")
        return projects[0]

    def get_project_owner(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
    ) -> mlrun.common.schemas.ProjectOwner:
        return self._leader_client.get_project_owner(self._sync_session, name)

    def list_projects(
        self,
        db_session: sqlalchemy.orm.Session,
        owner: str = None,
        format_: mlrun.common.formatters.ProjectFormat = mlrun.common.formatters.ProjectFormat.full,
        labels: list[str] = None,
        state: mlrun.common.schemas.ProjectState = None,
        # needed only for external usage when requesting leader format
        projects_role: typing.Optional[mlrun.common.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
        names: typing.Optional[list[str]] = None,
    ) -> mlrun.common.schemas.ProjectsOutput:
        if (
            format_ == mlrun.common.formatters.ProjectFormat.leader
            and not server.api.utils.helpers.is_request_from_leader(
                projects_role, leader_name=self._leader_name
            )
        ):
            raise mlrun.errors.MLRunAccessDeniedError(
                "Leader format is allowed only to the leader"
            )

        projects_output = server.api.crud.Projects().list_projects(
            db_session, owner, format_, labels, state, names
        )
        if format_ == mlrun.common.formatters.ProjectFormat.leader:
            leader_projects = [
                self._leader_client.format_as_leader_project(project)
                for project in projects_output.projects
            ]
            projects_output.projects = leader_projects
        return projects_output

    async def list_project_summaries(
        self,
        db_session: sqlalchemy.orm.Session,
        owner: str = None,
        labels: list[str] = None,
        state: mlrun.common.schemas.ProjectState = None,
        projects_role: typing.Optional[mlrun.common.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
        names: typing.Optional[list[str]] = None,
    ) -> mlrun.common.schemas.ProjectSummariesOutput:
        return await server.api.crud.Projects().list_project_summaries(
            db_session, owner, labels, state, names
        )

    async def get_project_summary(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        leader_session: typing.Optional[str] = None,
    ) -> mlrun.common.schemas.ProjectSummary:
        return await server.api.crud.Projects().get_project_summary(db_session, name)

    @server.api.utils.helpers.ensure_running_on_chief
    def _start_periodic_sync(self):
        # the > 0 condition is to allow ourselves to disable the sync from configuration
        if self._periodic_sync_interval_seconds > 0:
            logger.info(
                "Starting periodic projects sync",
                interval=self._periodic_sync_interval_seconds,
            )
            server.api.utils.periodic.run_function_periodically(
                self._periodic_sync_interval_seconds,
                self._sync_projects.__name__,
                False,
                self._sync_projects,
            )

    @server.api.utils.helpers.ensure_running_on_chief
    def _stop_periodic_sync(self):
        server.api.utils.periodic.cancel_periodic_function(self._sync_projects.__name__)

    @server.api.utils.helpers.ensure_running_on_chief
    def _sync_projects(self, full_sync=False):
        """
        :param full_sync: when set to true, in addition to syncing project creation/updates from the leader, we will
        also sync deletions that may occur without updating us the follower
        """
        db_session = server.api.db.session.create_session()

        try:
            leader_projects, latest_updated_at = self._list_projects_from_leader()
            db_projects = server.api.crud.Projects().list_projects(db_session)

            self._store_projects_from_leader(db_session, db_projects, leader_projects)

            if full_sync:
                self._archive_projects_missing_from_leader(
                    db_session, db_projects, leader_projects
                )

            self._update_latest_synced_datetime(latest_updated_at)
        finally:
            server.api.db.session.close_session(db_session)

    def _list_projects_from_leader(self):
        try:
            leader_projects, latest_updated_at = self._leader_client.list_projects(
                self._sync_session, self._synced_until_datetime
            )
        except Exception:
            # if we failed to get projects from the leader, we'll try get all the
            # projects without the updated_at filter
            leader_projects, latest_updated_at = self._leader_client.list_projects(
                self._sync_session
            )
        return leader_projects, latest_updated_at

    def _store_projects_from_leader(self, db_session, db_projects, leader_projects):
        db_projects_names = [project.metadata.name for project in db_projects.projects]

        # Don't add projects in non-terminal state if they didn't exist before, or projects that are currently being
        # deleted to prevent race conditions
        filtered_projects = []
        for leader_project in leader_projects:
            if (
                leader_project.status.state
                not in mlrun.common.schemas.ProjectState.terminal_states()
                and leader_project.metadata.name not in db_projects_names
            ) or self._project_deletion_background_task_exists(
                leader_project.metadata.name
            ):
                continue
            filtered_projects.append(leader_project)

        for project in filtered_projects:
            # if a project was previously archived, it's state will be overriden by the leader
            # and returned to normal here.
            server.api.crud.Projects().store_project(
                db_session, project.metadata.name, project
            )

    @staticmethod
    def _project_deletion_background_task_exists(project_name):
        background_task_kinds = [
            task_format.format(project_name)
            for task_format in [
                server.api.utils.background_tasks.BackgroundTaskKinds.project_deletion_wrapper,
                server.api.utils.background_tasks.BackgroundTaskKinds.project_deletion,
            ]
        ]
        return any(
            [
                server.api.utils.background_tasks.InternalBackgroundTasksHandler().get_active_background_task_by_kind(
                    background_task_kind,
                    raise_on_not_found=False,
                )
                for background_task_kind in background_task_kinds
            ]
        )

    def _archive_projects_missing_from_leader(
        self, db_session, db_projects, leader_projects
    ):
        logger.info("Performing full sync")
        leader_project_names = [project.metadata.name for project in leader_projects]
        projects_to_archive = {
            project.metadata.name: project for project in db_projects.projects
        }
        for project_name in leader_project_names:
            if project_name in projects_to_archive:
                del projects_to_archive[project_name]

        for project_to_archive in projects_to_archive:
            logger.info(
                "Found project in the DB that is not in leader. Archiving...",
                name=project_to_archive,
            )
            try:
                projects_to_archive[
                    project_to_archive
                ].status.state = mlrun.common.schemas.ProjectState.archived
                server.api.crud.Projects().patch_project(
                    db_session,
                    project_to_archive,
                    projects_to_archive[project_to_archive].dict(),
                )
            except Exception as exc:
                logger.warning(
                    "Failed to archive project from DB, continuing...",
                    name=project_to_archive,
                    exc=err_to_str(exc),
                )

    def _update_latest_synced_datetime(self, latest_updated_at):
        if latest_updated_at:
            # sanity and defensive programming - if the leader returned the latest_updated_at that is older
            # than the epoch, we'll set it to the epoch
            epoch = pytz.UTC.localize(datetime.datetime.utcfromtimestamp(0))
            if latest_updated_at < epoch:
                latest_updated_at = epoch
            self._synced_until_datetime = latest_updated_at
