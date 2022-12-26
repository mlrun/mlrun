# Copyright 2018 Iguazio
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

import mlrun.api.crud
import mlrun.api.db.session
import mlrun.api.schemas
import mlrun.api.utils.auth.verifier
import mlrun.api.utils.clients.iguazio
import mlrun.api.utils.clients.nuclio
import mlrun.api.utils.periodic
import mlrun.api.utils.projects.member
import mlrun.api.utils.projects.remotes.leader
import mlrun.api.utils.projects.remotes.nop_leader
import mlrun.config
import mlrun.errors
import mlrun.utils
import mlrun.utils.helpers
import mlrun.utils.regex
import mlrun.utils.singleton
from mlrun.errors import err_to_str
from mlrun.utils import logger


class Member(
    mlrun.api.utils.projects.member.Member,
    metaclass=mlrun.utils.singleton.AbstractSingleton,
):
    def initialize(self):
        logger.info("Initializing projects follower")
        self._leader_name = mlrun.mlconf.httpdb.projects.leader
        self._sync_session = None
        self._leader_client: mlrun.api.utils.projects.remotes.leader.Member
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
        # run one sync to start off on the right foot and fill out the cache but don't fail initialization on it
        try:
            # Basically the delete operation in our projects mechanism is fully consistent, meaning the leader won't
            # remove the project from its persistency (the source of truth) until it was successfully removed from all
            # followers. Therefore, when syncing projects from the leader, we don't need to search for the deletions
            # that may happened without us knowing about it (therefore full_sync by default is false). When we
            # introduced the chief/worker mechanism, we needed to change the follower to keep its projects in the DB
            # instead of in cache. On the switch, since we were using cache and the projects table in the DB was not
            # maintained, we know we may have projects that shouldn't be there anymore, ideally we would have trigger
            # the full sync only once on the switch, but since we don't have a good heuristic to identify the switch
            # we're doing a full_sync on every initialization
            full_sync = (
                mlrun.mlconf.httpdb.clusterization.role
                == mlrun.api.schemas.ClusterizationRole.chief
            )
            self._sync_projects(full_sync=full_sync)
        except Exception as exc:
            logger.warning(
                "Initial projects sync failed",
                exc=err_to_str(exc),
                traceback=traceback.format_exc(),
            )
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
        commit_before_get: bool = False,
    ) -> typing.Tuple[typing.Optional[mlrun.api.schemas.Project], bool]:
        if self._is_request_from_leader(projects_role):
            mlrun.api.crud.Projects().create_project(db_session, project)
            return project, False
        else:
            is_running_in_background = self._leader_client.create_project(
                leader_session, project, wait_for_completion
            )
            created_project = None
            if not is_running_in_background:
                # as part of the store_project flow we encountered an error related to the isolation level we use.
                # We use the default isolation level, I wasn't able to find exactly what is the default that sql alchemy
                # sets but its serializable(once you SELECT a series of rows in a transaction, you will get the
                # identical data back each time you re-emit that SELECT) or repeatable read isolation (you’ll see newly
                # added rows (and no longer see deleted rows), but for rows that you’ve already loaded, you won’t see
                # any change). Eventually, in the store_project flow, we already queried get_project and at the second
                # time(below), after the project created, we failed because we got the same result from first query.
                # Using session.commit ends the current transaction and start a new one which will result in a
                # new query to the DB.
                # for further read: https://docs-sqlalchemy.readthedocs.io/ko/latest/faq/sessions.html
                # https://docs-sqlalchemy.readthedocs.io/ko/latest/dialects/mysql.html#transaction-isolation-level
                # https://dev.mysql.com/doc/refman/8.0/en/innodb-transaction-isolation-levels.html
                # TODO: there are multiple isolation level we can choose, READ COMMITTED seems to solve our issue
                #  but will require deeper investigation and more test coverage
                if commit_before_get:
                    db_session.commit()

                created_project = self.get_project(
                    db_session, project.metadata.name, leader_session
                )
            return created_project, is_running_in_background

    def store_project(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        project: mlrun.api.schemas.Project,
        projects_role: typing.Optional[mlrun.api.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
        wait_for_completion: bool = True,
    ) -> typing.Tuple[typing.Optional[mlrun.api.schemas.Project], bool]:
        if self._is_request_from_leader(projects_role):
            mlrun.api.crud.Projects().store_project(db_session, name, project)
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
                return self.get_project(db_session, name, leader_session), False

    def patch_project(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        project: dict,
        patch_mode: mlrun.api.schemas.PatchMode = mlrun.api.schemas.PatchMode.replace,
        projects_role: typing.Optional[mlrun.api.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
        wait_for_completion: bool = True,
    ) -> typing.Tuple[typing.Optional[mlrun.api.schemas.Project], bool]:
        if self._is_request_from_leader(projects_role):
            # No real scenario for this to be useful currently - in iguazio patch is transformed to store request
            raise NotImplementedError("Patch operation not supported from leader")
        else:
            current_project = self.get_project(db_session, name, leader_session)
            strategy = patch_mode.to_mergedeep_strategy()
            current_project_dict = current_project.dict(exclude_unset=True)
            mergedeep.merge(current_project_dict, project, strategy=strategy)
            patched_project = mlrun.api.schemas.Project(**current_project_dict)
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
        deletion_strategy: mlrun.api.schemas.DeletionStrategy = mlrun.api.schemas.DeletionStrategy.default(),
        projects_role: typing.Optional[mlrun.api.schemas.ProjectsRole] = None,
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
        wait_for_completion: bool = True,
    ) -> bool:
        if self._is_request_from_leader(projects_role):
            mlrun.api.crud.Projects().delete_project(
                db_session, name, deletion_strategy
            )
        else:
            return self._leader_client.delete_project(
                auth_info.session,
                name,
                deletion_strategy,
                wait_for_completion,
            )
        return False

    def get_project(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        leader_session: typing.Optional[str] = None,
    ) -> mlrun.api.schemas.Project:
        return mlrun.api.crud.Projects().get_project(db_session, name)

    def get_project_owner(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
    ) -> mlrun.api.schemas.ProjectOwner:
        return self._leader_client.get_project_owner(self._sync_session, name)

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
        names: typing.Optional[typing.List[str]] = None,
    ) -> mlrun.api.schemas.ProjectsOutput:
        if (
            format_ == mlrun.api.schemas.ProjectsFormat.leader
            and not self._is_request_from_leader(projects_role)
        ):
            raise mlrun.errors.MLRunAccessDeniedError(
                "Leader format is allowed only to the leader"
            )

        projects_output = mlrun.api.crud.Projects().list_projects(
            db_session, owner, format_, labels, state, names
        )
        if format_ == mlrun.api.schemas.ProjectsFormat.leader:
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
        labels: typing.List[str] = None,
        state: mlrun.api.schemas.ProjectState = None,
        projects_role: typing.Optional[mlrun.api.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
        names: typing.Optional[typing.List[str]] = None,
    ) -> mlrun.api.schemas.ProjectSummariesOutput:
        return await mlrun.api.crud.Projects().list_project_summaries(
            db_session, owner, labels, state, names
        )

    async def get_project_summary(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        leader_session: typing.Optional[str] = None,
    ) -> mlrun.api.schemas.ProjectSummary:
        return await mlrun.api.crud.Projects().get_project_summary(db_session, name)

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

    def _sync_projects(self, full_sync=False):
        """
        :param full_sync: when set to true, in addition to syncing project creation/updates from the leader, we will
        also sync deletions that may occur without updating us the follower
        """
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

        db_session = mlrun.api.db.session.create_session()
        try:
            db_projects = mlrun.api.crud.Projects().list_projects(
                db_session, format_=mlrun.api.schemas.ProjectsFormat.name_only
            )
            # Don't add projects in non terminal state if they didn't exist before to prevent race conditions
            filtered_projects = []
            for leader_project in leader_projects:
                if (
                    leader_project.status.state
                    not in mlrun.api.schemas.ProjectState.terminal_states()
                    and leader_project.metadata.name not in db_projects.projects
                ):
                    continue
                filtered_projects.append(leader_project)

            for project in filtered_projects:
                mlrun.api.crud.Projects().store_project(
                    db_session, project.metadata.name, project
                )
            if full_sync:
                logger.info("Performing full sync")
                leader_project_names = [
                    project.metadata.name for project in leader_projects
                ]
                projects_to_remove = list(
                    set(db_projects.projects).difference(leader_project_names)
                )
                for project_to_remove in projects_to_remove:
                    logger.info(
                        "Found project in the DB that is not in leader. Removing",
                        name=project_to_remove,
                    )
                    mlrun.api.crud.Projects().delete_project(
                        db_session,
                        project_to_remove,
                        mlrun.api.schemas.DeletionStrategy.cascading,
                    )
            if latest_updated_at:

                # sanity and defensive programming - if the leader returned a latest_updated_at that is older
                # than the epoch, we'll set it to the epoch
                epoch = pytz.UTC.localize(datetime.datetime.utcfromtimestamp(0))
                if latest_updated_at < epoch:
                    latest_updated_at = epoch
                self._synced_until_datetime = latest_updated_at
        finally:
            mlrun.api.db.session.close_session(db_session)

    def _is_request_from_leader(
        self, projects_role: typing.Optional[mlrun.api.schemas.ProjectsRole]
    ) -> bool:
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
