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
import collections
import traceback
import typing

import humanfriendly
import sqlalchemy.orm

import mlrun.api.db.session
import mlrun.api.utils.clients.nuclio
import mlrun.api.utils.periodic
import mlrun.api.utils.projects.member
import mlrun.api.utils.projects.member as project_member
import mlrun.api.utils.projects.remotes.follower
import mlrun.api.utils.projects.remotes.nop_follower
import mlrun.common.schemas
import mlrun.config
import mlrun.errors
import mlrun.utils
import mlrun.utils.helpers
import mlrun.utils.regex
import mlrun.utils.singleton
from mlrun.errors import err_to_str
from mlrun.utils import logger


class Member(
    project_member.Member,
    metaclass=mlrun.utils.singleton.AbstractSingleton,
):
    def initialize(self):
        logger.info("Initializing projects leader")
        self._initialize_followers()
        self._periodic_sync_interval_seconds = humanfriendly.parse_timespan(
            mlrun.config.config.httpdb.projects.periodic_sync_interval
        )
        self._projects_in_deletion = set()
        # run one sync to start off on the right foot
        self._sync_projects()
        self._start_periodic_sync()

    def shutdown(self):
        logger.info("Shutting down projects leader")
        self._stop_periodic_sync()

    def create_project(
        self,
        db_session: sqlalchemy.orm.Session,
        project: mlrun.common.schemas.Project,
        projects_role: typing.Optional[mlrun.common.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
        wait_for_completion: bool = True,
        commit_before_get: bool = False,
    ) -> typing.Tuple[typing.Optional[mlrun.common.schemas.Project], bool]:
        self._enrich_and_validate_before_creation(project)
        self._run_on_all_followers(True, "create_project", db_session, project)
        return self.get_project(db_session, project.metadata.name), False

    def store_project(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        project: mlrun.common.schemas.Project,
        projects_role: typing.Optional[mlrun.common.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
        wait_for_completion: bool = True,
    ) -> typing.Tuple[typing.Optional[mlrun.common.schemas.Project], bool]:
        self._enrich_project(project)
        mlrun.projects.ProjectMetadata.validate_project_name(name)
        self._validate_body_and_path_names_matches(name, project)
        self._run_on_all_followers(True, "store_project", db_session, name, project)
        return self.get_project(db_session, name), False

    def patch_project(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        project: dict,
        patch_mode: mlrun.common.schemas.PatchMode = mlrun.common.schemas.PatchMode.replace,
        projects_role: typing.Optional[mlrun.common.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
        wait_for_completion: bool = True,
    ) -> typing.Tuple[mlrun.common.schemas.Project, bool]:
        self._enrich_project_patch(project)
        self._validate_body_and_path_names_matches(name, project)
        self._run_on_all_followers(
            True, "patch_project", db_session, name, project, patch_mode
        )
        return self.get_project(db_session, name), False

    def delete_project(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        deletion_strategy: mlrun.common.schemas.DeletionStrategy = mlrun.common.schemas.DeletionStrategy.default(),
        projects_role: typing.Optional[mlrun.common.schemas.ProjectsRole] = None,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
        wait_for_completion: bool = True,
    ) -> bool:
        self._projects_in_deletion.add(name)
        try:
            self._run_on_all_followers(
                False, "delete_project", db_session, name, deletion_strategy
            )
        finally:
            self._projects_in_deletion.remove(name)
        return False

    def get_project(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        leader_session: typing.Optional[str] = None,
    ) -> mlrun.common.schemas.Project:
        return self._leader_follower.get_project(db_session, name)

    def list_projects(
        self,
        db_session: sqlalchemy.orm.Session,
        owner: str = None,
        format_: mlrun.common.schemas.ProjectsFormat = mlrun.common.schemas.ProjectsFormat.full,
        labels: typing.List[str] = None,
        state: mlrun.common.schemas.ProjectState = None,
        projects_role: typing.Optional[mlrun.common.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
        names: typing.Optional[typing.List[str]] = None,
    ) -> mlrun.common.schemas.ProjectsOutput:
        return self._leader_follower.list_projects(
            db_session, owner, format_, labels, state, names
        )

    async def list_project_summaries(
        self,
        db_session: sqlalchemy.orm.Session,
        owner: str = None,
        labels: typing.List[str] = None,
        state: mlrun.common.schemas.ProjectState = None,
        projects_role: typing.Optional[mlrun.common.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
        names: typing.Optional[typing.List[str]] = None,
    ) -> mlrun.common.schemas.ProjectSummariesOutput:
        return await self._leader_follower.list_project_summaries(
            db_session, owner, labels, state, names
        )

    async def get_project_summary(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        leader_session: typing.Optional[str] = None,
    ) -> mlrun.common.schemas.ProjectSummary:
        return await self._leader_follower.get_project_summary(db_session, name)

    def get_project_owner(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
    ) -> mlrun.common.schemas.ProjectOwner:
        raise NotImplementedError()

    def _start_periodic_sync(self):
        # if no followers no need for sync
        # the > 0 condition is to allow ourselves to disable the sync from configuration
        if self._periodic_sync_interval_seconds > 0 and self._followers:
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
        db_session = mlrun.api.db.session.create_session()
        try:
            # re-generating all of the maps every time since _ensure_follower_projects_synced might cause changes
            leader_projects: mlrun.common.schemas.ProjectsOutput
            follower_projects_map: typing.Dict[str, mlrun.common.schemas.ProjectsOutput]
            leader_projects, follower_projects_map = self._run_on_all_followers(
                True, "list_projects", db_session
            )
            leader_project_names = {
                project.metadata.name for project in leader_projects.projects
            }
            # create reverse map project -> follower names
            project_follower_names_map = collections.defaultdict(set)
            for _follower_name, follower_projects in follower_projects_map.items():
                for project in follower_projects.projects:
                    project_follower_names_map[project.metadata.name].add(
                        _follower_name
                    )

            # create map - follower name -> project name -> project for easier searches
            followers_projects_map = collections.defaultdict(dict)
            for _follower_name, follower_projects in follower_projects_map.items():
                for project in follower_projects.projects:
                    followers_projects_map[_follower_name][
                        project.metadata.name
                    ] = project

            # create map - leader project name -> leader project for easier searches
            leader_projects_map = {}
            for leader_project in leader_projects.projects:
                leader_projects_map[leader_project.metadata.name] = leader_project

            all_project = leader_project_names.copy()
            all_project.update(project_follower_names_map.keys())

            for project in all_project:
                if project in self._projects_in_deletion:
                    continue
                self._ensure_project_synced(
                    db_session,
                    leader_project_names,
                    project_follower_names_map[project],
                    project,
                    followers_projects_map,
                    leader_projects_map,
                )
        finally:
            mlrun.api.db.session.close_session(db_session)

    def _ensure_project_synced(
        self,
        db_session: sqlalchemy.orm.Session,
        leader_project_names: typing.Set[str],
        follower_names: typing.Set[str],
        project_name: str,
        followers_projects_map: typing.Dict[
            str, typing.Dict[str, mlrun.common.schemas.Project]
        ],
        leader_projects_map: typing.Dict[str, mlrun.common.schemas.Project],
    ):
        # FIXME: This function only handles syncing project existence, i.e. if a user updates a project attribute
        #  through one of the followers this change won't be synced and the projects will be left with this discrepancy
        #  for ever
        project = None
        project_follower_name = None
        # first verify that the leader have this project
        if project_name not in leader_project_names:
            project_in_leader = False
            logger.debug(
                "Found project in some of the followers that is not in leader. Creating",
                follower_names=follower_names,
                project=project_name,
            )
            try:
                # Heuristically pick the first follower
                project_follower_name = list(follower_names)[0]
                project = followers_projects_map[project_follower_name][project_name]
                self._enrich_and_validate_before_creation(project)
                self._leader_follower.create_project(db_session, project)
            except Exception as exc:
                logger.warning(
                    "Failed creating missing project in leader",
                    project_follower_name=project_follower_name,
                    project_name=project_name,
                    exc=err_to_str(exc),
                    traceback=traceback.format_exc(),
                )
            else:
                project_in_leader = True
        else:
            project_in_leader = True
            project = leader_projects_map[project_name]

        # only if project in leader - align the rest of followers
        if project_in_leader:
            missing_followers = set(follower_names).symmetric_difference(
                self._followers.keys()
            )
            if self._should_sync_project_to_followers(project_name):
                if missing_followers:
                    self._create_project_in_missing_followers(
                        db_session,
                        missing_followers,
                        project_follower_name,
                        project_name,
                        project,
                    )

                # we possibly enriched the project we found in the follower, so let's update the followers that had it
                self._store_project_in_followers(
                    db_session, follower_names, project_name, project
                )

    def _store_project_in_followers(
        self,
        db_session: sqlalchemy.orm.Session,
        follower_names: typing.Set[str],
        project_name: str,
        project: mlrun.common.schemas.Project,
    ):
        for follower_name in follower_names:
            logger.debug(
                "Updating project in follower",
                follower_name=follower_name,
                project_name=project_name,
            )
            try:
                self._enrich_and_validate_before_creation(project)
                self._followers[follower_name].store_project(
                    db_session,
                    project_name,
                    project,
                )
            except Exception as exc:
                logger.warning(
                    "Failed updating project in follower",
                    follower_name=follower_name,
                    project_name=project_name,
                    exc=err_to_str(exc),
                    traceback=traceback.format_exc(),
                )

    def _create_project_in_missing_followers(
        self,
        db_session: sqlalchemy.orm.Session,
        missing_followers: typing.Set[str],
        # the name of the follower which we took the missing project from
        project_follower_name: str,
        project_name: str,
        project: mlrun.common.schemas.Project,
    ):
        for missing_follower in missing_followers:
            logger.debug(
                "Project is missing from follower. Creating",
                missing_follower_name=missing_follower,
                project_follower_name=project_follower_name,
                project_name=project_name,
            )
            try:
                self._enrich_and_validate_before_creation(project)
                self._followers[missing_follower].create_project(
                    db_session,
                    project,
                )
            except Exception as exc:
                logger.warning(
                    "Failed creating missing project in follower",
                    missing_follower_name=missing_follower,
                    project_follower_name=project_follower_name,
                    project_name=project_name,
                    exc=err_to_str(exc),
                    traceback=traceback.format_exc(),
                )

    def _should_sync_project_to_followers(self, project_name: str) -> bool:
        """
        projects name validation is enforced on creation, the only way for a project name to be invalid is if it was
        created prior to 0.6.0, and the version was upgraded we do not want to sync these projects since it will
        anyways fail (Nuclio doesn't allow these names as well)
        """
        return mlrun.projects.ProjectMetadata.validate_project_name(
            project_name, raise_on_failure=False
        )

    def _run_on_all_followers(
        self, leader_first: bool, method: str, *args, **kwargs
    ) -> typing.Tuple[typing.Any, typing.Dict[str, typing.Any]]:
        leader_response = None
        if leader_first:
            leader_response = getattr(self._leader_follower, method)(*args, **kwargs)
        # TODO: do it concurrently
        follower_responses = {
            follower_name: getattr(follower, method)(*args, **kwargs)
            for follower_name, follower in self._followers.items()
        }
        if not leader_first:
            leader_response = getattr(self._leader_follower, method)(*args, **kwargs)
        return leader_response, follower_responses

    def _initialize_followers(self):
        leader_name = mlrun.config.config.httpdb.projects.leader
        self._leader_follower = self._initialize_follower(leader_name)
        followers = (
            mlrun.config.config.httpdb.projects.followers.split(",")
            if mlrun.config.config.httpdb.projects.followers
            else []
        )
        self._followers = {
            follower: self._initialize_follower(follower) for follower in followers
        }
        logger.debug(
            "Initialized leader and followers",
            leader=leader_name,
            followers=list(self._followers.keys()),
        )

    def _initialize_follower(
        self, name: str
    ) -> mlrun.api.utils.projects.remotes.follower.Member:
        # importing here to avoid circular import (db using project member using mlrun follower using db)
        import mlrun.api.crud

        followers_classes_map = {
            "mlrun": mlrun.api.crud.Projects(),
            "nuclio": mlrun.api.utils.clients.nuclio.Client(),
            # for tests
            "nop-self-leader": mlrun.api.utils.projects.remotes.nop_follower.Member(),
            "nop": mlrun.api.utils.projects.remotes.nop_follower.Member(),
            "nop2": mlrun.api.utils.projects.remotes.nop_follower.Member(),
        }
        if name not in followers_classes_map:
            raise ValueError(f"Unknown follower name: {name}")
        return followers_classes_map[name]

    def _enrich_and_validate_before_creation(
        self, project: mlrun.common.schemas.Project
    ):
        self._enrich_project(project)
        mlrun.projects.ProjectMetadata.validate_project_name(project.metadata.name)

    @staticmethod
    def _enrich_project(project: mlrun.common.schemas.Project):
        project.status.state = project.spec.desired_state

    @staticmethod
    def _enrich_project_patch(project_patch: dict):
        if project_patch.get("spec", {}).get("desired_state"):
            project_patch.setdefault("status", {})["state"] = project_patch["spec"][
                "desired_state"
            ]

    @staticmethod
    def validate_project_name(name: str, raise_on_failure: bool = True) -> bool:
        try:
            mlrun.utils.helpers.verify_field_regex(
                "project.metadata.name", name, mlrun.utils.regex.project_name
            )
        except mlrun.errors.MLRunInvalidArgumentError:
            if raise_on_failure:
                raise
            return False
        return True

    @staticmethod
    def _validate_body_and_path_names_matches(
        path_name: str, project: typing.Union[mlrun.common.schemas.Project, dict]
    ):
        if isinstance(project, mlrun.common.schemas.Project):
            body_name = project.metadata.name
        elif isinstance(project, dict):
            body_name = project.get("metadata", {}).get("name")
        else:
            raise NotImplementedError("Unsupported project instance type")

        if body_name and path_name != body_name:
            message = "Conflict between name in body and name in path"
            logger.warning(message, path_name=path_name, body_name=body_name)
            raise mlrun.errors.MLRunConflictError(message)
