import collections
import typing

import humanfriendly
import sqlalchemy.orm

import mlrun.api.db.session
import mlrun.api.schemas
import mlrun.api.utils.clients.nuclio
import mlrun.api.utils.periodic
import mlrun.api.utils.projects.member
import mlrun.api.utils.projects.remotes.follower
import mlrun.api.utils.projects.remotes.nop_follower
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
        project: mlrun.api.schemas.Project,
        projects_role: typing.Optional[mlrun.api.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
        wait_for_completion: bool = True,
    ) -> typing.Tuple[mlrun.api.schemas.Project, bool]:
        self._enrich_and_validate_before_creation(project)
        self._run_on_all_followers(True, "create_project", db_session, project)
        return self.get_project(db_session, project.metadata.name), False

    def store_project(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        project: mlrun.api.schemas.Project,
        projects_role: typing.Optional[mlrun.api.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
        wait_for_completion: bool = True,
    ) -> typing.Tuple[mlrun.api.schemas.Project, bool]:
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
        patch_mode: mlrun.api.schemas.PatchMode = mlrun.api.schemas.PatchMode.replace,
        projects_role: typing.Optional[mlrun.api.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
        wait_for_completion: bool = True,
    ) -> typing.Tuple[mlrun.api.schemas.Project, bool]:
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
        deletion_strategy: mlrun.api.schemas.DeletionStrategy = mlrun.api.schemas.DeletionStrategy.default(),
        projects_role: typing.Optional[mlrun.api.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
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
        self, db_session: sqlalchemy.orm.Session, name: str
    ) -> mlrun.api.schemas.Project:
        return self._leader_follower.get_project(db_session, name)

    def list_projects(
        self,
        db_session: sqlalchemy.orm.Session,
        owner: str = None,
        format_: mlrun.api.schemas.Format = mlrun.api.schemas.Format.full,
        labels: typing.List[str] = None,
        state: mlrun.api.schemas.ProjectState = None,
    ) -> mlrun.api.schemas.ProjectsOutput:
        return self._leader_follower.list_projects(
            db_session, owner, format_, labels, state
        )

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
            leader_projects: mlrun.api.schemas.ProjectsOutput
            follower_projects_map: typing.Dict[str, mlrun.api.schemas.ProjectsOutput]
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
            str, typing.Dict[str, mlrun.api.schemas.Project]
        ],
        leader_projects_map: typing.Dict[str, mlrun.api.schemas.Project],
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
                    project=project,
                    project_name=project_name,
                    exc=str(exc),
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
            if missing_followers:
                # projects name validation is enforced on creation, the only way for a project name to be invalid is
                # if it was created prior to 0.6.0, and the version was upgraded
                # we do not want to sync these projects since it will anyways fail (Nuclio doesn't allow these names
                # as well)
                if not mlrun.projects.ProjectMetadata.validate_project_name(
                    project_name, raise_on_failure=False
                ):
                    return
                for missing_follower in missing_followers:
                    logger.debug(
                        "Project is missing from follower. Creating",
                        missing_follower_name=missing_follower,
                        project_follower_name=project_follower_name,
                        project_name=project_name,
                        project=project,
                    )
                    try:
                        self._enrich_and_validate_before_creation(project)
                        self._followers[missing_follower].create_project(
                            db_session, project,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Failed creating missing project in follower",
                            missing_follower_name=missing_follower,
                            project_follower_name=project_follower_name,
                            project_name=project_name,
                            project=project,
                            exc=str(exc),
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

    def _enrich_and_validate_before_creation(self, project: mlrun.api.schemas.Project):
        self._enrich_project(project)
        mlrun.projects.ProjectMetadata.validate_project_name(project.metadata.name)

    @staticmethod
    def _enrich_project(project: mlrun.api.schemas.Project):
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
        path_name: str, project: typing.Union[mlrun.api.schemas.Project, dict]
    ):
        if isinstance(project, mlrun.api.schemas.Project):
            body_name = project.metadata.name
        elif isinstance(project, dict):
            body_name = project.get("metadata", {}).get("name")
        else:
            raise NotImplementedError("Unsupported project instance type")

        if body_name and path_name != body_name:
            message = "Conflict between name in body and name in path"
            logger.warning(message, path_name=path_name, body_name=body_name)
            raise mlrun.errors.MLRunConflictError(message)
