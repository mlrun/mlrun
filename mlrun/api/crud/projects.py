import asyncio
import collections
import datetime
import typing

import fastapi.concurrency
import humanfriendly
import sqlalchemy.orm

import mlrun.api.crud
import mlrun.api.schemas
import mlrun.api.utils.projects.remotes.follower
import mlrun.api.utils.singletons.db
import mlrun.api.utils.singletons.scheduler
import mlrun.errors
import mlrun.utils.singleton
from mlrun.utils import logger


class Projects(
    mlrun.api.utils.projects.remotes.follower.Member,
    metaclass=mlrun.utils.singleton.AbstractSingleton,
):
    def __init__(self) -> None:
        super().__init__()
        self._cache = {
            "project_resources_counters": {"value": None, "ttl": datetime.datetime.min}
        }

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
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
        # In follower mode the store of the projects objects themselves is just a dict in the follower member class
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
        if (
            deletion_strategy.is_restricted()
            or deletion_strategy == mlrun.api.schemas.DeletionStrategy.check
        ):
            if not projects_store.is_project_exists(
                session, name, leader_session=auth_info.session
            ):
                return
            mlrun.api.utils.singletons.db.get_db().verify_project_has_no_related_resources(
                session, name
            )
            if deletion_strategy == mlrun.api.schemas.DeletionStrategy.check:
                return
        elif deletion_strategy.is_cascading():
            self.delete_project_resources(session, name)
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Unknown deletion strategy: {deletion_strategy}"
            )
        projects_store.delete_project(session, name, deletion_strategy)

    def delete_project_resources(
        self, session: sqlalchemy.orm.Session, name: str,
    ):
        # delete runtime resources
        mlrun.api.crud.RuntimeResources().delete_runtime_resources(
            session, label_selector=f"mlrun/project={name}", force=True,
        )

        mlrun.api.crud.Logs().delete_logs(name)

        mlrun.api.utils.singletons.scheduler.get_scheduler().delete_schedules(
            session, name
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
        format_: mlrun.api.schemas.ProjectsFormat = mlrun.api.schemas.ProjectsFormat.full,
        labels: typing.List[str] = None,
        state: mlrun.api.schemas.ProjectState = None,
        names: typing.Optional[typing.List[str]] = None,
    ) -> mlrun.api.schemas.ProjectsOutput:
        return mlrun.api.utils.singletons.db.get_db().list_projects(
            session, owner, format_, labels, state, names
        )

    async def list_project_summaries(
        self,
        session: sqlalchemy.orm.Session,
        owner: str = None,
        labels: typing.List[str] = None,
        state: mlrun.api.schemas.ProjectState = None,
        names: typing.Optional[typing.List[str]] = None,
    ) -> mlrun.api.schemas.ProjectSummariesOutput:
        projects_output = await fastapi.concurrency.run_in_threadpool(
            self.list_projects,
            session,
            owner,
            mlrun.api.schemas.ProjectsFormat.name_only,
            labels,
            state,
            names,
        )
        project_summaries = await self.generate_projects_summaries(
            session, projects_output.projects
        )
        return mlrun.api.schemas.ProjectSummariesOutput(
            project_summaries=project_summaries
        )

    async def get_project_summary(
        self, session: sqlalchemy.orm.Session, name: str
    ) -> mlrun.api.schemas.ProjectSummary:
        # Call get project so we'll explode if project doesn't exists
        await fastapi.concurrency.run_in_threadpool(self.get_project, session, name)
        project_summaries = await self.generate_projects_summaries(session, [name])
        return project_summaries[0]

    async def generate_projects_summaries(
        self, session: sqlalchemy.orm.Session, projects: typing.List[str]
    ) -> typing.List[mlrun.api.schemas.ProjectSummary]:
        (
            project_to_function_count,
            project_to_schedule_count,
            project_to_feature_set_count,
            project_to_models_count,
            project_to_recent_failed_runs_count,
            project_to_running_runs_count,
            project_to_running_pipelines_count,
        ) = await self._get_project_resources_counters(session)
        project_summaries = []
        for project in projects:
            project_summaries.append(
                mlrun.api.schemas.ProjectSummary(
                    name=project,
                    functions_count=project_to_function_count.get(project, 0),
                    schedules_count=project_to_schedule_count.get(project, 0),
                    feature_sets_count=project_to_feature_set_count.get(project, 0),
                    models_count=project_to_models_count.get(project, 0),
                    runs_failed_recent_count=project_to_recent_failed_runs_count.get(
                        project, 0
                    ),
                    runs_running_count=project_to_running_runs_count.get(project, 0),
                    pipelines_running_count=project_to_running_pipelines_count.get(
                        project, 0
                    ),
                )
            )
        return project_summaries

    async def _get_project_resources_counters(
        self, session: sqlalchemy.orm.Session
    ) -> typing.Tuple[
        typing.Dict[str, int],
        typing.Dict[str, int],
        typing.Dict[str, int],
        typing.Dict[str, int],
        typing.Dict[str, int],
        typing.Dict[str, int],
        typing.Dict[str, int],
    ]:
        now = datetime.datetime.now()
        if (
            not self._cache["project_resources_counters"]["ttl"]
            or self._cache["project_resources_counters"]["ttl"] < now
        ):
            logger.debug(
                "Project resources counter cache expired. Calculating",
                ttl=self._cache["project_resources_counters"]["ttl"],
            )

            results = await asyncio.gather(
                mlrun.api.utils.singletons.db.get_db().get_project_resources_counters(
                    session
                ),
                self._calculate_pipelines_counters(session),
            )
            (
                project_to_function_count,
                project_to_schedule_count,
                project_to_feature_set_count,
                project_to_models_count,
                project_to_recent_failed_runs_count,
                project_to_running_runs_count,
            ) = results[0]
            project_to_running_pipelines_count = results[1]
            self._cache["project_resources_counters"]["result"] = (
                project_to_function_count,
                project_to_schedule_count,
                project_to_feature_set_count,
                project_to_models_count,
                project_to_recent_failed_runs_count,
                project_to_running_runs_count,
                project_to_running_pipelines_count,
            )
            ttl_time = datetime.datetime.now() + datetime.timedelta(
                seconds=humanfriendly.parse_timespan(
                    mlrun.mlconf.httpdb.projects.counters_cache_ttl
                )
            )
            self._cache["project_resources_counters"]["ttl"] = ttl_time
        return self._cache["project_resources_counters"]["result"]

    async def _calculate_pipelines_counters(
        self, session: sqlalchemy.orm.Session,
    ) -> typing.Dict[str, int]:
        _, _, pipelines = await fastapi.concurrency.run_in_threadpool(
            mlrun.api.crud.Pipelines().list_pipelines,
            session,
            "*",
            format_=mlrun.api.schemas.PipelinesFormat.metadata_only,
        )
        project_to_running_pipelines_count = collections.defaultdict(int)
        for pipeline in pipelines:
            if pipeline["status"] not in mlrun.run.RunStatuses.stable_statuses():
                project_to_running_pipelines_count[pipeline["project"]] += 1
        return project_to_running_pipelines_count
