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
import asyncio
import collections
import datetime
import typing

import fastapi.concurrency
import humanfriendly
import sqlalchemy.orm

import mlrun.api.crud
import mlrun.api.db.session
import mlrun.api.schemas
import mlrun.api.utils.projects.remotes.follower
import mlrun.api.utils.singletons.db
import mlrun.api.utils.singletons.k8s
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
    ):
        logger.debug("Deleting project", name=name, deletion_strategy=deletion_strategy)
        if (
            deletion_strategy.is_restricted()
            or deletion_strategy == mlrun.api.schemas.DeletionStrategy.check
        ):
            if not mlrun.api.utils.singletons.db.get_db().is_project_exists(
                session, name
            ):
                return
            mlrun.api.utils.singletons.db.get_db().verify_project_has_no_related_resources(
                session, name
            )
            self._verify_project_has_no_external_resources(name)
            if deletion_strategy == mlrun.api.schemas.DeletionStrategy.check:
                return
        elif deletion_strategy.is_cascading():
            self.delete_project_resources(session, name)
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Unknown deletion strategy: {deletion_strategy}"
            )
        mlrun.api.utils.singletons.db.get_db().delete_project(
            session, name, deletion_strategy
        )

    def _verify_project_has_no_external_resources(self, project: str):
        # Resources which are not tracked in the MLRun DB need to be verified here. Currently these are project
        # secrets and model endpoints.
        mlrun.api.crud.ModelEndpoints().verify_project_has_no_model_endpoints(project)

        # Note: this check lists also internal secrets. The assumption is that any internal secret that relate to
        # an MLRun resource (such as model-endpoints) was already verified in previous checks. Therefore, any internal
        # secret existing here is something that the user needs to be notified about, as MLRun didn't generate it.
        # Therefore, this check should remain at the end of the verification flow.
        if (
            mlrun.mlconf.is_api_running_on_k8s()
            and mlrun.api.utils.singletons.k8s.get_k8s().get_project_secret_keys(
                project
            )
        ):
            raise mlrun.errors.MLRunPreconditionFailedError(
                f"Project {project} can not be deleted since related resources found: project secrets"
            )

    def delete_project_resources(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
    ):
        # Delete schedules before runtime resources - otherwise they will keep getting created
        mlrun.api.utils.singletons.scheduler.get_scheduler().delete_schedules(
            session, name
        )

        # delete runtime resources
        mlrun.api.crud.RuntimeResources().delete_runtime_resources(
            session,
            label_selector=f"mlrun/project={name}",
            force=True,
        )

        mlrun.api.crud.Logs().delete_logs(name)

        # delete db resources
        mlrun.api.utils.singletons.db.get_db().delete_project_related_resources(
            session, name
        )

        # delete model monitoring resources
        mlrun.api.crud.ModelEndpoints().delete_model_endpoints_resources(name)

        # delete project secrets - passing None will delete all secrets
        if mlrun.mlconf.is_api_running_on_k8s():
            mlrun.api.utils.singletons.k8s.get_k8s().delete_project_secrets(name, None)

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
            projects_output.projects
        )
        return mlrun.api.schemas.ProjectSummariesOutput(
            project_summaries=project_summaries
        )

    async def get_project_summary(
        self, session: sqlalchemy.orm.Session, name: str
    ) -> mlrun.api.schemas.ProjectSummary:
        # Call get project so we'll explode if project doesn't exists
        await fastapi.concurrency.run_in_threadpool(self.get_project, session, name)
        project_summaries = await self.generate_projects_summaries([name])
        return project_summaries[0]

    async def generate_projects_summaries(
        self, projects: typing.List[str]
    ) -> typing.List[mlrun.api.schemas.ProjectSummary]:
        (
            project_to_files_count,
            project_to_schedule_count,
            project_to_feature_set_count,
            project_to_models_count,
            project_to_recent_failed_runs_count,
            project_to_running_runs_count,
            project_to_running_pipelines_count,
        ) = await self._get_project_resources_counters()
        project_summaries = []
        for project in projects:
            project_summaries.append(
                mlrun.api.schemas.ProjectSummary(
                    name=project,
                    files_count=project_to_files_count.get(project, 0),
                    schedules_count=project_to_schedule_count.get(project, 0),
                    feature_sets_count=project_to_feature_set_count.get(project, 0),
                    models_count=project_to_models_count.get(project, 0),
                    runs_failed_recent_count=project_to_recent_failed_runs_count.get(
                        project, 0
                    ),
                    runs_running_count=project_to_running_runs_count.get(project, 0),
                    # project_to_running_pipelines_count is a defaultdict so it will return None if using dict.get()
                    # and the key wasn't set yet, so we need to use the [] operator to get the default value of the dict
                    pipelines_running_count=project_to_running_pipelines_count[project],
                )
            )
        return project_summaries

    async def _get_project_resources_counters(
        self,
    ) -> typing.Tuple[
        typing.Dict[str, int],
        typing.Dict[str, int],
        typing.Dict[str, int],
        typing.Dict[str, int],
        typing.Dict[str, int],
        typing.Dict[str, int],
        typing.Dict[str, typing.Union[int, None]],
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
                mlrun.api.utils.singletons.db.get_db().get_project_resources_counters(),
                self._calculate_pipelines_counters(),
            )
            (
                project_to_files_count,
                project_to_schedule_count,
                project_to_feature_set_count,
                project_to_models_count,
                project_to_recent_failed_runs_count,
                project_to_running_runs_count,
            ) = results[0]
            project_to_running_pipelines_count = results[1]
            self._cache["project_resources_counters"]["result"] = (
                project_to_files_count,
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

    @staticmethod
    def _list_pipelines(
        session,
        format_: mlrun.api.schemas.PipelinesFormat = mlrun.api.schemas.PipelinesFormat.metadata_only,
    ):
        return mlrun.api.crud.Pipelines().list_pipelines(session, "*", format_=format_)

    async def _calculate_pipelines_counters(
        self,
    ) -> typing.Dict[str, typing.Union[int, None]]:
        # creating defaultdict instead of a regular dict, because it possible that not all projects have pipelines
        # and we want to return 0 for those projects, or None if we failed to get the information
        project_to_running_pipelines_count = collections.defaultdict(lambda: 0)
        if not mlrun.mlconf.resolve_kfp_url():
            # If KFP is not configured, return dict with 0 counters (no running pipelines)
            return project_to_running_pipelines_count

        try:
            _, _, pipelines = await fastapi.concurrency.run_in_threadpool(
                mlrun.api.db.session.run_function_with_new_db_session,
                self._list_pipelines,
            )
        except Exception as exc:
            # If list pipelines failed, set counters to None (unknown) to indicate that we failed to get the information
            logger.warning(
                "Failed to list pipelines. Pipelines counters will be set to None",
                exc=mlrun.errors.err_to_str(exc),
            )
            return collections.defaultdict(lambda: None)

        for pipeline in pipelines:
            if pipeline["status"] not in mlrun.run.RunStatuses.stable_statuses():
                project_to_running_pipelines_count[pipeline["project"]] += 1
        return project_to_running_pipelines_count
