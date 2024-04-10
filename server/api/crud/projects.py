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
import asyncio
import collections
import datetime
import typing

import fastapi.concurrency
import humanfriendly
import sqlalchemy.orm

import mlrun.common.schemas
import mlrun.errors
import mlrun.utils.singleton
import server.api.crud
import server.api.db.session
import server.api.utils.background_tasks
import server.api.utils.clients.nuclio
import server.api.utils.events.events_factory as events_factory
import server.api.utils.projects.remotes.follower as project_follower
import server.api.utils.singletons.db
import server.api.utils.singletons.scheduler
from mlrun.utils import logger, retry_until_successful
from server.api.utils.singletons.k8s import get_k8s_helper


class Projects(
    project_follower.Member,
    metaclass=mlrun.utils.singleton.AbstractSingleton,
):
    def __init__(self) -> None:
        super().__init__()
        self._cache = {
            "project_resources_counters": {"value": None, "ttl": datetime.datetime.min}
        }

    def create_project(
        self, session: sqlalchemy.orm.Session, project: mlrun.common.schemas.Project
    ):
        logger.debug(
            "Creating project",
            name=project.metadata.name,
            owner=project.spec.owner,
            created_time=project.metadata.created,
            desired_state=project.spec.desired_state,
            state=project.status.state,
            function_amount=len(project.spec.functions or []),
            artifact_amount=len(project.spec.artifacts or []),
            workflows_amount=len(project.spec.workflows or []),
        )
        server.api.utils.singletons.db.get_db().create_project(session, project)

    def store_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        project: mlrun.common.schemas.Project,
    ):
        logger.debug(
            "Storing project",
            name=project.metadata.name,
            owner=project.spec.owner,
            created_time=project.metadata.created,
            desired_state=project.spec.desired_state,
            state=project.status.state,
            function_amount=len(project.spec.functions or []),
            artifact_amount=len(project.spec.artifacts or []),
            workflows_amount=len(project.spec.workflows or []),
        )
        server.api.utils.singletons.db.get_db().store_project(session, name, project)

    def patch_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        project: dict,
        patch_mode: mlrun.common.schemas.PatchMode = mlrun.common.schemas.PatchMode.replace,
    ):
        logger.debug(
            "Patching project", name=name, project=project, patch_mode=patch_mode
        )
        server.api.utils.singletons.db.get_db().patch_project(
            session, name, project, patch_mode
        )

    def delete_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        deletion_strategy: mlrun.common.schemas.DeletionStrategy = mlrun.common.schemas.DeletionStrategy.default(),
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
        background_task_name: str = None,
    ):
        logger.debug("Deleting project", name=name, deletion_strategy=deletion_strategy)
        self._enrich_project_with_deletion_background_task_name(
            session, name, background_task_name
        )
        if (
            deletion_strategy.is_restricted()
            or deletion_strategy == mlrun.common.schemas.DeletionStrategy.check
        ):
            if not server.api.utils.singletons.db.get_db().is_project_exists(
                session, name
            ):
                return
            self.verify_project_is_empty(session, name, auth_info)
            if deletion_strategy == mlrun.common.schemas.DeletionStrategy.check:
                return
        elif deletion_strategy.is_cascading():
            self.delete_project_resources(session, name, auth_info=auth_info)
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Unknown deletion strategy: {deletion_strategy}"
            )
        server.api.utils.singletons.db.get_db().delete_project(
            session, name, deletion_strategy
        )

    def verify_project_is_empty(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ):
        server.api.utils.singletons.db.get_db().verify_project_has_no_related_resources(
            session, name
        )
        self._verify_project_has_no_external_resources(session, name, auth_info)

    def _verify_project_has_no_external_resources(
        self,
        session: sqlalchemy.orm.Session,
        project: str,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ):
        # Resources which are not tracked in the MLRun DB need to be verified here. Currently these are project
        # secrets and model endpoints.
        server.api.crud.ModelEndpoints().verify_project_has_no_model_endpoints(project)

        # Note: this check lists also internal secrets. The assumption is that any internal secret that relate to
        # an MLRun resource (such as model-endpoints) was already verified in previous checks. Therefore, any internal
        # secret existing here is something that the user needs to be notified about, as MLRun didn't generate it.
        # Therefore, this check should remain at the end of the verification flow.
        if (
            mlrun.mlconf.is_api_running_on_k8s()
            and get_k8s_helper().get_project_secret_keys(project)
        ):
            raise mlrun.errors.MLRunPreconditionFailedError(
                f"Project {project} can not be deleted since related resources found: project secrets"
            )

        # verify project can be deleted in nuclio
        if mlrun.mlconf.nuclio_dashboard_url:
            nuclio_client = server.api.utils.clients.nuclio.Client()
            nuclio_client.delete_project(
                session,
                project,
                deletion_strategy=mlrun.common.schemas.DeletionStrategy.check,
                auth_info=auth_info,
            )

    def delete_project_resources(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ):
        # Delete schedules before runtime resources - otherwise they will keep getting created
        server.api.utils.singletons.scheduler.get_scheduler().delete_schedules(
            session, name
        )

        # delete runtime resources
        server.api.crud.RuntimeResources().delete_runtime_resources(
            session,
            label_selector=f"mlrun/project={name}",
            force=True,
        )
        if mlrun.mlconf.resolve_kfp_url():
            logger.debug("Removing KFP pipelines project resources", project_name=name)
            server.api.crud.pipelines.Pipelines().delete_pipelines_runs(
                db_session=session, project_name=name
            )

        # log collector service will delete the logs, so we don't need to do it here
        if (
            mlrun.mlconf.log_collector.mode
            == mlrun.common.schemas.LogsCollectorMode.legacy
        ):
            server.api.crud.Logs().delete_project_logs_legacy(name)

        # delete db resources
        server.api.utils.singletons.db.get_db().delete_project_related_resources(
            session, name
        )

        # wait for nuclio to delete the project as well, so it won't create new resources after we delete them
        self._wait_for_nuclio_project_deletion(name, session, auth_info)

        # delete model monitoring resources
        server.api.crud.ModelEndpoints().delete_model_endpoints_resources(name)

        # delete project secrets - passing None will delete all secrets
        if mlrun.mlconf.is_api_running_on_k8s():
            secrets = None
            (
                secret_name,
                action,
            ) = get_k8s_helper().delete_project_secrets(name, secrets)
            if action:
                events_client = events_factory.EventsFactory().get_events_client()
                events_client.emit(
                    events_client.generate_project_secret_event(
                        name,
                        secret_name,
                        action=action,
                    )
                )

            else:
                logger.debug(
                    "No project secrets to delete",
                    action=action,
                    secret_name=secret_name,
                )

    def get_project(
        self, session: sqlalchemy.orm.Session, name: str
    ) -> mlrun.common.schemas.Project:
        return server.api.utils.singletons.db.get_db().get_project(session, name)

    def list_projects(
        self,
        session: sqlalchemy.orm.Session,
        owner: str = None,
        format_: mlrun.common.schemas.ProjectsFormat = mlrun.common.schemas.ProjectsFormat.full,
        labels: list[str] = None,
        state: mlrun.common.schemas.ProjectState = None,
        names: typing.Optional[list[str]] = None,
    ) -> mlrun.common.schemas.ProjectsOutput:
        return server.api.utils.singletons.db.get_db().list_projects(
            session, owner, format_, labels, state, names
        )

    async def list_project_summaries(
        self,
        session: sqlalchemy.orm.Session,
        owner: str = None,
        labels: list[str] = None,
        state: mlrun.common.schemas.ProjectState = None,
        names: typing.Optional[list[str]] = None,
    ) -> mlrun.common.schemas.ProjectSummariesOutput:
        projects_output = await fastapi.concurrency.run_in_threadpool(
            self.list_projects,
            session,
            owner,
            mlrun.common.schemas.ProjectsFormat.name_only,
            labels,
            state,
            names,
        )
        project_summaries = await self.generate_projects_summaries(
            projects_output.projects
        )
        return mlrun.common.schemas.ProjectSummariesOutput(
            project_summaries=project_summaries
        )

    async def get_project_summary(
        self, session: sqlalchemy.orm.Session, name: str
    ) -> mlrun.common.schemas.ProjectSummary:
        # Call get project so we'll explode if project doesn't exists
        await fastapi.concurrency.run_in_threadpool(self.get_project, session, name)
        project_summaries = await self.generate_projects_summaries([name])
        return project_summaries[0]

    async def generate_projects_summaries(
        self, projects: list[str]
    ) -> list[mlrun.common.schemas.ProjectSummary]:
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
                mlrun.common.schemas.ProjectSummary(
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
    ) -> tuple[
        dict[str, int],
        dict[str, int],
        dict[str, int],
        dict[str, int],
        dict[str, int],
        dict[str, int],
        dict[str, typing.Union[int, None]],
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
                server.api.utils.singletons.db.get_db().get_project_resources_counters(),
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
        format_: mlrun.common.schemas.PipelinesFormat = mlrun.common.schemas.PipelinesFormat.metadata_only,
    ):
        return server.api.crud.Pipelines().list_pipelines(session, "*", format_=format_)

    async def _calculate_pipelines_counters(
        self,
    ) -> dict[str, typing.Union[int, None]]:
        # creating defaultdict instead of a regular dict, because it possible that not all projects have pipelines
        # and we want to return 0 for those projects, or None if we failed to get the information
        project_to_running_pipelines_count = collections.defaultdict(lambda: 0)
        if not mlrun.mlconf.resolve_kfp_url():
            # If KFP is not configured, return dict with 0 counters (no running pipelines)
            return project_to_running_pipelines_count

        try:
            _, _, pipelines = await fastapi.concurrency.run_in_threadpool(
                server.api.db.session.run_function_with_new_db_session,
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

    @staticmethod
    def _wait_for_nuclio_project_deletion(
        project_name: str,
        session: sqlalchemy.orm.Session,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ):
        if not mlrun.mlconf.nuclio_dashboard_url:
            return

        nuclio_client = server.api.utils.clients.nuclio.Client()

        def _check_nuclio_project_deletion():
            try:
                nuclio_client.get_project(session, project_name, auth_info=auth_info)
            except mlrun.errors.MLRunNotFoundError:
                logger.debug(
                    "Nuclio project deleted",
                    project_name=project_name,
                )
            else:
                raise Exception(
                    f"Project not deleted in nuclio yet. Project: {project_name}"
                )

        def _verify_no_project_function_pods():
            project_function_pods = server.api.utils.singletons.k8s.get_k8s_helper().list_pods(
                selector=f"nuclio.io/project-name={project_name},nuclio.io/class=function"
            )
            if not project_function_pods:
                logger.debug(
                    "No function pods found for project",
                    project_name=project_name,
                )
                return
            pod_names = [pod.metadata.name for pod in project_function_pods]
            first_three_pods = ", ".join(pod_names[:3])
            raise Exception(
                f"Project {project_name} still has '{len(pod_names)}' function pods; first 3: {first_three_pods}"
            )

        timeout = int(
            humanfriendly.parse_timespan(
                mlrun.mlconf.httpdb.projects.nuclio_project_deletion_verification_timeout
            )
        )
        interval = int(
            humanfriendly.parse_timespan(
                mlrun.mlconf.httpdb.projects.nuclio_project_deletion_verification_interval
            )
        )

        # ensure nuclio project CRD is deleted
        retry_until_successful(
            interval,
            timeout,
            logger,
            False,
            _check_nuclio_project_deletion,
        )

        # ensure no function pods are running
        # this is a bit hacky but should do the job
        # the reason we need it is that nuclio first delete the project CRD, and then
        # nuclio-controller deletes the function crds, and only then the function pods
        # to ensure that nuclio resources (read: functions) are completely deleted
        # we need to wait for the function pods to be deleted as well.
        retry_until_successful(
            interval,
            timeout,
            logger,
            False,
            _verify_no_project_function_pods,
        )

    @staticmethod
    def _enrich_project_with_deletion_background_task_name(
        session: sqlalchemy.orm.Session, name: str, background_task_name: str
    ):
        if not background_task_name:
            return

        project_patch = {
            "status": {"deletion_background_task_name": background_task_name}
        }

        server.api.utils.singletons.db.get_db().patch_project(
            session, name, project_patch
        )
