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

import ast
import concurrent.futures
import http
import tempfile
import threading
import traceback
import typing
from collections.abc import Iterable

import kfp_server_api
import sqlalchemy.orm
import yaml

import mlrun
import mlrun.auth.utils
import mlrun.common.constants as mlrun_constants
import mlrun.common.formatters
import mlrun.common.helpers
import mlrun.common.runtimes.constants
import mlrun.common.schemas
import mlrun.errors
import mlrun.utils
import mlrun.utils.helpers
import mlrun.utils.singleton
import mlrun_pipelines.client
import mlrun_pipelines.common.models
import mlrun_pipelines.common.ops
import mlrun_pipelines.imports
import mlrun_pipelines.mixins
import mlrun_pipelines.models
import mlrun_pipelines.utils
from mlrun.common.schemas import WorkflowResponse
from mlrun.k8s_utils import sanitize_label_value
from mlrun_pipelines.models import PipelineRun

import framework.api.utils
import framework.utils.singletons.db
import framework.utils.singletons.k8s
import services.api.crud
import services.api.utils.helpers
from services.api.crud.workflows import RerunRunner


class Pipelines(
    mlrun_pipelines.mixins.PipelineProviderMixin,
    metaclass=mlrun.utils.singleton.Singleton,
):
    def list_pipelines(
        self,
        db_session: sqlalchemy.orm.Session,
        project: typing.Union[str, list[str]] | None = None,
        namespace: str | None = None,
        sort_by: str | None = None,
        page_token: str | None = None,
        filter_json: str | None = None,
        name_contains: str | None = None,
        format_: mlrun.common.formatters.PipelineFormat = mlrun.common.formatters.PipelineFormat.metadata_only,
        page_size: int | None = None,
    ) -> tuple[int, int | None, list[dict]]:
        if format_ == mlrun.common.formatters.PipelineFormat.summary:
            # we don't support summary format in list pipelines since the returned runs doesn't include the workflow
            # manifest status that includes the nodes section we use to generate the DAG.
            # (There is a workflow manifest under the run's pipeline_spec field, but it doesn't include the status)
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Summary format is not supported for list pipelines, use get instead"
            )
        kfp_client = self._initialize_kfp_client(
            namespace=namespace,
        )
        runs = []
        next_page_token = page_token or None
        for page_runs, next_page_token in kfp_client.list_runs(
            project=project,
            page_token=next_page_token,
            page_size=page_size
            or mlrun.common.schemas.PipelinesPagination.default_page_size,
            sort_by=sort_by,
            filter_json=filter_json,
        ):
            if project and project != "*":
                if isinstance(project, str):
                    page_runs = [
                        run
                        for run in page_runs
                        if self._resolve_project_from_pipeline(run) == project
                    ]
                elif isinstance(project, list):
                    page_runs = [
                        run
                        for run in page_runs
                        if self._resolve_project_from_pipeline(run) in project
                    ]

            if name_contains:
                page_runs = self._filter_runs_by_name(
                    runs=page_runs,
                    target_name=name_contains,
                )

            page_runs = self._format_runs_concurrently(
                kfp_client=kfp_client,
                runs=page_runs,
                format_=format_,
            )
            runs.extend(page_runs)

        # In-memory filtering turns Kubeflow's counting inaccurate if there are multiple pages of data
        # so don't pass it to the client in such case
        total_size = -1 if next_page_token else len(runs)

        return total_size, next_page_token, runs

    def delete_pipelines_runs(
        self, db_session: sqlalchemy.orm.Session, project_name: str
    ):
        # Retry listing pipelines to handle transient KFP connection errors
        # (e.g. "invalid connection" from KFP's internal DB pool)
        _, _, project_pipeline_runs = mlrun.utils.helpers.retry_until_successful(
            backoff=2,
            timeout=60,
            logger=mlrun.utils.logger,
            verbose=True,
            _function=self.list_pipelines,
            fatal_exceptions=(mlrun.errors.MLRunInvalidArgumentError,),
            db_session=db_session,
            project=project_name,
            format_=mlrun.common.formatters.PipelineFormat.metadata_only,
        )
        kfp_client = self._initialize_kfp_client()

        if project_pipeline_runs:
            mlrun.utils.logger.debug(
                "Detected pipeline runs for project, deleting them",
                project_name=project_name,
                pipeline_run_count=len(project_pipeline_runs),
            )

        runs_succeeded = 0
        runs_failed = 0
        experiment_ids = set()
        delete_run_futures = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=mlrun.mlconf.workflows.concurrent_delete_worker_count,
            thread_name_prefix="delete_workflow_experiment_",
        ) as executor:
            for pipeline_run in project_pipeline_runs:
                pipeline_run = mlrun_pipelines.models.PipelineRun(pipeline_run)
                # delete pipeline run also terminates it if it is in progress
                delete_run_futures.append(
                    executor.submit(kfp_client._run_api.delete_run, pipeline_run.id)
                )
                if pipeline_run.experiment_id:
                    experiment_ids.add(pipeline_run.experiment_id)
            for future in concurrent.futures.as_completed(delete_run_futures):
                delete_run_exception = future.exception()
                if delete_run_exception is not None:
                    # we don't want to fail the entire delete operation if we failed to delete a single pipeline run
                    # so it won't fail the delete project operation. we will log the error and continue
                    mlrun.utils.logger.warning(
                        "Failed to delete pipeline run",
                        project_name=project_name,
                        pipeline_run_id=pipeline_run.id,
                        exc_info=delete_run_exception,
                    )
                    runs_failed += 1
                else:
                    runs_succeeded += 1
            else:
                mlrun.utils.logger.debug(
                    "Finished deleting pipeline runs",
                    project_name=project_name,
                    succeeded=runs_succeeded,
                    failed=runs_failed,
                )

        experiments_succeeded = 0
        experiments_failed = 0
        delete_experiment_futures = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=mlrun.mlconf.workflows.concurrent_delete_worker_count,
            thread_name_prefix="delete_workflow_experiment_",
        ) as executor:
            for experiment_id in experiment_ids:
                mlrun.utils.logger.debug(
                    f"Detected experiment for project {project_name} and deleting it",
                    project_name=project_name,
                    experiment_id=experiment_id,
                )
                delete_experiment_futures.append(
                    executor.submit(
                        kfp_client._experiment_api.delete_experiment,
                        experiment_id,
                    )
                )
            for future in concurrent.futures.as_completed(delete_experiment_futures):
                delete_experiment_exception = future.exception()
                if delete_experiment_exception is not None:
                    experiments_failed += 1
                    mlrun.utils.logger.warning(
                        "Failed to delete an experiment",
                        project_name=project_name,
                        experiment_id=experiment_id,
                        exc_info=mlrun.errors.err_to_str(delete_experiment_exception),
                    )
                else:
                    experiments_succeeded += 1
            else:
                mlrun.utils.logger.debug(
                    "Finished deleting project experiments",
                    project_name=project_name,
                    succeeded=experiments_succeeded,
                    failed=experiments_failed,
                )

    def get_run(
        self,
        run_id: str,
        project: str,
        namespace: str | None = None,
    ) -> mlrun_pipelines.models.PipelineRun:
        """
        Get a Kubeflow Pipeline (KFP) run by its ID.

        :param run_id: The unique identifier of the pipeline run.
        :param project: The name of the MLRun project associated with the pipeline run.
        :param namespace: (Optional) The Kubernetes namespace in which the pipeline is running.
                          Defaults to the configured namespace if not specified.
        :raises MLRunNotFoundError: If the pipeline run does not belong to the specified project
                                    or if the run ID is not found.
        :raises MLRunRuntimeError: If there is an error retrieving the pipeline run details.
        :raises MLRunHTTPStatusError: If there is an HTTP error interacting with KFP.
        :return: The pipeline run object.
        :rtype: mlrun_pipelines.models.PipelineRun
        """

        kfp_client = self._initialize_kfp_client(namespace)
        try:
            api_run_detail = kfp_client.get_run(run_id)
            run = mlrun_pipelines.models.PipelineRun(api_run_detail)
            if run:
                if project and project != "*":
                    run_project = self._resolve_project_from_pipeline(run)
                    if run_project != project:
                        raise mlrun.errors.MLRunNotFoundError(
                            f"Pipeline run with id {run_id} is not of project {project}"
                        )

        except kfp_server_api.ApiException as exc:
            raise mlrun.errors.err_for_status_code(
                exc.status, mlrun.errors.err_to_str(exc)
            ) from exc
        except mlrun.errors.MLRunHTTPStatusError:
            raise
        except Exception as exc:
            raise mlrun.errors.MLRunRuntimeError(
                f"Failed getting KFP run: {mlrun.errors.err_to_str(exc)}"
            ) from exc
        return run

    def get_formatted_pipeline(
        self,
        run_id: str,
        project: str | None = None,
        namespace: str | None = None,
        format_: mlrun.common.formatters.PipelineFormat = mlrun.common.formatters.PipelineFormat.summary,
    ) -> dict:
        kfp_client = self._initialize_kfp_client(namespace)
        try:
            run = self.get_run(
                run_id=run_id,
                project=project,
                namespace=namespace,
            )
            run = self._format_run(run, format_, kfp_client)
        except mlrun.errors.MLRunHTTPError as exc:
            raise exc
        except Exception as exc:
            raise mlrun.errors.MLRunRuntimeError(
                f"Failed getting KFP run: {mlrun.errors.err_to_str(exc)}"
            ) from exc
        return run

    def get_original_workflow_run(
        self,
        db_session: sqlalchemy.orm.Session,
        run_id: str,
        project: str,
    ) -> tuple[mlrun.model.RunObject | None, str]:
        """
        Given any KFP pipeline run UID (whether the very first run or a retry),
        resolve back to the *original* workflow‐runner RunObject and its workflow ID.

        1. Look for a workflow‐runner job whose "workflow-id" label == run_id.
           If found, that *is* our original runner.
        2. Otherwise, find rerun-runner (workflow-id == run_id), grab its original_workflow_id,
         then fetch that original workflow-runner.

        :returns:
          Tuple of:
            * the RunObject for the original workflow‐runner
            * the original_workflow_id (str)

        :raises:
            MLRunNotFoundError: If the run ID doesn't correspond to any remote workflow.
        """
        job_type_label = mlrun_constants.MLRunInternalLabels.job_type
        workflow_id_label = mlrun_constants.MLRunInternalLabels.workflow_id

        def _list_runs(labels: list[str]) -> list[mlrun.model.RunObject]:
            return services.api.crud.Runs().list_runs(
                db_session=db_session,
                project=project,
                labels=labels,
                sort=True,
                with_notifications=True,
            )

        def _first_or_none(labels: list[str]) -> mlrun.model.RunObject | None:
            runs = _list_runs(labels)
            return runs.to_objects()[0] if runs else None

        def _get_original_workflow(
            workflow_id: str,
        ) -> mlrun.model.RunObject | None:
            """Find a workflow‐runner run by its workflow_id."""
            labels = [
                f"{workflow_id_label}={workflow_id}",
                f"{job_type_label}={mlrun_constants.JOB_TYPE_WORKFLOW_RUNNER}",
            ]
            return _first_or_none(labels)

        # direct workflow-runner
        if original := _get_original_workflow(run_id):
            return original, run_id

        # rerun-runner → original_workflow_id → workflow-runner
        rerun_labels = [
            f"{workflow_id_label}={run_id}",
            f"{job_type_label}={mlrun_constants.JOB_TYPE_RERUN_WORKFLOW_RUNNER}",
        ]
        if rerun := _first_or_none(rerun_labels):
            original_workflow_id = rerun.metadata.labels[
                mlrun_constants.MLRunInternalLabels.original_workflow_id
            ]
            if original := _get_original_workflow(original_workflow_id):
                return original, original_workflow_id

        raise mlrun.errors.MLRunNotFoundError(
            f"No remote workflow runner found with workflow-id={run_id} in project '{project}'"
        )

    def rerun_pipeline_direct(
        self,
        run_id: str,
        project: str,
        namespace: str | None = None,
    ) -> str:
        """
        Retry a Kubeflow Pipeline (KFP) run.

        :param run_id: The unique identifier of the pipeline run to retry.
        :param project: The name of the MLRun project associated with the pipeline run.
        :param namespace: (Optional) The Kubernetes namespace in which the pipeline is running.
                          Defaults to the configured namespace if not specified.
        :raises MLRunBadRequestError: If the pipeline run is not in a retryable state.
        :raises MLRunNotFoundError: If the pipeline run does not belong to the specified project
                                    or if the run ID is not found.
        :raises MLRunRuntimeError: If there is an error retrieving the pipeline run details.
        :raises MLRunHTTPStatusError: If there is an HTTP error interacting with KFP.
        :return: The unique identifier of the retried pipeline run.
        :rtype: str
        """
        run = self.get_run(
            run_id=run_id,
            project=project,
            namespace=namespace,
        )

        # Check if the pipeline is in a completed state
        if (
            run.status
            not in mlrun_pipelines.common.models.RunStatuses.retryable_statuses()
        ):
            raise mlrun.errors.MLRunBadRequestError(
                f"Pipeline run {run_id} is not in a completed state. Current status: {run.status}"
            )

        mlrun.utils.logger.debug(
            "Retrying KFP run",
            run_id=run_id,
            run_name=run.get("name"),
            project=project,
        )
        kfp_client = self._initialize_kfp_client(namespace)
        return kfp_client.retry_run(
            run_id=run_id,
            project=project,
        )

    def rerun_pipeline_via_runner(
        self,
        db_session: sqlalchemy.orm.Session,
        run_id: str,
        project: mlrun.common.schemas.ProjectOut,
        original_runner: mlrun.run.RunObject,
        auth_info: mlrun.common.schemas.AuthInfo,
        client_version: str | None = None,
        rerun_index: int | None = None,
    ):
        """
        Re-run a completed KFP pipeline by launching an MLRun RerunRunner job.

        This will:
        1. Choose a KFP image (honoring any client_version header).
        2. Create & save an MLRun function (the RerunRunner) that invokes `mlrun.projects.rerun_workflow`
           with the original run_id.
        3. Submit that function to Kubernetes, returning a WorkflowResponse for the new MLRun-run.

        :param db_session:       SQLAlchemy session for persisting the runner function.
        :param run_id:           The pipeline run ID to retry (the original KFP run UID).
        :param project:          The MLRun project description (ProjectOut).
        :param original_runner:  The RunObject of the original workflow-runner function.
        :param auth_info:        Caller’s authentication info.
        :param client_version:   Optional SDK version header, to pin the runner image.
        :param rerun_index:      The index of this retry (e.g. 1 for first retry, 2 for second, etc.).
        :return:                 A WorkflowResponse with:
                                   - project: same project name
                                   - name:    the MLRun function name for the rerun
                                   - status:  `"running"`
                                   - run_id:  the new MLRun-run UID for the RerunRunner job
        """
        client_image = services.api.utils.helpers.resolve_client_default_kfp_image(
            project,
            workflow_spec=None,
            client_version=client_version,
        )
        run_name = f"rerun-runner-{run_id[:8]}"

        rerun_runner: mlrun.run.KubejobRuntime = RerunRunner().create_runner(
            run_name=run_name,
            project=project.metadata.name,
            db_session=db_session,
            auth_info=auth_info,
            image=client_image,
        )

        mlrun.utils.logger.debug(
            "Saved function for rerun workflow",
            project_name=rerun_runner.metadata.project,
            function_name=rerun_runner.metadata.name,
            kind=rerun_runner.kind,
            image=rerun_runner.spec.image,
        )

        if client_version is not None:
            rerun_runner.metadata.labels[
                mlrun_constants.MLRunInternalLabels.client_version
            ] = sanitize_label_value(client_version)

        original_runner_notifications = (
            original_runner.spec.notifications.to_dict()
            if original_runner.spec.notifications
            else []
        )

        rerun_notifications = [
            self._augment_notification_for_retry(n, rerun_index)
            for n in original_runner_notifications
        ]

        rerun_request = mlrun.common.schemas.RerunWorkflowRequest(
            run_name=run_name,
            run_id=run_id,
            notifications=rerun_notifications,
            workflow_runner_node_selector=original_runner.spec.node_selector,
            original_workflow_runner_uid=original_runner.metadata.uid,
            original_workflow_name=original_runner.spec.parameters["workflow_name"],
            rerun_index=rerun_index,
        )

        run = RerunRunner().run(
            runner=rerun_runner,
            project=project,
            run_uid=run_id,
            rerun_request=rerun_request,
            auth_info=auth_info,
            original_runner_owner=original_runner.metadata.labels.get(
                mlrun_constants.MLRunInternalLabels.owner
            ),
        )
        status = mlrun_pipelines.common.models.RunStatuses.running
        runner_uid = run.uid()

        return mlrun.common.schemas.WorkflowResponse(
            project=project.metadata.name,
            name=rerun_request.run_name,
            status=str(status),
            run_id=runner_uid,
        )

    def lock_run_and_mark_retrying(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        run_id: str,
        retrying: bool = True,
    ) -> int:
        """
        Lock the specified run row, toggle its `retrying` label (adding by default),
        bump the `rerun_counter` as needed, and return the updated counter.
        """
        run_struct = services.api.crud.RerunRunner().set_run_retrying_status(
            db_session=db_session,
            project=project,
            run_id=run_id,
            retrying=retrying,
        )
        return run_struct["metadata"]["labels"].get(
            mlrun_constants.MLRunInternalLabels.rerun_counter, 1
        )

    def get_running_rerun_runner(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        original_workflow_id: str,
    ):
        running_rerun_runners = services.api.crud.Runs().list_runs(
            db_session=db_session,
            project=project,
            labels=[
                f"{mlrun_constants.MLRunInternalLabels.job_type}={mlrun_constants.JOB_TYPE_RERUN_WORKFLOW_RUNNER}",
                f"{mlrun_constants.MLRunInternalLabels.original_workflow_id}={original_workflow_id}",
            ],
            states=[mlrun.common.runtimes.constants.RunStates.running],
        )
        if running_rerun_runners:
            run = running_rerun_runners.to_objects()[0]
            return WorkflowResponse(
                project=project,
                name=run.metadata.name,
                run_id=run.metadata.uid,
                status=str(run.status),
            )
        raise mlrun.errors.MLRunNotFoundError

    def terminate_pipeline(
        self,
        run_id: str,
        project: str,
        namespace: str | None = None,
    ) -> str:
        """
        Terminate a Kubeflow Pipeline (KFP) run.

        :param run_id: The unique identifier of the pipeline run to terminate.
        :param project: The name of the MLRun project associated with the pipeline run.
        :param namespace: (Optional) The Kubernetes namespace in which the pipeline is running.
                          Defaults to the configured namespace if not specified.
        :raises MLRunBadRequestError: If the pipeline run is not in a terminable state.
        :raises MLRunNotFoundError: If the pipeline run does not belong to the specified project
                                    or if the run ID is not found.
        :raises MLRunRuntimeError: If there is an error retrieving the pipeline run details.
        :raises MLRunHTTPStatusError: If there is an HTTP error interacting with KFP.
        :return: The unique identifier of the terminated pipeline run.
        :rtype: str
        """
        run = self.get_run(
            run_id=run_id,
            project=project,
            namespace=namespace,
        )

        # Check if the pipeline is in a terminable state
        if (
            run.status
            not in mlrun_pipelines.common.models.RunStatuses.terminable_statuses()
        ):
            raise mlrun.errors.MLRunBadRequestError(
                f"Pipeline run {run_id} is not in a terminable state. Current status: {run.status}"
            )

        mlrun.utils.logger.info(
            "Terminating KFP run",
            run_id=run_id,
            run_name=run.get("name"),
            project=project,
        )
        kfp_client = self._initialize_kfp_client(namespace)
        kfp_client.terminate_run(
            run_id=run_id,
        )

    def create_pipeline(
        self,
        experiment_name: str,
        run_name: str,
        content_type: str,
        data: bytes,
        arguments: dict | None = None,
        auth_info: mlrun.common.schemas.AuthInfo | None = None,
    ):
        if arguments is None:
            arguments = {}

        # Extract auth token name from YAML manifest before normalizing content_type
        token_name = None
        if "/yaml" in content_type:
            try:
                token_name = self.resolve_auth_token_name_from_workflow_manifest(data)
            except yaml.YAMLError as exc:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Failed to parse workflow manifest YAML: {mlrun.errors.err_to_str(exc)}"
                ) from exc
            content_type = ".yaml"
        elif " /zip" in content_type:
            content_type = ".zip"
        else:
            framework.api.utils.log_and_raise(
                http.HTTPStatus.BAD_REQUEST.value,
                reason=f"unsupported pipeline type {content_type}",
            )
        mlrun.utils.logger.debug(
            "Writing pipeline to temp file", content_type=content_type
        )

        # Workflows do not go through launcher/runtime handler
        # So enrichment, validation and secret retrieval need to be done here
        auth_secret_name = services.api.utils.helpers.resolve_auth_token_secret_name(
            provided_token_name=token_name, user_id=auth_info.user_id
        )

        data = mlrun_pipelines.common.ops.process_kfp_workflow_secret_references(
            byte_buffer=data,
            content_type=content_type,
            env_var_names=["MLRUN_AUTH_SESSION", "V3IO_ACCESS_KEY"],
            secrets_store=services.api.crud.Secrets(),
            auth_secret_name=auth_secret_name,
            auth_info=auth_info,
        )
        pipeline_file = tempfile.NamedTemporaryFile(suffix=content_type)
        with open(pipeline_file.name, "wb") as fp:
            fp.write(data)

        mlrun.utils.logger.info(
            "Creating pipeline",
            experiment_name=experiment_name,
            run_name=run_name,
            arguments=arguments,
        )

        try:
            kfp_client = self._initialize_kfp_client()
            experiment = mlrun_pipelines.models.PipelineExperiment(
                kfp_client.create_experiment(name=experiment_name)
            )
            run = mlrun_pipelines.models.PipelineRun(
                kfp_client.run_pipeline(
                    experiment.id, run_name, pipeline_file.name, params=arguments
                )
            )
        except Exception as exc:
            mlrun.utils.logger.warning(
                "Failed creating pipeline",
                traceback=traceback.format_exc(),
                exc=mlrun.errors.err_to_str(exc),
            )
            raise mlrun.errors.MLRunBadRequestError(
                f"Failed creating pipeline: {mlrun.errors.err_to_str(exc)}"
            )
        finally:
            pipeline_file.close()

        return run

    @staticmethod
    def _initialize_kfp_client(
        namespace: str | None = None,
    ) -> mlrun_pipelines.client.Client:
        if namespace is None:
            namespace = mlrun.mlconf.namespace
        return mlrun_pipelines.utils.get_client(
            logger=mlrun.utils.logger,
            url=mlrun.mlconf.kfp_url,
            namespace=namespace,
        )

    @staticmethod
    def _is_run_in_unsuccessful_status(
        pipeline_run: mlrun_pipelines.models.PipelineRun,
    ) -> bool:
        return (
            pipeline_run.status
            in mlrun_pipelines.common.models.RunStatuses.unsuccessful_statuses()
        )

    def _format_run(
        self,
        run: mlrun_pipelines.models.PipelineRun,
        format_: mlrun.common.formatters.PipelineFormat,
        kfp_client: mlrun_pipelines.client.Client | None = None,
    ) -> dict:
        run.project = self._resolve_project_from_pipeline(run)
        if self._is_run_in_unsuccessful_status(run) and kfp_client is not None:
            if err := self._get_error_from_pipeline(
                kfp_client=kfp_client,
                run=run,
            ):
                run.error = err
        return mlrun.common.formatters.PipelineFormat.format_obj(run, format_)

    def _format_runs_concurrently(
        self,
        kfp_client: mlrun_pipelines.client.Client,
        runs: list[mlrun_pipelines.models.PipelineRun],
        format_: mlrun.common.formatters.PipelineFormat = mlrun.common.formatters.PipelineFormat.metadata_only,
        *,
        max_workers: int = 32,
        queue_size: int | None = None,
    ) -> list[dict]:
        """
        Submit formatting tasks concurrently and emit results in discovery order.

        This function parallelizes run-formatting using a ThreadPoolExecutor.
        Two separate controls influence concurrency:

        * **max_workers** – limits the number of *active* threads executing
          formatting tasks at any moment. This caps CPU usage and prevents
          excessive I/O pressure against the KFP API.

        * **queue_size** – limits the number of *submitted but not yet started*
          tasks. Without this bound, submitting thousands of runs at once would
          allocate a large number of pending Future objects and unbounded
          memory growth. By default, the queue is `max_workers * 2`, providing
          a small buffer while still preventing runaway task submission.

        The internal **semaphore** enforces the queue bound: each submission
        acquires the semaphore, and each finished task releases it. This keeps
        the total number of in-flight tasks (running + waiting) under control,
        ensuring predictable memory usage even for very large run lists.
        """
        if not runs:
            return []
        if queue_size is None:
            queue_size = max_workers * 2

        semaphore = threading.Semaphore(queue_size) if queue_size else None
        runs = list(runs)
        futures_by_index = [None] * len(runs)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as thread_pool:
            for run_index, pipeline_run in enumerate(runs):
                if semaphore:
                    semaphore.acquire()
                future = thread_pool.submit(
                    self._format_run,
                    run=pipeline_run,
                    format_=format_,
                    kfp_client=kfp_client,
                )
                if semaphore:
                    future.add_done_callback(lambda _f: semaphore.release())
                futures_by_index[run_index] = future

            formatted_runs = []
            for future in futures_by_index:
                try:
                    formatted_runs.append(future.result())
                except Exception:
                    mlrun.utils.logger.error(
                        "Run formatting failed; skipping run", exc_info=True
                    )

        return formatted_runs

    def _resolve_project_from_command(
        self,
        command: list[str],
        hyphen_p_is_also_project: bool,
        has_func_url_flags: bool,
        has_runtime_flags: bool,
    ):
        # project has precedence over function url so search for it first
        for index, argument in enumerate(command):
            if (
                (argument == "-p" and hyphen_p_is_also_project)
                or argument == "--project"
            ) and index + 1 < len(command):
                return command[index + 1]
        if has_func_url_flags:
            for index, argument in enumerate(command):
                if (argument == "-f" or argument == "--func-url") and index + 1 < len(
                    command
                ):
                    function_url = command[index + 1]
                    if function_url.startswith("db://"):
                        (
                            project,
                            _,
                            _,
                            _,
                        ) = mlrun.common.helpers.parse_versioned_object_uri(
                            function_url[len("db://") :]
                        )
                        if project:
                            return project
        if has_runtime_flags:
            for index, argument in enumerate(command):
                if (argument == "-r" or argument == "--runtime") and index + 1 < len(
                    command
                ):
                    runtime = command[index + 1]
                    try:
                        parsed_runtime = ast.literal_eval(runtime)
                    except Exception as exc:
                        mlrun.utils.logger.warning(
                            "Failed parsing runtime. Skipping",
                            runtime=runtime,
                            exc=mlrun.errors.err_to_str(exc),
                        )
                    else:
                        if isinstance(parsed_runtime, dict):
                            project = parsed_runtime.get("metadata", {}).get("project")
                            if project:
                                return project

        return None

    def _resolve_project_from_pipeline(
        self,
        pipeline: mlrun_pipelines.models.PipelineRun,
    ):
        return self.resolve_project_from_workflow_manifest(pipeline.workflow_manifest())

    def _get_error_from_pipeline(
        self,
        kfp_client,
        run: mlrun_pipelines.models.PipelineRun,
    ):
        pipeline = kfp_client.get_run(run.id)
        return self.resolve_error_from_pipeline(pipeline)

    @staticmethod
    def _augment_notification_for_retry(
        notification: dict[str, typing.Any], rerun_index: int
    ) -> dict[str, typing.Any]:
        """
        Return a new notification dict with its `name` suffixed by "– Retry #<idx>".
        """
        return {
            **notification,
            "name": f"{notification.get('name', '')} – Retry #{rerun_index}",
        }

    def _filter_runs_by_name(
        self,
        runs: Iterable[PipelineRun],
        target_name: str,
    ) -> typing.Generator[PipelineRun, None, None]:
        """Filter runs by their name while ignoring the project string on them
        :param runs: list of runs to filter
        :param target_name: target name to filter by
        :return: generator of filtered runs
        """

        def filter_by(
            run_to_filter: PipelineRun,
        ) -> bool:
            project_prefix = self._resolve_project_from_pipeline(run_to_filter) + "-"
            run_name = run_to_filter.name.removeprefix(project_prefix)
            return target_name in run_name

        if not target_name:
            for run in runs:
                yield run

        for run in runs:
            if filter_by(run):
                yield run
