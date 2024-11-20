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
import os
import uuid
from abc import abstractmethod
from typing import Optional

from sqlalchemy.orm import Session

import mlrun.common.constants as mlrun_constants
import mlrun.common.runtimes
import mlrun.common.schemas
import mlrun.config as mlrun_config
import mlrun.model as mlrun_model
import mlrun.utils as mlrun_utils
import mlrun.utils.singleton
import mlrun_pipelines.common.models

import framework.api.utils
import framework.constants
import framework.utils.notifications
import framework.utils.notifications.notification_pusher
import services.api.crud
import services.api.utils.singletons.scheduler

JOB_TYPE_WORKFLOW_RUNNER = "workflow-runner"
JOB_TYPE_PROJECT_LOADER = "project-loader"


class BaseRunner(metaclass=mlrun.utils.singleton.Singleton):
    """
    Base class for workflow runners.
    """

    @staticmethod
    def create_runner(
        run_name: str,
        project: str,
        db_session: Session,
        auth_info: mlrun.common.schemas.AuthInfo,
        image: str,
    ) -> mlrun.run.KubejobRuntime:
        """
        Create the base object for the workflow runner function with
        all the necessary metadata to create it on the server side.

        :param run_name:   Workflow-runner function name.
        :param project:    Project name.
        :param db_session: Session that manages the current dialog with the database.
        :param auth_info:  Authentication information of the request.
        :param image:      Image for the workflow runner job.
        :return: Workflow runner object.
        """
        runner = mlrun.new_function(
            name=run_name,
            project=project,
            kind=mlrun.runtimes.RuntimeKinds.job,
            image=image,
        )

        runner.set_db_connection(framework.api.utils.get_run_db_instance(db_session))

        # Enrichment and validation require access key
        runner.metadata.credentials.access_key = (
            mlrun_model.Credentials.generate_access_key
        )

        framework.api.utils.apply_enrichment_and_validation_on_function(
            function=runner,
            auth_info=auth_info,
        )

        runner.save()
        return runner

    def prepare_and_run(
        self,
        runner: mlrun.run.KubejobRuntime,
        project: mlrun.common.schemas.ProjectOut,
        labels: dict[str, str],
        workflow_request: Optional[mlrun.common.schemas.WorkflowRequest] = None,
        auth_info: mlrun.common.schemas.AuthInfo = None,
        artifact_path: str = "",
    ) -> mlrun_model.RunObject:
        """
        Prepare the run object and execute the runner.

        :param runner:           Workflow runner function object.
        :param project:          MLRun project.
        :param labels:           Labels for the run.
        :param workflow_request: Workflow request containing the workflow spec.
        :param auth_info:        Authentication information of the request.
        :param artifact_path:    Artifact path for the run.
        :return: RunObject with run metadata, results, and status.
        """
        mlrun.runtimes.utils.enrich_run_labels(
            labels, [mlrun.common.runtimes.constants.RunLabels.owner]
        )

        run_object = self._prepare_run_object(
            project=project,
            labels=labels,
            workflow_request=workflow_request,
            run_name=runner.metadata.name,
        )

        # We want to store the secret params as k8s secret, so later we can access them with the project internal secret
        # key that was created.
        framework.utils.notifications.mask_notification_params_on_task_object(
            run_object, framework.constants.MaskOperations.CONCEAL
        )

        # TODO: Passing auth_info is required for server side launcher, but the runner is already enriched with the
        #  auth_info when it was created in create_runner. We should move the enrichment to the launcher and need to
        #  make sure it is safe for scheduling and project load endpoint.
        return runner.run(
            runspec=run_object,
            artifact_path=artifact_path,
            local=False,
            watch=False,
            auth_info=auth_info,
        )

    @abstractmethod
    def _prepare_run_object(
        self,
        project: mlrun.common.schemas.ProjectOut,
        labels: dict[str, str],
        workflow_request: mlrun.common.schemas.WorkflowRequest,
        run_name: Optional[str] = None,
    ) -> mlrun_model.RunObject:
        """
        Abstract method to prepare the run object.

        :param project:          MLRun project.
        :param labels:           Labels for the run.
        :param workflow_request: Workflow request containing the workflow spec.
        :param run_name:         Name of the run.
        :return: RunObject ready for execution.
        """
        ...

    def _create_run_object(
        self,
        source: str,
        project_name: str,
        save: bool,
        handler: str,
        parameters: dict,
        notifications: Optional[list[mlrun_model.Notification]] = None,
        run_name: Optional[str] = None,
        is_context: Optional[bool] = None,
        labels: Optional[dict[str, str]] = None,
        scrape_metrics: Optional[bool] = None,
        output_path: Optional[str] = None,
        uid: Optional[str] = None,
    ) -> mlrun_model.RunObject:
        """
        Create a RunObject with the given parameters.

        :param source:          Project source URL or path.
        :param project_name:    Name of the project.
        :param save:            Whether to save the project after loading.
        :param handler:         Handler function to run.
        :param parameters:      Parameters for the run.
        :param notifications:   List of notifications.
        :param run_name:        Name of the run.
        :param is_context:      Indicates if the source is a context path.
        :param labels:          Labels for the run.
        :param scrape_metrics:  Whether to scrape metrics.
        :param output_path:     Output path for artifacts.
        :param uid:             Unique identifier for the run.
        :return: RunObject ready for execution.
        """
        # Common parameters
        run_spec_parameters = {
            "url": source,
            "project_name": project_name,
            "save": save,
            "dirty": save,
            "wait_for_completion": True,
        }

        # Update with specific parameters
        run_spec_parameters.update(parameters)

        run_object = mlrun_model.RunObject(
            spec=mlrun_model.RunSpec(
                parameters=run_spec_parameters,
                handler=handler,
                notifications=notifications,
                scrape_metrics=scrape_metrics,
                output_path=output_path,
            ),
            metadata=mlrun_model.RunMetadata(
                name=run_name, project=project_name, uid=uid
            ),
        )

        if is_context:
            # The source is a context (local path contained in the image),
            # load the project from the context instead of a remote URL
            run_object.spec.parameters["project_context"] = source
            run_object.spec.parameters.pop("url", None)

        # Setting labels
        return self._label_run_object(run_object, labels)

    @staticmethod
    def _label_run_object(
        run_object: mlrun_model.RunObject,
        labels: dict[str, str],
    ) -> mlrun_model.RunObject:
        """
        Set labels on the run object.

        :param run_object: Run object to set labels on.
        :param labels:     Dictionary of labels.
        :return: RunObject with labels.
        """
        for key, value in labels.items():
            run_object = run_object.set_label(key, value)
        return run_object


class LoadRunner(BaseRunner, metaclass=mlrun.utils.singleton.Singleton):
    """
    Runner class for loading projects.
    """

    def run(
        self,
        runner: mlrun.run.KubejobRuntime,
        project: mlrun.common.schemas.ProjectOut,
        auth_info: mlrun.common.schemas.AuthInfo = None,
    ) -> mlrun_model.RunObject:
        """
        Run the project loader.

        :param runner:    Workflow runner function object.
        :param project:   MLRun project.
        :param auth_info: Authentication information of the request.
        :return: RunObject with run metadata, results, and status.
        """
        labels = {
            "project": project.metadata.name,
            mlrun_constants.MLRunInternalLabels.job_type: JOB_TYPE_PROJECT_LOADER,
        }

        return self.prepare_and_run(
            runner=runner,
            project=project,
            labels=labels,
            auth_info=auth_info,
        )

    def _prepare_run_object(
        self,
        project: mlrun.common.schemas.ProjectOut,
        labels: dict[str, str],
        run_name: Optional[str] = None,
        workflow_request: Optional[mlrun.common.schemas.WorkflowRequest] = None,
    ) -> mlrun_model.RunObject:
        """
        Prepare the RunObject for loading the project.

        :param project: MLRun project.
        :param labels:  Labels for the run.
        :param run_name: Name of the run.
        :return: RunObject ready for execution.
        """
        source, save, is_context = LoadRunner._validate_source(project, "")

        parameters = {
            "dirty": save,
        }

        run_object = self._create_run_object(
            source=source,
            project_name=project.metadata.name,
            save=save,
            handler="mlrun.projects.import_remote_project",
            parameters=parameters,
            run_name=run_name,
            is_context=is_context,
            labels=labels,
            scrape_metrics=mlrun_config.config.scrape_metrics,
        )

        return run_object

    @staticmethod
    def _validate_source(
        project: mlrun.common.schemas.ProjectOut,
        source: str,
    ) -> tuple[str, bool, bool]:
        """
        Validate the source for the project loader.

        :param project: MLRun project.
        :param source:  Source of the project.
        :return: Tuple with source, save flag, and is_context flag.
        """
        # In load-only flow, we always want to save the project
        save = True

        source = source or project.spec.source
        if not source:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Project source is required. Either specify the source in the project or provide it in the request."
            )

        if "://" not in source:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Invalid URL '{source}' for loading project '{project.metadata.name}'. Expected to be a remote URL."
            )

        return source, save, False


class WorkflowRunners(BaseRunner, metaclass=mlrun.utils.singleton.Singleton):
    """
    Runner class for workflows.
    """

    def schedule(
        self,
        runner: mlrun.run.KubejobRuntime,
        project: mlrun.common.schemas.ProjectOut,
        workflow_request: mlrun.common.schemas.WorkflowRequest,
        db_session: Session = None,
        auth_info: mlrun.common.schemas.AuthInfo = None,
    ):
        """
        Schedule workflow runner.

        :param runner:           Workflow runner function object.
        :param project:          MLRun project.
        :param workflow_request: Workflow request containing the workflow spec.
        :param db_session:       Database session.
        :param auth_info:        Authentication information of the request.
        """
        labels = {
            mlrun_constants.MLRunInternalLabels.job_type: JOB_TYPE_WORKFLOW_RUNNER,
            mlrun_constants.MLRunInternalLabels.workflow: workflow_request.spec.name,
        }

        # Generate unique UID
        meta_uid = uuid.uuid4().hex

        run_object = self._prepare_run_object(
            project=project,
            labels=labels,
            workflow_request=workflow_request,
            run_name=workflow_request.spec.name,
            uid=meta_uid,
            scrape_metrics=mlrun_config.config.scrape_metrics,
            url=project.spec.source,
        )

        # Mask notification parameters
        framework.utils.notifications.mask_notification_params_on_task_object(
            run_object, framework.constants.MaskOperations.CONCEAL
        )

        self._enrich_runner_node_selector(runner, workflow_request.spec)

        # Store function - this includes filling the spec.function which is required for submit run
        runner._store_function(
            runspec=run_object, meta=run_object.metadata, db=runner._get_db()
        )

        schedule = workflow_request.spec.schedule
        scheduled_object = {
            "task": run_object.to_dict(),
            "schedule": schedule,
        }

        services.api.utils.singletons.scheduler.get_scheduler().store_schedule(
            db_session=db_session,
            auth_info=auth_info,
            project=project.metadata.name,
            name=workflow_request.spec.name,
            kind=mlrun.common.schemas.ScheduleKinds.job,
            scheduled_object=scheduled_object,
            cron_trigger=schedule,
            labels=runner.metadata.labels,
        )

    def run(
        self,
        runner: mlrun.run.KubejobRuntime,
        project: mlrun.common.schemas.ProjectOut,
        auth_info: mlrun.common.schemas.AuthInfo = None,
        workflow_request: mlrun.common.schemas.WorkflowRequest = None,
    ) -> mlrun_model.RunObject:
        """
        Run workflow runner.

        :param runner:           Workflow runner function object.
        :param project:          MLRun project.
        :param auth_info:        Authentication information of the request.
        :param workflow_request: Workflow request containing the workflow spec.
        :return: RunObject with run metadata, results, and status.
        """
        labels = {
            "project": project.metadata.name,
            mlrun_constants.MLRunInternalLabels.job_type: JOB_TYPE_WORKFLOW_RUNNER,
            mlrun_constants.MLRunInternalLabels.workflow: runner.metadata.name,
        }

        self._enrich_runner_node_selector(runner, workflow_request.spec)

        return self.prepare_and_run(
            runner=runner,
            project=project,
            labels=labels,
            workflow_request=workflow_request,
            auth_info=auth_info,
            artifact_path=workflow_request.artifact_path,
        )

    def _prepare_run_object(
        self,
        project: mlrun.common.schemas.ProjectOut,
        labels: dict[str, str],
        workflow_request: mlrun.common.schemas.WorkflowRequest,
        run_name: Optional[str] = None,
        uid: Optional[str] = None,
        scrape_metrics: Optional[str] = None,
        url: str = "",
    ) -> mlrun_model.RunObject:
        """
        Prepare the RunObject for running the workflow.

        :param project:          MLRun project.
        :param labels:           Labels for the run.
        :param workflow_request: Workflow request containing the workflow spec.
        :param run_name:         Name of the run.
        :param uid:              Unique identifier for the run.
        :param scrape_metrics:   Whether to scrape metrics.
        :return: RunObject ready for execution.
        """
        source, save, is_context = WorkflowRunners._validate_source(
            project, workflow_request.source
        )

        notifications = [
            mlrun_model.Notification.from_dict(notification.dict())
            for notification in workflow_request.notifications or []
        ]

        output_path = (
            mlrun_utils.template_artifact_path(
                workflow_request.artifact_path or mlrun_config.config.artifact_path,
                project.metadata.name,
                uid,
            )
            if uid
            else None
        )

        parameters = dict(
            workflow_name=workflow_request.spec.name,
            workflow_path=workflow_request.spec.path,
            workflow_arguments=workflow_request.spec.args,
            artifact_path=workflow_request.artifact_path,
            workflow_handler=workflow_request.spec.handler,
            namespace=workflow_request.namespace,
            cleanup_ttl=workflow_request.spec.ttl,
            engine=workflow_request.spec.engine,
            local=workflow_request.spec.run_local,
            subpath=project.spec.subpath,
            url=url or source,
        )

        run_object = self._create_run_object(
            source=source,
            project_name=project.metadata.name,
            save=save,
            # TODO: We use 'load_and_run' for BC. Change it to 'load_and_run_workflow' in 1.10
            handler="mlrun.projects.load_and_run",
            parameters=parameters,
            notifications=notifications,
            run_name=run_name,
            is_context=is_context,
            labels=labels,
            uid=uid,
            output_path=output_path,
            scrape_metrics=scrape_metrics,
        )

        return run_object

    @staticmethod
    def _enrich_runner_node_selector(
        runner: mlrun.run.KubejobRuntime,
        workflow_spec: mlrun.common.schemas.WorkflowSpec,
    ):
        """
        Enrich the runner's node selector with the workflow's node selector.

        :param runner:        Workflow runner function object.
        :param workflow_spec: Workflow specification containing node selector information.
        """
        if workflow_spec.workflow_runner_node_selector:
            runner.spec.node_selector.update(
                workflow_spec.workflow_runner_node_selector
            )

    @staticmethod
    def _validate_source(
        project: mlrun.common.schemas.ProjectOut,
        source: str,
    ) -> tuple[str, bool, bool]:
        """
        Validate the source for the workflow runner.

        :param project: MLRun project.
        :param source:  Source of the project.
        :return: Tuple with source, save flag, and is_context flag.
        """
        save = bool(not source)
        source = source or project.spec.source

        if not source:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Project source is required. Either specify the source in the project or provide it in the request."
            )

        if source.startswith("/"):
            return source, save, True

        if source.startswith("./") or source == ".":
            build = project.spec.build
            source_code_target_dir = (
                build.get("source_code_target_dir") if build else ""
            )

            # When the source is relative, it is relative to the project's source_code_target_dir
            # If the project's source_code_target_dir is not set, the source is relative to the cwd
            if source_code_target_dir:
                source = os.path.normpath(os.path.join(source_code_target_dir, source))
            return source, save, True

        if "://" not in source:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Invalid source '{source}' for remote workflow. "
                f"Expected to be a remote URL or a path to the project context on the image. "
                f"For more details, see "
                f"https://docs.mlrun.org/en/latest/concepts/scheduled-jobs.html#scheduling-a-workflow"
            )

        return source, save, False

    @staticmethod
    def get_workflow_id(
        uid: str, project: str, engine: str, db_session: Session
    ) -> mlrun.common.schemas.GetWorkflowResponse:
        """
        Retrieve the actual workflow ID from the workflow runner.

        :param uid:        ID of the workflow runner job.
        :param project:    Name of the project.
        :param engine:     Pipeline runner engine (e.g., "kfp").
        :param db_session: Database session.
        :return: GetWorkflowResponse containing the workflow ID.
        """
        # Retrieve run
        run = services.api.crud.Runs().get_run(
            db_session=db_session, uid=uid, iter=0, project=project
        )
        run_object = mlrun_model.RunObject.from_dict(run)
        state = run_object.status.state
        workflow_id = None

        if isinstance(run_object.status.results, dict):
            workflow_id = run_object.status.results.get("workflow_id", None)

        if workflow_id is None:
            if (
                run_object.metadata.is_workflow_runner()
                and run_object.status.is_failed()
            ):
                state_text = run_object.status.error
                workflow_name = run_object.spec.parameters.get(
                    "workflow_name", "<unknown>"
                )
                raise mlrun.errors.MLRunPreconditionFailedError(
                    f"Failed to run workflow {workflow_name}, state: {state}, state_text: {state_text}"
                )
            elif (
                engine == "local"
                and state.casefold()
                == mlrun_pipelines.common.models.RunStatuses.running.casefold()
            ):
                workflow_id = run_object.metadata.uid
            else:
                raise mlrun.errors.MLRunNotFoundError(
                    f"Workflow ID of run {project}:{uid} not found"
                )

        return mlrun.common.schemas.GetWorkflowResponse(workflow_id=workflow_id)
