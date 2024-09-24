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
import os
import uuid

import mlrun_pipelines.common.models
from sqlalchemy.orm import Session

import mlrun.common.constants as mlrun_constants
import mlrun.common.schemas
import mlrun.utils.singleton
import server.api.api.utils
from mlrun.config import config
from mlrun.model import Credentials, RunMetadata, RunObject, RunSpec
from mlrun.utils import template_artifact_path


class WorkflowRunners(
    metaclass=mlrun.utils.singleton.Singleton,
):
    @staticmethod
    def create_runner(
        run_name: str,
        project: str,
        db_session: Session,
        auth_info: mlrun.common.schemas.AuthInfo,
        image: str,
    ) -> mlrun.run.KubejobRuntime:
        """
        Creating the base object for the workflow runner function with
        all the necessary metadata to create it on server-side.

        :param run_name:    workflow-runner function name
        :param project:     project name
        :param db_session:  session that manages the current dialog with the database
        :param auth_info:   auth info of the request
        :param image:       image for the workflow runner job

        :returns: workflow runner object
        """
        runner = mlrun.new_function(
            name=run_name,
            project=project,
            kind=mlrun.runtimes.RuntimeKinds.job,
            image=image,
        )

        runner.set_db_connection(server.api.api.utils.get_run_db_instance(db_session))

        # Enrichment and validation requires access key
        runner.metadata.credentials.access_key = Credentials.generate_access_key

        server.api.api.utils.apply_enrichment_and_validation_on_function(
            function=runner,
            auth_info=auth_info,
        )

        runner.save()
        return runner

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

        :param runner:              workflow runner function object
        :param project:             MLRun project
        :param workflow_request:    contains the workflow spec, that will be scheduled
        :param db_session:          session that manages the current dialog with the database
        :param auth_info:           auth info of the request
        """
        labels = {
            mlrun_constants.MLRunInternalLabels.job_type: "workflow-runner",
            mlrun_constants.MLRunInternalLabels.workflow: workflow_request.spec.name,
        }

        run_spec = self._prepare_run_object_for_scheduling(
            project=project,
            workflow_request=workflow_request,
            labels=labels,
        )
        workflow_spec = workflow_request.spec
        if workflow_spec.workflow_runner_node_selector:
            runner.spec.node_selector.update(
                workflow_spec.workflow_runner_node_selector
            )

        # this includes filling the spec.function which is required for submit run
        runner._store_function(
            runspec=run_spec, meta=run_spec.metadata, db=runner._get_db()
        )

        schedule = workflow_spec.schedule
        scheduled_object = {
            "task": run_spec.to_dict(),
            "schedule": schedule,
        }

        server.api.api.utils.get_scheduler().store_schedule(
            db_session=db_session,
            auth_info=auth_info,
            project=project.metadata.name,
            name=workflow_spec.name,
            kind=mlrun.common.schemas.ScheduleKinds.job,
            scheduled_object=scheduled_object,
            cron_trigger=schedule,
            labels=runner.metadata.labels,
        )

    def _prepare_run_object_for_scheduling(
        self,
        project: mlrun.common.schemas.ProjectOut,
        workflow_request: mlrun.common.schemas.WorkflowRequest,
        labels: dict[str, str],
    ) -> mlrun.run.RunObject:
        """
        Preparing all the necessary metadata and specifications for scheduling workflow from server-side.

        :param project:             MLRun project
        :param workflow_request:    contains the workflow spec and extra data for the run object
        :param labels:              dict of the task labels

        :returns: RunObject ready for schedule.
        """
        meta_uid = uuid.uuid4().hex

        notifications = [
            mlrun.model.Notification.from_dict(notification.dict())
            for notification in workflow_request.notifications or []
        ]

        source, save, is_context = self._validate_source(
            project, workflow_request.source
        )
        workflow_spec = workflow_request.spec
        run_object = RunObject(
            spec=RunSpec(
                parameters=dict(
                    url=project.spec.source,
                    project_name=project.metadata.name,
                    workflow_name=workflow_spec.name,
                    workflow_path=workflow_spec.path,
                    workflow_arguments=workflow_spec.args,
                    artifact_path=workflow_request.artifact_path,
                    workflow_handler=workflow_spec.handler,
                    namespace=workflow_request.namespace,
                    ttl=workflow_spec.ttl,
                    engine=workflow_spec.engine,
                    local=workflow_spec.run_local,
                    save=save,
                    # save=True modifies the project.yaml (by enrichment) so the local git repo is becoming dirty
                    dirty=save,
                    subpath=project.spec.subpath,
                    # remote pipeline pod stays alive for the whole lifetime of the pipeline.
                    # once the pipeline is done, the pod finishes (either successfully or not) and notifications
                    # can be sent.
                    wait_for_completion=True,
                ),
                handler="mlrun.projects.load_and_run",
                scrape_metrics=config.scrape_metrics,
                output_path=template_artifact_path(
                    workflow_request.artifact_path or config.artifact_path,
                    project.metadata.name,
                    meta_uid,
                ),
                notifications=notifications,
            ),
            metadata=RunMetadata(
                uid=meta_uid, name=workflow_spec.name, project=project.metadata.name
            ),
        )

        if is_context:
            # The source is a context (local path contained in the image),
            # load the project from the context instead of remote URL
            run_object.spec.parameters["project_context"] = source
            run_object.spec.parameters.pop("url", None)

        # Setting labels:
        return self._label_run_object(run_object, labels)

    def run(
        self,
        runner: mlrun.run.KubejobRuntime,
        project: mlrun.common.schemas.ProjectOut,
        workflow_request: mlrun.common.schemas.WorkflowRequest = None,
        load_only: bool = False,
        auth_info: mlrun.common.schemas.AuthInfo = None,
    ) -> RunObject:
        """
        Run workflow runner.

        :param runner:              workflow runner function object
        :param project:             MLRun project
        :param workflow_request:    contains the workflow spec, that will be executed
        :param load_only:           If True, will only load the project remotely (without running workflow)
        :param auth_info:           auth info of the request

        :returns: run context object (RunObject) with run metadata, results and status
        """
        labels = {"project": project.metadata.name}
        if load_only:
            labels[mlrun_constants.MLRunInternalLabels.job_type] = "project-loader"
        else:
            labels[mlrun_constants.MLRunInternalLabels.job_type] = "workflow-runner"
            labels[mlrun_constants.MLRunInternalLabels.workflow] = runner.metadata.name
        mlrun.runtimes.utils.enrich_run_labels(
            labels, [mlrun.common.runtimes.constants.RunLabels.owner]
        )

        run_spec = self._prepare_run_object_for_single_run(
            project=project,
            labels=labels,
            workflow_request=workflow_request,
            run_name=runner.metadata.name,
            load_only=load_only,
        )

        artifact_path = workflow_request.artifact_path if workflow_request else ""
        if workflow_request:
            workflow_spec_node_selector = (
                workflow_request.spec.workflow_runner_node_selector
            )
            if workflow_spec_node_selector:
                runner.spec.node_selector.update(workflow_spec_node_selector)

        # TODO: Passing auth_info is required for server side launcher, but the runner is already enriched with the
        #  auth_info when it was created in create_runner. We should move the enrichment to the launcher and need to
        #  make sure it is safe for scheduling and project load endpoint.
        return runner.run(
            runspec=run_spec,
            artifact_path=artifact_path,
            local=False,
            watch=False,
            auth_info=auth_info,
        )

    @staticmethod
    def get_workflow_id(
        uid: str, project: str, engine: str, db_session: Session
    ) -> mlrun.common.schemas.GetWorkflowResponse:
        """
        Retrieving the actual workflow id form the workflow runner

        :param uid:         the id of the workflow runner job
        :param project:     name of the project
        :param engine:      pipeline runner, for example: "kfp"
        :param db_session:  session that manages the current dialog with the database

        :return: The id of the workflow.
        """
        # Reading run:
        run = server.api.crud.Runs().get_run(
            db_session=db_session, uid=uid, iter=0, project=project
        )
        run_object = RunObject.from_dict(run)
        state = run_object.status.state
        workflow_id = None
        if isinstance(run_object.status.results, dict):
            workflow_id = run_object.status.results.get("workflow_id", None)

        if workflow_id is None:
            if (
                run_object.metadata.is_workflow_runner()
                and run_object.status.is_failed()
            ):
                state = run_object.status.state
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
                    f"Workflow id of run {project}:{uid} not found"
                )

        return mlrun.common.schemas.GetWorkflowResponse(workflow_id=workflow_id)

    def _prepare_run_object_for_single_run(
        self,
        project: mlrun.common.schemas.ProjectOut,
        labels: dict[str, str],
        workflow_request: mlrun.common.schemas.WorkflowRequest = None,
        run_name: str = None,
        load_only: bool = False,
    ) -> mlrun.run.RunObject:
        """
        Preparing all the necessary metadata and specifications for running workflow from server-side.

        :param project:             MLRun project
        :param labels:              dict of the task labels
        :param workflow_request:    contains the workflow spec and extra data for the run object
        :param run_name:            workflow-runner function name
        :param load_only:           if True, will only load the project remotely (without running workflow)

        :returns: RunObject ready for execution.
        """
        notifications = None
        if workflow_request:
            notifications = [
                mlrun.model.Notification.from_dict(notification.dict())
                for notification in workflow_request.notifications or []
            ]

        source = workflow_request.source if workflow_request else ""
        source, save, is_context = self._validate_source(project, source, load_only)
        run_object = RunObject(
            spec=RunSpec(
                parameters=dict(
                    url=source,
                    project_name=project.metadata.name,
                    load_only=load_only,
                    save=save,
                    # save=True modifies the project.yaml (by enrichment) so the local git repo is becoming dirty
                    dirty=save,
                    # remote pipeline pod stays alive for the whole lifetime of the pipeline.
                    # once the pipeline is done, the pod finishes (either successfully or not) and notifications
                    # can be sent.
                    wait_for_completion=True,
                ),
                handler="mlrun.projects.load_and_run",
                notifications=notifications,
            ),
            metadata=RunMetadata(name=run_name),
        )

        if is_context:
            # The source is a context (local path contained in the image),
            # load the project from the context instead of remote URL
            run_object.spec.parameters["project_context"] = source
            run_object.spec.parameters.pop("url", None)

        if not load_only:
            workflow_spec = workflow_request.spec
            run_object.spec.parameters.update(
                dict(
                    workflow_name=workflow_spec.name,
                    workflow_path=workflow_spec.path,
                    workflow_arguments=workflow_spec.args,
                    artifact_path=workflow_request.artifact_path,
                    workflow_handler=workflow_spec.handler,
                    namespace=workflow_request.namespace,
                    ttl=workflow_spec.ttl,
                    engine=workflow_spec.engine,
                    local=workflow_spec.run_local,
                    subpath=project.spec.subpath,
                )
            )

        # Setting labels:
        return self._label_run_object(run_object, labels)

    @staticmethod
    def _validate_source(
        project: mlrun.common.schemas.ProjectOut, source: str, load_only: bool = False
    ) -> tuple[str, bool, bool]:
        """
        In case the user provided a source we want to load the project from the source
        (like from a specific commit/branch from git repo) without changing the source of the project (save=False).

        :param project:     MLRun project
        :param source:      The source of the project, remote URL or context on image that contains the
                            project yaml file.
        :param load_only:   If we only load the project, it must be saved to ensure we are not running a pipeline
                            without a project as it's not supported.

        :returns: A tuple of:
              [0] = The source string.
              [1] = Bool if the project need to be saved afterward.
              [2] = Bool if the source is a path.
        """

        save = bool(load_only or not source)
        source = source or project.spec.source
        if not source:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Project source is required. Either specify the source in the project or provide it in the request."
            )

        if not load_only:
            # Path like source is not supported for load_only since it uses the mlrun default image
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
                    source = os.path.normpath(
                        os.path.join(source_code_target_dir, source)
                    )
                return source, save, True

        if "://" not in source:
            if load_only:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Invalid URL '{source}' for loading project '{project.metadata.name}'. "
                    f"Expected to be in format: <scheme>://<netloc>/<path>;<params>?<query>#<fragment>."
                )

            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Invalid source '{source}' for remote workflow."
                f"Expected to be a remote URL or a path to the project context on image."
                f" For more details, see"
                f" https://docs.mlrun.org/en/latest/concepts/scheduled-jobs.html#scheduling-a-workflow"
            )

        return source, save, False

    @staticmethod
    def _label_run_object(
        run_object: mlrun.run.RunObject,
        labels: dict[str, str],
    ) -> mlrun.run.RunObject:
        """
        Setting labels to the task

        :param run_object:  run object to set labels on
        :param labels:      dict of labels

        :returns: labeled RunObject
        """
        for key, value in labels.items():
            run_object = run_object.set_label(key, value)
        return run_object
