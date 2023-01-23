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
import uuid
from typing import List, Tuple

from sqlalchemy.orm import Session

import mlrun.api.schemas
import mlrun.utils.singleton
from mlrun.api.api.utils import (
    apply_enrichment_and_validation_on_function,
    get_run_db_instance,
    get_scheduler,
)
from mlrun.config import config
from mlrun.model import Credentials, RunMetadata, RunObject, RunSpec


class WorkflowRunners(
    metaclass=mlrun.utils.singleton.Singleton,
):
    @staticmethod
    def create_runner(
        run_name: str,
        project: str,
        db_session: Session,
        auth_info: mlrun.api.schemas.AuthInfo,
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
            # For preventing deployment:
            image=image,
        )

        runner.set_db_connection(get_run_db_instance(db_session))

        # Enrichment and validation requires access key
        runner.metadata.credentials.access_key = Credentials.generate_access_key

        apply_enrichment_and_validation_on_function(
            function=runner,
            auth_info=auth_info,
        )

        runner.save()
        return runner

    def schedule(
        self,
        runner: mlrun.run.KubejobRuntime,
        project: mlrun.api.schemas.Project,
        workflow_request: mlrun.api.schemas.WorkflowRequest,
        db_session: Session = None,
        auth_info: mlrun.api.schemas.AuthInfo = None,
    ):
        """
        Schedule workflow runner.

        :param runner:              workflow runner function object
        :param project:             MLRun project
        :param workflow_request:    contains the workflow spec, that will be scheduled
        :param db_session:          session that manages the current dialog with the database
        :param auth_info:           auth info of the request
        """
        labels = [
            ("job-type", "workflow-runner"),
            ("workflow", workflow_request.spec.name),
        ]

        run_spec = self._prepare_run_object_for_scheduling(
            project=project,
            workflow_request=workflow_request,
            labels=labels,
        )
        # this includes filling the spec.function which is required for submit run
        runner._store_function(
            runspec=run_spec, meta=run_spec.metadata, db=runner._get_db()
        )
        workflow_spec = workflow_request.spec
        schedule = workflow_spec.schedule
        scheduled_object = {
            "task": run_spec.to_dict(),
            "schedule": schedule,
        }

        if workflow_request.spec.override:
            get_scheduler().store_schedule(
                db_session=db_session,
                auth_info=auth_info,
                project=project.metadata.name,
                name=workflow_spec.name,
                scheduled_object=scheduled_object,
                cron_trigger=schedule,
                labels=runner.metadata.labels,
                kind=mlrun.api.schemas.ScheduleKinds.job,
            )
        else:
            try:
                get_scheduler().create_schedule(
                    db_session=db_session,
                    auth_info=auth_info,
                    project=project.metadata.name,
                    name=workflow_spec.name,
                    kind=mlrun.api.schemas.ScheduleKinds.job,
                    scheduled_object=scheduled_object,
                    cron_trigger=schedule,
                    labels=runner.metadata.labels,
                )
            except mlrun.errors.MLRunConflictError:
                raise mlrun.errors.MLRunConflictError(
                    f"There is already a schedule for workflow {workflow_spec.name}."
                    " If you want to override this schedule use override=True (SDK) or --override-workflow (CLI)"
                )

    def run(
        self,
        runner: mlrun.run.KubejobRuntime,
        project: mlrun.api.schemas.Project,
        workflow_request: mlrun.api.schemas.WorkflowRequest = None,
        load_only: bool = False,
    ) -> RunObject:
        """
        Run workflow runner.

        :param runner:              workflow runner function object
        :param project:             MLRun project
        :param workflow_request:    contains the workflow spec, that will be executed
        :param load_only:           If True, will only load the project remotely (without running workflow)

        :returns: run context object (RunObject) with run metadata, results and status
        """
        if load_only:
            labels = [
                ("job-type", "project-loader"),
                ("project", project.metadata.name),
            ]
        else:
            labels = [
                ("job-type", "workflow-runner"),
                ("workflow", runner.metadata.name),
            ]

        run_spec = self._prepare_run_object_for_single_run(
            project=project,
            labels=labels,
            workflow_request=workflow_request,
            run_name=runner.metadata.name,
            load_only=load_only,
        )

        artifact_path = workflow_request.artifact_path if workflow_request else ""
        return runner.run(
            runspec=run_spec,
            artifact_path=artifact_path,
            local=False,
        )

    @staticmethod
    def get_workflow_id(
        uid: str, project: str, db_session: Session
    ) -> mlrun.api.schemas.GetWorkflowResponse:
        """
        Retrieving the actual workflow id form the workflow runner

        :param uid:         the id of the workflow runner job
        :param project:     name of the project
        :param db_session:  session that manages the current dialog with the database

        :return: The id of the workflow.
        """
        # Reading run:
        run = mlrun.api.crud.Runs().get_run(
            db_session=db_session, uid=uid, iter=0, project=project
        )
        run_object = RunObject.from_dict(run)

        workflow_id = None
        if isinstance(run_object.status.results, dict):
            workflow_id = run_object.status.results.get("workflow_id")
        if not workflow_id:
            raise mlrun.errors.MLRunNotFoundError(
                f"workflow id of run {uid}:{project} not found"
            )

        return mlrun.api.schemas.GetWorkflowResponse(workflow_id=workflow_id)

    @staticmethod
    def _label_run_object(
        run_object: mlrun.run.RunObject,
        labels: List[Tuple[str, str]],
    ) -> mlrun.run.RunObject:
        """
        Setting labels to the task

        :param run_object:  run object to set labels on
        :param labels:      list that contains pairs of label keys and label values

        :returns: labeled RunObject
        """
        for key, value in labels:
            run_object = run_object.set_label(key, value)
        return run_object

    @staticmethod
    def _set_source(
        project: mlrun.api.schemas.Project, source: str, load_only: bool = False
    ) -> bool:
        """
        Setting the project source.
        In case the user provided a source we want to load the project from the source
        (like from a specific commit/branch from git repo) without changing the source of the project (save=False).

        :param project:     MLRun project
        :param source:      the source of the project, needs to be a remote URL that contains the project yaml file.
        :param load_only:   if we only load the project, the project must be saved.

        :returns: True if the project need to be saved afterwards.
        """

        save = True
        if source and not load_only:
            save = False
            project.spec.source = source

        if "://" not in project.spec.source:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"remote workflows can only be performed by a project with remote source,"
                f" the given source '{project.spec.source}' is not remote"
            )
        return save

    def _prepare_run_object_for_scheduling(
        self,
        project: mlrun.api.schemas.Project,
        workflow_request: mlrun.api.schemas.WorkflowRequest,
        labels: List[Tuple[str, str]],
    ) -> mlrun.run.RunObject:
        """
        Preparing all the necessary metadata and specifications for scheduling workflow from server-side.

        :param project:             MLRun project
        :param workflow_request:    contains the workflow spec and extra data for the run object
        :param labels:              pairs of label keys and label values for the task

        :returns: RunObject ready for schedule.
        """
        meta_uid = uuid.uuid4().hex

        save = self._set_source(project, workflow_request.source)
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
                ),
                handler="mlrun.projects.load_and_run",
                scrape_metrics=config.scrape_metrics,
                output_path=(
                    workflow_request.artifact_path or config.artifact_path
                ).replace("{{run.uid}}", meta_uid),
            ),
            metadata=RunMetadata(
                uid=meta_uid, name=workflow_spec.name, project=project.metadata.name
            ),
        )

        # Setting labels:
        return self._label_run_object(run_object, labels)

    def _prepare_run_object_for_single_run(
        self,
        project: mlrun.api.schemas.Project,
        labels: List,
        workflow_request: mlrun.api.schemas.WorkflowRequest = None,
        run_name: str = None,
        load_only: bool = False,
    ) -> mlrun.run.RunObject:
        """
        Preparing all the necessary metadata and specifications for running workflow from server-side.

        :param project:             MLRun project
        :param labels:              pairs of label keys and label values for the task
        :param workflow_request:    contains the workflow spec and extra data for the run object
        :param run_name:            workflow-runner function name
        :param load_only:           if True, will only load the project remotely (without running workflow)

        :returns: RunObject ready for execution.
        """
        source = workflow_request.source if workflow_request else ""
        save = self._set_source(project, source, load_only)
        run_object = RunObject(
            spec=RunSpec(
                parameters=dict(
                    url=project.spec.source,
                    project_name=project.metadata.name,
                    load_only=load_only,
                    save=save,
                ),
                handler="mlrun.projects.load_and_run",
            ),
            metadata=RunMetadata(name=run_name),
        )

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
                )
            )

        # Setting labels:
        return self._label_run_object(run_object, labels)
