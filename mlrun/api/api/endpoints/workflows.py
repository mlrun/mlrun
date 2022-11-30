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
import copy
import traceback
import uuid
from http import HTTPStatus
from typing import Dict

import fastapi
from sqlalchemy.orm import Session

import mlrun
import mlrun.api.api.deps
import mlrun.api.api.utils
import mlrun.api.crud
import mlrun.api.schemas
import mlrun.api.utils.auth.verifier
import mlrun.api.utils.clients.chief
import mlrun.api.utils.singletons.db
import mlrun.api.utils.singletons.project_member
import mlrun.projects.pipelines
from mlrun.api.api.utils import log_and_raise
from mlrun.config import config
from mlrun.utils.helpers import logger

router = fastapi.APIRouter()


def _get_workflow_by_name(project: mlrun.api.schemas.Project, workflow) -> Dict:
    for project_workflow in project.spec.workflows:
        if project_workflow["name"] == workflow:
            return project_workflow


@router.post(
    "/projects/{project}/workflows/{name}/submit",
    status_code=HTTPStatus.ACCEPTED.value,
    response_model=mlrun.api.schemas.WorkflowResponse,
)
def submit_workflow(
    project: str,
    name: str,
    workflow_request: mlrun.api.schemas.WorkflowRequest = mlrun.api.schemas.WorkflowRequest(),
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
    db_session: Session = fastapi.Depends(mlrun.api.api.deps.get_db_session),
):
    """
    Submitting a workflow of existing project.

    To support workflow scheduling, we use here an auxiliary function called 'load_and_run'.
    This function runs remotely (in a distinct pod), loads a project and then runs the workflow.
    In this way we can run the workflow remotely with the workflow's engine or
    schedule this function which in every time loads the project and runs the workflow.

    Notice:

    in case of simply running a workflow, the returned run_id value is the id of the run of the auxiliary function.
    For getting the id and status of the workflow, use the `get_workflow_id` endpoint with the returned run id.

    :param project:             name of the project
    :param name:                name of the workflow
    :param workflow_request:    the request includes: workflow spec, arguments for the workflow, artifact path
                                as the artifact target path of the workflow, source url of the project for overriding
                                the existing one, run name to override the default: 'workflow-runner-<workflow name>'
                                and kubernetes namespace if other than default
    :param auth_info:           auth info of the request
    :param db_session:          session that manages the current dialog with the database

    :returns: A response that contains the project name, workflow name, name of the workflow,
             status, run id (in case of a single run) and schedule (in case of scheduling)
    """
    # Getting project:
    project = (
        mlrun.api.utils.singletons.project_member.get_project_member().get_project(
            db_session=db_session, name=project, leader_session=auth_info.session
        )
    )

    # Permission checks:
    # 1. CREATE run
    mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        resource_type=mlrun.api.schemas.AuthorizationResourceTypes.run,
        project_name=project.metadata.name,
        resource_name=workflow_request.run_name or "",
        action=mlrun.api.schemas.AuthorizationAction.create,
        auth_info=auth_info,
    )
    # 2. READ workflow
    mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        resource_type=mlrun.api.schemas.AuthorizationResourceTypes.workflow,
        project_name=project.metadata.name,
        resource_name=name,
        action=mlrun.api.schemas.AuthorizationAction.read,
        auth_info=auth_info,
    )

    # Verifying that project has a workflow name:
    project_workflow_names = [workflow["name"] for workflow in project.spec.workflows]
    if name not in project_workflow_names:
        log_and_raise(
            reason=f"{name} workflow not found in project",
        )
    spec = workflow_request.spec
    workflow = _get_workflow_by_name(project, name)

    if spec:
        # Merge between the workflow spec provided in the request with existing
        # workflow while the provided workflow takes precedence over the existing workflow params
        workflow = copy.deepcopy(workflow)
        workflow = {key: val or workflow.get(key) for key, val in spec.dict().items()}

    workflow_spec = mlrun.projects.pipelines.WorkflowSpec.from_dict(workflow)
    run_name = workflow_request.run_name or f"workflow-runner-{workflow_spec.name}"

    # Scheduling must be performed only by chief:
    if (
        workflow_spec.schedule
        and mlrun.mlconf.httpdb.clusterization.role
        != mlrun.api.schemas.ClusterizationRole.chief
    ):
        try:
            # Check permission for scheduling:
            allowed_creating_schedule = mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
                resource_type=mlrun.api.schemas.AuthorizationResourceTypes.schedule,
                project_name=project.metadata.name,
                resource_name=run_name,
                action=mlrun.api.schemas.AuthorizationAction.create,
                auth_info=auth_info,
            )
        except mlrun.errors.MLRunAccessDeniedError:
            allowed_creating_schedule = False

        if not allowed_creating_schedule:
            chief_client = mlrun.api.utils.clients.chief.Client()
            return chief_client.submit_workflow(
                project=project.metadata.name, name=name, json=workflow_request.dict()
            )
    # Preparing inputs for load_and_run function
    # 1. To override the source of the project.
    # This is mainly for supporting loading project from a certain commits (GitHub)
    if workflow_request.source:
        project.spec.source = workflow_request.source

    # 2. Overriding arguments of the existing workflow:
    if workflow_request.arguments:
        workflow_spec.args = workflow_spec.args or {}
        workflow_spec.args.update(workflow_request.arguments)

    # This function is for loading the project and running workflow remotely.
    # In this way we support scheduling workflows (by scheduling a job that runs the workflow
    run_name = run_name or f"workflow-runner-{workflow_spec.name}"

    # Creating the auxiliary function for loading and running the workflow
    load_and_run_fn = mlrun.api.crud.Workflows().create_function(
        run_name=run_name,
        project=project.metadata.name,
        kind=mlrun.runtimes.RuntimeKinds.job,
        # For preventing deployment
        image=mlrun.mlconf.default_base_image,
        db_session=db_session,
        auth_info=auth_info,
        # Enrichment and validation requires an access key.
        access_key=mlrun.model.Credentials.generate_access_key,
    )
    logger.debug(
        "saved function for running workflow",
        project_name=load_and_run_fn.metadata.project,
        function_name=load_and_run_fn.metadata.name,
        workflow_name=workflow_spec.name,
        arguments=workflow_spec.args,
        source=project.spec.source,
        kind=load_and_run_fn.kind,
    )

    if workflow_spec.schedule:
        # This logic follows the one is performed in `BaseRuntime._enrich_run()`
        meta_uid = uuid.uuid4().hex

        # creating runspec for scheduling:
        spec = {
            "scrape_metrics": config.scrape_metrics,
            "output_path": (
                workflow_request.artifact_path or config.artifact_path
            ).replace("{{run.uid}}", meta_uid),
        }
        metadata = {"uid": meta_uid, "project": project.metadata.name}

        try:
            mlrun.api.crud.Workflows().execute_function(
                function=load_and_run_fn,
                project=project,
                workflow_spec=workflow_spec,
                artifact_path=workflow_request.artifact_path,
                namespace=workflow_request.namespace,
                spec=spec,
                metadata=metadata,
                db_session=db_session,
                auth_info=auth_info,
            )

        except Exception as error:
            logger.error(traceback.format_exc())
            log_and_raise(
                reason=f"Scheduling workflow {workflow_spec.name} failed with the following error: {error}"
            )

        return mlrun.api.schemas.WorkflowResponse(
            project=project.metadata.name,
            name=workflow_spec.name,
            status="scheduled",
            schedule=workflow_spec.schedule,
        )

    else:
        run_arguments = {"local": False}
        run = mlrun.api.crud.Workflows().execute_function(
            function=load_and_run_fn,
            project=project,
            workflow_spec=workflow_spec,
            artifact_path=workflow_request.artifact_path,
            namespace=workflow_request.namespace,
            workflow_name=run_name,
            run_kwargs=run_arguments,
        )

        state = mlrun.run.RunStatuses.running

        return mlrun.api.schemas.WorkflowResponse(
            project=project.metadata.name,
            name=workflow_spec.name,
            status=state,
            run_id=run.uid(),
        )


@router.get(
    "/projects/{project}/{uid}", response_model=mlrun.api.schemas.GetWorkflowResponse
)
def get_workflow_id(
    project: str,
    uid: str,
    name: str,
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
    db_session: Session = fastapi.Depends(mlrun.api.api.deps.get_db_session),
):
    """
    Retrieve workflow id from the uid of the workflow runner.

    When creating a remote workflow we are creating an auxiliary function
    which is responsible for actually running the workflow,
    as we don't know beforehand the workflow uid but only the run uid of the auxiliary function we ran,
    we have to wait until the running function will log the workflow id it created.
    Because we don't know how long it will take for the run to create the workflow
    we decided to implement that in an asynchronous mechanism which at first,
    client will get the run uid and then will pull the workflow id from the run id
    kinda as you would use a background task to query if it finished.

    Supporting workflows that executed by the remote engine **only**.

    :param project:     name of the project
    :param uid:         the id of the running job that runs the workflow
    :param name:        name of the workflow
    :param auth_info:   auth info of the request
    :param db_session:  session that manages the current dialog with the database

    :returns: The id and status of the workflow.
    """
    # Check permission READ run:
    mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.run,
        project,
        uid,
        mlrun.api.schemas.AuthorizationAction.read,
        auth_info,
    )
    # Check permission READ workflow:
    mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.workflow,
        project,
        name,
        mlrun.api.schemas.AuthorizationAction.read,
        auth_info,
    )

    # Reading run:
    run = mlrun.api.crud.Runs().get_run(
        db_session=db_session, uid=uid, iter=0, project=project
    )
    run_object = mlrun.RunObject.from_dict(run)

    if not isinstance(run_object.status.results, dict):
        # in case the run object did not instantiate with the "results" field at this point
        return {"workflow_id": None}

    workflow_id = run_object.status.results.get("workflow_id", "")

    return mlrun.api.schemas.GetWorkflowResponse(workflow_id=workflow_id)
