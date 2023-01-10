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
import typing
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
from mlrun.utils.helpers import logger

router = fastapi.APIRouter()


def _get_workflow_by_name(
    project: mlrun.api.schemas.Project, name: str
) -> typing.Optional[Dict]:
    for project_workflow in project.spec.workflows:
        if project_workflow["name"] == name:
            return project_workflow
    return None


def _enrich_workflow(
    project: mlrun.api.schemas.Project,
    name: str,
    spec: mlrun.api.schemas.WorkflowSpec,
    arguments,
):
    # Verifying workflow exists in project:
    workflow = _get_workflow_by_name(project, name)
    if not workflow:
        log_and_raise(
            reason=f"workflow {name} not found in project",
        )
    if spec:
        # Merge between the workflow spec provided in the request with existing
        # workflow while the provided workflow takes precedence over the existing workflow params
        workflow = copy.deepcopy(workflow)
        workflow = {key: val or workflow.get(key) for key, val in spec.dict().items()}

    workflow_spec = mlrun.api.schemas.WorkflowSpec(**workflow)
    # Overriding arguments of the existing workflow:
    if arguments:
        workflow_spec.args = workflow_spec.args or {}
        workflow_spec.args.update(arguments)

    return workflow_spec


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

    # Verifying permissions:
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

    workflow_spec = _enrich_workflow(
        project=project,
        name=name,
        spec=workflow_request.spec,
        arguments=workflow_request.arguments,
    )
    updated_request = workflow_request.copy()
    updated_request.spec = workflow_spec

    # This function is for loading the project and running workflow remotely.
    # In this way we can schedule workflows (by scheduling a job that runs the workflow)
    workflow_runner = mlrun.api.crud.WorkflowRunners().create_runner(
        run_name=updated_request.run_name or f"workflow-runner-{workflow_spec.name}",
        project=project.metadata.name,
        db_session=db_session,
        auth_info=auth_info,
    )
    logger.debug(
        "saved function for running workflow",
        project_name=workflow_runner.metadata.project,
        function_name=workflow_runner.metadata.name,
        workflow_name=workflow_spec.name,
        arguments=workflow_spec.args,
        source=project.spec.source,
        kind=workflow_runner.kind,
    )

    run_uid = None
    status = None
    workflow_action = ""
    try:
        if workflow_spec.schedule:
            # Re-route to chief in case of schedule:
            if (
                mlrun.mlconf.httpdb.clusterization.role
                != mlrun.api.schemas.ClusterizationRole.chief
            ):
                chief_client = mlrun.api.utils.clients.chief.Client()
                return chief_client.submit_workflow(
                    project=project.metadata.name,
                    name=name,
                    json=workflow_request.dict(),
                )
            workflow_action = "schedule"
            mlrun.api.crud.WorkflowRunners().schedule(
                runner=workflow_runner,
                project=project,
                workflow_request=updated_request,
                db_session=db_session,
                auth_info=auth_info,
            )
            status = "scheduled"

        else:
            workflow_action = "run"
            run = mlrun.api.crud.WorkflowRunners().run(
                runner=workflow_runner,
                project=project,
                workflow_request=updated_request,
            )
            status = mlrun.run.RunStatuses.running
            run_uid = run.uid()
    except Exception as error:
        logger.error(traceback.format_exc())
        log_and_raise(
            reason=f"Workflow {workflow_spec.name} {workflow_action} failed!, error: {error}"
        )

    return mlrun.api.schemas.WorkflowResponse(
        project=project.metadata.name,
        name=workflow_spec.name,
        status=status,
        run_id=run_uid,
        schedule=workflow_spec.schedule,
    )


@router.get(
    "/projects/{project}/{name}/{uid}",
    response_model=mlrun.api.schemas.GetWorkflowResponse,
)
def get_workflow_id(
    project: str,
    name: str,
    uid: str,
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
    :param name:        name of the workflow
    :param uid:         the id of the running job that runs the workflow
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
