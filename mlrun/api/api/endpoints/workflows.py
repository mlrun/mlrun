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
from typing import Any, Dict, Optional, Union

import fastapi
from sqlalchemy.orm import Session

import mlrun
import mlrun.api.api.deps
import mlrun.api.api.utils
import mlrun.api.utils.singletons.db
import mlrun.api.schemas
import mlrun.api.utils.auth.verifier
import mlrun.api.utils.clients.chief
import mlrun.api.utils.singletons.project_member
import mlrun.projects.pipelines
from mlrun.api.api.utils import (
    apply_enrichment_and_validation_on_function,
    get_run_db_instance,
    get_scheduler,
    log_and_raise,
)
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
)
def submit_workflow(
    project: str,
    name: str,
    spec: Optional[mlrun.api.schemas.WorkflowSpec] = None,
    arguments: Optional[Dict] = None,
    artifact_path: Optional[str] = None,
    source: Optional[str] = None,
    run_name: Optional[str] = None,
    namespace: Optional[str] = None,
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
    db_session: Session = fastapi.Depends(mlrun.api.api.deps.get_db_session),
) -> Union[Dict[str, Any], fastapi.Response]:
    """
    Submitting a workflow of existing project.

    :param project:         name of the project
    :param name:            name of the workflow
    :param spec:            workflow's spec. Can include only `mlrun.api.schemas.WorkflowSpec` fields.
    :param arguments:       arguments for the workflow. Overrides the arguments given in spec.
    :param artifact_path:   Target path/url for workflow artifacts, the string
                            '{{workflow.uid}}' will be replaced by workflow id
    :param source:          name (in DB) or git or tar.gz or .zip sources archive path e.g.:
                            git://github.com/mlrun/demo-xgb-project.git
                            http://mysite/archived-project.zip
                            <project-name>
                            the git project should include the project yaml file.
                            if the project yaml file is in a subdirectory, must specify the sub-directory.
    :param run_name:        name of the run object. Default to "workflow-runner-{workflow_spec.name}"
    :param namespace:       kubernetes namespace if other than default
    :param auth_info:       auth info of the request
    :param db_session:      session that manages the current dialog with the database
    :return:
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
        resource_name=run_name or "",
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

    project_workflow_names = [workflow["name"] for workflow in project.spec.workflows]

    # Taking from spec input or looking inside the project's workflows:
    if spec:
        workflow_name = spec.name or name
        spec.name = workflow_name
        if workflow_name in project_workflow_names:
            # Update with favor to the workflow's spec from the input:
            workflow = _get_workflow_by_name(project, spec.name)
            workflow_spec = copy.deepcopy(workflow)
            workflow_spec = {
                key: val or workflow_spec.get(key) for key, val in spec.dict().items()
            }
        else:
            workflow_spec = spec.dict()
    else:
        workflow_spec = _get_workflow_by_name(project, name)

    # Verifying that project has a workflow name:
    if not workflow_spec:
        log_and_raise(
            reason=f"{name} workflow not found in project",
        )

    workflow_spec = mlrun.projects.pipelines.WorkflowSpec.from_dict(workflow_spec)

    # Scheduling must be performed only by chief:
    if (
        workflow_spec.schedule and
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.api.schemas.ClusterizationRole.chief
    ):
        chief_client = mlrun.api.utils.clients.chief.Client()
        submit_workflow_params = {
            "spec": spec and spec.dict(),
            "arguments": arguments,
            "artifact_path": artifact_path,
            "source": source,
            "run_name": run_name,
            "namespace": namespace,
        }
        return chief_client.submit_workflow(
            project=project.metadata.name, name=name, json=submit_workflow_params
        )
    # Preparing inputs for load_and_run function:
    if source:
        project.spec.source = source

    if arguments:
        if not workflow_spec.args:
            workflow_spec.args = {}
        workflow_spec.args.update(arguments)

    # Creating the load and run function in the server-side way:
    load_and_run_fn = mlrun.new_function(
        name=run_name or f"workflow-runner-{workflow_spec.name}",
        project=project.metadata.name,
        kind="job",
        image=mlrun.mlconf.default_base_image,
    )

    run_db = get_run_db_instance(db_session)

    try:
        # Setting a connection between the function and the DB:
        load_and_run_fn.set_db_connection(run_db)

        # Setting "generate" for the function's access key:
        load_and_run_fn.metadata.credentials.access_key = "$generate"

        # Enrichment + validation to function:
        apply_enrichment_and_validation_on_function(
            function=load_and_run_fn,
            auth_info=auth_info,
        )
        load_and_run_fn.save()

        logger.info(f"Fn:\n{load_and_run_fn.to_yaml()}")

        if workflow_spec.schedule:

            meta_uid = uuid.uuid4().hex

            # creating runspec for scheduling:
            run_object_kwargs = {
                "spec": {
                    "scrape_metrics": config.scrape_metrics,
                    "output_path": (artifact_path or config.artifact_path).replace(
                        "{{run.uid}}", meta_uid
                    ),
                },
                "metadata": {"uid": meta_uid, "project": project.metadata.name},
            }

            runspec = mlrun.projects.pipelines._create_run_object_for_workflow_runner(
                project=project,
                workflow_spec=workflow_spec,
                artifact_path=artifact_path,
                namespace=namespace,
                **run_object_kwargs,
            )

            # Creating scheduled object:
            scheduled_object = {
                "task": runspec.to_dict(),
                "schedule": workflow_spec.schedule,
            }

            # Creating schedule:
            get_scheduler().create_schedule(
                db_session=db_session,
                auth_info=auth_info,
                project=project.metadata.name,
                name=load_and_run_fn.metadata.name,
                kind=mlrun.api.schemas.ScheduleKinds.job,
                scheduled_object=scheduled_object,
                cron_trigger=workflow_spec.schedule,
                labels=load_and_run_fn.metadata.labels,
            )

            return {
                "project": project.metadata.name,
                "name": workflow_spec.name,
                "status": "created",
                "schedule": workflow_spec.schedule,
            }

        else:
            # Running workflow from the remote engine:
            run_status = mlrun.projects.pipelines._RemoteRunner.run(
                project=project,
                workflow_spec=workflow_spec,
                name=name,
                workflow_handler=workflow_spec.handler,
                artifact_path=artifact_path,
                namespace=namespace,
                api_function=load_and_run_fn,
            )
            # if run_status:
            #     run: mlrun.RunObject = run_status.run_object
            #     workflow_id = _get_workflow_id(run, run_db)

            return {
                "project": project.metadata.name,
                "name": workflow_spec.name,
                "status": run_status.state,
                "run_id": run_status.run_id,
            }

    except Exception as err:
        logger.error(traceback.format_exc())
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason=f"runtime error: {err}")


@router.get(
    "/projects/{project}/{uid}"
)
def get_workflow_id(
    project: str,
    uid: str,
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(mlrun.api.api.deps.authenticate_request),
    db_session: Session = fastapi.Depends(mlrun.api.api.deps.get_db_session),
):
    # Check permission READ run:
    mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.run,
        project,
        uid,
        mlrun.api.schemas.AuthorizationAction.read,
        auth_info,
    )
    # Reading run:
    run_db = get_run_db_instance(db_session)
    run = run_db.read_run(uid=uid, project=project)
    # To get the workflow id, we need to use session.commit() for getting the updated results.
    # db_session.commit()
    run_object = mlrun.RunObject.from_dict(run)
    workflow_id = run_object.status.results.get("workflow_id", None)
    state = run_object.state()
    return {
        "response": {
            "workflow_id": workflow_id,
            "state": state
            }
    }
