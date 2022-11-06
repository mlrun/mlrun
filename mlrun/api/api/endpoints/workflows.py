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
from typing import Dict, Optional

import fastapi
from sqlalchemy.orm import Session

import mlrun
import mlrun.api.api.deps
import mlrun.api.api.utils
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
):
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

    # Scheduling must be performed only by chief:
    if (
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

    # Preparing inputs for load_and_run function:
    if source:
        project.spec.source = source

    if arguments:
        if not workflow_spec.args:
            workflow_spec.args = {}
        workflow_spec.args.update(arguments)

    # Creating the load and run function in the server-side way:
    load_and_run_fn = mlrun.new_function(
        name=f"workflow-runner-{workflow_spec.name}",
        project=project.metadata.name,
        kind="job",
        image=mlrun.mlconf.default_base_image,
    )

    try:
        # Setting a connection between the function and the DB:
        run_db = get_run_db_instance(db_session)
        load_and_run_fn.set_db_connection(run_db)
        # Generating an access key:
        load_and_run_fn.metadata.credentials.access_key = "$generate"

        apply_enrichment_and_validation_on_function(
            function=load_and_run_fn,
            auth_info=auth_info,
        )
        load_and_run_fn.save()

        logger.info(f"Fn:\n{load_and_run_fn.to_yaml()}")

        if workflow_spec.schedule:
            meta_uid = uuid.uuid4().hex

            # creating runspec for scheduling:
            runspec = mlrun.RunObject.from_dict(
                {
                    "spec": {
                        "function": load_and_run_fn.uri,
                        "parameters": {
                            "url": project.spec.source,
                            "project_name": project.metadata.name,
                            "workflow_name": workflow_spec.name,
                            "workflow_path": workflow_spec.path,
                            "workflow_arguments": workflow_spec.args,
                            "artifact_path": artifact_path,
                            "workflow_handler": workflow_spec.handler
                            or workflow_spec.handler,
                            "namespace": namespace,
                            "ttl": workflow_spec.ttl,
                            "engine": workflow_spec.engine,
                            "local": workflow_spec.run_local,
                        },
                        "handler": "mlrun.projects.load_and_run",
                        "scrape_metrics": config.scrape_metrics,
                        "output_path": (artifact_path or config.artifact_path).replace(
                            "{{run.uid}}", meta_uid
                        ),
                    },
                    "metadata": {
                        "name": workflow_spec.name,
                        "uid": meta_uid,
                        "project": project.metadata.name,
                    },
                }
            )

            runspec.set_label("job-type", "workflow-runner").set_label(
                "workflow", workflow_spec.name
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
            response = {
                "schedule": workflow_spec.schedule,
                "project": project.metadata.name,
                "name": load_and_run_fn.metadata.name,
            }
            return {
                "result": f"The workflow was scheduled successfully, response: {response}"
            }
        else:
            # Creating the load and run function in the server-side way:
            load_and_run_fn = mlrun.new_function(
                name=f"workflow-runner-{workflow_spec.name}",
                project=project.metadata.name,
                kind="job",
                image=mlrun.mlconf.default_base_image,
            )
            # Running workflow from the remote engine:
            run = mlrun.projects.pipelines._RemoteRunner.run(
                project=project,
                workflow_spec=workflow_spec,
                name=name,
                workflow_handler=workflow_spec.handler,
                artifact_path=artifact_path,
                namespace=namespace,
                api_function=load_and_run_fn,
            )
            # run is None for scheduled workflows:
            return {"workflow_id": run.run_id}

    except Exception as err:
        logger.error(traceback.format_exc())
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason=f"runtime error: {err}")
