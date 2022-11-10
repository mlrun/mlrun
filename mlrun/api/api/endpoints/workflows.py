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
from typing import Any, Callable, Dict, Optional, Union

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
    response_model=mlrun.api.schemas.SubmitWorkflowResponse,
)
def submit_workflow(
    project: str,
    name: str,
    request: Optional[mlrun.api.schemas.SubmitWorkflowRequest] = None,
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
    db_session: Session = fastapi.Depends(mlrun.api.api.deps.get_db_session),
) -> Union[Dict[str, Any], fastapi.Response]:
    """
    Submitting a workflow of existing project.
    todo: add full flow description
    :param project:         name of the project
    :param name:            name of the workflow
    :param request:         the request includes:
                                - workflow spec
                                - arguments for the workflow
                                - artifact path as the artifact target path of the workflow
                                - source url of the project for overriding the existing one
                                - run name to override the default: 'workflow-runner-<workflow name>'
                                - kubernetes namespace if other than default
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

    request = request or mlrun.api.schemas.SubmitWorkflowRequest()
    (
        spec,
        arguments,
        artifact_path,
        source,
        run_name,
        namespace,
    ) = request.dict().values()
    spec = spec or mlrun.api.schemas.WorkflowSpec()
    spec = mlrun.api.schemas.WorkflowSpec.parse_obj(spec)

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
            workflow_spec = None
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
        workflow_spec.schedule
        and mlrun.mlconf.httpdb.clusterization.role
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
    # Preparing inputs for load_and_run function
    # 1. To override the source of the project.
    # This is mainly for supporting loading project from a certain commits (GitHub)
    if source:
        project.spec.source = source

    # 2. Overriding arguments of the existing workflow:
    if arguments:
        if not workflow_spec.args:
            workflow_spec.args = {}
        workflow_spec.args.update(arguments)

    # This function is for loading the project and running workflow remotely.
    # In this way we support scheduling workflows (by scheduling a job that runs the workflow
    load_and_run_fn = mlrun.new_function(
        name=run_name or f"workflow-runner-{workflow_spec.name}",
        project=project.metadata.name,
        kind="job",
        image=mlrun.mlconf.default_base_image,  # To prevent deploy
    )

    run_db = get_run_db_instance(db_session)

    try:
        # Setting a connection between the function and the DB:
        load_and_run_fn.set_db_connection(run_db)

        # Enrichment and validation requires an access key.
        # By using `$generate` the enriching process will generate an access key for this function.
        load_and_run_fn.metadata.credentials.access_key = "$generate"
        apply_enrichment_and_validation_on_function(
            function=load_and_run_fn,
            auth_info=auth_info,
        )
        load_and_run_fn.save()
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

            # Why do we need here to create uid?
            # update run metadata (uid, labels) and store in DB
            # This logic follows the one is performed in `BaseRuntime._enrich_run()`
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

            runspec = _create_run_object_for_workflow_runner(
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
                "status": "scheduled",
                "schedule": workflow_spec.schedule,
            }

        else:
            print("1" * 100)
            runspec = _create_run_object_for_workflow_runner(
                project=project,
                workflow_spec=workflow_spec,
                artifact_path=artifact_path,
                namespace=namespace,
                workflow_name=workflow_spec.name,
                workflow_handler=workflow_spec.handler,
            )
            print("2" * 100)
            run = load_and_run_fn.run(
                runspec=runspec,
                local=False,
                schedule=workflow_spec.schedule,
                artifact_path=artifact_path,
            )
            print("3" * 100)
            state = mlrun.run.RunStatuses.running
            # Running workflow from the remote engine:
            # run_status = mlrun.projects.pipelines._RemoteRunner.run(
            #     project=project,
            #     workflow_spec=workflow_spec,
            #     name=name,
            #     workflow_handler=workflow_spec.handler,
            #     artifact_path=artifact_path,
            #     namespace=namespace,
            #     api_function=load_and_run_fn,
            # )

            return {
                "project": project.metadata.name,
                "name": workflow_spec.name,
                "status": state,
                "run_id": run.uid(),
            }

    except Exception as err:
        logger.error(traceback.format_exc())
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason=f"runtime error: {err}")


@router.get(
    "/projects/{project}/{uid}", response_model=mlrun.api.schemas.GetWorkflowResponse
)
def get_workflow_id(
    project: str,
    uid: str,
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
    db_session: Session = fastapi.Depends(mlrun.api.api.deps.get_db_session),
):
    """
    Retrieve workflow id by the uid of the runner.
    Supporting workflows that executed by the remote engine **only**.
    :param project:     name of the project
    :param uid:         the id of the running job that runs the workflow
    :param auth_info:   auth info of the request
    :param db_session:  session that manages the current dialog with the database
    :return:
    """
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

    run_object = mlrun.RunObject.from_dict(run)
    workflow_id = run_object.status.results.get("workflow_id", None)

    # Check permission READ run:
    mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.run,
        project,
        workflow_id,
        mlrun.api.schemas.AuthorizationAction.read,
        auth_info,
    )
    engine = run_object.status.results["engine"]
    engine = mlrun.projects.pipelines.get_workflow_engine(engine)
    status = engine.get_state(workflow_id, project)
    # status = ""
    # if engine == "kfp":
    #     # Getting workflow state for kfp:
    #     pipeline = mlrun.api.crud.Pipelines().get_pipeline(
    #         db_session=db_session, run_id=workflow_id, project=project
    #     )
    #     status = pipeline["run"].get("status", "")

    return {"workflow_id": workflow_id, "status": status}


# def load_and_run(
#     context,
#     url: str = None,
#     project_name: str = "",
#     init_git: bool = None,
#     subpath: str = None,
#     clone: bool = False,
#     workflow_name: str = None,
#     workflow_path: str = None,
#     workflow_arguments: Dict[str, Any] = None,
#     artifact_path: str = None,
#     workflow_handler: Union[str, Callable] = None,
#     namespace: str = None,
#     sync: bool = False,
#     dirty: bool = False,
#     ttl: int = None,
#     engine: str = None,
#     local: bool = None,
# ):
#     project = mlrun.load_project(
#         context=f"./{project_name}",
#         url=url,
#         name=project_name,
#         init_git=init_git,
#         subpath=subpath,
#         clone=clone,
#     )
#     context.logger.info(f"Loaded project {project.name} from remote successfully")
#
#     workflow_log_message = workflow_name or workflow_path
#     context.logger.info(f"Running workflow {workflow_log_message} from remote")
#     run = project.run(
#         name=workflow_name,
#         workflow_path=workflow_path,
#         arguments=workflow_arguments,
#         artifact_path=artifact_path,
#         workflow_handler=workflow_handler,
#         namespace=namespace,
#         sync=sync,
#         watch=False,  # Required for fetching the workflow_id
#         dirty=dirty,
#         ttl=ttl,
#         engine=engine,
#         local=local,
#     )
#     context.log_result(key="workflow_id", value=run.run_id)
#
#     context.log_result(key="engine", value=run._engine.engine, commit=True)


def _create_run_object_for_workflow_runner(
    project,
    workflow_spec,
    artifact_path: Optional[str] = None,
    namespace: Optional[str] = None,
    workflow_name: Optional[str] = None,
    workflow_handler: Union[str, Callable] = None,
    **kwargs,
) -> mlrun.RunObject:
    """
    Creating run object for the load_and_run function.
    :param project:             project object that matches the workflow
    :param workflow_spec:       spec of the workflow to run
    :param artifact_path:       artifact path target for the run
    :param namespace:           kubernetes namespace if other than default
    :param workflow_name:       name of the workflow to override the one in the workflow spec.
    :param workflow_handler:    handler of the workflow to override the one in the workflow spec.
    :param kwargs:              dictionary with "spec" and "metadata" keys with dictionaries as values that are
                                corresponding to the keys.
    :return:    a RunObject with the desired spec and metadata with labels.
    """
    spec_kwargs, metadata_kwargs = (
        kwargs.get("spec"),
        kwargs.get("metadata"),
    ) if kwargs else {}, {}
    spec = {
        "parameters": {
            "url": project.spec.source,
            "project_name": project.metadata.name,
            "workflow_name": workflow_name or workflow_spec.name,
            "workflow_path": workflow_spec.path,
            "workflow_arguments": workflow_spec.args,
            "artifact_path": artifact_path,
            "workflow_handler": workflow_handler or workflow_spec.handler,
            "namespace": namespace,
            "ttl": workflow_spec.ttl,
            "engine": workflow_spec.engine,
            "local": workflow_spec.run_local,
        },
        "handler": "mlrun.projects.load_and_run",
    }
    metadata = {"name": workflow_name}
    spec.update(spec_kwargs)
    metadata.update(metadata_kwargs)

    # Creating object:
    run_object = mlrun.RunObject.from_dict({"spec": spec, "metadata": metadata})

    # Setting labels:
    return run_object.set_label("job-type", "workflow-runner").set_label(
        "workflow", workflow_name or workflow_spec.name
    )
