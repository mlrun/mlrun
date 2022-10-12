import copy
import traceback
from http import HTTPStatus
from typing import Dict, Optional

import fastapi
from sqlalchemy.orm import Session

import mlrun
import mlrun.api.api.deps
import mlrun.api.api.utils
import mlrun.api.schemas
import mlrun.api.utils.auth.verifier
import mlrun.api.utils.singletons.project_member
import mlrun.projects.pipelines
from mlrun.utils.helpers import logger
from mlrun.api.api.utils import log_and_raise, get_run_db_instance, apply_enrichment_and_validation_on_function

router = fastapi.APIRouter()


def _get_workflow_by_name(project: mlrun.api.schemas.Project, workflow) -> Dict:
    for wf in project.spec.workflows:
        if wf["name"] == workflow:
            return wf


def print_debug(key, val):
    prefix = "<DEBUG YONI>\n"
    suffix = "\n<END DEBUG YONI>"
    msg = f"{key}: {val}\n type: {type(val)}"
    print(prefix + msg + suffix)


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
    print_debug("input workflow_name", name)  # TODO: Remove!
    # Getting project:
    project = (
        mlrun.api.utils.singletons.project_member.get_project_member().get_project(
            db_session=db_session, name=project, leader_session=auth_info.session
        )
    )
    # Debugging:
    print_debug("project_name", project.metadata.name)  # TODO: Remove!
    # Checking CREATE run permission
    mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        resource_type=mlrun.api.schemas.AuthorizationResourceTypes.run,
        project_name=project.metadata.name,
        resource_name=run_name or "",
        action=mlrun.api.schemas.AuthorizationAction.create,
        auth_info=auth_info,
    )
    # Checking READ workflow permission
    mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        resource_type=mlrun.api.schemas.AuthorizationResourceTypes.workflow,
        project_name=project.metadata.name,
        resource_name=name,
        action=mlrun.api.schemas.AuthorizationAction.read,
        auth_info=auth_info,
    )

    existing_workflows = [workflow["name"] for workflow in project.spec.workflows]

    # Taking from spec input or looking inside the project's workflows:
    if spec:
        workflow_name = spec.name or name
        spec.name = workflow_name
        if workflow_name in existing_workflows:
            # Update with favor to the workflow's spec from the input.:
            workflow = _get_workflow_by_name(project, spec.name)
            workflow_spec = copy.deepcopy(workflow)
            workflow_spec.update(spec.dict())
        else:
            workflow_spec = spec.dict()
    else:
        workflow_spec = _get_workflow_by_name(project, name)

    print_debug("project's workflows", project.spec.workflows)  # TODO: Remove!
    print_debug("main workflow", project.spec.workflows[0])  # TODO: Remove!
    # Verifying that project has a workflow name:

    if not workflow_spec:
        log_and_raise(
            reason=f"{name} workflow not found in project",
        )
    workflow_spec = mlrun.projects.pipelines.WorkflowSpec.from_dict(workflow_spec)

    print_debug("workflow_spec", workflow_spec)  # TODO: Remove!
    # Preparing inputs for _RemoteRunner.run():
    if source:
        project.spec.source = source

    if arguments:
        if not workflow_spec.args:
            workflow_spec.args = {}
        workflow_spec.args.update(arguments)

    # workflow_spec.run_local = True  # Running remotely local workflows
    print_debug("final workflow spec", workflow_spec)  # TODO: Remove!
    # Printing input for debug:  # TODO: Remove!
    print_debug("project", project)  # TODO: Remove!
    print_debug("handler", workflow_spec.handler)  # TODO: Remove!
    print_debug("artifact_path", artifact_path)  # TODO: Remove!
    print_debug("namespace", namespace)  # TODO: Remove!

    # Creating the load and run function in the server-side way:
    load_and_run_fn = mlrun.new_function(
        name=f"workflow-runner-{workflow_spec.name}",
        project=project.metadata.name,
        kind="job",
        image=mlrun.mlconf.default_base_image,
        kfp=('kfp' in workflow_spec.engine),
    )

    try:
        run_db = get_run_db_instance(db_session)
        load_and_run_fn.set_db_connection(run_db)
        load_and_run_fn.metadata.credentials.access_key = "$generate"
        apply_enrichment_and_validation_on_function(
            function=load_and_run_fn,
            auth_info=auth_info,
        )
        load_and_run_fn.save()
        logger.info(f"Fn:\n{load_and_run_fn.to_yaml()}")
    except Exception as err:
        logger.error(traceback.format_exc())
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason=f"runtime error: {err}")

    print_debug('workflow spec', workflow_spec)  # TODO: Remove!
    run = mlrun.projects.pipelines._RemoteRunner.run(
        project=project,
        workflow_spec=workflow_spec,
        name=name,
        workflow_handler=workflow_spec.handler,
        artifact_path=artifact_path,
        namespace=namespace,
        api_function=load_and_run_fn,
    )

    print_debug("run result", run)  # TODO: Remove!
    if run:
        return run.run_id
