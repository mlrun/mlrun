import mlrun
import mlrun.api.schemas
import mlrun.api.api.deps
import mlrun.api.api.utils
import mlrun.api.utils.singletons.project_member
import mlrun.api.utils.auth.verifier
import mlrun.projects.pipelines
from typing import Dict, Optional

import fastapi
from sqlalchemy.orm import Session

router = fastapi.APIRouter()


@router.post(
    "/projects/{project}/workflows/{name}/submit",
    response_model=mlrun.api.schemas.PipelinesOutput,
)
def submit_workflow(
    project: str,
    name: str,
    spec: mlrun.api.schemas.WorkflowSpec,
    arguments: Dict,
    artifact_path: Optional[str] = None,
    source: Optional[str] = None,
    run_name: Optional[str] = None,
    namespace: Optional[str] = None,
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(mlrun.api.api.deps.authenticate_request),
    db_session: Session = fastapi.Depends(mlrun.api.api.deps.get_db_session),
):
    # Getting project:
    project = mlrun.api.utils.singletons.project_member.get_project_member().get_project(
        db_session=db_session,
        name=project,
        leader_session=auth_info.session
    )

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
        resource_name=name or spec.name,  # todo: seem to be unnecessary
        action=mlrun.api.schemas.AuthorizationAction.read,
        auth_info=auth_info,
    )

    # Verifying that project has a workflow name:
    workflow_names = [workflow.name for workflow in project.spec.workflows]
    workflow_name = name or spec.name  # todo: seem to be unnecessary
    if workflow_name not in workflow_names:
        raise mlrun.errors.MLRunInvalidArgumentError(f'{workflow_name} workflow not found in project')

    # Preparing inputs for _RemoteRunner.run():
    if source:
        project.spec.source = source
    if arguments:
        if spec.args is None:
            spec.args = {}
        else:
            spec.args.update(arguments)

    workflow_spec = mlrun.projects.pipelines.WorkflowSpec.from_dict(spec.dict())

    workflow_spec.run_local = True  # Running remotely local workflows

    run = mlrun.projects.pipelines._RemoteRunner.run(
        project=project,
        workflow_spec=workflow_spec,
        name=workflow_name,
        workflow_handler=spec.handler,
        artifact_path=artifact_path,
        namespace=namespace,
        db_session=mlrun.api.api.utils.get_run_db_instance(db_session),
    )
    return run.run_id


# questions:
# 1. which arguments to pass?
# 2. which arguments are optional / obligatory ?
# 3. which permission checks to do?
# 4. path of router discussion
# put in the run object status the run id.
