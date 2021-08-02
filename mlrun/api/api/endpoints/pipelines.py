import ast
import typing
from datetime import datetime
from http import HTTPStatus

from fastapi import APIRouter, Depends, Query, Request
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun.api.crud
import mlrun.api.schemas
import mlrun.api.utils.clients.opa
import mlrun.errors
from mlrun.api.api import deps
from mlrun.api.api.utils import log_and_raise
from mlrun.config import config
from mlrun.k8s_utils import get_k8s_helper

router = APIRouter()


@router.get(
    "/projects/{project}/pipelines", response_model=mlrun.api.schemas.PipelinesOutput
)
def list_pipelines(
    project: str,
    namespace: str = config.namespace,
    sort_by: str = "",
    page_token: str = "",
    filter_: str = Query("", alias="filter"),
    format_: mlrun.api.schemas.PipelinesFormat = Query(
        mlrun.api.schemas.PipelinesFormat.metadata_only, alias="format"
    ),
    page_size: int = Query(None, gt=0, le=200),
    auth_verifier: mlrun.api.api.deps.AuthVerifier = Depends(
        mlrun.api.api.deps.AuthVerifier
    ),
):
    total_size, next_page_token, runs = None, None, []
    if get_k8s_helper(silent=True).is_running_inside_kubernetes_cluster():
        # we need to resolve the project from the returned run for the opa enforcement (project query param might be
        # "*", so we can't really get back only the names here
        computed_format = (
            mlrun.api.schemas.PipelinesFormat.metadata_only
            if format_ == mlrun.api.schemas.PipelinesFormat.name_only
            else format_
        )
        total_size, next_page_token, runs = mlrun.api.crud.Pipelines().list_pipelines(
            project,
            namespace,
            sort_by,
            page_token,
            filter_,
            computed_format,
            page_size,
        )
    allowed_runs = mlrun.api.utils.clients.opa.Client().filter_resources_by_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.pipeline,
        runs,
        lambda run: (run["project"], run["id"],),
        auth_verifier.auth_info,
    )
    if format_ == mlrun.api.schemas.PipelinesFormat.name_only:
        allowed_runs = [run["name"] for run in allowed_runs]
    return mlrun.api.schemas.PipelinesOutput(
        runs=allowed_runs,
        total_size=total_size or 0,
        next_page_token=next_page_token or None,
    )


# curl -d@/path/to/pipe.yaml http://localhost:8080/submit_pipeline
@router.post("/submit_pipeline")
@router.post("/submit_pipeline/")
# TODO: remove when 0.6.6 is no longer relevant
async def submit_pipeline_legacy(
    request: Request,
    namespace: str = config.namespace,
    experiment_name: str = Query("Default", alias="experiment"),
    run_name: str = Query("", alias="run"),
    auth_verifier: mlrun.api.api.deps.AuthVerifier = Depends(
        mlrun.api.api.deps.AuthVerifier
    ),
):
    response = await _submit_pipeline(
        auth_verifier.auth_info,
        request,
        namespace,
        experiment_name,
        run_name,
        allow_without_project=True,
    )
    return response


@router.post("/projects/{project}/pipelines")
async def create_pipeline(
    project: str,
    request: Request,
    namespace: str = config.namespace,
    experiment_name: str = Query("Default", alias="experiment"),
    run_name: str = Query("", alias="run"),
    auth_verifier: mlrun.api.api.deps.AuthVerifier = Depends(
        mlrun.api.api.deps.AuthVerifier
    ),
):
    response = await _submit_pipeline(
        auth_verifier.auth_info, request, namespace, experiment_name, run_name, project
    )
    return response


async def _submit_pipeline(
    auth_info: mlrun.api.schemas.AuthInfo,
    request: Request,
    namespace: str,
    experiment_name: str,
    run_name: str,
    project: typing.Optional[str] = None,
    allow_without_project: bool = False,
):
    if not project and not allow_without_project:
        raise mlrun.errors.MLRunInvalidArgumentError("Project must be provided")
    if project:
        await run_in_threadpool(
            mlrun.api.utils.clients.opa.Client().query_resource_permissions,
            mlrun.api.schemas.AuthorizationResourceTypes.pipeline,
            project,
            "",
            mlrun.api.schemas.AuthorizationAction.create,
            auth_info,
        )
    run_name = run_name or experiment_name + " " + datetime.now().strftime(
        "%Y-%m-%d %H-%M-%S"
    )

    data = await request.body()
    if not data:
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason="Request body is empty")

    arguments = {}
    # TODO: stop reading "pipeline-arguments" header when 0.6.6 is no longer relevant
    arguments_data = request.headers.get("pipeline-arguments") or request.headers.get(
        mlrun.api.schemas.HeaderNames.pipeline_arguments
    )
    if arguments_data:
        arguments = ast.literal_eval(arguments_data)

    content_type = request.headers.get("content-type", "")
    run = await run_in_threadpool(
        mlrun.api.crud.Pipelines().create_pipeline,
        experiment_name,
        run_name,
        content_type,
        data,
        arguments,
        namespace,
    )

    return {
        "id": run.id,
        "name": run.name,
    }


# curl http://localhost:8080/pipelines/:id
@router.get("/pipelines/{run_id}")
@router.get("/pipelines/{run_id}/")
# TODO: remove when 0.6.6 is no longer relevant
def get_pipeline_legacy(
    run_id: str,
    namespace: str = Query(config.namespace),
    db_session: Session = Depends(deps.get_db_session),
):
    return mlrun.api.crud.Pipelines().get_pipeline(
        db_session, run_id, namespace=namespace
    )


@router.get("/projects/{project}/pipelines/{run_id}")
def get_pipeline(
    run_id: str,
    project: str,
    namespace: str = Query(config.namespace),
    format_: mlrun.api.schemas.PipelinesFormat = Query(
        mlrun.api.schemas.PipelinesFormat.summary, alias="format"
    ),
    db_session: Session = Depends(deps.get_db_session),
):
    return mlrun.api.crud.Pipelines().get_pipeline(
        db_session, run_id, project, namespace, format_
    )
