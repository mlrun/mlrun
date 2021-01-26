import ast
import tempfile
from datetime import datetime
from http import HTTPStatus
from os import remove

from fastapi import APIRouter, Request, Query
from fastapi.concurrency import run_in_threadpool
from kfp import Client as kfclient

import mlrun.api.crud
import mlrun.api.schemas
from mlrun.api.api.utils import log_and_raise
from mlrun.config import config
from mlrun.k8s_utils import get_k8s_helper
from mlrun.utils import logger

router = APIRouter()


@router.get(
    "/projects/{project}/pipelines", response_model=mlrun.api.schemas.PipelinesOutput
)
def list_pipelines(
    project: str,
    namespace: str = None,
    sort_by: str = "",
    page_token: str = "",
    filter_: str = Query("", alias="filter"),
    format_: mlrun.api.schemas.Format = Query(
        mlrun.api.schemas.Format.metadata_only, alias="format"
    ),
    page_size: int = Query(None, gt=0, le=200),
):
    total_size, next_page_token, runs = None, None, None
    if get_k8s_helper(silent=True).is_running_inside_kubernetes_cluster():
        total_size, next_page_token, runs = mlrun.api.crud.list_pipelines(
            project, namespace, sort_by, page_token, filter_, format_, page_size,
        )
    return mlrun.api.schemas.PipelinesOutput(
        runs=runs or [],
        total_size=total_size or 0,
        next_page_token=next_page_token or None,
    )


# curl -d@/path/to/pipe.yaml http://localhost:8080/submit_pipeline
@router.post("/submit_pipeline")
@router.post("/submit_pipeline/")
async def submit_pipeline(
    request: Request,
    namespace: str = config.namespace,
    experiment_name: str = Query("Default", alias="experiment"),
    run_name: str = Query("", alias="run"),
):
    run_name = run_name or experiment_name + " " + datetime.now().strftime(
        "%Y-%m-%d %H-%M-%S"
    )

    data = await request.body()
    if not data:
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason="post data is empty")

    run = await run_in_threadpool(
        _submit_pipeline, request, data, namespace, experiment_name, run_name
    )

    return {
        "id": run.id,
        "name": run.name,
    }


# curl http://localhost:8080/pipelines/:id
@router.get("/pipelines/{run_id}")
@router.get("/pipelines/{run_id}/")
def get_pipeline(run_id, namespace: str = Query(config.namespace)):

    client = kfclient(namespace=namespace)
    try:
        run = client.get_run(run_id)
        if run:
            run = run.to_dict()
    except Exception as e:
        log_and_raise(
            HTTPStatus.INTERNAL_SERVER_ERROR.value, reason="get kfp error: {}".format(e)
        )

    return run


def _submit_pipeline(request, data, namespace, experiment_name, run_name):
    arguments = {}
    arguments_data = request.headers.get("pipeline-arguments")
    if arguments_data:
        arguments = ast.literal_eval(arguments_data)
        logger.info("pipeline arguments {}".format(arguments_data))

    ctype = request.headers.get("content-type", "")
    if "/yaml" in ctype:
        ctype = ".yaml"
    elif " /zip" in ctype:
        ctype = ".zip"
    else:
        log_and_raise(
            HTTPStatus.BAD_REQUEST.value,
            reason="unsupported pipeline type {}".format(ctype),
        )

    logger.info("writing file {}".format(ctype))

    print(str(data))
    pipe_tmp = tempfile.mktemp(suffix=ctype)
    with open(pipe_tmp, "wb") as fp:
        fp.write(data)

    run = None
    try:
        client = kfclient(namespace=namespace)
        experiment = client.create_experiment(name=experiment_name)
        run = client.run_pipeline(experiment.id, run_name, pipe_tmp, params=arguments)
    except Exception as e:
        remove(pipe_tmp)
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason="kfp err: {}".format(e))

    remove(pipe_tmp)

    return run
