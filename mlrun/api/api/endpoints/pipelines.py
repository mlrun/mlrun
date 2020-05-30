import ast
import tempfile
from datetime import datetime
from http import HTTPStatus
from os import remove

from fastapi import APIRouter, Request, Query
from fastapi.concurrency import run_in_threadpool
from kfp import Client as kfclient

from mlrun.api.api.utils import log_and_raise
from mlrun.config import config
from mlrun.utils import logger

router = APIRouter()


# curl -d@/path/to/pipe.yaml http://localhost:8080/submit_pipeline
@router.post("/submit_pipeline")
@router.post("/submit_pipeline/")
async def submit_pipeline(
        request: Request,
        namespace: str = config.namespace,
        experiment_name: str = Query("Default", alias="experiment"),
        run_name: str = Query("", alias="run")):
    run_name = run_name or experiment_name + " " + datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    data = await request.body()
    if not data:
        log_and_raise(HTTPStatus.BAD_REQUEST, reason="post data is empty")

    run = await run_in_threadpool(_submit_pipeline, request, data, namespace, experiment_name, run_name)

    return {
        "id": run.id,
        "name": run.name,
    }


# curl http://localhost:8080/pipelines/:id
@router.get("/pipelines/{run_id}")
@router.get("/pipelines/{run_id}/")
def get_pipeline(run_id,
                 namespace: str = Query(config.namespace)):

    client = kfclient(namespace=namespace)
    try:
        run = client.get_run(run_id)
        if run:
            run = run.to_dict()
    except Exception as e:
        log_and_raise(HTTPStatus.INTERNAL_SERVER_ERROR, reason="get kfp error: {}".format(e))

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
        log_and_raise(HTTPStatus.BAD_REQUEST, reason="unsupported pipeline type {}".format(ctype))

    logger.info("writing file {}".format(ctype))

    print(str(data))
    pipe_tmp = tempfile.mktemp(suffix=ctype)
    with open(pipe_tmp, "wb") as fp:
        fp.write(data)

    run = None
    try:
        client = kfclient(namespace=namespace)
        experiment = client.create_experiment(name=experiment_name)
        run = client.run_pipeline(experiment.id, run_name, pipe_tmp,
                                  params=arguments)
    except Exception as e:
        remove(pipe_tmp)
        log_and_raise(HTTPStatus.BAD_REQUEST, reason="kfp err: {}".format(e))

    remove(pipe_tmp)

    return run
