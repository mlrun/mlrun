import ast
import asyncio
import tempfile
from datetime import datetime
from http import HTTPStatus
from os import remove

from fastapi import APIRouter, Request, Query
from kfp import Client as kfclient

from mlrun.api.api.utils import log_and_raise
from mlrun.config import config
from mlrun.utils import logger

router = APIRouter()


# curl -d@/path/to/pipe.yaml http://localhost:8080/submit_pipeline
@router.post("/submit_pipeline")
@router.post("/submit_pipeline/")
def submit_pipeline(
        request: Request,
        namespace: str = config.namespace,
        experiment_name: str = Query("Default", alias="experiment"),
        run_name: str = Query("", alias="run")):
    run_name = run_name or experiment_name + " " + datetime.now().strftime("%Y-%m-%d %H-%M-%S")

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
    data = asyncio.run(request.body())
    if not data:
        log_and_raise(HTTPStatus.BAD_REQUEST, reason="post data is empty")

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
    return {
        "id": run.id,
        "name": run.name,
    }


# curl http://localhost:8080/pipelines/:id
@router.get("/pipelines/{run_id}")
@router.get("/pipelines/{run_id}/")
def get_pipeline(run_id,
                 namespace: str = Query(config.namespace),
                 wait: int = Query(None)):

    client = kfclient(namespace=namespace)
    try:
        if wait:
            run = client.wait_for_run_completion(run_id, wait)
        else:
            run = client.get_run(run_id)
    except Exception as e:
        log_and_raise(HTTPStatus.INTERNAL_SERVER_ERROR, reason="get kfp error: {}".format(e))

    return run
