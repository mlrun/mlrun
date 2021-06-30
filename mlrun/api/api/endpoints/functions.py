import traceback
from distutils.util import strtobool
from http import HTTPStatus
from typing import List, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Cookie,
    Depends,
    Query,
    Request,
    Response,
)
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun.api.db.session
import mlrun.api.schemas
import mlrun.api.utils.background_tasks
from mlrun.api.api import deps
from mlrun.api.api.utils import get_run_db_instance, log_and_raise
from mlrun.api.utils.singletons.db import get_db
from mlrun.api.utils.singletons.k8s import get_k8s
from mlrun.builder import build_runtime
from mlrun.config import config
from mlrun.run import new_function
from mlrun.runtimes import RuntimeKinds, runtime_resources_map
from mlrun.runtimes.function import deploy_nuclio_function, get_nuclio_deploy_status
from mlrun.utils import get_in, logger, parse_versioned_object_uri, update_in

router = APIRouter()


# curl -d@/path/to/func.json http://localhost:8080/func/prj/7?tag=0.3.2
@router.post("/func/{project}/{name}")
async def store_function(
    request: Request,
    project: str,
    name: str,
    tag: str = "",
    versioned: bool = False,
    iguazio_session: Optional[str] = Cookie(None, alias="session"),
    db_session: Session = Depends(deps.get_db_session),
):
    data = None
    try:
        data = await request.json()
    except ValueError:
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason="bad JSON body")

    logger.debug(data)
    logger.info("store function: project=%s, name=%s, tag=%s", project, name, tag)
    hash_key = await run_in_threadpool(
        get_db().store_function,
        db_session,
        data,
        name,
        project,
        tag=tag,
        versioned=versioned,
        leader_session=iguazio_session,
    )
    return {
        "hash_key": hash_key,
    }


# curl http://localhost:8080/log/prj/7?tag=0.2.3
@router.get("/func/{project}/{name}")
def get_function(
    project: str,
    name: str,
    tag: str = "",
    hash_key="",
    db_session: Session = Depends(deps.get_db_session),
):
    func = get_db().get_function(db_session, name, project, tag, hash_key)
    return {
        "func": func,
    }


@router.delete(
    "/projects/{project}/functions/{name}", status_code=HTTPStatus.NO_CONTENT.value
)
def delete_function(
    project: str, name: str, db_session: Session = Depends(deps.get_db_session),
):
    get_db().delete_function(db_session, project, name)
    return Response(status_code=HTTPStatus.NO_CONTENT.value)


# curl http://localhost:8080/funcs?project=p1&name=x&label=l1&label=l2
@router.get("/funcs")
def list_functions(
    project: str = config.default_project,
    name: str = None,
    tag: str = None,
    labels: List[str] = Query([], alias="label"),
    db_session: Session = Depends(deps.get_db_session),
):
    funcs = get_db().list_functions(db_session, name, project, tag, labels)
    return {
        "funcs": list(funcs),
    }


# curl -d@/path/to/job.json http://localhost:8080/build/function
@router.post("/build/function")
@router.post("/build/function/")
async def build_function(
    request: Request,
    iguazio_session: Optional[str] = Cookie(None, alias="session"),
    db_session: Session = Depends(deps.get_db_session),
):
    data = None
    try:
        data = await request.json()
    except ValueError:
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason="bad JSON body")

    logger.info(f"build_function:\n{data}")
    function = data.get("function")
    with_mlrun = strtobool(data.get("with_mlrun", "on"))
    skip_deployed = data.get("skip_deployed", False)
    mlrun_version_specifier = data.get("mlrun_version_specifier")
    fn, ready = await run_in_threadpool(
        _build_function,
        db_session,
        function,
        with_mlrun,
        skip_deployed,
        mlrun_version_specifier,
        iguazio_session,
    )
    return {
        "data": fn.to_dict(),
        "ready": ready,
    }


# curl -d@/path/to/job.json http://localhost:8080/start/function
@router.post("/start/function", response_model=mlrun.api.schemas.BackgroundTask)
@router.post("/start/function/", response_model=mlrun.api.schemas.BackgroundTask)
async def start_function(
    request: Request,
    background_tasks: BackgroundTasks,
    iguazio_session: Optional[str] = Cookie(None, alias="session"),
    db_session: Session = Depends(deps.get_db_session),
):
    data = None
    try:
        data = await request.json()
    except ValueError:
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason="bad JSON body")

    logger.info("Got request to start function", body=data)

    function = await run_in_threadpool(_parse_start_function_body, db_session, data)

    background_task = await run_in_threadpool(
        mlrun.api.utils.background_tasks.Handler().create_background_task,
        db_session,
        iguazio_session,
        function.metadata.project,
        background_tasks,
        _start_function,
        function,
        iguazio_session,
    )

    return background_task


# curl -d@/path/to/job.json http://localhost:8080/status/function
@router.post("/status/function")
@router.post("/status/function/")
async def function_status(request: Request):
    data = None
    try:
        data = await request.json()
    except ValueError:
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason="bad JSON body")

    resp = await run_in_threadpool(_get_function_status, data)
    return {
        "data": resp,
    }


# curl -d@/path/to/job.json http://localhost:8080/build/status
@router.get("/build/status")
@router.get("/build/status/")
def build_status(
    name: str = "",
    project: str = "",
    tag: str = "",
    offset: int = 0,
    logs: bool = True,
    last_log_timestamp: float = 0.0,
    verbose: bool = False,
    iguazio_session: Optional[str] = Cookie(None, alias="session"),
    db_session: Session = Depends(deps.get_db_session),
):
    fn = get_db().get_function(db_session, name, project, tag)
    if not fn:
        log_and_raise(HTTPStatus.NOT_FOUND.value, name=name, project=project, tag=tag)

    # nuclio deploy status
    if fn.get("kind") in RuntimeKinds.nuclio_runtimes():
        (
            state,
            address,
            nuclio_name,
            last_log_timestamp,
            text,
        ) = get_nuclio_deploy_status(
            name, project, tag, last_log_timestamp=last_log_timestamp, verbose=verbose
        )
        if state == "ready":
            logger.info("Nuclio function deployed successfully", name=name)
        if state == "error":
            logger.error(f"Nuclio deploy error, {text}", name=name)
        update_in(fn, "status.nuclio_name", nuclio_name)
        update_in(fn, "status.state", state)
        update_in(fn, "status.address", address)

        versioned = False
        if state == "ready":
            # Versioned means the version will be saved in the DB forever, we don't want to spam
            # the DB with intermediate or unusable versions, only successfully deployed versions
            versioned = True
        get_db().store_function(
            db_session,
            fn,
            name,
            project,
            tag,
            versioned=versioned,
            leader_session=iguazio_session,
        )
        return Response(
            content=text,
            media_type="text/plain",
            headers={
                "x-mlrun-function-status": state,
                "x-mlrun-last-timestamp": str(last_log_timestamp),
                "x-mlrun-address": address,
                "x-mlrun-name": nuclio_name,
            },
        )

    # job deploy status
    state = get_in(fn, "status.state", "")
    pod = get_in(fn, "status.build_pod", "")
    image = get_in(fn, "spec.build.image", "")
    out = b""
    if not pod:
        if state == "ready":
            image = image or get_in(fn, "spec.image")
        return Response(
            content=out,
            media_type="text/plain",
            headers={
                "function_status": state,
                "function_image": image,
                "builder_pod": pod,
            },
        )

    logger.info(f"get pod {pod} status")
    state = get_k8s().get_pod_status(pod)
    logger.info(f"pod state={state}")

    if state == "succeeded":
        logger.info("build completed successfully")
        state = mlrun.api.schemas.FunctionState.ready
    if state in ["failed", "error"]:
        logger.error(f"build {state}, watch the build pod logs: {pod}")
        state = mlrun.api.schemas.FunctionState.error

    if logs and state != "pending":
        resp = get_k8s().logs(pod)
        if resp:
            out = resp.encode()[offset:]

    update_in(fn, "status.state", state)
    if state == mlrun.api.schemas.FunctionState.ready:
        update_in(fn, "spec.image", image)

    versioned = False
    if state == mlrun.api.schemas.FunctionState.ready:
        versioned = True
    get_db().store_function(
        db_session,
        fn,
        name,
        project,
        tag,
        versioned=versioned,
        leader_session=iguazio_session,
    )

    return Response(
        content=out,
        media_type="text/plain",
        headers={
            "x-mlrun-function-status": state,
            "function_status": state,
            "function_image": image,
            "builder_pod": pod,
        },
    )


def _build_function(
    db_session,
    function,
    with_mlrun,
    skip_deployed,
    mlrun_version_specifier,
    leader_session,
):
    fn = None
    ready = None
    try:
        fn = new_function(runtime=function)

        run_db = get_run_db_instance(db_session, leader_session)
        fn.set_db_connection(run_db)
        fn.save(versioned=False)
        if fn.kind in RuntimeKinds.nuclio_runtimes():
            deploy_nuclio_function(fn)
            # deploy only start the process, the get status API is used to check readiness
            ready = False
        else:
            ready = build_runtime(
                fn, with_mlrun, mlrun_version_specifier, skip_deployed
            )
        fn.save(versioned=True)
        logger.info("Fn:\n %s", fn.to_yaml())
    except Exception as err:
        logger.error(traceback.format_exc())
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason=f"runtime error: {err}")
    return fn, ready


def _parse_start_function_body(db_session, data):
    url = data.get("functionUrl")
    if not url:
        log_and_raise(
            HTTPStatus.BAD_REQUEST.value,
            reason="runtime error: functionUrl not specified",
        )

    project, name, tag, hash_key = parse_versioned_object_uri(url)
    runtime = get_db().get_function(db_session, name, project, tag, hash_key)
    if not runtime:
        log_and_raise(
            HTTPStatus.BAD_REQUEST.value,
            reason=f"runtime error: function {url} not found",
        )

    return new_function(runtime=runtime)


def _start_function(function, leader_session: Optional[str] = None):
    db_session = mlrun.api.db.session.create_session()
    try:
        resource = runtime_resources_map.get(function.kind)
        if "start" not in resource:
            log_and_raise(
                HTTPStatus.BAD_REQUEST.value,
                reason="runtime error: 'start' not supported by this runtime",
            )
        try:
            run_db = get_run_db_instance(db_session, leader_session)
            function.set_db_connection(run_db)
            #  resp = resource["start"](fn)  # TODO: handle resp?
            resource["start"](function)
            function.save(versioned=False)
            logger.info("Fn:\n %s", function.to_yaml())
        except Exception as err:
            logger.error(traceback.format_exc())
            log_and_raise(HTTPStatus.BAD_REQUEST.value, reason=f"runtime error: {err}")
    finally:
        mlrun.api.db.session.close_session(db_session)


def _get_function_status(data):
    logger.info(f"function_status:\n{data}")
    selector = data.get("selector")
    kind = data.get("kind")
    if not selector or not kind:
        log_and_raise(
            HTTPStatus.BAD_REQUEST.value,
            reason="runtime error: selector or runtime kind not specified",
        )

    resource = runtime_resources_map.get(kind)
    if "status" not in resource:
        log_and_raise(
            HTTPStatus.BAD_REQUEST.value,
            reason="runtime error: 'status' not supported by this runtime",
        )

    resp = None
    try:
        resp = resource["status"](selector)
        logger.info("status: %s", resp)
    except Exception as err:
        logger.error(traceback.format_exc())
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason=f"runtime error: {err}")
